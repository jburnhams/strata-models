"""Jobs sent through a vendor's batch API (the Claude Message Batches API: half the price of the same calls made at once, answered within
24 hours, most within one), for bulk work that can wait: an archive of photos to caption, an eval's candidate run.

An app asks for it per job: `batch = true` in the params of a job on a model whose runtime has a batch API (claude-api). Nothing else
changes for the app: it submits jobs and waits for each as it always does, and each job ends done or failed with the same result as a call
made at once (plus `batch`, the vendor's id). On a model without a batch API the flag is dropped and the job runs at once. What to batch,
how many, and what to do with an entry that failed (retry it at once, or later) stay with the app: this only sends the calls cheaper.

The batcher (one per scheduler, stepped on its own thread) takes the queued jobs that asked for it out of the CPU lane and:
  submit  once the oldest queued batch job of a model has waited `wait_s` (so a stream of submissions becomes one batch), or `max_jobs` of
          them are queued: each job's request is built as the runtime would send it, its worst case (at the batch price) held of the budget
          in one check (spend.reserve_many: the jobs that do not fit fail with OverBudget, the rest go), the batch row written (api_batches),
          the batch sent; the jobs are running from then on, their `note` naming the batch. A batch stays under `max_bytes` of requests
          (the API takes 256 MB) and `max_jobs` (it takes 100 000); what is left over waits for the next.
  poll    each open batch every `poll_s`; once it has ended, every entry's outcome is recorded in the ledger at the batch price (spend.record,
          freeing its hold) and its job finished. An entry that errored, expired or was cancelled fails its job (nothing is billed for it).
  resume  after a restart: open batches keep their jobs running (Store.recover leaves them), and their holds are taken again (spend.hold,
          no check: the calls are made). A batch whose row was written but that was never confirmed sent fails its jobs.

The runtime side is the `batch_*` functions of runtimes/claude.py: batch_request, batch_submit, batch_poll, batch_results, batch_answer."""

import json

from stratamodels import runtimes, spend
from stratamodels.store import FINISHED

BATCH = "batch"  # a job's params key: send it through the vendor's batch API
PREFIX = "job-"  # an entry's custom_id is PREFIX + the job id


def wants(params):
    return bool((params or {}).get(BATCH))


def clean(params):
    """A batch job's params as the runtime reads them (without the scheduler's own keys)."""
    from stratamodels.scheduler import AFTER, OPTIONS

    return {k: v for k, v in (params or {}).items() if k not in (BATCH, OPTIONS, AFTER)}


class Batcher:
    def __init__(self, scheduler, wait_s=60.0, poll_s=60.0, max_jobs=10_000, max_bytes=200 * 1024 * 1024):
        self.s = scheduler
        self.wait_s, self.poll_s, self.max_jobs, self.max_bytes = (
            float(wait_s),
            float(poll_s),
            int(max_jobs),
            int(max_bytes),
        )
        self.handles = {}  # model key -> (settings hash, handle): the batcher's own client, apart from the CPU lane's
        self.tickets = {}  # job id -> its spend hold, until its outcome is recorded

    @property
    def store(self):
        return self.s.store

    def _runtime(self, model):
        try:
            rt = self.s.get_runtime(model.runtime)
        except KeyError:
            return None
        return rt if hasattr(rt, "batch_submit") else None

    def takes(self, j):
        """Whether a queued job is the batcher's (a run that asked for a batch, on a served model with a batch API), not the CPU lane's."""
        if j["kind"] != "run" or not wants(j["params"]):
            return False
        model = self.s._model_of(j)
        return model is not None and model.served and self._runtime(model) is not None

    def _handle(self, model):
        rt = self.s.get_runtime(model.runtime)
        have = self.handles.get(model.key)
        if have is not None and have[0] == model.settings_hash:
            return have[1]
        if have is not None:
            rt.unload(have[1])
        h = rt.load(model, self.s.models_dir)
        self.handles[model.key] = (model.settings_hash, h)
        return h

    def _fail(self, job_id, error, claim=False):
        now = self.s.clock()
        if claim and self.store.claim(job_id, status="running", started_at=now) is None:
            return  # withdrawn meanwhile
        self.store.update(job_id, status="failed", error=str(error), finished_at=now, load_s=0.0)

    def _release(self, job_id):
        t = self.tickets.pop(job_id, None)
        if t is not None:
            spend.release(t)

    # ---- submit ---------------------------------------------------------------------------------------------------------------------
    def due(self, now=None):
        """{model key: (model, queued jobs, oldest first)} of the models whose batch is due."""
        now = self.s.clock() if now is None else now
        groups = {}
        for j in self.store.queued():
            if self.takes(j) and not self.s._waiting_on_step(j):
                m = self.s._model_of(j)
                groups.setdefault(m.key, (m, []))[1].append(j)
        return {
            k: (m, js)
            for k, (m, js) in groups.items()
            if len(js) >= self.max_jobs or now - min(j["submitted_at"] for j in js) >= self.wait_s
        }

    def submit(self, model, jobs):
        """Send one batch of `jobs` (queued, on `model`); returns its row, or None when nothing was sent. Jobs past the size limits stay queued."""
        from stratamodels.scheduler import rank

        rt, now = self._runtime(model), self.s.clock()
        try:
            handle = self._handle(model)
        except runtimes.RunError as e:
            for j in jobs:
                self._fail(j["id"], e, claim=True)
            return None
        entries, size = [], 0
        for j in sorted(jobs, key=lambda q: (rank(q["priority"]), q["id"]))[: self.max_jobs]:
            try:
                body, worst = rt.batch_request(handle, clean(j["params"]))
            except runtimes.RunError as e:
                self._fail(j["id"], e, claim=True)
                continue
            n = len(json.dumps(body))
            if entries and size + n > self.max_bytes:
                break
            entries.append((j, body, worst))
            size += n
        if not entries:
            return None
        try:
            tickets, refusal = spend.reserve_many(rt.PROVIDER, [w for _, _, w in entries])
        except spend.OverBudget as e:
            tickets, refusal = [], e
        for j, _, _ in entries[len(tickets) :]:
            self._fail(j["id"], refusal, claim=True)
        sent = []
        provenance = dict(options=dict(model.options), settings_hash=model.settings_hash)
        for (j, body, worst), t in zip(entries, tickets):
            if self.store.claim(j["id"], status="running", started_at=now, load_s=0.0, **provenance) is None:
                spend.release(t)  # withdrawn after the queue was read
                continue
            self.tickets[j["id"]] = t
            sent.append((j, body, worst))
        if not sent:
            return None
        b = self.store.add_batch(
            provider=rt.PROVIDER,
            model_key=model.key,
            submitted_at=now,
            status="submitted",
            jobs={str(j["id"]): w for j, _, w in sent},
        )
        try:
            vendor_id = rt.batch_submit(handle, [(f"{PREFIX}{j['id']}", body) for j, body, _ in sent])
        except runtimes.RunError as e:
            for j, _, _ in sent:
                self._release(j["id"])
                self._fail(j["id"], e)
            self.s._event(f"batch {b['id']} of {len(sent)} {model.key} jobs was refused: {e}")
            return self.store.update_batch(b["id"], status="failed", error=str(e), ended_at=self.s.clock())
        for j, _, _ in sent:
            self.store.update(j["id"], note=f"in batch {vendor_id}")
        self.s._event(
            f"sent {len(sent)} {model.key} jobs as batch {vendor_id} ({size / 1e6:.1f} MB, holding ${sum(w for _, _, w in sent):.4f})"
        )
        return self.store.update_batch(b["id"], vendor_id=vendor_id, vendor_status="in_progress", polled_at=now)

    # ---- poll and collect -----------------------------------------------------------------------------------------------------------
    def _model_of_batch(self, b):
        for k in b["jobs"]:
            j = self.store.get(int(k), result=False)
            if j is not None:
                m = self.s._model_of(j)
                if m is not None and self._runtime(m) is not None:
                    return m
        return None

    def poll(self, b, now=None):
        """Ask the vendor how batch row `b` stands; collect its outcomes once it has ended. Returns the batch's row."""
        now = self.s.clock() if now is None else now
        model = self._model_of_batch(b)
        if model is None:
            return self.store.update_batch(
                b["id"], polled_at=now, error=f"model {b['model_key']} is no longer served: the batch is not polled"
            )
        rt = self._runtime(model)
        try:
            handle = self._handle(model)
            status, counts = rt.batch_poll(handle, b["vendor_id"])
        except Exception as e:  # a poll that fails is tried again next time
            return self.store.update_batch(b["id"], polled_at=now, error=f"{type(e).__name__}: {e}"[:2000])
        b = self.store.update_batch(b["id"], polled_at=now, vendor_status=status, counts=counts, error=None)
        return self.collect(b, model, rt, handle, now) if status == "ended" else b

    def collect(self, b, model, rt, handle, now):
        jobs = {int(k): float(w) for k, w in b["jobs"].items()}
        seconds = round(now - b["submitted_at"], 2)
        seen = set()
        try:
            for cid, outcome in rt.batch_results(handle, b["vendor_id"]):
                jid = int(cid[len(PREFIX) :]) if cid.startswith(PREFIX) and cid[len(PREFIX) :].isdigit() else None
                if jid not in jobs:
                    continue
                seen.add(jid)
                j = self.store.get(jid, result=False)
                if j is None or j["status"] in FINISHED:
                    self._release(jid)
                    continue  # already collected (a collection cut short and done again)
                ticket = self.tickets.pop(jid, None) or spend.hold(b["provider"], jobs[jid])
                with spend.job(j["app"], jid):
                    try:
                        result = rt.batch_answer(handle, clean(j["params"]), outcome, ticket, b["vendor_id"], seconds)
                    except Exception as e:
                        spend.release(ticket)  # already freed when the row was written; a no-op then
                        self.store.update(
                            jid,
                            status="failed",
                            error=str(e) if isinstance(e, runtimes.RunError) else f"{type(e).__name__}: {e}",
                            finished_at=now,
                        )
                        continue
                self.store.update(jid, status="done", result=result, finished_at=now, run_s=seconds)
        except (
            Exception
        ) as e:  # the results stopped half way: what was collected stands, the rest is read again at the next poll
            self.s._notify()
            return self.store.update_batch(b["id"], error=f"reading the results: {type(e).__name__}: {e}"[:2000])
        for jid in set(jobs) - seen:
            self._release(jid)
            j = self.store.get(jid, result=False)
            if j is not None and j["status"] == "running":
                self.store.update(
                    jid,
                    status="failed",
                    error=f"batch {b['vendor_id']} ended without an answer for this job",
                    finished_at=now,
                )
        self.s._event(f"batch {b['vendor_id']} ended: {b['counts']}")
        self.s._notify()
        return self.store.update_batch(b["id"], status="ended", ended_at=now)

    # ---- the loop -------------------------------------------------------------------------------------------------------------------
    def resume(self):
        """At start (after Store.recover): hold the budget again for the open batches' jobs; fail those of a batch never confirmed sent."""
        for b in self.store.batches(status="submitted"):
            for k, w in b["jobs"].items():
                jid = int(k)
                if b["vendor_id"] is None:
                    j = self.store.get(jid, result=False)
                    if j is not None and j["status"] == "running":
                        self._fail(jid, "the service restarted while this job's batch was being sent")
                elif jid not in self.tickets:
                    self.tickets[jid] = spend.hold(b["provider"], float(w))
            if b["vendor_id"] is None:
                self.store.update_batch(
                    b["id"],
                    status="failed",
                    error="the service restarted while it was being sent",
                    ended_at=self.s.clock(),
                )

    def step(self, now=None):
        """Poll the open batches that are due, then send the batches that are due. Returns the batch rows it touched."""
        now = self.s.clock() if now is None else now
        out = []
        for b in self.store.batches(status="submitted"):
            if b["vendor_id"] is not None and (b["polled_at"] is None or now - b["polled_at"] >= self.poll_s):
                out.append(self.poll(b, now))
        for model, jobs in self.due(now).values():
            b = self.submit(model, jobs)
            if b is not None:
                out.append(b)
        if out:
            self.s._notify()
        return out

    def status(self):
        """The open batches, for /api/status: id, model, jobs, the vendor's counts."""
        return [
            dict(
                id=b["id"],
                vendor_id=b["vendor_id"],
                model=b["model_key"],
                jobs=len(b["jobs"]),
                submitted_at=b["submitted_at"],
                vendor_status=b["vendor_status"],
                counts=b["counts"],
                polled_at=b["polled_at"],
                error=b["error"],
            )
            for b in self.store.batches(status="submitted")
        ]
