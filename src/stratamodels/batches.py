"""Jobs sent through a vendor's batch API (the Claude Message Batches API: half the price of the same calls made at once, answered within
24 hours, most within one), for bulk work that can wait: an archive of photos to caption, an eval's candidate run.

Every job on a model whose runtime has a batch API (claude-api) is batched, however it was queued, with one exception: a single job at high
priority (POST /api/jobs, someone waiting on it now) is sent at once (Jonathan, 2026-11-28: "fix so all models and evals use batch. only
thing allow single run is individual high priority calls"; a batch usually answers within minutes). A bulk call's `batch` flag and an item's
own `batch` key are accepted and not read: the priority and the count decide. On a model without a batch API nothing changes.
Submit and submit_many mark each such job (`batch = true` in its params); the batcher sends the marked jobs of each model at its next step. Nothing
else changes for the app: it waits for each job as it always does, and each ends done or failed with the same result as a call made at once
(plus `batch`, the vendor's id). What to do with an entry that failed (retry it at once, or later) stays with the app.

The batcher (one per scheduler, stepped on its own thread) takes the queued jobs it holds out of the CPU lane and:
  submit  each job's request is built as the runtime would send it, its worst case (at the batch price) held of the budget in one check
          (spend.reserve_many: the jobs that do not fit fail with OverBudget, the rest go), the batch row written (api_batches), the batch
          sent; the jobs are running from then on, their `note` naming the batch. A batch stays under `max_bytes` of requests (the API
          takes 256 MB) and `max_jobs` (it takes 100 000); what is left over waits for the next.
  poll    each open batch every `poll_s`; once it has ended, every entry's outcome is recorded in the ledger at the batch price (spend.record,
          freeing its hold) and its job finished. An entry that errored, expired or was cancelled fails its job (nothing is billed for it).
  resume  after a restart: open batches keep their jobs running (Store.recover leaves them), and their holds are taken again (spend.hold,
          no check: the calls are made). A batch whose row was written but that was never confirmed sent fails its jobs. Held jobs are
          still queued, so they are simply decided again.

The runtime side is the `batch_*` functions of runtimes/claude.py: batch_request, batch_submit, batch_poll, batch_results, batch_answer."""

import json
from types import SimpleNamespace

from stratamodels import runtimes, spend
from stratamodels.store import FINISHED

BATCH = "batch"  # a job's params key: send it through the vendor's batch API
PREFIX = "job-"  # an entry's custom_id is PREFIX + the job id


def wants(params):
    """Whether a job's params mark it for the batch API (submit_many sets the mark)."""
    return bool((params or {}).get(BATCH))


def clean(params):
    """A batch job's params as the runtime reads them (without the scheduler's own keys)."""
    from stratamodels.scheduler import AFTER, OPTIONS

    return {k: v for k, v in (params or {}).items() if k not in (BATCH, OPTIONS, AFTER)}


def snapshot(model, logs):
    """What collecting a batch needs, as it was sent: the model's runtime and load options (its price, its vendor id) and each job's
    request as the ledger logs it (the prompt with its presets filled in, the schema): a preset or a price edited before the answers come
    is not read. The requests are kept once each (a bulk job sends the same prompt thousands of times), the pictures per job."""
    distinct, index, of = [], {}, {}
    for jid, log in logs:
        shared = {k: v for k, v in log.items() if k != "images"}
        sig = json.dumps(shared, sort_keys=True, default=str)
        if sig not in index:
            index[sig] = len(distinct)
            distinct.append(shared)
        of[str(jid)] = [index[sig], log.get("images") or []]
    return dict(runtime=model.runtime, options=dict(model.options), requests=distinct, of=of)


def requests(snap):
    """{job id: its request as logged} from a snapshot."""
    return {int(k): dict(snap["requests"][i], images=imgs) for k, (i, imgs) in snap["of"].items()}


class Batcher:
    def __init__(
        self,
        scheduler,
        poll_s=60.0,
        max_jobs=10_000,
        max_bytes=200 * 1024 * 1024,
    ):
        self.s = scheduler
        self.poll_s, self.max_jobs, self.max_bytes = float(poll_s), int(max_jobs), int(max_bytes)
        self.handles = {}  # model key -> (settings hash, handle): the batcher's own client, apart from the CPU lane's
        self.tickets = {}  # job id -> its spend hold, until its outcome is recorded

    @property
    def store(self):
        return self.s.store

    def batchable(self, model):
        """Whether jobs on `model` can go through a batch API (a served model whose runtime has one)."""
        return model is not None and model.served and self._runtime(model) is not None

    def one_lot(self, model):
        """Whether an eval queues every job on `model` at once: a batched model, or one whose jobs only wait on a batched model (`waits`:
        image-kind or vectorise asking Claude), which the scheduler's wait lane runs together so their delegated calls are one batch."""
        if self.batchable(model):
            return True
        return asks_batched(model, self.s.registry.models, self.s.get_runtime)

    def decide(self, model, n, priority):
        """Whether `n` jobs queued together on `model` at `priority` are batched: always on a model with a batch API, except one job at high
        priority (someone waiting on that one answer)."""
        return self.batchable(model) and not (priority == "high" and n == 1)

    def _runtime(self, model):
        try:
            rt = self.s.get_runtime(model.runtime)
        except KeyError:
            return None
        return rt if hasattr(rt, "batch_submit") else None

    def takes(self, j):
        """Whether a queued job is the batcher's, not the CPU lane's: a run marked for the batch API on a model that has one."""
        if j["kind"] != "run" or not wants(j["params"]):
            return False
        return self.batchable(self.s._model_of(j))

    def _handle(self, model):
        return self._handle_for(model.runtime, model.key, model.options)

    def _handle_for(self, runtime, key, options):
        """A client for a model as it was when a batch was sent (its runtime, key and load options: price and vendor model id), cached:
        a batch is collected with the settings it was sent with, whatever the registry says now."""
        rt = self.s.get_runtime(runtime)
        sig = json.dumps([runtime, options], sort_keys=True, default=str)
        have = self.handles.get(key)
        if have is not None and have[0] == sig:
            return have[1]
        if have is not None:
            rt.unload(have[1])
        h = rt.load(SimpleNamespace(key=key, runtime=runtime, options=dict(options)), self.s.models_dir)
        self.handles[key] = (sig, h)
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
    def due(self):
        """{model key: (model, jobs to batch)}: the queued jobs marked for the batch API, by model (those waiting on a step stay)."""
        groups = {}
        for j in self.store.queued():
            if self.takes(j) and not self.s._waiting_on_step(j):
                m = self.s._model_of(j)
                groups.setdefault(m.key, (m, []))[1].append(j)
        return groups

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
        entries, size, logs = [], 0, {}
        for j in sorted(jobs, key=lambda q: (rank(q["priority"]), q["id"]))[: self.max_jobs]:
            try:
                body, worst, log = rt.batch_request(handle, clean(j["params"]))
            except Exception as e:  # a picture that cannot be read fails its own job, not the batch
                self._fail(j["id"], e if isinstance(e, runtimes.RunError) else f"{type(e).__name__}: {e}", claim=True)
                continue
            n = len(json.dumps(body))
            if entries and size + n > self.max_bytes:
                break
            entries.append((j, body, worst))
            logs[j["id"]] = log
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
            snapshot=snapshot(model, [(j["id"], logs[j["id"]]) for j, _, _ in sent]),
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
    def poll(self, b, now=None):
        """Ask the vendor how batch row `b` stands; collect its outcomes once it has ended. Returns the batch's row."""
        now = self.s.clock() if now is None else now
        snap = b["snapshot"] or {}
        try:
            rt = self.s.get_runtime(snap["runtime"])
            handle = self._handle_for(snap["runtime"], b["model_key"], snap["options"])
            status, counts = rt.batch_poll(handle, b["vendor_id"])
        except Exception as e:  # a poll that fails is tried again next time
            return self.store.update_batch(b["id"], polled_at=now, error=f"{type(e).__name__}: {e}"[:2000])
        b = self.store.update_batch(b["id"], polled_at=now, vendor_status=status, counts=counts, error=None)
        return self.collect(b, rt, handle, now) if status == "ended" else b

    def collect(self, b, rt, handle, now):
        logs = requests(b["snapshot"])
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
                        result = rt.batch_answer(handle, logs[jid], outcome, ticket, b["vendor_id"], seconds)
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
        for model, jobs in self.due().values():
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


def asks_batched(model, models, get_runtime):
    """Whether `model`'s jobs only wait on a model of `models` (its options `vlm` or `llm`) that has a batch API."""
    if model is None or not model.served or model.vram_gb > 0:
        return False
    o = model.options or {}
    try:
        waits = getattr(get_runtime(model.runtime), "waits", None)
        asks = models.get(str(o.get("vlm") or o.get("llm") or ""))
        return bool(
            callable(waits)
            and waits(o)
            and asks is not None
            and asks.served
            and callable(getattr(get_runtime(asks.runtime), "batch_submit", None))
        )
    except Exception:  # an unknown runtime: the job's own run says why
        return False
