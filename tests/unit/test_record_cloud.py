"""scripts/evals/record_cloud.py: the subscription-CLI recorder (codex exec, agy -p) that writes outputs for the `recorded` runtime. The CLIs are
faked: tests check the command lines, what reaches the CLI (pictures, schema, no API keys), answer parsing, quota stops and resuming."""

import importlib.util
import json
import os
import subprocess
import types

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
spec = importlib.util.spec_from_file_location("record_cloud", os.path.join(ROOT, "scripts", "evals", "record_cloud.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def done(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def test_parse_json_takes_the_object_out_of_fences_and_prose():
    assert R.parse_json('{"a": 1}') == {"a": 1}
    assert R.parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert R.parse_json('Here it is: {"a": 3} hope that helps') == {"a": 3}
    assert R.parse_json("no json here") is None
    assert R.parse_json("") is None


def test_prompt_puts_the_system_first_and_says_where_the_pictures_are():
    p = R.prompt_of("SYS", "describe", ["img0.jpg"], {"type": "object"})
    assert (
        p.startswith("SYS") and "working directory: img0.jpg" in p and p.index("describe") < p.index("JSON object only")
    )
    assert "attached" in R.prompt_of(None, "describe", ["a.jpg", "b.jpg"], None, attached=True)
    assert R.prompt_of(None, "just text", [], None) == "just text"


def test_codex_attaches_pictures_passes_the_schema_and_never_an_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    img = tmp_path / "p.jpg"
    img.write_bytes(b"x")
    seen = {}

    def run(cmd, input, cwd, env, **kw):
        seen.update(cmd=cmd, input=input, env=env, schema=json.load(open(cmd[cmd.index("--output-schema") + 1])))
        open(cmd[cmd.index("-o") + 1], "w").write('{"caption": "c"}')
        return done()

    r = R.call_codex("SYS", "go", [str(img)], {"type": "object"}, model="gpt-x", effort="low", run=run)
    assert r["text"] == '{"caption": "c"}'
    c = seen["cmd"]
    assert c[:2] == ["codex", "exec"] and f"--image={img}" in c and c[c.index("-m") + 1] == "gpt-x" and c[-1] == "-"
    assert "--sandbox" in c and c[c.index("--sandbox") + 1] == "read-only"
    assert seen["schema"] == {"type": "object"} and "SYS" in seen["input"] and "OPENAI_API_KEY" not in seen["env"]


def test_codex_quota_message_stops_and_other_failures_are_per_item():
    with pytest.raises(R.QuotaError):
        R.call_codex(
            None,
            "go",
            [],
            None,
            run=lambda *a, **k: done(stderr="You've hit your usage limit. Try again in 3h", code=1),
        )
    with pytest.raises(R.CallError):
        R.call_codex(None, "go", [], None, run=lambda *a, **k: done(stderr="boom", code=2))


def test_agy_copies_pictures_into_its_workspace_and_reads_the_structured_output(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    img = tmp_path / "Photo.JPG"
    img.write_bytes(b"jpeg")
    seen = {}

    def run(cmd, cwd, env, **kw):
        seen.update(cmd=cmd, files=sorted(os.listdir(cwd)), env=env)
        return done(
            stdout=json.dumps(
                dict(
                    status="SUCCESS",
                    response="ignored",
                    structured_output={"caption": "c"},
                    usage=dict(input_tokens=5, output_tokens=7),
                )
            )
        )

    r = R.call_agy("SYS", "go", [str(img)], {"type": "object"}, model="gem", run=run)
    assert json.loads(r["text"]) == {"caption": "c"} and r["usage"]["output_tokens"] == 7
    c = seen["cmd"]
    assert (
        c[:2] == ["agy", "-p"]
        and "img0.jpg" in c[2]
        and c[c.index("--output-format") + 1] == "json"
        and c[c.index("--model") + 1] == "gem"
    )
    assert seen["files"] == ["img0.jpg", "schema.json"] and "GEMINI_API_KEY" not in seen["env"]


def test_agy_error_status_is_a_failure_and_exhausted_is_a_quota_stop():
    with pytest.raises(R.CallError, match="ERROR"):
        R.call_agy(None, "go", [], None, run=lambda *a, **k: done(stdout=json.dumps(dict(status="ERROR", error="bad"))))
    with pytest.raises(R.QuotaError):
        R.call_agy(
            None,
            "go",
            [],
            None,
            run=lambda *a, **k: done(
                stdout=json.dumps(dict(status="ERROR", error="RESOURCE_EXHAUSTED: quota")), code=1
            ),
        )


def test_record_one_builds_the_same_job_as_llama_server_and_times_it(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"x")
    got = {}

    def call(system, text, images, schema, **kw):
        got.update(text=text, images=images, schema=schema)
        return dict(text='{"caption": "c"}', usage=None)

    t = iter([10.0, 12.5])
    o = R.record_one(
        dict(preset="photo-describe-v1", context={"iso": 100}, image=str(img)),
        "codex",
        call=call,
        clock=lambda: next(t),
    )
    assert '"iso": 100' in got["text"] and got["images"] == [str(img)] and got["schema"]["type"] == "object"
    assert o["json"] == {"caption": "c"} and o["seconds"] == 2.5 and o["cli"] == "codex"


def test_main_skips_recorded_items_and_stops_on_quota(tmp_path, monkeypatch):
    from stratamodels.evals import core as C

    items = [dict(id=f"p{i}", input=dict(image=f"/img/p{i}.jpg", context={})) for i in range(4)]
    cand = types.SimpleNamespace(name="local", params=dict(preset="photo-describe-v1"))
    sp = types.SimpleNamespace(name="describe", dataset="ds", candidate=lambda n: cand)
    monkeypatch.setattr(C, "load_spec", lambda n: sp)
    monkeypatch.setattr(C, "load_dataset", lambda n: types.SimpleNamespace(items=items))
    out = tmp_path / "rec.jsonl"
    out.write_text(json.dumps(dict(key="/img/p0.jpg", output={})) + "\n")
    calls = []

    def rec(params, cli, *a):
        calls.append(params["image"])
        if len(calls) == 3:
            raise R.QuotaError("usage limit")
        return dict(text="{}", json={}, seconds=1.0)

    monkeypatch.setattr(R, "record_one", rec)
    monkeypatch.setattr(R, "low_bucket", lambda *a: None)
    code = R.main(["describe", "--like", "local", "--cli", "codex", "--out", str(out)], log=lambda m: None)
    assert code == R.TEMPFAIL and calls == ["/img/p1.jpg", "/img/p2.jpg", "/img/p3.jpg"]
    assert [json.loads(line)["key"] for line in out.read_text().splitlines()] == [
        "/img/p0.jpg",
        "/img/p1.jpg",
        "/img/p2.jpg",
    ]


def test_an_empty_answer_is_a_failure_so_a_rerun_tries_again(tmp_path):
    with pytest.raises(R.CallError, match="answered nothing"):
        R.record_one(dict(prompt="hi"), "agy", call=lambda *a, **k: dict(text="", usage=None))


def test_latency_of_a_recorded_output_is_the_time_it_took_when_made():
    from stratamodels.evals import metrics as M

    lat = M.METRICS["latency_s"].score
    assert (
        lat(dict(_run_s=0.001), dict(json={}, seconds=18.5, cli="agy"), None) == 18.5
    )  # made by record_cloud: its own seconds
    assert (
        lat(dict(_run_s=0.001), dict(json={}, recorded=True), None) is None
    )  # replayed with no time recorded: none, not 0
    assert lat(dict(_run_s=4.2), dict(text="x", seconds=4.2), None) == 4.2  # a model run here: the job's run time
    assert lat(dict(_run_s=4.2), None, "boom") is None


AGY_USAGE = dict(
    status="SUCCESS",
    command=dict(
        name="usage",
        data=dict(
            groups=[
                dict(
                    name="Gemini Models",
                    buckets=[
                        dict(id="gemini-weekly", window="weekly", remaining_fraction=0.95, reset_time="w"),
                        dict(id="gemini-5h", window="5h", remaining_fraction=0.08, reset_time="h"),
                    ],
                ),
                dict(
                    name="Claude and GPT models",
                    buckets=[dict(id="3p-weekly", window="weekly", remaining_fraction=0.5, reset_time="w")],
                ),
                dict(
                    name="Gemini 4 Argon",
                    buckets=[
                        dict(id="gemini-argon-wtu-weekly", window="weekly", remaining_fraction=0.84, reset_time="a")
                    ],
                ),
            ]
        ),
    ),
)


def test_agy_usage_reads_the_models_own_group():
    run = lambda *a, **k: done(stdout=json.dumps(AGY_USAGE))
    assert R.usage_agy("gemini-3.8-flash-low", run=run) == [
        ("Gemini Models weekly", 0.95, "w"),
        ("Gemini Models 5h", 0.08, "h"),
    ]
    assert R.usage_agy("gemini-4-argon-medium", run=run) == [("Gemini 4 Argon weekly", 0.84, "a")]
    assert R.usage_agy("claude-opus-5-5-low", run=run)[0][1] == 0.5
    assert R.low_bucket("agy", "gemini-3.8-flash-low", 0.10, usage=lambda m: R.usage_agy(m, run=run)) == (
        "Gemini Models 5h",
        0.08,
        "h",
    )
    assert R.low_bucket("agy", "gemini-4-argon-low", 0.10, usage=lambda m: R.usage_agy(m, run=run)) is None
    with pytest.raises(R.UsageError):
        R.usage_agy("x", run=lambda *a, **k: done(stdout="not json"))


class FakeAppServer:
    """codex app-server --stdio: answers initialize and account/rateLimits/read."""

    def __init__(self, used=(94, 15), error=None):
        import io

        self.stdin, self.used, self.error, self.sent = io.StringIO(), used, error, []
        self.stdout = self
        self.lines = None

    def readline(self):
        if self.lines is None:
            self.sent = [json.loads(x) for x in self.stdin.getvalue().splitlines()]
            r = dict(
                rateLimits=dict(
                    primary=dict(usedPercent=self.used[0], resetsAt=1794461330),
                    secondary=dict(usedPercent=self.used[1], resetsAt=1795048130),
                )
            )
            self.lines = [
                json.dumps(dict(id=1, result={})),
                json.dumps(dict(method="account/updated")),
                json.dumps(dict(id=2, error=self.error) if self.error else dict(id=2, result=r)),
            ]
        return (self.lines.pop(0) + "\n") if self.lines else ""

    def kill(self):
        pass

    def wait(self):
        pass


def test_codex_usage_comes_from_the_app_server_without_a_model_call():
    fake = FakeAppServer()
    u = R.usage_codex(popen=lambda *a, **k: fake)
    assert [m.get("method") for m in fake.sent] == ["initialize", "initialized", "account/rateLimits/read"]
    assert [(n, round(r, 2)) for n, r, _ in u] == [("codex 5h", 0.06), ("codex weekly", 0.85)] and u[0][2].startswith(
        "2026-"
    )
    assert R.low_bucket("codex", None, 0.10, usage=lambda m: u)[0] == "codex 5h"
    with pytest.raises(R.UsageError):
        R.usage_codex(popen=lambda *a, **k: FakeAppServer(error=dict(message="no")))


def test_main_stops_before_the_allowance_runs_dry(tmp_path, monkeypatch):
    from stratamodels.evals import core as C

    items = [dict(id=f"p{i}", input=dict(image=f"/img/p{i}.jpg", context={})) for i in range(3)]
    cand = types.SimpleNamespace(name="local", params=dict(preset="photo-describe-v1"))
    monkeypatch.setattr(
        C, "load_spec", lambda n: types.SimpleNamespace(name="d", dataset="ds", candidate=lambda n: cand)
    )
    monkeypatch.setattr(C, "load_dataset", lambda n: types.SimpleNamespace(items=items))
    left = iter([None, ("Gemini Models 5h", 0.09, "h")])
    monkeypatch.setattr(R, "low_bucket", lambda *a: next(left))
    monkeypatch.setattr(R, "record_one", lambda *a: dict(text="{}", json={}, seconds=1.0))
    out, log = tmp_path / "r.jsonl", []
    assert R.main(["d", "--like", "local", "--cli", "agy", "--out", str(out)], log=log.append) == R.TEMPFAIL
    assert len(out.read_text().splitlines()) == 1 and "9% left" in log[-1]

    def broken(*a):
        raise R.UsageError("no usage")

    monkeypatch.setattr(R, "low_bucket", broken)
    assert (
        R.main(["d", "--like", "local", "--cli", "agy", "--out", str(out)], log=log.append) == 1
        and "cannot read" in log[-1]
    )
    monkeypatch.setattr(R, "low_bucket", lambda *a: pytest.fail("checked with --min-left 0"))
    assert R.main(["d", "--like", "local", "--cli", "agy", "--out", str(out), "--min-left", "0"], log=log.append) == 0
