"""scripts/evals/record_cloud.py: the subscription-CLI recorder (codex exec, agy -p) that writes outputs for the `recorded` runtime. The CLIs are
faked: tests check the command lines, what reaches the CLI (pictures, schema, no API keys), answer parsing, quota stops and resuming."""

import importlib.util
import json
import os
import subprocess
import types

import pytest

from stratamodels.evals import cloud

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
    sp = types.SimpleNamespace(name="describe", role="vision-language", dataset="ds", candidate=lambda n: cand)
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


class SilentAppServer(FakeAppServer):
    """Starts, then never answers: readline blocks until the process is killed."""

    def __init__(self):
        import threading

        super().__init__()
        self.killed = threading.Event()

    def readline(self):
        self.killed.wait(5)
        return ""

    def kill(self):
        self.killed.set()


def test_a_silent_app_server_is_killed_at_the_deadline_not_waited_on_forever():
    fake = SilentAppServer()
    with pytest.raises(R.UsageError, match="within 0.05 s"):
        R.usage_codex(popen=lambda *a, **k: fake, timeout=0.05)
    assert fake.killed.is_set()


def test_no_readable_bucket_stops_the_recorder_rather_than_spending_blind():
    with pytest.raises(R.UsageError, match="no readable allowance"):
        R.low_bucket("agy", "gemini-3.8-flash-low", 0.10, usage=lambda m: [])
    empty = json.dumps(dict(command=dict(data=dict(groups=[dict(name="Gemini Models", buckets=[])]))))
    with pytest.raises(R.UsageError):
        R.low_bucket("agy", None, 0.10, usage=lambda m: R.usage_agy(m, run=lambda *a, **k: done(empty)))


def test_main_stops_before_the_allowance_runs_dry(tmp_path, monkeypatch):
    from stratamodels.evals import core as C

    items = [dict(id=f"p{i}", input=dict(image=f"/img/p{i}.jpg", context={})) for i in range(3)]
    cand = types.SimpleNamespace(name="local", params=dict(preset="photo-describe-v1"))
    monkeypatch.setattr(
        C,
        "load_spec",
        lambda n: types.SimpleNamespace(name="d", role="vision-language", dataset="ds", candidate=lambda n: cand),
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


# ---- pictures and speech ---------------------------------------------------------------------------------------------------------------------
def png(path, w=8, h=6):
    from PIL import Image

    Image.new("RGB", (w, h)).save(path)


def test_codex_makes_a_picture_in_its_writable_workspace_and_it_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    def run(cmd, input=None, cwd=None, env=None, **k):
        seen.update(cmd=cmd, input=input, env=env)
        png(os.path.join(cwd, "out.png"))
        return done("saved out.png")

    r = R.image_codex("a red fox", str(tmp_path / "keep" / "abc"), 512, 768, model="gpt-6.1-terra", run=run)
    assert r == dict(path=str(tmp_path / "keep" / "abc.png"), width=8, height=6) and os.path.isfile(r["path"])
    assert "workspace-write" in seen["cmd"] and "gpt-6.1-terra" in seen["cmd"] and "OPENAI_API_KEY" not in seen["env"]
    assert "512x768" in seen["input"] and "a red fox" in seen["input"]
    with pytest.raises(R.CallError, match="made no picture"):
        R.image_codex("x", str(tmp_path / "k2"), run=lambda *a, **k: done("sorry"))


def test_agy_picture_is_found_in_the_conversations_own_folder(tmp_path):
    brain = tmp_path / "brain"

    def run(cmd, cwd=None, **k):
        (brain / "conv1" / "artifacts").mkdir(parents=True)
        png(brain / "conv1" / "artifacts" / "fox.jpg", 4, 4)
        return done(json.dumps(dict(status="SUCCESS", conversation_id="conv1", response="made it")))

    r = R.image_agy("a fox", str(tmp_path / "out" / "k"), run=run, brain=str(brain))
    assert r["path"].endswith("k.jpg") and (r["width"], r["height"]) == (4, 4)
    with pytest.raises(R.CallError, match="made no picture"):  # an old picture of another turn is not this one's
        R.image_agy(
            "a fox",
            str(tmp_path / "out" / "k2"),
            run=lambda *a, **k: done(json.dumps(dict(status="SUCCESS", conversation_id="none"))),
            brain=str(brain),
        )


def test_keep_reads_no_size_from_a_file_that_is_not_a_picture(tmp_path):
    (tmp_path / "x.png").write_text("not a png")
    assert cloud._keep(str(tmp_path / "x.png"), str(tmp_path / "o" / "y"))["width"] is None


def test_agy_transcribes_a_clip_it_reads_from_its_workspace(tmp_path):
    cuts, seen = [], {}

    def cut(path, clip, out):
        cuts.append((path, clip))
        open(out, "wb").close()

    def run(cmd, cwd=None, **k):
        seen.update(prompt=cmd[cmd.index("-p") + 1], files=os.listdir(cwd))
        return done(json.dumps(dict(status="SUCCESS", response=" bonjour à tous \n", usage=dict(input_tokens=9))))

    r = R.transcribe_agy("/a/race.mp3", [3, 9.5], "fr", run=run, cut=cut)
    assert r == dict(text="bonjour à tous", usage=dict(input_tokens=9)) and cuts == [("/a/race.mp3", [3, 9.5])]
    assert "clip.wav" in seen["files"] and "in French" in seen["prompt"] and "shell" in seen["prompt"]


def test_cut_audio_asks_ffmpeg_for_16k_mono_of_the_clip():
    seen = []
    cloud.cut_audio("/a.mp3", [1, 2.5], "/o.wav", run=lambda cmd, **k: seen.append(cmd) or done())
    assert seen[0][-1] == "/o.wav" and seen[0][5:9] == ["-ss", "1.000", "-to", "2.500"] and "16000" in seen[0]
    cloud.cut_audio("/a.mp3", None, "/o.wav", run=lambda cmd, **k: seen.append(cmd) or done())
    assert "-ss" not in seen[1]
    with pytest.raises(R.CallError, match="could not cut"):
        cloud.cut_audio("/a.mp3", None, "/o.wav", run=lambda *a, **k: done(stderr="bad", code=1))


def test_record_image_and_speech_shape_outputs_like_the_local_runtimes(tmp_path):
    t = iter([0.0, 12.5, 0.0, 3.0])
    o = R.record_image(
        dict(prompt="fox", width=640),
        "agy",
        "/d/k",
        make=lambda p, d, w, h, **k: dict(path=d + ".png", width=w, height=h),
        clock=lambda: next(t),
    )
    assert o == dict(path="/d/k.png", width=640, height=1024, seconds=12.5, cli="agy", cli_model="default")
    o = R.record_speech(
        dict(path="/a.wav", language="en"),
        "agy",
        model="gemini-3.8-flash-low",
        hear=lambda *a, **k: dict(text="hello"),
        clock=lambda: next(t),
    )
    assert o["segments"] == [dict(text="hello")] and o["language"] == "en" and o["seconds"] == 3.0
    with pytest.raises(R.CallError):
        R.record_speech(dict(path="/a.wav"), "agy", hear=lambda *a, **k: dict(text=""))
    with pytest.raises(SystemExit):
        R.record_speech(dict(path="/a.wav"), "codex")


def test_main_picks_the_job_and_the_key_from_the_specs_role(tmp_path, monkeypatch):
    from stratamodels.evals import core as C

    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path))
    cand = types.SimpleNamespace(name="local", params=dict(width=1024, height=1024))
    sp = types.SimpleNamespace(name="img", role="image-generate", dataset="ds", candidate=lambda n: cand)
    monkeypatch.setattr(C, "load_spec", lambda n: sp)
    monkeypatch.setattr(
        C, "load_dataset", lambda n: types.SimpleNamespace(items=[dict(id="fox", input=dict(prompt="a fox", seed=1))])
    )
    monkeypatch.setattr(R, "low_bucket", lambda *a: None)
    dests = []
    monkeypatch.setattr(
        R, "record_image", lambda p, cli, dest, *a: dests.append(dest) or dict(path=dest + ".png", seconds=1.0)
    )
    out = tmp_path / "evals" / "recorded" / "flash-img.jsonl"
    assert R.main(["img", "--like", "local", "--cli", "agy", "--out", str(out)], log=lambda m: None) == 0
    assert (
        json.loads(out.read_text())["key"] == "a fox" and os.path.join("generated", "recorded", "flash-img") in dests[0]
    )

    sp.role, cand.params = "transcribe", {}
    monkeypatch.setattr(
        C, "load_dataset", lambda n: types.SimpleNamespace(items=[dict(id="a", input=dict(path="/a.wav", clip=[1, 2]))])
    )
    monkeypatch.setattr(R, "record_speech", lambda p, *a: dict(text="hi", segments=[dict(text="hi")], seconds=1.0))
    out = tmp_path / "speech.jsonl"
    assert R.main(["t", "--like", "local", "--cli", "agy", "--out", str(out)], log=lambda m: None) == 0
    assert json.loads(out.read_text())["key"] == json.dumps(["/a.wav", [1, 2]])
    log = []
    assert (
        R.main(["t", "--like", "local", "--cli", "codex", "--out", str(out)], log=log.append) == 2
        and "cannot hear" in log[-1]
    )


def test_a_filter_notice_inside_an_answer_is_a_failure_and_an_image_quota_a_stop(tmp_path):
    blocked = "She rose This request was blocked by Gemini's filters. They can occasionally trigger by mistake."
    with pytest.raises(R.CallError, match="filters"):
        R.transcribe_agy(
            "/a.wav",
            run=lambda *a, **k: done(json.dumps(dict(status="SUCCESS", response=blocked))),
            cut=lambda p, c, o: o,
        )
    full = "429 RESOURCE_EXHAUSTED (Capacity exhausted on `gemini-3.1-flash-image`; quota resets in ~4 hours)"
    with pytest.raises(R.QuotaError):  # the image model's own quota, which /usage does not show
        R.image_agy(
            "fox",
            str(tmp_path / "k"),
            run=lambda *a, **k: done(json.dumps(dict(status="SUCCESS", response=full))),
            brain=str(tmp_path),
        )
