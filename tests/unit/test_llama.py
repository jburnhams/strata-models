"""The llama.cpp runtime (stratamodels.runtimes.llama) without the binary: files, presets, the request, a run against a fake server, start-up
failures; and the recorded runtime that replays outputs made elsewhere."""

import json
import os
import subprocess
import sys
import types

import pytest

from stratamodels.registry import Model
from stratamodels.runtimes import RunError, llama, recorded


class FakeProc:
    def __init__(self, cmd, exits=None, **kw):
        self.cmd, self.pid, self.returncode, self._exits, self.env = cmd, 4242, None, exits, kw.get("env")

    def poll(self):
        return self._exits

    def terminate(self):
        self.returncode = 0

    def wait(self, t=None):
        return 0

    def kill(self):
        pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    b = tmp_path / "bin" / "llama" / "llama-b1"
    b.mkdir(parents=True)
    (b / "llama-server").write_text("#!/bin/sh\n")
    (tmp_path / "bin" / "llama" / "cudart-x").mkdir()
    monkeypatch.delenv("STRATAMODELS_LLAMA_BIN", raising=False)
    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path / "data"))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, path, local_dir: os.path.join(local_dir, path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(llama, "_get", lambda url, timeout=5: (200, b"{}"))
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8jpeg")
    monkeypatch.setattr(llama, "LIVE", set())
    return tmp_path, img


def model(**o):
    return Model(
        key="vl",
        runtime="llama-server",
        name="vl",
        roles=("vision-language",),
        vram_gb=7,
        options=dict(model="org/m-GGUF:m.gguf", mmproj="org/m-GGUF:mmproj.gguf", **o),
    )


def test_load_starts_the_server_with_its_files_and_libraries(setup):
    tmp, _ = setup
    procs = []
    h = llama.load(
        model(args=["--fit", "on"]), str(tmp), popen=lambda cmd, **kw: procs.append(FakeProc(cmd, **kw)) or procs[-1]
    )
    cmd = procs[0].cmd
    assert (
        cmd[0].endswith(os.path.join("llama-b1", "llama-server"))
        and cmd[cmd.index("-m") + 1].endswith(os.path.join("llama", "org--m-GGUF", "m.gguf"))
        and "--mmproj" in cmd
        and cmd[-2:] == ["--fit", "on"]
        and cmd[cmd.index("-c") + 1] == "8192"
    )
    assert "cudart-x" in procs[0].env["LD_LIBRARY_PATH"] and h["vision"] and h["url"].startswith("http://127.0.0.1:")
    llama.unload(h)
    assert h == {}


def test_servers_still_running_are_stopped_at_exit(setup):
    tmp, _ = setup
    procs = []

    class Stubborn(FakeProc):
        def wait(self, t=None):
            if not getattr(self, "killed", False):
                raise subprocess.TimeoutExpired(self.cmd, t)

        def kill(self):
            self.killed = True

    a = llama.load(model(), str(tmp), popen=lambda cmd, **kw: procs.append(FakeProc(cmd, **kw)) or procs[-1])
    llama.load(model(), str(tmp), popen=lambda cmd, **kw: procs.append(Stubborn(cmd, **kw)) or procs[-1])
    assert set(procs) == llama.LIVE
    llama.unload(a)
    assert {procs[1]} == llama.LIVE  # an unloaded server is no longer tracked
    llama.stop_all()
    assert set() == llama.LIVE and procs[1].killed  # one that ignores terminate is killed


def test_load_refuses_what_is_missing(setup, monkeypatch):
    tmp, _ = setup
    with pytest.raises(RunError, match="names no model"):
        llama.load(
            Model(key="x", runtime="llama-server", name="x", roles=(), vram_gb=1, options={}), str(tmp), popen=FakeProc
        )
    with pytest.raises(RunError, match="exited while loading"):
        llama.load(model(), str(tmp), popen=lambda cmd, **kw: FakeProc(cmd, exits=1))
    with pytest.raises(RunError, match="no such model file"):
        llama.fetch(str(tmp / "nope.gguf"), str(tmp))
    with pytest.raises(RunError, match="repo:path"):
        llama.fetch("nope", str(tmp))
    (tmp / "llama" / "org--m-GGUF").mkdir(parents=True)
    (tmp / "llama" / "org--m-GGUF" / "have.gguf").write_text("x")
    assert llama.fetch("org/m-GGUF:have.gguf", str(tmp)).endswith("have.gguf") and llama.fetch(
        str(tmp / "a.jpg"), str(tmp)
    ) == str(tmp / "a.jpg")

    def boom(*a, **k):
        raise OSError("offline")

    good = sys.modules["huggingface_hub"].hf_hub_download
    sys.modules["huggingface_hub"].hf_hub_download = boom
    with pytest.raises(RunError, match="could not fetch"):
        llama.fetch("org/m-GGUF:other.gguf", str(tmp))
    sys.modules["huggingface_hub"].hf_hub_download = good
    monkeypatch.setattr(llama, "_get", lambda url, timeout=5: (_ for _ in ()).throw(OSError("not yet")))
    with pytest.raises(RunError, match="did not answer"):
        llama.load(model(), str(tmp), popen=FakeProc, wait_s=0)
    monkeypatch.setenv("STRATAMODELS_LLAMA_BIN", str(tmp / "none"))
    with pytest.raises(RunError, match="not installed"):
        llama.load(model(), str(tmp), popen=FakeProc)


def test_requests_presets_and_refusals(setup):
    _, img = setup
    body, schema = llama.request(dict(image=str(img), preset="photo-describe-v1", context=dict(camera="NEX-5N")))
    text = body["messages"][0]["content"][-1]["text"]
    assert (
        "NEX-5N" in text
        and "{context}" not in text
        and schema["required"][0] == "caption"
        and body["response_format"]["json_schema"]["schema"] is schema
    )
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,") and body[
        "chat_template_kwargs"
    ] == {"enable_thinking": False}
    body, schema = llama.request(dict(prompt="what is it?", system="be brief", schema=False, think=True))
    assert (
        schema is False
        and "response_format" not in body
        and body["messages"][0]["role"] == "system"
        and body["chat_template_kwargs"]["enable_thinking"]
    )
    body, _ = llama.request(dict(prompt="a cat on a mat", system_preset="image-prompt-bfl"))
    assert (
        body["messages"][0]["role"] == "system"
        and body["messages"][0]["content"].startswith("You are the prompt upsampler")
        and body["messages"][1]["content"][-1]["text"] == "a cat on a mat"
    )
    for name in (
        "image-prompt-simple",
        "image-prompt-structured",
        "image-prompt-photographer",
        "image-prompt-fewshot",
        "image-prompt-concise",
    ):
        assert llama.request(dict(prompt="x", system_preset=name))[0]["messages"][0]["content"]
    with pytest.raises(RunError, match="not both"):
        llama.request(dict(prompt="x", system="y", system_preset="image-prompt-bfl"))
    for p, msg in [
        (dict(prompt="x", colour=1), "unknown params"),
        (dict(), "give a prompt"),
        (dict(prompt="x", preset="photo-describe-v1"), "not both"),
        (dict(preset="nope"), "no preset"),
        (dict(preset="../x"), "no preset"),
        (dict(prompt="x", image="/no/such.jpg"), "no such image"),
        (dict(prompt="x", images="a.jpg"), "list of paths"),
    ]:
        with pytest.raises(RunError, match=msg):
            llama.request(p)


def test_run_parses_the_answer_and_measures_vram(setup):
    tmp, img = setup
    h = llama.load(model(), str(tmp), popen=FakeProc)
    sent = []

    def post(url, body):
        sent.append(body)
        return dict(
            choices=[dict(message=dict(content='{"caption": "a dog"}'), finish_reason="stop")],
            usage=dict(prompt_tokens=900, completion_tokens=12),
        )

    probe = lambda: dict(processes=[dict(pid=4242, used_mb=7000.0), dict(pid=1, used_mb=50.0)])
    r = llama.run(h, dict(image=str(img), prompt="describe", schema={"type": "object"}), probe=probe, post=post)
    assert (
        r["json"] == {"caption": "a dog"}
        and r["tokens_in"] == 900
        and r["finish"] == "stop"
        and sent[0]["messages"][0]["content"][1]["text"] == "describe"
    )
    bad = lambda url, body: dict(choices=[dict(message=dict(content='{"capt'))], usage={})
    assert llama.run(h, dict(prompt="x", schema={"type": "object"}), probe=lambda: None, post=bad)["json"] is None
    assert "json" not in llama.run(h, dict(prompt="x"), probe=lambda: None, post=bad)
    h["vision"] = False
    with pytest.raises(RunError, match="no vision projector"):
        llama.run(h, dict(prompt="x", image=str(img)), post=post)


def test_post_reports_a_refusal(monkeypatch):
    import io
    import urllib.error

    def refuse(req, timeout):
        raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"image too big"))

    monkeypatch.setattr(llama.urllib.request, "urlopen", refuse)
    with pytest.raises(RunError, match="refused the request \\(400\\): image too big"):
        llama._post("http://x", {})


def test_recorded_replays_by_key(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path))
    (tmp_path / "r.jsonl").write_text(json.dumps(dict(key="/a.jpg", output=dict(json=dict(caption="x")))) + "\n\n")
    m = Model(
        key="sonnet", runtime="recorded", name="s", roles=("vision-language",), vram_gb=0, options=dict(file="r.jsonl")
    )
    h = recorded.load(m, str(tmp_path))
    assert recorded.run(h, dict(image="/a.jpg", preset="p")) == dict(
        json=dict(caption="x"), model="sonnet", recorded=True
    )
    with pytest.raises(RunError, match="nothing recorded"):
        recorded.run(h, dict(image="/b.jpg"))
    with pytest.raises(RunError, match="give image"):
        recorded.run(h, dict())
    with pytest.raises(RunError, match="names no file"):
        recorded.load(Model(key="s", runtime="recorded", name="s", roles=(), vram_gb=0), str(tmp_path))
    with pytest.raises(RunError, match="cannot read"):
        recorded.load(
            Model(key="s", runtime="recorded", name="s", roles=(), vram_gb=0, options=dict(file="none.jsonl")),
            str(tmp_path),
        )
    recorded.unload(h)
    assert h == {}


def test_recorded_can_match_on_several_params(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATAMODELS_DATA", str(tmp_path))
    key = json.dumps(["/a.wav", [1.0, 2.5]])
    (tmp_path / "r.jsonl").write_text(json.dumps(dict(key=key, output=dict(text="hi"))) + "\n")
    m = Model(
        key="f",
        runtime="recorded",
        name="f",
        roles=("transcribe",),
        vram_gb=0,
        options=dict(file="r.jsonl", by=["path", "clip"]),
    )
    h = recorded.load(m, str(tmp_path))
    assert recorded.run(h, dict(path="/a.wav", clip=[1.0, 2.5], language="en"))["text"] == "hi"
    with pytest.raises(RunError, match="nothing recorded"):
        recorded.run(h, dict(path="/a.wav"))  # the whole file is another key than a clip of it
