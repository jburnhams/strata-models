"""The cloud API runtimes (claude-api, gemini-api, nvidia-nim) and what they share (runtimes/cloud.py), with the vendors faked."""

import io
import json
import urllib.error
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from cloudfakes import FakeAnthropic, Post, claude_reply, key_file, ledger  # noqa: F401  (ledger: fixture)

from stratamodels import runtimes, settings, spend
from stratamodels.runtimes import RunError, claude, cloud, gemini, nim


def model(key="m", **options):
    return SimpleNamespace(key=key, options=options)


# ---- cloud.py ----------------------------------------------------------------------------------------------------------------------------


def test_parse_prompt_preset_images_and_refusals(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG")
    j = cloud.parse(
        dict(
            prompt="hi",
            system="be brief",
            image=str(img),
            max_tokens=10,
            effort="low",
            temperature=0.2,
            seed=1,
            think=False,
        )
    )
    assert (j["text"], j["system"], j["max_tokens"], j["effort"], j["schema"]) == ("hi", "be brief", 10, "low", None)
    assert j["images"] == [(str(img), "image/png", "iVBORw==")]
    assert cloud.logged(j) == dict(
        system="be brief", prompt="hi", images=[str(img)], schema=None, max_tokens=10, effort="low"
    )
    p = cloud.parse(dict(preset="photo-describe-v1", context={"year": 1990}, system_preset="image-prompt-bfl"))
    assert p["schema"] and "1990" in p["text"] and p["system"] and p["max_tokens"] == 2048
    assert cloud.parse(dict(preset="photo-describe-v1", schema=False))["schema"] is None
    for bad, msg in (
        (dict(prompt="x", nope=1), "unknown params"),
        (dict(), "give a prompt"),
        (dict(prompt="x", preset="photo-describe-v1"), "not both"),
        (dict(prompt="x", system="a", system_preset="image-prompt-bfl"), "not both"),
        (dict(prompt="x", images="a.png"), "list of paths"),
        (dict(prompt="x", image="/no/such.png"), "no such image"),
        (dict(prompt="x", effort="max"), "effort is one of"),
        (dict(prompt="x", max_tokens="lots"), "whole number"),
        (dict(prompt="x", max_tokens=200_000), "between 1 and 128000"),
    ):
        with pytest.raises(RunError, match=msg):
            cloud.parse(bad)


def test_answer_parses_json_only_with_a_schema():
    assert cloud.answer(dict(schema=None), '{"a": 1}') == {"text": '{"a": 1}'}
    assert cloud.answer(dict(schema={"type": "object"}), '{"a": 1}') == {"text": '{"a": 1}', "json": {"a": 1}}
    assert cloud.answer(dict(schema={"type": "object"}), "not json") == {"text": "not json"}


def test_post_success_http_error_and_unreachable(monkeypatch, tmp_path):
    key_file(tmp_path, monkeypatch, NVIDIA_API_KEY="nv-secret-0123456789")
    seen = {}

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def ok(req, timeout):
        seen.update(url=req.full_url, body=json.loads(req.data), auth=req.get_header("Authorization"))
        return Reply(b'{"ok": true}')

    monkeypatch.setattr(cloud.urllib.request, "urlopen", ok)
    assert cloud.post("https://x/v1/chat", {"a": 1}, {"authorization": "Bearer k"}) == {"ok": True}
    assert seen == dict(url="https://x/v1/chat", body={"a": 1}, auth="Bearer k")

    def err(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 429, "slow", {"retry-after": "3"}, io.BytesIO(b"echo nv-secret-0123456789")
        )  # type: ignore[arg-type]

    monkeypatch.setattr(cloud.urllib.request, "urlopen", err)
    with pytest.raises(cloud.HttpError) as e:
        cloud.post("https://x", {}, {})
    assert (
        e.value.status == 429
        and e.value.retry_after == 3.0
        and e.value.body == "echo [REDACTED]"
        and "HTTP 429" in str(e.value)
    )

    def down(req, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(cloud.urllib.request, "urlopen", down)
    with pytest.raises(RunError, match="could not reach https://x/a"):
        cloud.post("https://x/a?key=1", {}, {})


def test_registered_and_keys_reported_missing(monkeypatch, tmp_path):
    for rt in ("claude-api", "gemini-api", "nvidia-nim"):
        assert runtimes.get(rt).__name__ == runtimes.RUNTIMES[rt]
    av = runtimes.available()
    assert (
        any("ANTHROPIC_API_KEY" in m for m in av["claude-api"])
        and any("NVIDIA_API_KEY" in m for m in av["nvidia-nim"])
        and av["gemini-api"]
    )
    key_file(tmp_path, monkeypatch, ANTHROPIC_API_KEY="k" * 20, NVIDIA_API_KEY="n" * 20, GEMINI_FREE_API_KEY="g" * 20)
    av = runtimes.available()
    assert av["claude-api"] == [] and av["nvidia-nim"] == [] and av["gemini-api"] == []


# ---- claude-api --------------------------------------------------------------------------------------------------------------------------

PRICE = dict(input=1.0, output=5.0)


def test_claude_load_needs_model_price_and_key(monkeypatch):
    with pytest.raises(RunError, match="no Claude model id"):
        claude.load(model(price=PRICE), "/m")
    with pytest.raises(RunError, match="no price"):
        claude.load(model(model="claude-haiku-5-5"), "/m")
    with pytest.raises(RunError, match="no ANTHROPIC_API_KEY"):
        claude.load(model(model="claude-haiku-5-5", price=PRICE), "/m")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-0123456789")
    h = claude.load(model(model="claude-haiku-5-5", price=PRICE, effort="low"), "/m")
    assert isinstance(h["client"], anthropic.Anthropic) and h["effort"] == "low"
    claude.unload(h)
    assert h == {}


def test_claude_run_counts_reserves_and_records(ledger):  # noqa: F811
    settings.save(dict(claude_budget_usd=1.0))
    c = FakeAnthropic(count=1000, reply=claude_reply('{"a": 1}', tokens_in=900, tokens_out=50, cache_read=100))
    h = claude.load(model("haiku-api", model="claude-haiku-5-5", price=PRICE, effort="medium"), "/m", client=c)
    with spend.job("evals", 3):
        out = claude.run(h, dict(prompt="hi", system="s", schema={"type": "object"}, max_tokens=100))
    assert out == dict(
        text='{"a": 1}',
        json={"a": 1},
        tokens_in=1000,
        tokens_out=50,
        cost_usd=pytest.approx((900 + 250 + 10) / 1e6),
        seconds=out["seconds"],
        model="haiku-api",
        provider="claude",
        finish="end_turn",
    )
    sent = c.messages.created[0]
    assert (
        sent["system"] == "s"
        and sent["max_tokens"] == 100
        and sent["output_config"]
        == {"effort": "medium", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    )
    row = ledger.api_call(ledger.api_log()[0]["id"])
    assert (
        row["app"],
        row["job_id"],
        row["model_key"],
        row["tokens_in"],
        row["cache_read"],
        row["response"],
        row["request"]["prompt"],
    ) == ("evals", 3, "haiku-api", 900, 100, '{"a": 1}', "hi")
    claude.unload(h)
    assert c.closed


def test_claude_refuses_a_call_that_does_not_fit(ledger):  # noqa: F811
    settings.save(dict(claude_budget_usd=0.001))
    c = FakeAnthropic(count=1000, reply=claude_reply())
    h = claude.load(model(model="claude-haiku-5-5", price=PRICE), "/m", client=c)
    with pytest.raises(spend.OverBudget, match="could cost"):
        claude.run(h, dict(prompt="hi", max_tokens=1000))  # 1000 in + 1000 out at $1/$5 per M = $0.006
    assert c.messages.created == [] and ledger.api_log() == []


def test_claude_refusal_and_errors_are_recorded(ledger):  # noqa: F811
    settings.save(dict(claude_budget_usd=1.0))
    h = claude.load(
        model(model="claude-haiku-5-5", price=PRICE),
        "/m",
        client=FakeAnthropic(reply=claude_reply("", stop="refusal", category="cyber")),
    )
    with pytest.raises(RunError, match="declined this request \\(refusal, cyber\\)"):
        claude.run(h, dict(prompt="hi"))
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx2.Response(429, request=req)
    for err, msg in (
        (anthropic.RateLimitError("slow down", response=resp, body=None), "Claude API 429: slow down"),
        (anthropic.APIConnectionError(request=req), "could not reach"),
    ):
        h = claude.load(model(model="claude-haiku-5-5", price=PRICE), "/m", client=FakeAnthropic(error=err))
        with pytest.raises(RunError, match=msg):
            claude.run(h, dict(prompt="hi"))
    assert [(c["ok"], c["error"][:7]) for c in ledger.api_log()][::-1] == [
        (False, "refused"),
        (False, "slow do"),
        (False, "Connect"),
    ]
    assert spend.state("claude")["reserved"] == 0.0
    h = claude.load(
        model(model="claude-haiku-5-5", price=PRICE),
        "/m",
        client=FakeAnthropic(count=anthropic.APIConnectionError(request=req)),
    )
    with pytest.raises(RunError, match="refused to count"):
        claude.run(h, dict(prompt="hi"))


# ---- gemini-api --------------------------------------------------------------------------------------------------------------------------

G = dict(input=1.0, output=10.0)


def reply(text="ok", finish="STOP", prompt=100, out=5, thoughts=20, cached=0):
    return {
        "candidates": [
            {"content": {"parts": [{"text": "thinking", "thought": True}, {"text": text}]}, "finishReason": finish}
        ],
        "usageMetadata": {
            "promptTokenCount": prompt,
            "candidatesTokenCount": out,
            "thoughtsTokenCount": thoughts,
            "cachedContentTokenCount": cached,
        },
        "modelVersion": "gemini-3.8-flash",
    }


def test_gemini_load_checks(tmp_path, monkeypatch):
    with pytest.raises(RunError, match="no Gemini model id"):
        gemini.load(model(), "/m")
    with pytest.raises(RunError, match="keys are among"):
        gemini.load(model(model="g", keys=["other"]), "/m")
    with pytest.raises(RunError, match="no price"):
        gemini.load(model(model="g"), "/m")
    with pytest.raises(RunError, match="no Google AI Studio key"):
        gemini.load(model(model="g", price=G), "/m")
    key_file(tmp_path, monkeypatch, GEMINI_FREE_API_KEY="free-0123456789")
    h = gemini.load(model(model="g", keys=["gemini-free"]), "/m")
    assert h["keys"] == ["gemini-free"] and h["price"] is None
    gemini.unload(h)
    assert h == {}


def test_gemini_free_first_then_paid(ledger, tmp_path, monkeypatch):  # noqa: F811
    key_file(tmp_path, monkeypatch, GEMINI_FREE_API_KEY="free-0123456789", GEMINI_API_KEY="paid-0123456789")
    settings.save(dict(gemini_budget_usd=1.0))
    h = gemini.load(model("flash", model="gemini-3.8-flash", price=G), "/m")
    post = Post(reply(cached=40))
    out = gemini.run(h, dict(prompt="hi", system="s", effort="low", schema={"type": "object"}), post=post)
    assert (out["text"], out["provider"], out["cost_usd"], out["tokens_in"], out["tokens_out"]) == (
        "ok",
        "gemini-free",
        0.0,
        100,
        25,
    )
    url, body, hdr = post.calls[0]
    assert url.endswith("gemini-3.8-flash:generateContent") and hdr == {"x-goog-api-key": "free-0123456789"}
    assert body["systemInstruction"] == {"parts": [{"text": "s"}]} and body["generationConfig"] == dict(
        maxOutputTokens=2048,
        thinkingConfig={"thinkingLevel": "low"},
        responseMimeType="application/json",
        responseJsonSchema={"type": "object"},
    )
    post = Post(cloud.HttpError(429, "quota"), {"totalTokens": 100}, reply('{"a": 2}'))
    out = gemini.run(h, dict(prompt="hi", schema={"type": "object"}), post=post)
    assert (
        out["provider"] == "gemini"
        and out["json"] == {"a": 2}
        and out["cost_usd"] == pytest.approx((100 + 25 * 10) / 1e6)
    )
    assert [c[2]["x-goog-api-key"] for c in post.calls] == [
        "free-0123456789",
        "paid-0123456789",
        "paid-0123456789",
    ] and post.calls[1][0].endswith(":countTokens")
    assert [(c["provider"], c["ok"]) for c in ledger.api_log()] == [
        ("gemini", True),
        ("gemini-free", False),
        ("gemini-free", True),
    ]


def test_gemini_budget_errors_and_blocks(ledger, tmp_path, monkeypatch):  # noqa: F811
    key_file(tmp_path, monkeypatch, GEMINI_API_KEY="paid-0123456789")
    settings.save(dict(gemini_budget_usd=0.001))
    h = gemini.load(model(model="gemini-4-argon", price=G), "/m")
    with pytest.raises(spend.OverBudget):
        gemini.run(h, dict(prompt="hi"), post=Post({"totalTokens": 100}))  # 2048 tokens out at $10/M is $0.02
    with pytest.raises(RunError, match="refused to count the request \\(400\\)"):
        gemini.run(h, dict(prompt="hi"), post=Post(cloud.HttpError(400, "bad")))
    settings.save(dict(gemini_budget_usd=1.0))
    with pytest.raises(RunError, match=r"Google AI Studio \(gemini\): HTTP 500"):
        gemini.run(h, dict(prompt="hi"), post=Post({"totalTokens": 1}, cloud.HttpError(500, "oops")))
    with pytest.raises(RunError, match="declined this request \\(SAFETY\\)"):
        gemini.run(h, dict(prompt="hi"), post=Post({"totalTokens": 1}, reply("", finish="SAFETY")))
    with pytest.raises(RunError, match="declined this request \\(OTHER\\)"):
        gemini.run(h, dict(prompt="hi"), post=Post({"totalTokens": 1}, {"promptFeedback": {"blockReason": "OTHER"}}))
    monkeypatch.delenv("STRATAMODELS_SECRETS")
    with pytest.raises(RunError, match="none is set"):
        gemini.run(h, dict(prompt="hi"), post=Post())


def test_gemini_free_only_runs_out(ledger, tmp_path, monkeypatch):  # noqa: F811
    key_file(tmp_path, monkeypatch, GEMINI_FREE_API_KEY="free-0123456789")
    h = gemini.load(model(model="g", keys=["gemini-free"]), "/m")
    with pytest.raises(RunError, match="no Google AI Studio key could take this call: gemini-free: HTTP 503"):
        gemini.run(h, dict(prompt="hi", image=None), post=Post(cloud.HttpError(503, "busy")))


# ---- nvidia-nim --------------------------------------------------------------------------------------------------------------------------


def chat(text="ok"):
    return {
        "model": "nvidia/x",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 21, "completion_tokens": 4},
    }


def test_nim_load_and_body(tmp_path, monkeypatch):
    with pytest.raises(RunError, match="no NIM model id"):
        nim.load(model(), "/m")
    with pytest.raises(RunError, match="no NVIDIA_API_KEY"):
        nim.load(model(model="nvidia/x"), "/m")
    key_file(tmp_path, monkeypatch, NVIDIA_API_KEY="nv-0123456789")
    h = nim.load(model(model="nvidia/x", base_url="http://local/v1/", extra={"top_k": 1}), "/m")
    assert h["url"] == "http://local/v1/chat/completions"
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    b = nim.body(h, cloud.parse(dict(prompt="hi", system="s", image=str(img), schema={"type": "object"})))
    assert (
        b["messages"][0] == {"role": "system", "content": "s"}
        and b["messages"][1]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,eA=="
    )
    assert b["top_k"] == 1 and b["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert nim.body(h, cloud.parse(dict(prompt="hi")))["messages"] == [{"role": "user", "content": "hi"}]
    nim.unload(h)
    assert h == {}


def test_nim_run_retries_429_and_records(ledger, tmp_path, monkeypatch):  # noqa: F811
    key_file(tmp_path, monkeypatch, NVIDIA_API_KEY="nv-0123456789")
    h = nim.load(model("nem", model="nvidia/x"), "/m")
    waits = []
    post = Post(cloud.HttpError(429, "slow", retry_after=2.0), chat())
    out = nim.run(h, dict(prompt="hi"), post=post, sleep=waits.append)
    assert out == dict(
        text="ok",
        tokens_in=21,
        tokens_out=4,
        cost_usd=0.0,
        seconds=out["seconds"],
        model="nem",
        provider="nim",
        finish="stop",
    )
    assert waits == [2.0] and post.calls[0][2] == {"authorization": "Bearer nv-0123456789"}
    with pytest.raises(RunError, match="NVIDIA NIM 429"):
        nim.run(h, dict(prompt="hi"), post=Post(*[cloud.HttpError(429, "slow")] * 3), sleep=waits.append)
    assert waits[1:] == [5.0, 10.0]
    with pytest.raises(RunError, match="could not reach"):
        nim.run(h, dict(prompt="hi"), post=Post(RunError("could not reach x")))
    assert [(c["ok"], c["tokens_in"]) for c in ledger.api_log()] == [(False, 0), (False, 0), (True, 21)]


def test_a_picture_over_the_limit_is_shrunk_and_a_small_one_sent_as_it_is(tmp_path, monkeypatch):
    import base64
    import io

    from PIL import Image

    from stratamodels.runtimes import cloud

    small = tmp_path / "small.png"
    Image.new("L", (100, 150), 255).save(small)
    assert cloud.encode(str(small)) == ("image/png", base64.b64encode(small.read_bytes()).decode())
    big = tmp_path / "page.png"
    Image.effect_noise((3000, 4500), 120).save(big)  # noise does not compress: a strata-scan page's size
    mime, b64 = cloud.encode(str(big))
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert (
        len(b64) < cloud.MAX_B64
        and max(im.size) == cloud.LONG_EDGE
        and im.mode == "L"
        and mime in ("image/png", "image/jpeg")
    )
    monkeypatch.setattr(cloud, "MAX_B64", 100_000)  # even the scaled PNG is too big: JPEG
    colour = tmp_path / "photo.png"
    Image.merge("RGB", [Image.effect_noise((400, 300), 100)] * 3).save(colour)
    mime, b64 = cloud.encode(str(colour))
    assert mime == "image/jpeg" and Image.open(io.BytesIO(base64.b64decode(b64))).mode == "RGB"
