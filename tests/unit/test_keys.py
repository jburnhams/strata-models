import os

import pytest
from cloudfakes import key_file

from stratamodels import keys


def test_env_wins_then_file_and_redact(tmp_path, monkeypatch):
    key_file(tmp_path, monkeypatch, ANTHROPIC_API_KEY="sk-file-0123456789")
    (tmp_path / "secrets.env").write_text(
        '# keys\nexport NVIDIA_API_KEY="nv-quoted-0123456789"\nnot a line\n\nANTHROPIC_API_KEY=sk-file-0123456789\n'
    )
    assert keys.get("claude") == "sk-file-0123456789" and keys.get("nim") == "nv-quoted-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-0123456789")
    assert keys.get("claude") == "sk-env-0123456789"
    assert (
        keys.redact("a sk-env-0123456789 b nv-quoted-0123456789 c sk-file-0123456789")
        == "a [REDACTED] b [REDACTED] c [REDACTED]"
    )
    assert keys.have("claude") and not keys.have("gemini")
    with pytest.raises(keys.MissingKey, match="no GEMINI_FREE_API_KEY"):
        keys.get("gemini-free")


def test_default_path_and_missing_file(monkeypatch):
    monkeypatch.delenv("STRATAMODELS_SECRETS", raising=False)
    assert keys.path() == os.path.join(os.path.expanduser("~"), ".config", "strata-models", "secrets.env")
    assert not keys.have("claude") and keys.redact("nothing") == "nothing"


@pytest.mark.skipif(os.name == "nt", reason="file modes are POSIX")
def test_a_file_others_can_read_is_refused(tmp_path, monkeypatch):
    p = key_file(tmp_path, monkeypatch, ANTHROPIC_API_KEY="sk-0123456789")
    os.chmod(p, 0o644)
    with pytest.raises(keys.MissingKey, match="chmod 600"):
        keys.get("claude")
    assert not keys.have("claude") and keys.redact("sk-0123456789") == "sk-0123456789"
