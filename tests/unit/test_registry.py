"""The registry (stratamodels.registry): the built-in file, the machine's override, resolving a role to its model, and refusing bad files."""

import pytest

from stratamodels import settings
from stratamodels.registry import Registry, RegistryError


def test_the_builtin_registry_names_a_model_for_every_role():
    r = Registry.load()
    assert r.resolve("faces").key == "buffalo_l" and r.resolve("transcribe").key == "whisper-large-v3-turbo"
    assert r.resolve("transcribe-fast", "whisper-large-v3-turbo").runtime == "faster-whisper"
    assert not r.models["qwen3.5-9b-vl"].served and "strata-360" in r.models["qwen3.5-9b-vl"].used_by
    d = r.to_json()
    assert {x["key"] for x in d["roles"]} >= {"faces", "transcribe"} and all("served" in m for m in d["models"])


def test_an_override_in_the_data_folder_changes_one_role_and_one_option(tmp_path):
    (tmp_path / "models.toml").write_text(
        '[roles.transcribe]\nmodel = "whisper-large-v3-turbo"\n[models.buffalo_l]\nvram_gb = 2.5\n[roles.transcribe-fast]\ndescription = "lyrics"\n',
        encoding="utf8",
    )
    import os

    os.environ["STRATAMODELS_DATA"] = str(tmp_path)
    r = Registry.load()
    assert (
        r.resolve("transcribe").key == "whisper-large-v3-turbo"
        and r.models["buffalo_l"].vram_gb == 2.5
        and r.models["buffalo_l"].runtime == "insightface"
    )
    assert r.roles["transcribe-fast"].description == "lyrics" and r.roles["transcribe-fast"].model == "whisper-small"
    assert settings.data_dir() == str(tmp_path)


def test_bad_registries_are_refused(tmp_path):
    with pytest.raises(RegistryError, match="not in the registry"):
        Registry.from_dict(dict(roles=dict(a=dict(model="x")), models={}))
    with pytest.raises(RegistryError, match="whose roles"):
        Registry.from_dict(dict(roles=dict(a=dict(model="x")), models=dict(x=dict(runtime="fake", roles=["b"]))))
    with pytest.raises(RegistryError, match="names no model"):
        Registry.from_dict(dict(roles=dict(a=dict()), models={}))
    with pytest.raises(RegistryError, match="model 'x'"):
        Registry.from_dict(dict(models=dict(x=dict(roles=["a"]))))
    with pytest.raises(RegistryError, match="model 'x'"):
        Registry.from_dict(dict(models=dict(x=dict(runtime="fake", vram_gb="lots"))))
    bad = tmp_path / "bad.toml"
    bad.write_text("[roles\n", encoding="utf8")
    with pytest.raises(RegistryError, match="bad.toml"):
        Registry.load(builtin=str(bad))
