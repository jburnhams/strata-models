"""The registry (stratamodels.registry): the built-in file, the machine's override, resolving a role to its model, and refusing bad files."""

import pytest

from stratamodels import settings
from stratamodels.registry import Registry, RegistryError


def test_the_builtin_registry_names_a_model_for_every_role():
    r = Registry.load()
    assert r.resolve("faces").key == "buffalo_l" and r.resolve("transcribe").key == "whisper-large-v3-turbo"
    assert r.resolve("transcribe-fast", "whisper-large-v3-turbo").runtime == "venv-worker"
    assert (
        r.models["qwen3.5-9b-vl"].served
        and not r.models["yolo11s-pose"].served
        and "strata-360" in r.models["yolo11s-pose"].used_by
    )
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


@pytest.fixture
def engine():
    from stratamodels.store import memory_store

    return memory_store().engine


def _toml(tmp_path, text, name="models.toml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf8")
    return str(p)


BASE = '[roles.echo]\nmodel = "small"\ndescription = "echo"\n[models.small]\nruntime = "fake"\nroles = ["echo"]\nvram_gb = 2\n[models.big]\nruntime = "fake"\nroles = ["echo"]\nvram_gb = 6\n'


def test_the_builtin_file_seeds_the_tables_and_the_tables_drive_the_registry(engine):
    from stratamodels import registry as reg

    out = reg.seed(engine)
    r = Registry.from_db(engine)
    f = Registry.load()
    assert (
        "model buffalo_l" in out["added"]
        and sorted(r.to_json()["roles"], key=str) == sorted(f.to_json()["roles"], key=str)
        and {m["key"]: m for m in r.to_json()["models"]} == {m["key"]: m for m in f.to_json()["models"]}
    )
    assert reg.seed(engine) == dict(added=[], updated=[], kept=[])  # nothing changed: nothing written


def test_a_changed_file_updates_rows_nobody_edited_and_keeps_edited_ones(engine, tmp_path, clock):
    from stratamodels import registry as reg

    b = _toml(tmp_path, BASE)
    none = str(tmp_path / "none.toml")
    logs = []
    reg.seed(engine, b, none, clock, log=logs.append)
    assert "added 3" in logs[0]
    reg.save_model(engine, "small", vram_gb=1.5)
    b = _toml(
        tmp_path,
        BASE.replace("vram_gb = 2", "vram_gb = 3")
        .replace("vram_gb = 6", "vram_gb = 7")
        .replace('model = "small"', 'model = "big"'),
    )
    out = reg.seed(engine, b, none, clock)
    assert out == dict(added=[], updated=["model big", "role echo"], kept=["model small"])
    r = Registry.from_db(engine)
    assert r.models["small"].vram_gb == 1.5 and r.models["big"].vram_gb == 7 and r.resolve("echo").key == "big"
    reg.follow_seed(engine, "model", "small")
    reg.seed(engine, b, none, clock)
    assert Registry.from_db(engine).models["small"].vram_gb == 3
    with pytest.raises(RegistryError, match="no model"):
        reg.follow_seed(engine, "model", "nope")
    with pytest.raises(RegistryError, match="names model 'gone'"):
        reg.seed(engine, _toml(tmp_path, '[roles.x]\nmodel = "gone"\n', "bad.toml"), none, clock)


def test_edits_choose_models_add_models_and_refuse_broken_ones(engine, tmp_path):
    from stratamodels import registry as reg

    reg.seed(engine, _toml(tmp_path, BASE), str(tmp_path / "none.toml"))
    assert reg.set_role(engine, "echo", "big").resolve("echo").key == "big"
    with pytest.raises(RegistryError, match="not in the registry"):
        reg.set_role(engine, "echo", "nope")
    assert Registry.from_db(engine).resolve("echo").key == "big"  # a refused edit wrote nothing
    r = reg.save_model(engine, "tiny", roles=["echo", "other"], runtime="fake", vram_gb=0.5, options=dict(a=1))
    assert (
        r.models["tiny"].roles == ("echo", "other")
        and r.models["tiny"].options == dict(a=1)
        and r.resolve("echo", "tiny").key == "tiny"
    )
    assert (
        reg.set_role(engine, "other", "tiny", description="new").roles["other"].description == "new"
        and reg.set_role(engine, "other", description="newer").roles["other"].model == "tiny"
    )
    with pytest.raises(RegistryError, match="does not list"):
        reg.save_model(engine, "tiny", roles=["echo"])
    with pytest.raises(RegistryError, match="needs a model"):
        reg.set_role(engine, "fresh")
    with pytest.raises(RegistryError, match="needs a runtime"):
        reg.save_model(engine, "new")
    with pytest.raises(RegistryError, match="unknown model fields"):
        reg.save_model(engine, "tiny", colour="red")
    with pytest.raises(RegistryError, match="variant"):
        reg.save_model(engine, "tiny~1", runtime="fake")
    with pytest.raises(RegistryError, match="model 'tiny'"):
        reg.save_model(engine, "tiny", vram_gb="lots")


def test_preferences_give_an_app_its_own_model_and_default_params(engine, tmp_path, clock):
    from stratamodels import registry as reg
    from stratamodels.runtimes import fake
    from stratamodels.scheduler import Scheduler
    from stratamodels.store import Store

    reg.seed(engine, _toml(tmp_path, BASE), str(tmp_path / "none.toml"))
    r = reg.set_preference(engine, "bh", "echo", "big", dict(lang="en"), "bigger for bh")
    assert (
        r.preference("bh", "echo").to_json()
        == dict(app="bh", role="echo", model="big", params=dict(lang="en"), note="bigger for bh")
        and r.to_json()["preferences"][0]["app"] == "bh"
    )
    s = Scheduler(Store(engine, clock), r, str(tmp_path), clock=clock)
    fake.LOADED.clear()
    j = s.submit("echo", dict(x=1), "bh")
    assert j["model"] == "big" and j["params"] == dict(lang="en", x=1)
    assert (
        s.submit("echo", dict(lang="fr"), "bh")["params"] == dict(lang="fr")
        and s.submit("echo", {}, "bh", model="small")["model"] == "small"
    )
    assert s.submit("echo", {}, "other")["model"] == "small"
    assert reg.set_preference(engine, "bh", "echo", None, dict(lang="de")).preference("bh", "echo").model is None
    assert (
        reg.set_preference(engine, "bh", "echo").preference("bh", "echo") is None
        and reg.set_preference(engine, "bh", "echo").preferences == {}
    )
    with pytest.raises(RegistryError, match="no role"):
        reg.set_preference(engine, "bh", "nope", "big")
    with pytest.raises(RegistryError, match="does not serve"):
        reg.set_preference(engine, "bh", "echo", "missing")
    with pytest.raises(RegistryError, match="object"):
        reg.set_preference(engine, "bh", "echo", params=[1])
    s.registry.variant("small", dict(z=1))
    assert any("~" in k for k in s.reload_registry().models)  # variants survive a reload


def test_a_seed_that_would_break_an_edited_role_is_refused_and_writes_nothing(engine, tmp_path, clock):
    from stratamodels import registry as reg

    none = str(tmp_path / "none.toml")
    reg.seed(engine, _toml(tmp_path, BASE), none, clock)
    reg.set_role(engine, "echo", model="big")  # edited here: echo on big
    b = _toml(
        tmp_path,
        BASE.replace(
            '[models.big]\nruntime = "fake"\nroles = ["echo"]', '[models.big]\nruntime = "fake"\nroles = []'
        ).replace("vram_gb = 6", "vram_gb = 7"),
    )
    with pytest.raises(RegistryError, match="nothing written"):
        reg.seed(engine, b, none, clock)
    r = Registry.from_db(engine)
    assert r.resolve("echo").key == "big" and r.models["big"].vram_gb == 6 and "echo" in r.models["big"].roles


def test_an_edited_row_on_a_removed_runtime_moves_to_its_successor(engine, tmp_path, clock):
    from stratamodels import registry as reg

    old = '[roles.t]\nmodel = "w"\n[models.w]\nruntime = "faster-whisper"\nname = "small"\nroles = ["t"]\nvram_gb = 1\noptions = { compute_type = "float16" }\n'
    none = str(tmp_path / "none.toml")
    reg.seed(engine, _toml(tmp_path, old), none, clock)
    reg.save_model(engine, "w", vram_gb=0.8)  # edited here: the seed would normally leave it alone
    new = old.replace('runtime = "faster-whisper"', 'runtime = "venv-worker"').replace(
        "options = { compute_type", 'options = { worker = "whisper", venv = "whisper", compute_type'
    )
    out = reg.seed(engine, _toml(tmp_path, new), none, clock)
    m = Registry.from_db(engine).models["w"]
    assert out["updated"] == ["model w (runtime venv-worker)"] and m.runtime == "venv-worker" and m.vram_gb == 0.8
    assert m.options == dict(worker="whisper", venv="whisper", compute_type="float16") and m.served
