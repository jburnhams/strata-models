import os

import pytest

from stratamodels import settings


def test_defaults_env_and_file(tmp_path, monkeypatch):
    monkeypatch.delenv("STRATAMODELS_DATA", raising=False)
    monkeypatch.delenv("STRATAMODELS_HOME", raising=False)
    assert settings.home() == os.path.join(os.path.expanduser("~"), ".strata-models")
    assert settings.data_dir() == os.path.abspath(
        settings.DEFAULTS["data_dir"]
    ) and settings.models_dir() == os.path.join(os.path.expanduser("~"), "models")
    assert settings.vram_reserve_gb() == 1.0
    monkeypatch.setenv("STRATAMODELS_MODELS", str(tmp_path / "w"))
    monkeypatch.setenv("STRATAMODELS_VRAM_RESERVE_GB", "2.5")
    assert (
        settings.models_dir() == str(tmp_path / "w") and (tmp_path / "w").is_dir() and settings.vram_reserve_gb() == 2.5
    )
    monkeypatch.setenv("STRATAMODELS_VRAM_RESERVE_GB", "lots")
    assert settings.vram_reserve_gb() == 1.0


def test_save_load_reset_and_refusals():
    assert settings.load() == {}
    assert settings.save(dict(vram_reserve_gb=1.5, models_dir="/tmp/w")) == dict(
        vram_reserve_gb="1.5", models_dir="/tmp/w"
    )
    assert settings.vram_reserve_gb() == 1.5 and settings.all_settings()["models_dir"] == os.path.abspath("/tmp/w")
    assert "models_dir" not in settings.save(dict(models_dir=None))
    with pytest.raises(KeyError, match="unknown setting"):
        settings.save(dict(greeting="x"))
    with pytest.raises(ValueError, match="number"):
        settings.save(dict(vram_reserve_gb="x"))
    with pytest.raises(ValueError, match="between"):
        settings.save(dict(vram_reserve_gb=100))
    assert settings.whole_card_gb() == 10.0 and settings.save(dict(whole_card_gb=9))["whole_card_gb"] == "9"
    assert settings.all_settings()["whole_card_gb"] == 9.0
    with pytest.raises(ValueError, match="whole_card_gb must be a number"):
        settings.save(dict(whole_card_gb="big"))


def test_job_slots_default_validate_and_come_from_the_environment(monkeypatch):
    assert settings.all_settings()["gpu_jobs"] == 2 and settings.jobs("cpu_jobs") == 4
    assert settings.save(dict(gpu_jobs=" 3 "))["gpu_jobs"] == "3" and settings.jobs("gpu_jobs") == 3
    with pytest.raises(ValueError, match="whole number"):
        settings.save(dict(cpu_jobs="many"))
    with pytest.raises(ValueError, match="between 1 and 16"):
        settings.save(dict(gpu_jobs=0))
    monkeypatch.setenv("STRATAMODELS_CPU_JOBS", "8")
    assert settings.jobs("cpu_jobs") == 8
    monkeypatch.setenv("STRATAMODELS_GPU_JOBS", "lots")
    assert settings.jobs("gpu_jobs") == 2  # a bad value falls back to the default


def test_corrupt_or_non_dict_settings_file():
    os.makedirs(settings.home(), exist_ok=True)
    p = os.path.join(settings.home(), "settings.json")
    with open(p, "w", encoding="utf8") as f:
        f.write("{not valid json")
    assert settings.load() == {}
    with open(p, "w", encoding="utf8") as f:
        f.write("[1, 2, 3]")
    assert settings.load() == {} and settings.all_settings()["vram_reserve_gb"] == 1.0


def test_bound_settings_live_in_the_database_and_start_from_the_file():
    from stratamodels.store import memory_store

    settings.save(dict(vram_reserve_gb=2.5))
    e = memory_store().engine
    settings.bind(e)
    assert settings.load() == dict(vram_reserve_gb="2.5") and settings.vram_reserve_gb() == 2.5
    assert settings.save(dict(models_dir="/tmp/w", vram_reserve_gb=None)) == dict(
        models_dir="/tmp/w"
    ) and settings.load() == dict(models_dir="/tmp/w")
    assert (
        settings.save(dict(models_dir="/tmp/v"))["models_dir"] == "/tmp/v" and settings.load()["models_dir"] == "/tmp/v"
    )
    settings.bind(None)
    assert settings.load() == dict(vram_reserve_gb="2.5")  # the file was left alone
    settings.bind(e)
    assert settings.load() == dict(models_dir="/tmp/v")  # a table with rows is not refilled from the file


def test_the_file_is_imported_once_so_settings_reset_to_defaults_stay_reset():
    from stratamodels.store import memory_store

    settings.save(dict(vram_reserve_gb=2.5))
    e = memory_store().engine
    settings.bind(e)
    assert settings.load() == dict(vram_reserve_gb="2.5")
    settings.save(dict(vram_reserve_gb=None))
    assert settings.load() == {}
    settings.bind(None)
    settings.bind(e)
    assert settings.load() == {} and settings.vram_reserve_gb() == 1.0
    settings.bind(None)
