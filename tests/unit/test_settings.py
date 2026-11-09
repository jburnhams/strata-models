import os
import pytest
from stratamodels import settings


def test_defaults_env_and_file(tmp_path, monkeypatch):
    monkeypatch.delenv('STRATAMODELS_DATA', raising=False); monkeypatch.delenv('STRATAMODELS_HOME', raising=False)
    assert settings.home() == os.path.join(os.path.expanduser('~'), '.strata-models')
    assert settings.data_dir() == os.path.abspath(settings.DEFAULTS['data_dir']) and settings.models_dir() == os.path.join(os.path.expanduser('~'), 'models')
    assert settings.vram_reserve_gb() == 1.0
    monkeypatch.setenv('STRATAMODELS_MODELS', str(tmp_path / 'w')); monkeypatch.setenv('STRATAMODELS_VRAM_RESERVE_GB', '2.5')
    assert settings.models_dir() == str(tmp_path / 'w') and (tmp_path / 'w').is_dir() and settings.vram_reserve_gb() == 2.5
    monkeypatch.setenv('STRATAMODELS_VRAM_RESERVE_GB', 'lots'); assert settings.vram_reserve_gb() == 1.0


def test_save_load_reset_and_refusals():
    assert settings.load() == {}
    assert settings.save(dict(vram_reserve_gb=1.5, models_dir='/tmp/w')) == dict(vram_reserve_gb='1.5', models_dir='/tmp/w')
    assert settings.vram_reserve_gb() == 1.5 and settings.all_settings()['models_dir'] == '/tmp/w'
    assert 'models_dir' not in settings.save(dict(models_dir=None))
    with pytest.raises(KeyError, match='unknown setting'): settings.save(dict(greeting='x'))
    with pytest.raises(ValueError, match='number'): settings.save(dict(vram_reserve_gb='x'))
    with pytest.raises(ValueError, match='between'): settings.save(dict(vram_reserve_gb=100))


def test_corrupt_or_non_dict_settings_file():
    os.makedirs(settings.home(), exist_ok=True); p = os.path.join(settings.home(), 'settings.json')
    with open(p, 'w', encoding='utf8') as f: f.write('{not valid json')
    assert settings.load() == {}
    with open(p, 'w', encoding='utf8') as f: f.write('[1, 2, 3]')
    assert settings.load() == {} and settings.all_settings()['vram_reserve_gb'] == 1.0
