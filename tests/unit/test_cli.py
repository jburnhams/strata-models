import importlib, json
import pytest
from app import cli


def test_doctor_success_and_missing_module(capsys, monkeypatch):
    assert cli.main(['doctor']) == 0
    assert 'fastapi' in capsys.readouterr().out

    real_import = importlib.import_module
    def fake_import(name):
        if name == 'uvicorn': raise ImportError('no uvicorn')
        return real_import(name)
    monkeypatch.setattr(importlib, 'import_module', fake_import)
    assert cli.main(['doctor']) == 1
    assert 'MISSING  uvicorn' in capsys.readouterr().out


def test_hello_and_messages_commands(capsys):
    assert cli.main(['hello']) == 0
    assert capsys.readouterr().out.strip() == 'Hello, World!'

    assert cli.main(['hello', 'Alice', '--greeting', 'Hi', '--save', '--json']) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec['message'] == 'Hi, Alice!' and rec['id'] == 1

    assert cli.main(['messages']) == 0
    assert 'Hi, Alice!' in capsys.readouterr().out

    assert cli.main(['messages', '--json']) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1

    assert cli.main(['messages', '--clear']) == 0
    assert 'cleared 1 message(s)' in capsys.readouterr().out

    assert cli.main(['messages']) == 0
    assert 'no messages yet' in capsys.readouterr().out

    assert cli.main(['messages', '--clear', '--json']) == 0
    assert json.loads(capsys.readouterr().out) == {'cleared': 0}

    with pytest.raises(SystemExit, match='name is too long'):
        cli.main(['hello', 'x' * 121])


def test_set_and_serve_commands(capsys, monkeypatch):
    assert cli.main(['set', 'greeting']) == 0
    assert 'greeting = Hello' in capsys.readouterr().out

    assert cli.main(['set', 'greeting', 'Howdy']) == 0
    assert 'greeting = Howdy' in capsys.readouterr().out

    assert cli.main(['set', 'greeting', 'default']) == 0
    assert 'greeting = Hello' in capsys.readouterr().out

    with pytest.raises(SystemExit, match='unknown setting'):
        cli.main(['set', 'bad_key', 'val'])

    from app.server import app as server_app
    called = []
    monkeypatch.setattr(server_app, 'main', lambda argv: called.append(argv))
    assert cli.main(['serve', '--host', '0.0.0.0', '--port', '9000']) == 0
    assert called == [['--host', '0.0.0.0', '--port', '9000']]
