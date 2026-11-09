import json
from app import settings


def test_doctor_reports_the_environment(cli):
    r = cli('doctor', check=False)
    assert 'app 0.1.0' in r.stdout and 'fastapi' in r.stdout and r.code == 0


def test_hello_and_message_log_from_the_command_line(cli):
    assert cli('hello', 'World').stdout.strip() == 'Hello, World!'
    rec = json.loads(cli('hello', 'Grace', '--greeting', 'Welcome', '--save', '--json').stdout)
    assert rec['message'] == 'Welcome, Grace!' and rec['id'] == 1
    msgs = json.loads(cli('messages', '--json').stdout)
    assert len(msgs) == 1 and msgs[0]['name'] == 'Grace'
    assert 'cleared 1 message(s)' in cli('messages', '--clear').stdout
    assert json.loads(cli('messages', '--json').stdout) == []


def test_settings_are_viewed_and_updated_from_the_command_line(cli, tmp_path, monkeypatch):
    monkeypatch.delenv('APP_DATA', raising=False)
    elsewhere = tmp_path / 'elsewhere'
    assert str(elsewhere) in cli('set', 'data_dir', elsewhere).stdout and settings.data_dir() == str(elsewhere.resolve())
    assert 'data_dir' in cli('set', 'data_dir', 'default').stdout
    assert cli('set', 'nope', 'x', check=False).code != 0


def test_the_command_line_runs_as_a_real_process(cli_subprocess):
    assert 'app 0.1.0' in cli_subprocess('doctor').stdout
    assert cli_subprocess('hello', 'Subprocess').stdout.strip() == 'Hello, Subprocess!'
