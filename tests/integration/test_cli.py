import json


def test_doctor_and_registry(cli):
    r = cli('doctor', check=False)
    assert 'strata-models 0.1.0' in r.stdout and 'runtime fake' in r.stdout
    roles = {x['key']: x['model'] for x in json.loads(cli('registry', '--json').stdout)['roles']}
    assert roles['faces'] == 'buffalo_l' and roles['transcribe'] == 'whisper-large-v3-turbo'


def test_status_and_jobs_read_a_running_service(cli, service):
    import strata_models_client as sm
    assert sm.run('echo', dict(n=1), app='cli-test') == dict(model='echo-small', echo=dict(n=1))
    assert 'echo-small' in cli('status', '--url', service).stdout
    assert 'cli-test' in cli('jobs', '--url', service).stdout


def test_settings_from_the_command_line(cli):
    assert '= 3.0' in cli('set', 'vram_reserve_gb', '3').stdout and '= 1.0' in cli('set', 'vram_reserve_gb', 'default').stdout
    assert cli('set', 'nope', 'x', check=False).code != 0
