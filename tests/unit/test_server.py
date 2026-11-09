import pytest
import uvicorn
from fastapi.testclient import TestClient
from app.server import app as server_app
from app.server.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_landing_page_and_built_static_index(tmp_path, monkeypatch):
    monkeypatch.setattr(server_app, 'STATIC', str(tmp_path / 'no_static'))
    c = TestClient(create_app())
    assert 'python-app-template' in c.get('/').text and 'Say hello' in c.get('/').text
    assert c.get('/api/health').json()['ok'] is True

    static = tmp_path / 'static'; (static / 'assets').mkdir(parents=True)
    (static / 'index.html').write_text('<h1>built react app</h1>', encoding='utf8')
    (static / 'assets' / 'main.js').write_text('console.log(1)', encoding='utf8')
    monkeypatch.setattr(server_app, 'STATIC', str(static))
    c2 = TestClient(create_app())
    assert c2.get('/').text == '<h1>built react app</h1>'
    assert 'console.log(1)' in c2.get('/assets/main.js').text


def test_hello_and_messages_lifecycle(client):
    assert client.get('/api/hello').json() == dict(name='World', greeting='Hello', message='Hello, World!')
    assert client.get('/api/hello', params=dict(name='Ada', greeting='Hi')).json()['message'] == 'Hi, Ada!'
    assert client.get('/api/hello', params=dict(name='x' * 121)).status_code == 400

    assert client.get('/api/messages').json() == []
    m = client.post('/api/messages', json=dict(name='Ada', greeting='Hi')).json()
    assert m['id'] == 1 and m['message'] == 'Hi, Ada!'
    assert client.post('/api/messages', json=dict(name='x' * 121)).status_code == 400
    assert len(client.get('/api/messages').json()) == 1
    assert client.delete('/api/messages').json() == dict(cleared=1)
    assert client.get('/api/messages').json() == []


def test_settings_roundtrip_and_server_main(client, tmp_path, monkeypatch):
    assert client.put('/api/settings', json=dict(data_dir='relative/path')).status_code == 400
    new_dir = str(tmp_path / 'custom_data')
    monkeypatch.delenv('APP_DATA', raising=False)
    res = client.put('/api/settings', json=dict(data_dir=new_dir, greeting='Hey')).json()
    assert res == dict(data_dir=new_dir, greeting='Hey')
    assert client.get('/api/settings').json() == dict(data_dir=new_dir, greeting='Hey')

    runs = []
    monkeypatch.setattr(uvicorn, 'run', lambda app, host, port, log_level: runs.append((host, port, log_level)))
    server_app.main(['--host', '127.0.0.1', '--port', '8360'])
    assert runs == [('127.0.0.1', 8360, 'warning')]
