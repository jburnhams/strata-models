"""The HTTP API over real on-disk files: health check, greetings, message log, and settings."""


def test_health_and_hello_endpoints(client):
    assert client.get('/api/health').json() == dict(ok=True, version='0.1.0')
    assert client.get('/api/hello', params=dict(name='Linus')).json() == dict(name='Linus', greeting='Hello', message='Hello, Linus!')


def test_messages_created_via_http_persist_and_can_be_cleared(client, cli):
    assert client.get('/api/messages').json() == []
    m1 = client.post('/api/messages', json=dict(name='Alice')).json()
    m2 = client.post('/api/messages', json=dict(name='Bob', greeting='Hi')).json()
    assert m1['id'] == 1 and m2['id'] == 2
    assert [m['message'] for m in client.get('/api/messages').json()] == ['Hi, Bob!', 'Hello, Alice!']
    assert 'Hi, Bob!' in cli('messages').stdout
    assert client.delete('/api/messages').json() == dict(cleared=2)
    assert client.get('/api/messages').json() == []


def test_settings_changes_affect_subsequent_greetings(client):
    assert client.put('/api/settings', json=dict(greeting='Bonjour')).json()['greeting'] == 'Bonjour'
    assert client.get('/api/hello', params=dict(name='Monde')).json()['message'] == 'Bonjour, Monde!'
    assert client.put('/api/settings', json=dict(data_dir='relative')).status_code == 400
