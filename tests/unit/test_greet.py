import os
import pytest
from hypothesis import given, strategies as st
from app import greet, settings


def test_format_greeting_defaults_and_custom():
    assert greet.format_greeting() == dict(name='World', greeting='Hello', message='Hello, World!')
    assert greet.format_greeting('  Alice  ', '  Hi ') == dict(name='Alice', greeting='Hi', message='Hi, Alice!')
    assert greet.format_greeting('', '') == dict(name='World', greeting='Hello', message='Hello, World!')


def test_format_greeting_validation():
    with pytest.raises(greet.GreetingError, match='name is too long'):
        greet.format_greeting('x' * 121)
    with pytest.raises(greet.GreetingError, match='greeting is too long'):
        greet.format_greeting('Alice', 'g' * 61)


@given(name=st.text(min_size=1, max_size=40).filter(lambda s: s.strip()), greeting=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()))
def test_format_greeting_property(name, greeting):
    res = greet.format_greeting(name, greeting)
    assert res['name'] == name.strip()
    assert res['greeting'] == greeting.strip()
    assert res['message'] == f"{greeting.strip()}, {name.strip()}!"


def test_message_log_persistence_and_clear():
    assert greet.list_messages() == []
    m1 = greet.add_message('Alice')
    m2 = greet.add_message('Bob', 'Welcome')
    assert m1['id'] == 1 and m1['message'] == 'Hello, Alice!'
    assert m2['id'] == 2 and m2['message'] == 'Welcome, Bob!'
    msgs = greet.list_messages()
    assert [m['id'] for m in msgs] == [2, 1]

    assert greet.clear_messages() == dict(cleared=2)
    assert greet.list_messages() == []
    assert greet.clear_messages() == dict(cleared=0)


def test_list_messages_handles_corrupt_or_non_list_file():
    p = os.path.join(settings.data_dir(), 'messages.json')
    with open(p, 'w', encoding='utf8') as f: f.write('{bad json')
    assert greet.list_messages() == []
    with open(p, 'w', encoding='utf8') as f: f.write('{"not": "a list"}')
    assert greet.list_messages() == []
