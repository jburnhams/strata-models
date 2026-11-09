"""stratamodels.evals.similarity: text (words and sound), images (perceptual hash, structure, colour), JSON, and picking the kind."""
import pytest
from PIL import Image

from stratamodels.evals import similarity as S


def tx(t): return dict(segments=[dict(text=t)])


def test_phonetic_keys_match_words_that_sound_alike():
    assert S.phonetic('there') == S.phonetic('their') and S.phonetic('four') == S.phonetic('for') and S.phonetic('knight') == S.phonetic('night')
    assert S.phonetic('phone') == S.phonetic('fone') and S.phonetic('cat') != S.phonetic('dog') and S.phonetic('42') == '42' and S.phonetic("'") == "'"


def test_text_similarity_ranks_what_a_listener_would():
    same = S.similarity(tx(" Anders' face grew red."), tx("anders' face grew red"))
    spelling = S.similarity(tx('their bikes are over there'), tx('there bikes are over their'))
    words = S.similarity(tx('their bikes are over there'), tx('the hikes are over here'))
    assert same == 1.0 and 1.0 > spelling > words > 0 and S.similarity('', '', 'text') == 1.0
    assert S.edit_similarity([], []) == 1.0 and S.edit_similarity('abc', 'abd') == pytest.approx(2 / 3)


def test_image_similarity_orders_pictures(tmp_path):
    def pic(name, f):
        im = Image.new('RGB', (128, 128)); px = im.load()
        for x in range(128):
            for y in range(128): px[x, y] = f(x, y)
        p = tmp_path / name; im.save(p); return dict(path=str(p))
    a = pic('a.png', lambda x, y: (x * 2, y * 2, 100)); a2 = pic('a2.png', lambda x, y: (min(x * 2 + 3, 255), y * 2, 100))
    b = pic('b.png', lambda x, y: (255 - y * 2, 30, x * 2))
    near, far = S.similarity(a, a2), S.similarity(a, b)
    assert S.similarity(a, a) == 1.0 and 1.0 > near > far and S.kind_of(a, b) == 'image'
    assert S.image_similarity(a['path'], str(tmp_path / 'missing.png')) == 0.0 and S.similarity(a, dict(path='x.txt'), 'image') == 0.0


def test_image_similarity_without_pillow_compares_bytes(tmp_path, monkeypatch):
    p, q = tmp_path / 'p.png', tmp_path / 'q.png'; p.write_bytes(b'x'); q.write_bytes(b'y')
    def no_pil(*a): raise ImportError('PIL')
    monkeypatch.setattr(S, '_image_features', no_pil)
    assert S.image_similarity(str(p), str(q)) == 0.0 and S.image_similarity(str(p), str(tmp_path / 'p.png')) == 1.0
    assert S.image_similarity(str(p), str(tmp_path / 'none.png')) == 0.0


def test_json_and_kinds():
    assert S.similarity(dict(a=1, seconds=2.0), dict(a=1, seconds=9.0)) == 1.0 and 0 < S.similarity(dict(a=1), dict(a=2)) < 1
    assert S.kind_of(tx('a'), tx('b')) == 'text' and S.kind_of('a', 'b') == 'text' and S.kind_of(dict(a=1), [1]) == 'json'
    with pytest.raises(ValueError, match='no similarity'): S.similarity('a', 'b', 'audio')
