from translatorhoi4.translator.cache import DiskCache


def test_cache(tmp_path):
    p = tmp_path / 'cache.json'
    c = DiskCache(str(p))
    c.set('a', 'b')
    assert c.get('a') == 'b'
    c.save()
    c2 = DiskCache(str(p))
    assert c2.get('a') == 'b'
