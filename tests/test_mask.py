from translatorhoi4.translator.mask import mask_tokens, unmask_tokens, _extract_marked


def test_mask_unmask_roundtrip():
    src = 'Hello $NAME$ §Ytest§!'
    masked, mapping, idx = mask_tokens(src)
    assert masked != src and mapping
    unmasked = unmask_tokens(masked, mapping, idx)
    assert unmasked == src


def test_extract_marked():
    text = '<<SEG 1>>Hello<<END 1>>'
    assert _extract_marked(text, '1') == 'Hello'
