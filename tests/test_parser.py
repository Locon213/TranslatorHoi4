from translatorhoi4.parsers.paradox_yaml import LOCALISATION_LINE_RE, HEADER_RE


def test_header_re():
    assert HEADER_RE.match('l_english:')


def test_loc_line():
    line = ' KEY:0 "Value" #comment'
    m = LOCALISATION_LINE_RE.match(line)
    assert m
    assert m.group(2).strip() == 'KEY'
    assert m.group(4) == 'Value'
