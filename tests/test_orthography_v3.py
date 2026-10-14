import pytest

from nasn.orthography_v3 import Condition, TOKEN, is_cue, perturb, rewrite

TEXT = 'الفيلم كان جميل جدا ولا أحب النهاية https://t.co/x @user'


def test_condition_names_round_trip():
    for name in ['tatweel/single/non_cue', 'elongationx3/dense/non_cue', 'elongationx3/single/cue', 'fold/dense/non_cue']:
        assert Condition.parse(name).name == name
    with pytest.raises(ValueError):
        Condition('typo', 'single', 'cue')
    with pytest.raises(ValueError):
        Condition('elongation', 'single', 'non_cue')


@pytest.mark.parametrize('word,family,repeat,expected', [
    ('حلو', 'tatweel', 0, 'حلـو'), ('كثير', 'elongation', 3, 'كثيييير'), ('حلو', 'elongation', 2, 'حلووو'),
    ('لا', 'elongation', 3, 'لاااا'), ('أحب', 'fold', 0, 'احب'), ('جميلة', 'fold', 0, 'جميله'),
    ('على', 'fold', 0, 'علي'), ('ورد', 'tatweel', 0, None), ('كتب', 'fold', 0, None)])
def test_rewrite(word, family, repeat, expected):
    assert rewrite(word, family, repeat) == expected


def test_cues_are_untouched_by_non_cue_conditions():
    for name in ['tatweel/dense/non_cue', 'elongationx3/dense/non_cue', 'fold/dense/non_cue', 'typo/dense/non_cue']:
        changed, edits = perturb(TEXT, 'r1', Condition.parse(name))
        assert edits and 'ولا' in changed.split()
        assert changed.endswith('https://t.co/x @user')
        assert all(not is_cue(e['source']) for e in edits)


def test_cue_target_edits_only_cue():
    changed, edits = perturb(TEXT, 'r1', Condition.parse('elongationx3/single/cue'))
    assert [e['source'] for e in edits] == ['ولا'] and 'ولاااا' in changed
    assert perturb('الفيلم جميل', 'r1', Condition.parse('tatweel/single/cue')) == ('الفيلم جميل', [])


def test_nested_selection_and_offsets():
    single = perturb(TEXT, 'r9', Condition.parse('tatweel/single/non_cue'))[1]
    dense = perturb(TEXT, 'r9', Condition.parse('tatweel/dense/non_cue'))[1]
    assert {e['start'] for e in single} <= {e['start'] for e in dense}
    assert len(dense) == 3  # ceil(0.3 * 7 Arabic words)
    x2 = perturb(TEXT, 'r9', Condition.parse('elongationx2/single/non_cue'))[1]
    x5 = perturb(TEXT, 'r9', Condition.parse('elongationx5/single/non_cue'))[1]
    assert x2[0]['start'] == x5[0]['start']
    changed, edits = perturb(TEXT, 'r9', Condition.parse('fold/dense/non_cue'))
    for e in edits:
        assert changed[e['target_start']:e['target_start'] + len(e['target'])] == e['target']


def test_deterministic_and_order_independent():
    c = Condition.parse('typo/single/non_cue')
    assert perturb(TEXT, 'a', c) == perturb(TEXT, 'a', c)
    assert len({perturb(TEXT, f'id{i}', c)[0] for i in range(30)}) > 1


def test_skips_attached_latin_and_marked_words():
    assert perturb('abcحلو', 'x', Condition.parse('tatweel/single/non_cue')) == ('abcحلو', [])
    assert perturb('حُلو', 'x', Condition.parse('tatweel/single/non_cue')) == ('حُلو', [])
    assert TOKEN.findall('hi') == []
