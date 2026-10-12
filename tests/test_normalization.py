import json

import pytest

from nasn.normalization import (STRATEGIES, normalize, restore, load_rules,
                                text_hash, negation_guard)


@pytest.mark.parametrize('strategy',STRATEGIES)
@pytest.mark.parametrize('text',['لا أحب هذا أبداً', 'مو جيد ومش حلو', 'لم ولن وليس بدون غير',
    'ما أحبّه وما زال، ليس إلا', 'لا بأس يا سلام', 'لـا أُحِبّ هـذا',
    'فيلم كتااااب في الـشارع', 'إلى على علي ة ه ي ى أ إ آ ء ؤ ئ',
    '  كتاب\t\nالوقت  ', '', '@كتاااب #كتاااب https://example.com/كتاااب'] )
def test_exact_reversal_and_repeatability(strategy,text):
    result=normalize(text,strategy)
    assert restore(result.text,result)==text
    assert normalize(text,strategy)==result
    assert result.original_sha256==text_hash(text)
    assert result.semantic_status=='provisional_normalization_not_human_validated'


def test_baseline_is_byte_identity():
    text='  لــا\tإِلى كتااااب\n'
    assert normalize(text,'B0').text==text
    assert not normalize(text,'B0').edits


def test_fixed_baselines_do_only_declared_operations():
    text='  الـيوم\tا\u0654حب\n'
    assert normalize(text,'B1').text=='  اليوم\tا\u0654حب\n'
    assert normalize(text,'B2').text=='اليوم أحب'


@pytest.mark.parametrize('strategy',['B3','B4'])
def test_preserve_sensitive_words_expressions_and_hamza(strategy):
    text='لـا بـأس مـا شـاء الـله يا سـلام أُحـب جـيد رائـع إلى علي على ة ه ي ى'
    assert normalize(text,strategy).text==text
    assert normalize('جمييييل سيييئ أكرررره',strategy).text=='جمييييل سيييئ أكرررره'


def test_selective_rule_is_allowlisted_not_blanket_repeat_deletion():
    assert normalize('كتااااب اليييوم','B3').text=='كتاب اليوم'
    assert normalize('مدرررسة جميييل','B3').text=='مدرررسة جميييل'
    assert normalize('كتااب','B3').text=='كتااب'  # Two letters are below threshold.
    assert normalize('كتاااابx 2كتااااب @كتااااب #كتااااب','B3').text=='كتاااابx 2كتااااب @كتااااب #كتااااب'


def test_negation_scope_ablation_changes_only_guard_decision():
    text='مو كتااااب في الـبيت، اليييوم'
    assert normalize(text,'B3').text=='مو كتاب في البيت، اليوم'
    assert normalize(text,'B4').text=='مو كتااااب في الـبيت، اليوم'
    assert normalize('مو كتااااب. اليييوم','B4').text=='مو كتااااب. اليوم'


def test_guard_is_bounded_and_handles_attached_dialect_cues():
    text='وماأحبش كتااااب في الـبيت بعد اليييوم'
    assert normalize(text,'B4').text=='وماأحبش كتااااب في الـبيت بعد اليوم'
    spans=negation_guard(text,load_rules())
    assert spans and text[spans[0][0]:spans[0][1]]=='وماأحبش كتااااب في الـبيت'


def test_reversal_rejects_changed_output():
    result=normalize('كتااااب','B3')
    with pytest.raises(ValueError,match='altered output'):restore(result.text+'!',result)


def test_invalid_input_and_rule_change_fail(tmp_path):
    with pytest.raises(ValueError):normalize(None)
    with pytest.raises(ValueError):normalize('نص','B9')
    rules=load_rules();rules['scope_following_arabic_tokens']=4
    path=tmp_path/'rules.json';path.write_text(json.dumps(rules))
    with pytest.raises(ValueError,match='parameters changed'):load_rules(path)
