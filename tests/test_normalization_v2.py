import pytest

from nasn.normalization_v2 import STRATEGIES,normalize,explain
from nasn.normalization import restore


@pytest.mark.parametrize('strategy',STRATEGIES)
@pytest.mark.parametrize('text',['لااا أحبببب هذا','لـا مووو ومششش ليسسس','لا بأس ما شاء الله يا سلام',
    'ومااااحبش كتتتاب في الـبيت','إلى على علي ة ه ي ى أ إ آ ء ؤ ئ',
    '@كتتتاب #كتتتاب https://site.test/كتتتاب 123كتتتاب', '  كتتتاب\t\n', '',
    'مَدْرررسة حزييين جمييييل', 'للللغة الله للله'])
def test_reversible_and_deterministic(strategy,text):
    result=normalize(text,strategy)
    assert restore(result.text,result)==text
    assert normalize(text,strategy)==result
    assert result.semantic_status=='provisional_normalization_not_human_validated'


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_marker_bytes_are_never_changed(strategy):
    text='لـا مووو مششش ومااااحبش ليسسس'
    assert normalize(text,strategy).text==text


def test_run_rule_and_morphological_counterexample():
    assert normalize('كتتتاب','REP2').text=='كتتاب'
    assert normalize('كتتاب','REP2').text=='كتتاب'  # Cannot presume two letters are an error.
    assert normalize('للللغة','REP2').text=='للغة'  # Retain two; do not turn لل into ل.
    assert normalize('اااا ئئئء ييي ةةة','REP2').text=='اااا ئئئء ييي ةةة'
    assert normalize('مَدْرررسة','REP2').text=='مَدْرررسة'  # Marked words abstain.


def test_scope_ablation_only_blocks_repetition_not_typography():
    text='مو كتتتاب في الـبيت'
    assert normalize(text,'NO_CONTEXT').text=='مو كتتاب في البيت'
    assert normalize(text,'NASN2').text=='مو كتتتاب في البيت'
    assert normalize('مو فيلم في الـبيت','NASN2').text=='مو فيلم في البيت'
    assert normalize('مو كتتتاب. كتتتاب','NASN2').text=='مو كتتتاب. كتتاب'


def test_sensitive_words_and_counterexamples_are_not_auto_validated():
    assert normalize('أحببب أكره جمييييل ليسسس جيد','NASN2').text=='أحببب أكره جمييييل ليسسس جيد'
    # A limited lexicon cannot certify generic cap semantics/emphasis.
    result=normalize('مننن','REP2')
    assert result.text=='منن' and 'not_human_validated' in result.semantic_status
    assert normalize('ماكر كـتاب','TAT').text=='ماكر كتاب'  # Broad marker guard preserves ماكر itself.


def test_explanation_has_rule_and_scope_abstention():
    result,decisions=explain('مو كتتتاب في الـبيت','NASN2')
    assert any(d.get('repetition_decision')=='abstain_negation_context' for d in decisions)
    assert any('tatweel_outside_marker' in e.rule for e in result.edits)


def test_invalid_strategy_fails():
    with pytest.raises(ValueError):normalize('نص','UNKNOWN')


def test_possible_conjunction_negator_false_positive_is_not_posthoc_repaired():
    # فلممم can be a stretched dialect spelling of فيلم, but ف + لم is also
    # caught by the deliberately broad possible-marker heuristic. Abstain,
    # record the limit, and do not add a dataset-specific exception.
    result,decisions=explain('فلممم','NO_CONTEXT')
    assert result.text=='فلممم'
    assert decisions[0]['marker'] is True
