import pandas as pd
import pytest

from nasn.task06 import token_signature,comparison_tables
from nasn.normalization import STRATEGIES,normalize


class ExactCharacterTokenizer:
    def __call__(self,text,truncation=False,max_length=128,**kwargs):
        ids=[101]+[ord(c) for c in text]+[102]
        return {'input_ids':ids[:max_length] if truncation else ids}


def test_character_to_token_audit_and_truncation():
    tokenizer=ExactCharacterTokenizer()
    text='كـتاب'
    assert token_signature(tokenizer,text)!=token_signature(tokenizer,normalize(text,'B1').text)
    # Change beyond model context: full IDs change while effective IDs do not.
    text='ا'*130+'ـكتاب'
    raw=token_signature(tokenizer,text); normalized=token_signature(tokenizer,normalize(text,'B1').text)
    assert raw['full_sha256']!=normalized['full_sha256']
    assert raw['effective_sha256']==normalized['effective_sha256']
    assert raw['truncated']


def test_paired_strategy_metrics_use_same_source_ids():
    rows=[]
    for strategy in STRATEGIES:
        for i in range(30):
            base={'strategy':strategy,'source_row_id':str(i),'gold_id':i%3,'high_risk':False,
                'source_human_confirmed':True,'source_review_status':'eligible_user_declared_human',
                'human_negation':'no','lexical_negation':False,'rule':'tatweel','intensity':1}
            rows.append({**base,'input_id':'clean:'+str(i),'kind':'clean','predicted_id':i%3})
            rows.append({**base,'input_id':str(i),'kind':'perturbed','predicted_id':i%3 if strategy!='B0' else (i+1)%3})
    table,delta=comparison_tables(pd.DataFrame(rows),100)
    part=delta[delta.strategy.eq('B4')&delta.subset.eq('all_development')&delta.subgroup.eq('all')].set_index('metric')
    assert part.loc['perturbed_macro_f1_gain','estimate']==1
    assert part.loc['flip_rate_reduction','estimate']==1
    # Synthetic outputs test the metrics; never reported as research evidence.
    assert table.normalization_human_validated.eq(False).all()


def test_no_duplicate_prediction_ids_or_missing_strategies():
    frame=pd.DataFrame([{'input_id':'x','strategy':'B0'},{'input_id':'x','strategy':'B0'}])
    with pytest.raises(ValueError,match='Repeated'):comparison_tables(frame,100)
    with pytest.raises(ValueError,match='Incomplete'):comparison_tables(frame.iloc[:1],100)


def test_reviewed_exclusions_never_receive_robustness_metrics():
    rows=[]
    for s in STRATEGIES:
        common={'strategy':s,'source_row_id':'x','gold_id':0,'high_risk':False,
            'source_human_confirmed':False,'source_review_status':'reviewed_excluded',
            'human_negation':'uncertain','lexical_negation':True,'rule':'tatweel','intensity':1}
        rows.extend([{**common,'input_id':'clean:x','kind':'clean','predicted_id':0},
                     {**common,'input_id':'p','kind':'perturbed','predicted_id':1}])
    metrics,deltas=comparison_tables(pd.DataFrame(rows),100)
    assert metrics.status.eq('excluded_by_source_human_review').all()
    assert 'estimate' not in metrics
    assert deltas.empty
