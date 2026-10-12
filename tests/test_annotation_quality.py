import pandas as pd
import pytest

from nasn.annotation_quality import audit_tables,FREE_FIELDS,digest
from nasn.task05 import DECISIONS


def records():
    rows=[]
    for role in ['A','B','FINAL']:
        for i in range(3):
            record={'variant_id':str(i),'role':role,'semantic_validity':'preserved',
                'sentiment_preserved':'yes','negation_present':'no','negation_affects_sentiment':'no',
                'eligible_confirmed':'True' if role=='FINAL' else '',
                'decision_basis':'two_explicit_independent_human_decisions' if role=='FINAL' else ''}
            record.update({field:digest('') for field in FREE_FIELDS});rows.append(record)
    # One literal scope disagreement, resolved explicitly to a third wording.
    b=next(r for r in rows if r['role']=='B' and r['variant_id']=='0');b['original_negation_scope']=digest('second')
    f=next(r for r in rows if r['role']=='FINAL' and r['variant_id']=='0')
    f['original_negation_scope']=digest('third');f['decision_basis']='explicit_third_reviewer_adjudication'
    adj={**f,'role':'ADJ','eligible_confirmed':'','decision_basis':''};rows.append(adj)
    return pd.DataFrame(rows)


def test_pre_adjudication_agreement_and_literal_conflict():
    a,c,r=audit_tables(records())
    assert a.set_index('dimension').loc['semantic_validity','kappa_status']=='undefined_constant'
    scope=a.set_index('dimension').loc['original_negation_scope']
    assert scope.observed_agreement==pytest.approx(2/3)
    assert scope.kappa_status=='not_applicable_free_text'
    assert len(r)==1 and r.iloc[0].category=='scope_wording_only'
    assert r.iloc[0].original_negation_scope_resolution=='different_from_both'
    assert c[c.dimension=='semantic_validity'].n.sum()==3


def test_missing_original_or_false_adjudication_fails():
    frame=records()
    with pytest.raises(ValueError,match='Missing original'):audit_tables(frame[frame.role!='B'])
    with pytest.raises(ValueError,match='Resolution differs'):
        changed=frame.copy();changed.loc[changed.role.eq('FINAL')&changed.variant_id.eq('0'),'original_negation_scope']=digest('invented')
        audit_tables(changed)


def test_duplicate_assignment_and_invalid_category_fail():
    frame=records()
    with pytest.raises(ValueError,match='Duplicate'):audit_tables(pd.concat([frame,frame.iloc[:1]]))
    frame.loc[0,'semantic_validity']='unknown'
    with pytest.raises(ValueError,match='Invalid annotation'):audit_tables(frame)
