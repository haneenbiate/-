"""Formatting repairs never rewrite human judgments or meaning-bearing text."""
import hashlib

import pandas as pd
import pytest

from arsarcasm_audit.audit import DataValidationError
from nasn.task05 import review_template
from nasn.adjudication_intake import canonicalize


def test_end_to_end_intake_handles_nonzero_conflict_index(tmp_path,pair):
    import datetime
    import openpyxl
    from nasn.task05 import HUMAN,import_reviews,seal,csv
    from nasn.adjudication_intake import intake
    sample,one=pair
    template=pd.concat([one,one],ignore_index=True)
    template.loc[1,'variant_id']='v2';template.loc[1,'source_row_id']='s2'
    package=tmp_path/'package';package.mkdir()
    template.to_csv(package/'reviewer_a.csv',index=False)
    template.to_csv(package/'reviewer_b.csv',index=False)
    template[['variant_id','rule','intensity']].assign(gold_label='NEG',dialect='msa',sarcasm='FALSE',has_negation='True').to_csv(package/'queue_registry.csv',index=False)
    seal(package)
    a,b=template.copy(),template.copy()
    for frame,who in [(a,'R1'),(b,'R2')]:
        for field in HUMAN:frame[field]=sample[field].iloc[0]
        frame['reviewer_id']=who;frame['review_timestamp']='2026-10-12T08:00:00+03:00'
    b.loc[1,'semantic_validity']='ambiguous'
    a.to_csv(tmp_path/'a.csv',index=False);b.to_csv(tmp_path/'b.csv',index=False)
    independent=tmp_path/'independent'
    import_reviews(package,tmp_path/'a.csv',tmp_path/'b.csv',None,independent)
    adjud=csv(independent/'adjudication_queue.csv')
    for field in HUMAN:adjud[field]=sample[field].iloc[0]
    adjud['reviewer_id']='R3';adjud['original_text']=adjud.original_text.str.strip(' ')
    adjud['perturbed_text']=adjud.perturbed_text.str.strip(' ');adjud['diffs']='auxiliary only'
    workbook=tmp_path/'adjudication.xlsx';w=openpyxl.Workbook();s=w.active
    s.append(adjud.columns.tolist())
    for row in adjud.itertuples(index=False,name=None):s.append(list(row))
    s['O2']=datetime.datetime(2026,10,12,8,45);w.save(workbook)
    status=intake(package,independent,workbook,tmp_path/'result','Asia/Baghdad',True)
    assert status['human_review_completed'] and status['confirmed_n']==2
    assert csv(tmp_path/'result'/'canonical_adjudication.csv').review_timestamp.iloc[0]=='2026-10-12T08:45:00+03:00'


@pytest.fixture
def pair():
    records=pd.DataFrame({'variant_id':['v1'],'source_row_id':['s1'],
        'original_text':['  لا أحب الفيلم  '],'perturbed_text':['  لا أحب الفيلـم  '],
        'rule':['tatweel'],'intensity':['1']})
    for field,digest in [('original_text','original_sha256'),('perturbed_text','perturbed_sha256')]:
        records[digest]=records[field].map(lambda x:hashlib.sha256(x.encode()).hexdigest())
    template=review_template(records)
    frame=template.copy()
    frame['original_text']=frame.original_text.str.strip(' ')
    frame['perturbed_text']=frame.perturbed_text.str.strip(' ')
    frame['semantic_validity']='preserved';frame['sentiment_preserved']='yes'
    frame['negation_present']='yes';frame['negation_affects_sentiment']='yes'
    frame['linguistic_notes']='Synthetic human rationale'
    frame['negation_cue']='لا';frame['original_negation_scope']='أحب الفيلم';frame['perturbed_negation_scope']='أحب الفيلـم'
    frame['reviewer_id']='R3';frame['review_timestamp']='2026-10-12 08:45:00';frame['diffs']="[('insert', '', 'ـ')]"
    return frame,template


def test_explicit_repairs_restore_only_registered_text_and_declared_timezone(pair):
    frame,template=pair
    corrected,changes=canonicalize(frame,template,'Asia/Baghdad',True)
    assert corrected.original_text.tolist()==template.original_text.tolist()
    assert corrected.perturbed_text.tolist()==template.perturbed_text.tolist()
    assert corrected.review_timestamp.iloc[0]=='2026-10-12T08:45:00+03:00'
    assert len(changes)==3 and 'diffs' not in corrected
    for field in ['semantic_validity','linguistic_notes','negation_cue','original_negation_scope','reviewer_id']:
        assert corrected[field].tolist()==frame[field].tolist()


def test_naive_timestamp_needs_explicit_timezone(pair):
    frame,template=pair
    with pytest.raises(DataValidationError,match='timezone required'):
        canonicalize(frame,template,None,True)


def test_spaces_never_restored_without_explicit_option(pair):
    frame,template=pair
    with pytest.raises(DataValidationError,match='protected field differs'):
        canonicalize(frame,template,'Asia/Baghdad',False)


@pytest.mark.parametrize('field,value',[('original_text','أحب الفيلم'),
    ('original_text','لا أحبالفيلم'),('original_sha256','changed'),('source_row_id','s2')])
def test_meaning_sensitive_changes_and_hash_changes_are_rejected(pair,field,value):
    frame,template=pair;frame.loc[0,field]=value
    with pytest.raises(DataValidationError,match='protected field differs'):
        canonicalize(frame,template,'Asia/Baghdad',True)


def test_existing_timezone_is_preserved(pair):
    frame,template=pair;frame.loc[0,'review_timestamp']='2026-10-12T08:45:00+00:00'
    corrected,changes=canonicalize(frame,template,None,True)
    assert corrected.review_timestamp.iloc[0]==frame.review_timestamp.iloc[0]
    assert set(changes.field)=={'original_text','perturbed_text'}


def test_unknown_extra_columns_and_duplicate_ids_fail(pair):
    frame,template=pair;frame['model_prediction']='NEG'
    with pytest.raises(DataValidationError,match='columns changed'):
        canonicalize(frame,template,'Asia/Baghdad',True)
    frame=frame.drop(columns='model_prediction');frame=pd.concat([frame,frame],ignore_index=True)
    with pytest.raises(DataValidationError,match='duplicated'):
        canonicalize(frame,template,'Asia/Baghdad',True)
