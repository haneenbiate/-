"""Formatting repairs never rewrite human judgments or meaning-bearing text."""
import hashlib

import pandas as pd
import pytest

from arsarcasm_audit.audit import DataValidationError
from nasn.task05 import review_template
from nasn.adjudication_intake import canonicalize


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
