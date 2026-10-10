"""Synthetic annotation fixtures verify software, never claim human review results."""
import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nasn.task05 import (HUMAN, DECISIONS, VISIBLE, PROBS, csv, review_template,
    validate_decisions, resolve_reviews, agreement, confirmed_tables, tatweel_comparison,
    html_review, seal, registry, analyze, verify_preparation, validate_saved_predictions, import_reviews)
from arsarcasm_audit.audit import DataValidationError


@pytest.fixture
def template():
    data = pd.DataFrame({'variant_id':['v1','v2','v3'], 'source_row_id':['s1','s2','s3'],
        'original_sha256':['a'*64]*3,'perturbed_sha256':['b'*64]*3,
        'original_text':['لا أحب الكتاب','الكتاب جميل','ذهب إلى المدرسة'],
        'perturbed_text':['لا أحب الكـتاب','الكتـاب جميل','ذهب الى المدرسة'],
        'rule':['tatweel','tatweel','alef_allowlist'],'intensity':['1','2','1'],
        'gold_label':['NEG','POS','NEU'],'predicted_id':[0,2,1]})
    return review_template(data)


def decisions(template, reviewer):
    result = template.copy()
    result['semantic_validity']='preserved';result['sentiment_preserved']='yes'
    result['negation_present']='no';result['negation_affects_sentiment']='no'
    result['linguistic_notes']='نفس المعنى والمشاعر (synthetic test only)'
    result['reviewer_id']=reviewer;result['review_timestamp']='2026-10-10T10:00:00+03:00'
    return result


def write(tmp_path, frame, name='decisions.csv'):
    path=tmp_path/name;frame.to_csv(path,index=False,encoding='utf-8-sig');return path


def test_template_blinded_blank_and_utf8(template,tmp_path):
    assert list(template.columns)==VISIBLE+HUMAN
    assert template[HUMAN].eq('').all().all()
    assert not any('probability' in c or c in ['gold_label','predicted_id','has_negation'] for c in template)
    path=write(tmp_path,template)
    assert path.read_bytes().startswith(b'\xef\xbb\xbf')
    assert csv(path).equals(template)
    assert validate_decisions(path,template).empty


@pytest.mark.parametrize('change', ['text','missing','duplicate','hash','added_prediction','invalid','partial','timestamp','contradiction','scope'])
def test_invalid_decisions_rejected(template,tmp_path,change):
    frame=decisions(template,'A')
    if change=='text':frame.loc[0,'original_text']='أحب الكتاب'
    if change=='missing':frame=frame.iloc[1:]
    if change=='duplicate':frame=pd.concat([frame.iloc[:2],frame.iloc[:1]])
    if change=='hash':frame.loc[0,'perturbed_sha256']='c'*64
    if change=='added_prediction':frame['predicted_id']=0
    if change=='invalid':frame.loc[0,'semantic_validity']='valid'
    if change=='partial':frame.loc[0,'sentiment_preserved']=''
    if change=='timestamp':frame.loc[0,'review_timestamp']='2026-10-10T10:00:00'
    if change=='contradiction':frame.loc[0,'negation_affects_sentiment']='yes'
    if change=='scope':frame.loc[0,'negation_present']='yes'
    with pytest.raises((DataValidationError,ValueError)):
        validate_decisions(write(tmp_path,frame),template)


def test_order_relocation_and_negation_scope(template,tmp_path):
    frame=decisions(template,'A')
    for k,v in {'negation_present':'yes','negation_affects_sentiment':'yes',
                'negation_cue':'لا','original_negation_scope':'أحب الكتاب',
                'perturbed_negation_scope':'أحب الكـتاب'}.items():frame.loc[0,k]=v
    loaded=validate_decisions(write(tmp_path,frame.iloc[::-1]),template)
    assert loaded.variant_id.tolist()==template.variant_id.tolist()
    assert loaded.loc[0,'negation_cue']=='لا'


def test_independence_and_undefined_agreement(template):
    a=decisions(template,'A');b=decisions(template,'B')
    _,reports,coverage=agreement(a,b,3)
    assert coverage['paired_coverage']==1
    assert all(r['exact_agreement']==1 and r['cohen_kappa'] is None for r in reports)
    with pytest.raises(DataValidationError,match='distinct'):
        agreement(a,a,3)


def test_kappa_before_adjudication(template):
    a=decisions(template,'A');b=decisions(template,'B')
    a['semantic_validity']=['preserved','changed','ambiguous']
    b['semantic_validity']=['preserved','changed','preserved']
    _,reports,_=agreement(a,b,3)
    semantic=reports[0]
    assert semantic['exact_agreement']==pytest.approx(2/3)
    assert semantic['cohen_kappa']==pytest.approx(.5)


def test_disagreements_require_third_human(template,tmp_path):
    a,b=decisions(template,'A'),decisions(template,'B')
    b.loc[0,'semantic_validity']='changed'
    final,conflicts,_,coverage=resolve_reviews(template,a,b)
    assert conflicts.variant_id.tolist()==['v1']
    assert not coverage['human_review_completed'] and coverage['unresolved_disagreements']==1
    assert set(final.variant_id)=={'v2','v3'}
    third=decisions(conflicts,'C')
    final,_,_,coverage=resolve_reviews(template,a,b,write(tmp_path,third))
    assert coverage['human_review_completed'] and coverage['confirmed_n']==3
    assert final.loc[final.variant_id.eq('v1'),'decision_basis'].iloc[0]=='explicit_third_reviewer_adjudication'
    third['reviewer_id']='A'
    with pytest.raises(DataValidationError,match='Adjudicator'):
        resolve_reviews(template,a,b,write(tmp_path,third))


def test_uncertain_and_changed_cases_not_confirmed(template):
    a=decisions(template,'A');b=decisions(template,'B')
    for frame in [a,b]:
        frame.loc[0,'negation_present']='uncertain'
        frame.loc[1,'semantic_validity']='changed'
        frame.loc[2,'semantic_validity']='ambiguous'
    final,_,_,coverage=resolve_reviews(template,a,b)
    assert coverage['human_review_completed'] and not final.eligible_confirmed.any()


def test_scope_wording_difference_requires_adjudication(template):
    a,b=decisions(template,'A'),decisions(template,'B')
    a.loc[0,'original_negation_scope']='Scope 1'
    b.loc[0,'original_negation_scope']='Scope 2'
    _,conflicts,_,coverage=resolve_reviews(template,a,b)
    assert len(conflicts)==1 and not coverage['human_review_completed']


def test_no_human_pairs_no_agreement(template):
    a=template.iloc[:0]
    _,reports,coverage=agreement(a,a,3)
    assert coverage['paired_completed']==0 and all(r['exact_agreement'] is None for r in reports)


def test_import_blank_assignments_reports_coverage_without_decisions(template,tmp_path):
    package=tmp_path/'package';package.mkdir()
    template.to_csv(package/'reviewer_a.csv',index=False)
    template.to_csv(package/'reviewer_b.csv',index=False)
    metadata=template[['variant_id','rule','intensity']].assign(gold_label='NEU',dialect='msa',sarcasm='FALSE',has_negation='False')
    metadata.to_csv(package/'queue_registry.csv',index=False);seal(package)
    status=import_reviews(package,package/'reviewer_a.csv',package/'reviewer_b.csv',None,tmp_path/'audit')
    assert status['status']=='prepared_review_incomplete'
    assert status['confirmed_n']==status['paired_completed']==0
    assert csv(tmp_path/'audit'/'resolved_decisions.csv').empty


def test_confirmed_paired_bootstrap_and_small_strata():
    n=35
    changed=pd.DataFrame({'variant_id':[f'v{i}' for i in range(n)],'source_row_id':[f's{i}' for i in range(n)],
        'gold_id':np.arange(n)%3,'clean_id':np.arange(n)%3,'predicted_id':np.arange(n)%3,
        'rule':'tatweel','intensity':1,'high_risk_near_duplicate':False,'lexical_negation':np.arange(n)%2==0,'applicable':True})
    human=pd.DataFrame({'variant_id':changed.variant_id,'eligible_confirmed':True,
        'negation_present':['yes']*3+['no']*(n-3),'negation_affects_sentiment':'no'})
    tables=confirmed_tables(changed,human,100)
    overall=tables[(tables.subgroup=='all')&(tables.subset=='all_reviewed')]
    assert len(overall)>0 and (overall.estimate[overall.metric.eq('flip_rate')]==0).all()
    neg=tables[(tables.subgroup=='negation_present')&(tables.value=='yes')]
    assert set(neg.status)=={'insufficient_sample_size'} and set(neg.n)=={3}
    repeated=confirmed_tables(changed.iloc[::-1],human,100)
    pd.testing.assert_frame_equal(tables,repeated)


def test_tatweel_vectors_predictions_and_token_ids():
    changed=pd.DataFrame({'variant_id':['v1','v2'],'source_row_id':['s','s'],'rule':'tatweel','intensity':[1,2],
        'applicable':True,'predicted_id':[0,2],PROBS[0]:[.8,.1],PROBS[1]:[.1,.1],PROBS[2]:[.1,.8]})
    clean=pd.DataFrame({'source_row_id':['s'],'predicted_id':[0],PROBS[0]:[.8],PROBS[1]:[.1],PROBS[2]:[.1]})
    manifest=pd.DataFrame({'variant_id':['v1','v2'],'original_text':['اب','اب'],'perturbed_text':['اـب','اــب']})
    def tokenizer(text,**kwargs):return {'input_ids':[ord(c) for c in text]}
    details,summary=tatweel_comparison(changed,clean,manifest,tokenizer)
    between=summary[summary.comparison.eq('1_vs_2')].iloc[0]
    assert between.changed_effective_token_sequences==between.changed_probability_vectors_exact==between.changed_final_predictions==1
    assert 'left_token_ids' in details and 'probability_delta_NEG' in details
    _,missing=tatweel_comparison(changed,clean,manifest)
    assert missing.changed_effective_token_sequences.isna().all()
    assert set(missing.token_status)=={'unavailable_verified_tokenizer'}


def test_review_incomplete_stops_before_inference(tmp_path):
    package=tmp_path/'package';package.mkdir();(package/'placeholder').write_text('x');seal(package)
    decisions_dir=tmp_path/'decisions';decisions_dir.mkdir()
    (decisions_dir/'review_coverage.json').write_text(json.dumps({'human_review_completed':False}));seal(decisions_dir)
    args=SimpleNamespace(review_package=str(package),decisions=str(decisions_dir))
    with pytest.raises(DataValidationError,match='Human review incomplete'):
        analyze(args,{})


def test_artifact_tampering_detected(tmp_path):
    (tmp_path/'review.csv').write_text('original');seal(tmp_path)
    registry(tmp_path)
    (tmp_path/'review.csv').write_text('altered')
    with pytest.raises(DataValidationError,match='checksum mismatch'):
        registry(tmp_path)


def test_offline_html_escapes_script_and_rtl(template,tmp_path):
    template.loc[0,'original_text']='</script><script>alert(1)</script>'
    destination=tmp_path/'review.html';html_review(template,destination)
    source=destination.read_text()
    assert 'dir="rtl"' in source and '__REVIEW_DATA__' not in source
    assert '</script><script>alert(1)</script>' not in source
    assert 'localStorage' not in source and 'fetch(' not in source
    node=shutil.which('node')
    if node:
        script=source.split('<script>')[1].split('</script>')[0]
        path=tmp_path/'interface.js';path.write_text(script)
        subprocess.run([node,'--check',str(path)],check=True,capture_output=True)
        harness = '''
const assert=require('node:assert/strict');
class Element {constructor(){this.value='';this.children=[];} append(x){this.children.push(x);}}
const elements=new Map();
global.document={getElementById:k=>{if(!elements.has(k))elements.set(k,new Element());return elements.get(k);},createElement:()=>new Element()};
''' + script + '''
assert.deepEqual(parseCSV('\\ufeff"a","b"\\r\\n"سطر\\nثاني","quote""end"'),[['a','b'],['سطر\\nثاني','quote"end']]);
assert.equal(rows[0].semantic_validity,'');
$('reviewer').value='reviewer-A';$('semantic_validity').value='preserved';persist();
assert.equal(rows[0].reviewer_id,'reviewer-A');assert.equal(rows[0].semantic_validity,'preserved');
assert.notEqual(rows[0].review_timestamp,'');assert.equal(rows[1].semantic_validity,'');
'''
        path.write_text(harness)
        subprocess.run([node,str(path)],check=True,capture_output=True)


@pytest.mark.parametrize('mutation',['none','probability','pairing','identity','checkpoint','status'])
def test_saved_predictions_integrity(tmp_path,mutation):
    evidence={'manifest_sha256':{'validation':'a'},'checkpoint_sha256':{'model':'b'},
              'source_commit':'rev','source_training_sha256':'c'}
    pd.DataFrame().to_csv(tmp_path/'robustness_metrics.csv',index=False)
    status={'status':'completed','training_performed':False,'test_evaluated':False,'test_text_read':False}
    saved=evidence.copy()
    if mutation=='checkpoint':saved={**saved,'checkpoint_sha256':{'model':'changed'}}
    if mutation=='status':status['status']='blocked_or_failed'
    (tmp_path/'run_status.json').write_text(json.dumps(status));(tmp_path/'evidence.json').write_text(json.dumps(saved))
    clean=pd.DataFrame({'source_row_id':['s1'],'gold_id':[0],'predicted_id':[0],
                       PROBS[0]:[.8],PROBS[1]:[.1],PROBS[2]:[.1]})
    manifest=pd.DataFrame({'variant_id':['v1'],'source_row_id':['s1'],'original_sha256':['a'],
        'perturbed_sha256':['b'],'gold_id':['0'],'rule':['tatweel'],'intensity':['1'],
        'high_risk_near_duplicate':['False'],'has_negation':['True'],'applicable':['True']})
    changed=manifest.copy();changed['clean_id']=0;changed['predicted_id']=0
    for key in PROBS:changed[key]=clean[key]
    if mutation=='probability':changed[PROBS[0]]=2
    if mutation=='pairing':changed['clean_id']=1
    if mutation=='identity':changed['perturbed_sha256']='altered'
    clean.to_csv(tmp_path/'clean_predictions.csv',index=False)
    changed.to_csv(tmp_path/'perturbed_predictions.csv',index=False);seal(tmp_path)
    if mutation=='none':
        a,b,_=validate_saved_predictions(tmp_path,manifest,clean,evidence)
        assert b.clean_id.tolist()==[0] and b.lexical_negation.tolist()==[True]
    else:
        with pytest.raises(DataValidationError):validate_saved_predictions(tmp_path,manifest,clean,evidence)


def test_real_preparation_reads_no_official_test(monkeypatch):
    if not Path('runs/task04-final-v2-preparation/preparation.json').exists():
        pytest.skip('Original full-text Task 04 private preparation not present; run documented preparation first')
    import nasn.task04 as task04
    original=task04.load_csv
    def guarded(path):
        assert 'testing_data' not in str(path)
        return original(path)
    monkeypatch.setattr(task04,'load_csv',guarded)
    queue,manifest,_,evidence=verify_preparation('runs/task04-final-v2-preparation',
        'reports/task03b/marbert-v1-20261009T213143Z-be8b8811',json.loads(Path('configs/marbert_v1.json').read_text()))
    assert len(queue)==404 and len(manifest)==12545
    assert evidence['checkpoint_bytes_verified'] is False and evidence['test_evaluated'] is False
