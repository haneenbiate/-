"""Synthetic fixtures test software only, not MARBERT performance or semantic validity."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from arsarcasm_audit.audit import sha256, DataValidationError
from nasn.orthography import RULES, lexical_negation, perturb
from nasn.task04 import (build_manifest, review_sample, apply_reviews, paired_scores,
    paired_bootstrap, robustness_tables, validation_inputs, verify_run, run_study, REVISION)
from nasn.metrics import calculate, prediction_frame


@pytest.fixture
def validation():
    return pd.DataFrame({'source_row_id':['official_train:00001','official_train:00002','official_train:00003'],
        'row_id':['1','2','3'],'group_id':['g1','g2','g3'],'partition':['validation']*3,
        'sentiment':['NEG','NEU','POS'],'dialect':['msa','gulf','msa'],'sarcasm':['FALSE','TRUE','FALSE'],
        'tweet':['لم أحب هذا العمل الجميل','ذهب محمد إلى المدرسة','أحب الكتاب كثيرا'],
        'high_risk_near_duplicate':[True,False,False],'has_negation':[True,False,False]})


@pytest.mark.parametrize('text',['لا أحب الكتاب','وما أحب الكتاب','فلم أحب الكتاب','لَـن أكره الحياة','ماحبيتش هذا الكتاب','ليس إلا العمل'])
@pytest.mark.parametrize('rule,intensity',RULES)
def test_negation_and_spans_preserved(text,rule,intensity):
    original=lexical_negation(text)
    assert original
    changed,edit=perturb(text,'id',rule,intensity)
    assert [x[2] for x in lexical_negation(changed)]==[x[2] for x in original]
    if edit:
        assert text[edit.start:edit.end]==edit.source_span
        assert changed==text[:edit.start]+edit.target_span+text[edit.end:]
        assert changed[edit.start:edit.start+len(edit.target_span)]==edit.target_span
        assert text[:edit.start]==changed[:edit.start]


def test_deterministic_nested_intensity():
    for rule in ['elongation','tatweel']:
        a,e=perturb('الكتاب جميل جدا','id',rule,1)
        b,f=perturb('الكتاب جميل جدا','id',rule,2)
        assert e.start==f.start and e.end==f.end
        assert len(b)-len(a)==1
        assert perturb('الكتاب جميل جدا','id',rule,1)==(a,e)


def test_hamza_sensitive_not_blanket_mapped():
    text='سأل أحمد عن أهم أثر'
    assert perturb(text,'id','alef_allowlist',1)==(text,None)
    text='ذهب إلى المدرسة'
    changed,edit=perturb(text,'id','alef_allowlist',1)
    assert changed=='ذهب الى المدرسة' and edit.ambiguity


@pytest.mark.parametrize('text',['https://example.com/كتاب','#الكتاب','@الكتاب','abcكتاب123','لا لم لن مو مش','123 abc'])
def test_protected_and_inapplicable(text):
    for rule,intensity in RULES:
        assert perturb(text,'id',rule,intensity)==(text,None)


def test_invalid_rule_or_text():
    with pytest.raises(ValueError): perturb('كتاب','id','blanket_hamza',1)
    with pytest.raises(ValueError): perturb('','id','tatweel',1)


def test_manifest_identity_order_and_provisional(validation):
    manifest=build_manifest(validation)
    assert len(manifest)==15 and not manifest.variant_id.duplicated().any()
    assert set(manifest.semantic_status)<={'provisional','not_applicable'}
    for row in manifest.itertuples():
        source=validation.set_index('source_row_id').loc[row.source_row_id]
        assert row.original_text==source.tweet and row.gold_label==source.sentiment
    reversed_manifest=build_manifest(validation.iloc[::-1]).set_index('variant_id').sort_index()
    pd.testing.assert_frame_equal(manifest.set_index('variant_id').sort_index(),reversed_manifest)


def test_review_is_per_example_and_hash_verified(validation):
    manifest=build_manifest(validation)
    review=review_sample(manifest,1)
    assert set(review.review_status)=={'pending'}
    review=review.iloc[:1].copy()
    review['review_status']='valid'; review['reviewer']='Fixture reviewer'; review['rationale']='Synthetic fixture only'
    updated=apply_reviews(manifest,review)
    assert updated.semantic_status.eq('confirmed').sum()==1
    assert updated.semantic_status.eq('provisional').any()
    review['perturbed_sha256']='wrong'
    with pytest.raises(DataValidationError,match='hash'):apply_reviews(manifest,review)


@pytest.mark.parametrize('decision',['invalid','ambiguous'])
def test_invalid_and_ambiguous_never_confirmed(validation,decision):
    manifest=build_manifest(validation); review=review_sample(manifest).iloc[:1].copy()
    review['review_status']=decision; review['reviewer']='fixture'; review['rationale']='fixture'
    updated=apply_reviews(manifest,review)
    assert updated.semantic_status.eq(decision).sum()==1
    assert not updated.semantic_status.eq('confirmed').any()


def test_human_decision_requires_traceability(validation):
    manifest=build_manifest(validation); review=review_sample(manifest).iloc[:1].copy()
    review['review_status']='valid'
    with pytest.raises(DataValidationError,match='reviewer'):apply_reviews(manifest,review)
    with pytest.raises(DataValidationError,match='Duplicate'):apply_reviews(manifest,pd.concat([review,review]))


def test_paired_scores_match_sklearn():
    rng=np.random.default_rng(4)
    for n in [1,20,100]:
        gold=rng.integers(0,3,n); a=rng.integers(0,3,n); b=rng.integers(0,3,n)
        result=paired_scores(gold,a,b)
        before,classes,_=calculate(gold,a); after,_,_=calculate(gold,b)
        assert result['macro_f1_drop']==pytest.approx(before['macro_f1']-after['macro_f1'])
        for item in classes: assert result[f'clean_{item["label"]}_f1']==pytest.approx(item['f1'])


def test_paired_bootstrap_identity_and_reproducibility():
    frame=pd.DataFrame({'source_row_id':['a','b','c'],'gold_id':[0,1,2],'clean_id':[0,1,2],'predicted_id':[0,1,2]})
    a=paired_bootstrap(frame,100); assert a==paired_bootstrap(frame,100)
    for key in ['accuracy_drop','macro_f1_drop','flip_rate','NEG_f1_drop']:
        assert a[key]=={'estimate':0.,'ci_low':0.,'ci_high':0.}
    with pytest.raises(DataValidationError,match='original ID'):paired_bootstrap(pd.concat([frame,frame]),100)


def test_robustness_sensitivity_exclusions(validation):
    frame=build_manifest(validation).query('applicable').copy()
    frame['clean_id']=frame.gold_id; frame['predicted_id']=frame.gold_id
    result=robustness_tables(frame,100,minimum=1)
    assert set(result.subset)=={'all_validation','exclude_high_risk'}
    assert set(result.semantic_status)=={'provisional','provisional_population'}
    frame['semantic_status']='invalid'
    assert robustness_tables(frame,100,minimum=1).empty


def test_population_denominator_includes_noop(validation):
    frame=build_manifest(validation)
    frame['clean_id']=frame.gold_id;frame['predicted_id']=frame.gold_id
    result=robustness_tables(frame,100,minimum=1)
    overall=result[(result.subset=='all_validation') & (result.subgroup=='all') & (result.coverage=='population_with_noop')]
    assert set(overall.n)=={3}
    assert set(overall.semantic_status)=={'provisional_population'}


def test_attached_dialect_negator_overprotection():
    # Guard need not call unknown compounds a semantic negation subgroup.
    for word in ['مااحب','موحلو','ماحبيت']:
        for rule,intensity in RULES:
            changed,_=perturb(f'{word} الكتاب الجميل','id',rule,intensity)
            assert changed.startswith(word+' ')


def test_archive_import_hashes_and_path_safety(tmp_path):
    import hashlib,zipfile
    from nasn.task04 import import_reports
    payload=b'{"status":"fixture_only"}'
    archive=tmp_path/'reports.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('run_status.json',payload)
        z.writestr('report_checksums.json',json.dumps({'run_status.json':hashlib.sha256(payload).hexdigest()}))
    result=import_reports(archive,tmp_path/'ok')
    assert result['status']=='imported'
    assert (tmp_path/'ok'/'run_status.json').read_bytes()==payload
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('../raw.csv',b'unsafe')
    with pytest.raises(DataValidationError,match='Unsafe'):import_reports(archive,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('run_status.json',payload)
        z.writestr('report_checksums.json',json.dumps({'run_status.json':'0'*64}))
    with pytest.raises(DataValidationError,match='checksum mismatch'):import_reports(archive,tmp_path/'bad')


def test_reports_only_is_not_local_weight_verification(baseline,validation):
    run,checkpoint,evidence,config=baseline
    # The explicit report analysis mode requires no weight bytes; GPU evaluation does.
    for path in checkpoint.iterdir():path.unlink()
    verify_run(run,None,validation,evidence,config)
    with pytest.raises(DataValidationError,match='checkpoint files'):verify_run(run,checkpoint,validation,evidence,config)


def test_no_cpu_or_model_fallback_when_gpu_unavailable(tmp_path,monkeypatch,validation):
    import nasn.task04 as module
    import torch
    from transformers import AutoModelForSequenceClassification
    monkeypatch.setattr(module,'validation_inputs',lambda config:(validation,pd.DataFrame(),{}))
    monkeypatch.setattr(module,'verify_run',lambda *args:(pd.DataFrame(),{},{}))
    monkeypatch.setattr(module,'validate_near_report',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def prohibited(*args,**kwargs):pytest.fail('Model must not load before CUDA gate')
    monkeypatch.setattr(AutoModelForSequenceClassification,'from_pretrained',prohibited)
    args=SimpleNamespace(output=tmp_path/'out',run=tmp_path/'run',checkpoint=tmp_path/'checkpoint',reviews=None)
    with pytest.raises(DataValidationError,match='CUDA GPU'):run_study(args,{})
    status=json.loads((args.output/'run_status.json').read_text())
    assert status['metrics'] is None and status['training_performed'] is False


def test_clean_to_wrong_and_correction_directions():
    result=paired_scores([0,1,2],[0,0,0],[1,1,1])
    assert result['flip_rate']==1
    assert result['correct_to_wrong_rate']==pytest.approx(1/3)
    assert result['wrong_to_correct_rate']==pytest.approx(1/3)
    assert result['wrong_to_wrong_flip_rate']==pytest.approx(1/3)


def test_metric_reports_tolerate_roundoff_but_not_changed_results():
    from nasn.task04 import verify_metric_report
    actual=[{'value':'FALSE','n':2049,'macro_f1':0.75,'weighted_f1':0.7475,'status':'computed'}]
    saved=json.loads(json.dumps(actual));saved[0]['weighted_f1']=float(np.nextafter(0.7475,1.))
    verify_metric_report(saved,actual)
    saved[0]['weighted_f1']+=1e-5
    with pytest.raises(DataValidationError,match='weighted_f1'):verify_metric_report(saved,actual)
    saved=json.loads(json.dumps(actual));saved[0]['n']=2048
    with pytest.raises(DataValidationError,match='metadata'):verify_metric_report(saved,actual)
    saved=json.loads(json.dumps(actual));saved[0]['value']='TRUE'
    with pytest.raises(DataValidationError,match='metadata'):verify_metric_report(saved,actual)
    saved=json.loads(json.dumps(actual));saved[0]['macro_f1']=float('nan')
    with pytest.raises(DataValidationError,match='macro_f1'):verify_metric_report(saved,actual)


def test_analysis_failure_preserves_diagnostic_status(tmp_path,monkeypatch):
    import nasn.task04 as module
    def invalid(config):raise DataValidationError('Fixture: manifest checksum mismatch')
    monkeypatch.setattr(module,'validation_inputs',invalid)
    output=tmp_path/'analysis'
    with pytest.raises(DataValidationError,match='checksum'):module.analyze_reports(tmp_path/'run',output,{})
    status=json.loads((output/'run_status.json').read_text())
    assert status['status']=='blocked_or_failed' and 'checksum mismatch' in status['reason']
    assert status['analysis_completed'] is False and status['training_performed'] is False


@pytest.fixture
def baseline(tmp_path,validation):
    """Complete synthetic evidence, never claimed as an actual model experiment."""
    run=tmp_path/'run'; run.mkdir(); checkpoint=run/'best_checkpoint'; checkpoint.mkdir()
    config=json.loads(Path('configs/marbert_v1.json').read_text())
    hashes={}
    for name in ['model.safetensors','config.json','vocab.txt','tokenizer_config.json','special_tokens_map.json']:
        file=checkpoint/name; file.write_text('synthetic'); hashes[name]=sha256(file)
    def put(name,value): (run/name).write_text(json.dumps(value))
    put('run_status.json',{'status':'completed','checkpoint_reload_verified':True,'test_evaluated':False,'model_revision':REVISION,'best_epoch':1})
    put('resolved_training_config.json',config)
    evidence={'counts':{'train':10037,'validation':2509,'quarantine':2,'test':3000},'manifest_sha256':{'fixture':'hash'},'source_training_sha256':'train-hash'}
    from arsarcasm_audit.audit import COMMIT
    put('preflight.json',{'status':'passed','counts':evidence['counts'],'manifest_sha256':evidence['manifest_sha256'],
        'source_commit':COMMIT,'source_sha256':{'official_train':'train-hash'},'overlap':[{'exact_groups':0,'canonical_groups':0}]})
    put('checkpoint_hashes.json',hashes)
    prediction=prediction_frame(validation,np.eye(3))
    prediction.to_csv(run/'validation_predictions.csv',index=False)
    metrics,classes,matrix=calculate(prediction.gold_id,prediction.predicted_id)
    put('validation_metrics.json',{'status':'completed','checkpoint_epoch':1,**metrics})
    pd.DataFrame(classes).to_csv(run/'per_class_results.csv',index=False)
    pd.DataFrame(matrix,index=['NEG','NEU','POS']).to_csv(run/'confusion_matrix.csv')
    pd.DataFrame([{'epoch':1,'macro_f1':metrics['macro_f1']}]).to_csv(run/'epoch_history.csv',index=False)
    return run,checkpoint,evidence,config


def test_complete_evidence_verification_and_weight_corruption(baseline,validation):
    run,checkpoint,evidence,config=baseline
    verify_run(run,checkpoint,validation,evidence,config)
    (checkpoint/'model.safetensors').write_text('corrupt')
    with pytest.raises(DataValidationError,match='Checkpoint hash'):verify_run(run,checkpoint,validation,evidence,config)


@pytest.mark.parametrize('change',['tokenizer','extra_file','test','protocol','manifest','selection','confidence'])
def test_invalid_evidence_rejected(baseline,validation,change):
    run,checkpoint,evidence,config=baseline
    if change=='tokenizer':(checkpoint/'vocab.txt').write_text('tampered')
    elif change=='extra_file':(checkpoint/'pytorch_model.bin').write_text('unverified')
    elif change=='test':
        value=json.loads((run/'run_status.json').read_text()); value['test_evaluated']=True
        (run/'run_status.json').write_text(json.dumps(value))
    elif change=='protocol':
        value=json.loads((run/'resolved_training_config.json').read_text()); value['seed']=99
        (run/'resolved_training_config.json').write_text(json.dumps(value))
    elif change=='manifest':evidence['manifest_sha256']={'fixture':'different'}
    elif change=='selection':
        value=json.loads((run/'run_status.json').read_text()); value['best_epoch']=2
        (run/'run_status.json').write_text(json.dumps(value))
    elif change=='confidence':
        frame=pd.read_csv(run/'validation_predictions.csv');frame['confidence']=0.1;frame.to_csv(run/'validation_predictions.csv',index=False)
    with pytest.raises(DataValidationError):verify_run(run,checkpoint,validation,evidence,config)


def test_missing_run_records_blocker_without_predictions(tmp_path,monkeypatch,validation):
    import nasn.task04 as module
    monkeypatch.setattr(module,'validation_inputs',lambda config:(validation,pd.DataFrame(),{}))
    args=SimpleNamespace(output=tmp_path/'out',run=tmp_path/'absent',checkpoint=tmp_path/'absent-weight')
    with pytest.raises(DataValidationError,match='missing'):run_study(args,{})
    status=json.loads((args.output/'run_status.json').read_text())
    assert status['status']=='blocked_or_failed' and status['metrics'] is None
    assert not list(args.output.glob('*predictions*'))


def test_real_validation_never_opens_test_csv(monkeypatch):
    # Actual pinned data readiness check; if absent, skip explicitly.
    if not Path('data/raw/training_data.csv').exists():pytest.skip('Pinned official training CSV not present')
    import nasn.task04 as module
    original=module.load_csv
    def guard(path):
        assert Path(path).name=='training_data.csv'
        return original(path)
    monkeypatch.setattr(module,'load_csv',guard)
    config=json.loads(Path('configs/marbert_v1.json').read_text())
    config['raw_testing']='/does/not/exist/testing_data.csv'
    validation,near,evidence=validation_inputs(config)
    assert len(validation)==2509 and len(near)==753
    assert evidence['test_text_read'] is False


def test_published_artifacts_are_anchored_and_text_free():
    directory=Path('reports/task04')
    if not directory.exists():pytest.skip('Published Task 04 artifacts not present')
    registry=json.loads((directory/'artifact_hashes.json').read_text())
    for name,digest in registry.items():assert sha256(directory/name)==digest
    for name in ['perturbation_manifest.csv','human_review_queue.csv']:
        frame=pd.read_csv(directory/name,dtype={'sarcasm':str},keep_default_na=False)
        assert 'original_text' not in frame and 'perturbed_text' not in frame
        assert not frame.variant_id.duplicated().any()
        assert 'confirmed' not in set(frame.semantic_status)
        assert set(frame.sarcasm)<={'FALSE','TRUE'}
    preparation=json.loads((directory/'preparation.json').read_text())
    assert preparation['confirmed_variants']==0 and preparation['human_review_completed'] is False


def test_task04_notebook_is_pinned_unexecuted_and_inference_only():
    import ast,nbformat
    notebook=nbformat.read('notebooks/task04_robustness_colab.ipynb',as_version=4)
    nbformat.validate(notebook)
    source='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
    assert 'bde827f741a625e5b8b4ceaca7c6dd6bc36901a6' in source
    assert "'nasn.task04','evaluate'" in source
    assert 'nasn.marbert' not in source and 'testing_data.csv' not in source
    assert 'AutoModel' not in source and '.fit(' not in source and 'torch.optim' not in source
    assert 'checkpoint_access.json' in source and 'gpu_runtime.json' in source
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is None and cell.outputs==[]
            ast.parse(cell.source)


def test_notebook_command_failure_exposes_stderr(tmp_path,capsys):
    import ast,subprocess,sys,uuid,nbformat
    notebook=nbformat.read('notebooks/task04_robustness_colab.ipynb',as_version=4)
    cell=next(c for c in notebook.cells if c.cell_type=='code' and 'def call(' in c.source)
    function=next(node for node in ast.parse(cell.source).body if isinstance(node,ast.FunctionDef) and node.name=='call')
    scope={'subprocess':subprocess,'json':json,'uuid':uuid,'ROOT':tmp_path}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook-call>','exec'),scope)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        scope['call']([sys.executable,'-c',"import sys;print('ERROR: synthetic diagnostic',file=sys.stderr);sys.exit(2)"])
    assert 'ERROR: synthetic diagnostic' in failure.value.stderr
    assert 'ERROR: synthetic diagnostic' in capsys.readouterr().out
    reports=list(tmp_path.glob('command-failure-*.json'))
    assert len(reports)==1
    saved=json.loads(reports[0].read_text())
    assert saved['exit_code']==2 and 'synthetic diagnostic' in saved['stderr']
    source='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
    assert "'--analysis',ANALYSIS_DIR" in source
