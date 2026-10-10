"""Validation-only orthographic feasibility study. Never trains or loads test text."""
import argparse
import hashlib
import json
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import COMMIT, HASHES, load_csv, sha256
from .preflight import LABEL2ID, EXPECTED, require, load_partition, audit_near, library_versions
from .metrics import calculate, prediction_frame, subgroup_metrics
from .orthography import VERSION, RULES, SEED, perturb, lexical_negation, failure_reason

REVISION = '88e1fa192dd723cf0b3563500aec46209762eb22'


def code_hashes():
    directory=Path(__file__).parent
    return {name:sha256(directory/name) for name in ['task04.py','orthography.py','marbert.py','metrics.py','preflight.py']}


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def fresh(path):
    path=Path(path)
    require(not path.exists(), f'Output exists; preserve evidence and select a new directory: {path}')
    path.mkdir(parents=True)
    return path


def read(path):
    require(Path(path).is_file(),f'Required evidence missing: {path}')
    return json.loads(Path(path).read_text())


def verify_metric_report(saved, actual, path='subgroups'):
    """Exact report structure/counts/labels; float metrics tolerate 1e-12 roundoff."""
    if isinstance(actual,dict):
        require(isinstance(saved,dict) and set(saved)==set(actual),f'Saved metric report keys mismatch: {path}')
        for key in actual:verify_metric_report(saved[key],actual[key],f'{path}.{key}')
    elif isinstance(actual,list):
        require(isinstance(saved,list) and len(saved)==len(actual),f'Saved metric report length mismatch: {path}')
        for i,(a,b) in enumerate(zip(saved,actual)):verify_metric_report(a,b,f'{path}[{i}]')
    elif isinstance(actual,float):
        require(isinstance(saved,(int,float)) and not isinstance(saved,bool) and
                np.isfinite(saved) and np.isfinite(actual) and abs(saved-actual)<=1e-12,
                f'Saved metric report value mismatch: {path}')
    else:
        require(type(saved) is type(actual) and saved==actual,f'Saved metric report metadata mismatch: {path}')


def validation_inputs(config):
    """Verify split registry bytes; read only official training CSV, never test CSV."""
    directory=Path(config['split_directory'])
    require(sha256(directory/'metadata.json')==config['task02_metadata_sha256'],'Task 02 metadata changed')
    require(sha256(directory/'methodology.md')==config['task02_methodology_sha256'],'Task 02 methodology changed')
    meta=read(directory/'metadata.json')
    frozen={'seed':42,'max_length':128,'model_name':'UBC-NLP/MARBERT','model_revision':REVISION,
            'learning_rate':2e-5,'batch_size':16,'max_epochs':5,'early_stopping_patience':2,
            'checkpoint_selection':'validation_macro_f1','label2id':LABEL2ID}
    require(all(config.get(k)==v for k,v in frozen.items()),'Baseline protocol config differs')
    versions=library_versions()
    require(all(versions[k]==meta['versions'][k] for k in ['python','pandas','numpy','scikit-learn']),
            'Task 02 runtime version mismatch')
    require(unicodedata.unidata_version==meta['versions']['unicode'],'Unicode database differs from Task 02')
    require(meta['source_commit']==COMMIT and meta['seed']==42,'Incorrect source revision or seed')
    require(all(meta['counts'][k]==v for k,v in EXPECTED.items()),'Incorrect Task 02 counts')
    for name,digest in meta['artifact_sha256'].items():
        require(sha256(directory/name)==digest,f'Task 02 manifest hash changed: {name}')
    require(sha256(config['raw_training'])==HASHES['training_data.csv']==meta['source_sha256']['official_train'],
            'Official training CSV hash changed')
    require(meta['source_sha256']['official_test']==HASHES['testing_data.csv'],'Recorded test checksum changed')
    raw=load_csv(config['raw_training'])
    require(len(raw)==12548,'Official training row count changed')
    manifest=pd.read_csv(directory/'validation_manifest.csv',dtype=str,keep_default_na=False)
    require(len(manifest)==2509,'Expected 2509 validation rows')
    validation=load_partition(raw,manifest,'validation')
    # Overlap checks operate on immutable registered text/group hashes, not test text.
    parts={name:pd.read_csv(directory/f'{name}_manifest.csv',dtype=str) for name in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for key in ['text_sha256','group_id']:
            require(not set(parts[a][key]) & set(parts[b][key]),f'Active {a}/{b} overlap: {key}')
    pairs=pd.read_csv(directory/'near_duplicate_review.csv',keep_default_na=False)
    master=pd.read_csv(directory/'split_manifest.csv',dtype=str,keep_default_na=False)
    near=audit_near(pairs,master,threshold=.95)
    near['similarity_bin']=pd.cut(near.cosine_similarity.astype(float),[.9,.95,.99,1.000001],
        labels=['0.90-0.95','0.95-0.99','0.99-1.00'],include_lowest=True,right=False).astype(str)
    require(len(near)==753,'Unexpected near-duplicate candidate count')
    high=near[near.high_risk_manual_review]
    flagged=set(high.left_source_row_id)|set(high.right_source_row_id)
    validation['high_risk_near_duplicate']=validation.source_row_id.isin(flagged)
    validation['has_negation']=validation.tweet.map(lambda t:bool(lexical_negation(t)))
    evidence={'source_commit':COMMIT,'source_training_sha256':sha256(config['raw_training']),
              'recorded_test_sha256_not_opened':meta['source_sha256']['official_test'],
              'task02_metadata_sha256':sha256(directory/'metadata.json'),
              'manifest_sha256':meta['artifact_sha256'],'counts':{k:meta['counts'][k] for k in EXPECTED},
              'seed':42,'validation_high_risk_rows':int(validation.high_risk_near_duplicate.sum()),
              'near_pairs':len(near),'high_risk_pairs':len(high),'test_text_read':False,'test_evaluated':False,
              'code_sha256':code_hashes(),'versions':versions}
    return validation, near, evidence


def review_sample(manifest, per_stratum=2):
    applicable=manifest[manifest.applicable & manifest.semantic_status.eq('provisional')].copy()
    keys=['rule','intensity','gold_label','dialect','sarcasm','has_negation']
    # Hash ranking gives stable selection independent of input ordering.
    applicable['_rank']=applicable.variant_id.map(lambda x:hashlib.sha256(f'42|review|{x}'.encode()).hexdigest())
    result=applicable.sort_values('_rank').groupby(keys,sort=True,dropna=False).head(per_stratum).drop(columns='_rank')
    result=result.sort_values('variant_id').reset_index(drop=True)
    result['review_status']='pending'
    result['reviewer']=''
    result['rationale']=''
    return result


def build_manifest(validation):
    records=[]
    for row in validation.itertuples(index=False):
        for rule,intensity in RULES:
            changed,edit=perturb(row.tweet,row.source_row_id,rule,intensity)
            variant_id='ov1_'+hashlib.sha256(f'{row.source_row_id}|{rule}|{intensity}|42|{VERSION}'.encode()).hexdigest()
            record={'variant_id':variant_id,'source_row_id':row.source_row_id,'row_id':str(row.row_id),
                    'group_id':row.group_id,'partition':'validation','gold_label':row.sentiment,
                    'gold_id':LABEL2ID[row.sentiment],'dialect':row.dialect,'sarcasm':row.sarcasm,
                    'has_negation':row.has_negation,'high_risk_near_duplicate':row.high_risk_near_duplicate,
                    'negation_spans':json.dumps(lexical_negation(row.tweet),ensure_ascii=False),
                    'rule':rule,'intensity':intensity,'seed':42,'rule_version':VERSION,
                    'original_text':row.tweet,'perturbed_text':changed,'original_sha256':hashlib.sha256(row.tweet.encode()).hexdigest(),
                    'perturbed_sha256':hashlib.sha256(changed.encode()).hexdigest(),
                    'applicable':edit is not None,'semantic_status':'provisional' if edit else 'not_applicable',
                    'failure_reason':'' if edit else failure_reason(row.tweet,rule,intensity),
                    'start':None,'end':None,'target_start':None,'target_end':None,
                    'source_span':'','target_span':'','ambiguity':''}
            if edit:
                record.update(edit.record(),target_start=edit.start,target_end=edit.start+len(edit.target_span))
            records.append(record)
    return pd.DataFrame(records)


def apply_reviews(manifest, reviews):
    """Require explicit per-example human decisions; never propagate sample approval."""
    result=manifest.copy()
    if reviews is None:
        return result
    needed={'variant_id','review_status','reviewer','rationale','original_sha256','perturbed_sha256'}
    require(needed<=set(reviews),'Review file missing identity/decision columns')
    require(not reviews.variant_id.duplicated().any(),'Duplicate human review decisions')
    require(set(reviews.variant_id)<=set(result.variant_id),'Review refers to unknown variant')
    lookup=result.set_index('variant_id')
    mapping={}
    for row in reviews.itertuples(index=False):
        original=lookup.loc[row.variant_id]
        require(original.applicable,'Review for inapplicable variant')
        require(row.original_sha256==original.original_sha256 and row.perturbed_sha256==original.perturbed_sha256,
                'Review text hashes mismatch')
        require(row.review_status in {'pending','valid','invalid','ambiguous'},'Unknown review decision')
        if row.review_status!='pending':
            require(bool(str(row.reviewer).strip()) and bool(str(row.rationale).strip()),'Human decision requires reviewer and rationale')
            mapping[row.variant_id]={'valid':'confirmed','invalid':'invalid','ambiguous':'ambiguous'}[row.review_status]
    result['semantic_status']=[mapping.get(r.variant_id,r.semantic_status) for r in result.itertuples()]
    decisions=reviews.set_index('variant_id')
    for key in ['reviewer','rationale','review_status']:
        result[key]=result.variant_id.map(decisions[key]).fillna('')
    return result


def prepare(output, config, per_stratum=2):
    require(per_stratum>=1,'Review sample size must be positive')
    output=fresh(output)
    validation,near,evidence=validation_inputs(config)
    manifest=build_manifest(validation)
    manifest.to_csv(output/'perturbation_manifest.csv',index=False)
    review_sample(manifest,per_stratum).to_csv(output/'human_review_queue.csv',index=False)
    near.to_csv(output/'near_duplicate_audit.csv',index=False)
    summary=manifest.groupby(['rule','intensity','gold_label','has_negation'],dropna=False).agg(
        rows=('variant_id','size'),applicable=('applicable','sum')).reset_index()
    summary['not_applicable']=summary.rows-summary.applicable
    summary.to_csv(output/'applicability.csv',index=False)
    manifest[~manifest.applicable].groupby(['rule','intensity','failure_reason']).size().rename('n').reset_index().to_csv(output/'failure_cases.csv',index=False)
    evidence.update(status='prepared_only',rule_version=VERSION,variant_rows=len(manifest),
                    applicable_variants=int(manifest.applicable.sum()),
                    confirmed_variants=0,human_review_completed=False,baseline_error_analysis='not_run_by_prepare_command')
    save(output/'preparation.json',evidence)
    save(output/'artifact_hashes.json',{f.name:sha256(f) for f in sorted(output.iterdir()) if f.is_file()})
    return evidence


def verify_run(run, checkpoint, validation, evidence, original):
    """Allow relocation of Drive artifacts, not reconstruction or checkpoint reselection."""
    run=Path(run)
    status=read(run/'run_status.json')
    require(status.get('status')=='completed' and status.get('checkpoint_reload_verified') is True,
            'Task 03B is not completed with verified checkpoint reload')
    require(status.get('test_evaluated') is False and status.get('model_revision')==REVISION,
            'Incorrect baseline revision or test evaluation')
    cfg=read(run/'resolved_training_config.json')
    operational={'device','cache_directory','micro_batch_size','gradient_accumulation_steps','raw_training','raw_testing','split_directory'}
    for key,value in original.items():
        if key not in operational:
            require(cfg.get(key)==value,f'Baseline protocol changed: {key}')
    require(cfg['micro_batch_size']*cfg['gradient_accumulation_steps']==16,'Effective batch changed')
    pre=read(run/'preflight.json')
    require(pre['status']=='passed' and pre['counts']==evidence['counts'],'Task 03B preflight invalid')
    require(pre['manifest_sha256']==evidence['manifest_sha256'],'Task 03B split hashes differ')
    require(pre['source_commit']==COMMIT and pre['source_sha256']['official_train']==evidence['source_training_sha256'],
            'Task 03B source differs')
    require(all(x['exact_groups']==x['canonical_groups']==0 for x in pre['overlap']), 'Task 03B overlap check failed')
    hashes=read(run/'checkpoint_hashes.json')
    required={'model.safetensors','config.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'}
    require(required<=set(hashes),'Encoder/tokenizer checkpoint hash registry incomplete')
    for name,digest in hashes.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and bool(re.fullmatch('[0-9a-f]{64}',digest)),
                'Invalid checkpoint hash registry entry')
    if checkpoint is not None:
        checkpoint=Path(checkpoint)
        require(checkpoint.is_dir(),'Selected trained checkpoint directory unavailable')
        actual={str(f.relative_to(checkpoint)) for f in checkpoint.rglob('*') if f.is_file()}
        require(actual==set(hashes),'Unregistered/missing checkpoint files')
        for name,digest in hashes.items():
            file=(checkpoint/name).resolve()
            require(file.is_relative_to(checkpoint.resolve()) and sha256(file)==digest,f'Checkpoint hash mismatch: {name}')
    predictions=pd.read_csv(run/'validation_predictions.csv',dtype={'sarcasm':str,'row_id':str},keep_default_na=False)
    require(not predictions.source_row_id.duplicated().any(),'Duplicated validation prediction IDs')
    require(predictions.source_row_id.tolist()==validation.source_row_id.tolist(),'Validation prediction IDs/order changed')
    require(predictions.gold_label.tolist()==validation.sentiment.tolist(),'Validation prediction labels changed')
    for col in ['row_id','group_id','sarcasm','dialect']:
        require(predictions[col].astype(str).tolist()==validation[col].astype(str).tolist(),f'Prediction metadata changed: {col}')
    require(predictions.gold_id.tolist()==validation.sentiment.map(LABEL2ID).tolist(),'Gold numeric mapping changed')
    require(predictions.predicted_label.tolist()==predictions.predicted_id.map({v:k for k,v in LABEL2ID.items()}).tolist(),
            'Predicted numeric mapping changed')
    checked=prediction_frame(validation,predictions[[f'probability_{x}' for x in LABEL2ID]].to_numpy())
    require(checked.predicted_id.tolist()==predictions.predicted_id.tolist(),'Predictions disagree with saved probabilities')
    require(np.allclose(checked.confidence,predictions.confidence,atol=1e-8,rtol=0),'Confidence mismatch')
    metrics,classes,matrix=calculate(predictions.gold_id,predictions.predicted_id)
    saved=read(run/'validation_metrics.json')
    require(saved['status']=='completed','Missing completed validation metrics')
    for key in ['accuracy','macro_f1','weighted_f1','n']:
        require(abs(saved[key]-metrics[key])<1e-12,f'Saved validation metric mismatch: {key}')
    require(np.array_equal(pd.read_csv(run/'confusion_matrix.csv',index_col=0).to_numpy(),matrix),'Saved confusion matrix mismatch')
    saved_classes=pd.read_csv(run/'per_class_results.csv')
    require(saved_classes.label.tolist()==list(LABEL2ID),'Saved per-class label order mismatch')
    for key in ['precision','recall','f1','support']:
        require(np.allclose(saved_classes[key],[r[key] for r in classes],atol=1e-12,rtol=0),'Saved per-class results mismatch')
    history=pd.read_csv(run/'epoch_history.csv')
    require(1<=len(history)<=5 and history.epoch.tolist()==list(range(1,len(history)+1)),'Invalid epoch history')
    best=int(history.loc[history.macro_f1.idxmax(),'epoch'])
    require(best==status['best_epoch']==saved['checkpoint_epoch'],'Checkpoint selection differs from original best epoch')
    require(abs(history.macro_f1.max()-metrics['macro_f1'])<1e-12,'Selected checkpoint score mismatch')
    if (run/'subgroup_metrics.json').is_file():
        verify_metric_report(read(run/'subgroup_metrics.json'),subgroup_metrics(predictions))
    return predictions,hashes,status


def import_reports(archive, output):
    """Flat allowlisted report bundle; verify every checksum before extracting."""
    from tools.colab_runtime import REPORT_FILES
    allowed=REPORT_FILES | {'gpu_runtime.json','runtime_provenance.json','snapshot_verification.json',
        'encoder_verification.json','dependency_freeze.txt','dependency_check.txt','report_checksums.json'}
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        require(len(names)==len(set(names)) and set(names)<=allowed,'Unsafe/unknown/duplicate archive members')
        require('report_checksums.json' in names,'Report archive checksum registry missing')
        require(all(i.file_size<=10*1024**2 for i in bundle.infolist()) and
                sum(i.file_size for i in bundle.infolist())<50*1024**2,'Report archive too large')
        checks=json.loads(bundle.read('report_checksums.json'))
        require(set(checks)==set(names)-{'report_checksums.json'},'Report checksum coverage incomplete')
        files={name:bundle.read(name) for name in names}
        for name,digest in checks.items():
            require(hashlib.sha256(files[name]).hexdigest()==digest,f'Uploaded report checksum mismatch: {name}')
    output=fresh(output)
    for name,data in files.items(): (output/name).write_bytes(data)
    save(output/'import_provenance.json',{'source_archive_name':Path(archive).name,'source_archive_sha256':sha256(archive),
         'status':'all_report_bytes_verified','checkpoint_bytes_in_archive':False,'files':checks})
    return {'status':'imported','reports':len(files),'source_archive_sha256':sha256(archive)}


def verify_external_reports(run):
    run=Path(run)
    checks=read(run/'report_checksums.json')
    for name,value in checks.items():
        require(Path(name).name==name and sha256(run/name)==value,f'External report checksum mismatch: {name}')
    provenance=read(run/'runtime_provenance.json')
    require(provenance['reference_commit']=='c8dc698780cb06c8838c557f159bdaed7507aeda' and
            provenance['training_completed'] is True,'Incorrect/incomplete external baseline provenance')
    require(provenance['run_status']==read(run/'run_status.json'),'External provenance status inconsistent')
    from tools.colab_runtime import WEIGHTS_SHA256
    snapshot=read(run/'snapshot_verification.json')
    require(snapshot['status']=='verified' and snapshot['revision']==REVISION and
            snapshot['files_sha256']['pytorch_model.bin']==WEIGHTS_SHA256,'Unverified original encoder snapshot provenance')
    loading=read(run/'encoder_verification.json')
    require(loading['status']=='encoder_verified' and loading['revision']==REVISION,'Encoder load provenance incomplete')
    info=loading['loading_info']
    require(not info.get('error_msgs') and not info.get('mismatched_keys') and
            all(k.startswith(('classifier.','bert.pooler.')) for k in info['missing_keys']), 'Encoder load recorded missing pretrained weights')
    return provenance


def analyze_reports(run, output, config):
    output=fresh(output)
    try:
        return _analyze_reports(run,output,config)
    except Exception as exc:
        save(output/'run_status.json',{'status':'blocked_or_failed',
            'reason':f'{type(exc).__name__}: {exc}','analysis_completed':False,
            'training_performed':False,'test_text_read':False,'test_evaluated':False})
        raise


def _analyze_reports(run, output, config):
    validation,near,evidence=validation_inputs(config)
    provenance=verify_external_reports(run)
    predictions,hashes,status=verify_run(run,None,validation,evidence,config)
    validate_near_report(run,near)
    predictions['has_negation']=validation.has_negation
    predictions['high_risk_near_duplicate']=validation.high_risk_near_duplicate
    predictions['original_text']=validation.tweet
    matrix=error_analysis(predictions,output)
    figures(matrix,pd.DataFrame(),output)
    near.to_csv(output/'near_duplicate_audit.csv',index=False)
    save(output/'evidence.json',{**evidence,'external_provenance':provenance,'checkpoint_hash_registry':hashes,
         'checkpoint_bytes_verified_here':False,'checkpoint_verification':'external runner reports reload and durable hash checks; bytes unavailable locally',
         'baseline_status':status,'report_sha256':read(Path(run)/'report_checksums.json')})
    result={'status':'actual_saved_prediction_analysis_completed','checkpoint_bytes_verified_here':False,
            'robustness_inference':'blocked_missing_checkpoint_and_gpu','test_text_read':False,'test_evaluated':False}
    save(output/'run_status.json',result)
    save(output/'artifact_hashes.json',{p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    return result


def publish(preparation, analysis, output):
    """Publish aggregates and text-free manifests, retaining full texts in private outputs."""
    import shutil
    preparation,analysis=Path(preparation),Path(analysis)
    for directory in [preparation,analysis]:
        registry=read(directory/'artifact_hashes.json')
        for name,value in registry.items():
            require(Path(name).name==name and sha256(directory/name)==value,'Generated artifact changed before publication')
    output=fresh(output)
    for name in ['perturbation_manifest.csv','human_review_queue.csv']:
        frame=pd.read_csv(preparation/name,dtype={'sarcasm':str,'row_id':str},keep_default_na=False)
        require({'original_text','perturbed_text'}<=set(frame),'Full-text preparation manifest missing')
        frame=frame.drop(columns=['original_text','perturbed_text'])
        frame.to_csv(output/name,index=False)
    for name in ['preparation.json','applicability.csv','failure_cases.csv']:
        shutil.copyfile(preparation/name,output/name)
    names=['baseline_metrics.json','baseline_per_class.csv','baseline_per_class.tex','baseline_confusion.csv',
        'baseline_subgroups.json','baseline_subgroups.csv','baseline_sensitivity.json','baseline_confidence_intervals.json',
        'baseline_error_analysis.md','error_patterns.csv','evidence.json','validation_confusion.pdf','validation_confusion.svg']
    for name in names: shutil.copyfile(analysis/name,output/name)
    shutil.copyfile(analysis/'run_status.json',output/'analysis_status.json')
    save(output/'publication.json',{'raw_text_committed':False,'model_weights_committed':False,
        'full_text_manifests':str(preparation),'full_text_error_queue':str(analysis/'error_review_queue.csv'),
        'instruction':'Regenerate full-text review queues using prepare/analyze; text-free queues are not human validation',
        'source_artifact_registries':{str(d):sha256(d/'artifact_hashes.json') for d in [preparation,analysis]}})
    save(output/'artifact_hashes.json',{p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    return {'status':'published_text_free_reports','directory':str(output)}


def validate_near_report(run, near):
    saved=pd.read_csv(Path(run)/'near_duplicate_audit.csv',keep_default_na=False)
    require(len(saved)==len(near) and set(saved)==set(near),'Baseline near-duplicate audit differs')
    for key in near:
        if key=='cosine_similarity':
            require(np.allclose(saved[key],near[key],atol=1e-14,rtol=0),'Near-pair scores changed')
        else:
            require(saved[key].astype(str).tolist()==near[key].astype(str).tolist(),f'Near-pair audit changed: {key}')


def paired_scores(gold, clean, changed):
    gold,clean,changed=map(np.asarray,(gold,clean,changed))
    require(gold.ndim==1 and len(gold)>0 and gold.shape==clean.shape==changed.shape,'Paired metric shape mismatch')
    require(all(np.isin(x,[0,1,2]).all() for x in [gold,clean,changed]),'Invalid paired metric labels')
    def scores(pred):
        counts=np.bincount((3*gold+pred).astype(int),minlength=9).reshape(3,3)
        denominator=counts.sum(axis=0)+counts.sum(axis=1)
        per_class=np.divide(2*np.diag(counts),denominator,out=np.zeros(3),where=denominator!=0)
        return {'accuracy':float(np.trace(counts)/counts.sum()),'macro_f1':float(per_class.mean())},per_class
    (a,af1),(b,bf1)=scores(clean),scores(changed)
    result={'clean_accuracy':a['accuracy'],'perturbed_accuracy':b['accuracy'],
            'accuracy_drop':a['accuracy']-b['accuracy'],'clean_macro_f1':a['macro_f1'],
            'perturbed_macro_f1':b['macro_f1'],'macro_f1_drop':a['macro_f1']-b['macro_f1'],
            'flip_rate':float(np.mean(clean!=changed)),
            'correct_to_wrong_rate':float(np.mean((clean==gold)&(changed!=gold))),
            'wrong_to_correct_rate':float(np.mean((clean!=gold)&(changed==gold))),
            'wrong_to_wrong_flip_rate':float(np.mean((clean!=gold)&(changed!=gold)&(clean!=changed)))}
    for label,i in LABEL2ID.items():
        for name,f1 in [('clean',af1),('perturbed',bf1)]:
            result[f'{name}_{label}_f1']=float(f1[i])
        result[f'{label}_f1_drop']=result[f'clean_{label}_f1']-result[f'perturbed_{label}_f1']
    return result


def paired_bootstrap(frame, repetitions=2000, seed=42):
    require(repetitions>=100,'At least 100 paired bootstrap repetitions required')
    require(len(frame)>0 and not frame.source_row_id.duplicated().any(),'One perturbation per original ID required in each analysis stratum')
    frame=frame.sort_values('source_row_id')
    gold=frame.gold_id.to_numpy(); clean=frame.clean_id.to_numpy(); changed=frame.predicted_id.to_numpy()
    point=paired_scores(gold,clean,changed)
    rng=np.random.default_rng(seed)
    samples={key:[] for key in point}
    for _ in range(repetitions):
        idx=rng.integers(0,len(frame),len(frame))
        scores=paired_scores(gold[idx],clean[idx],changed[idx])
        for key,value in scores.items(): samples[key].append(value)
    return {key:{'estimate':value,'ci_low':float(np.quantile(samples[key],.025)),
                 'ci_high':float(np.quantile(samples[key],.975))} for key,value in point.items()}


def error_analysis(predictions, output):
    metrics,classes,matrix=calculate(predictions.gold_id,predictions.predicted_id)
    save(output/'baseline_metrics.json',metrics)
    pd.DataFrame(classes).to_csv(output/'baseline_per_class.csv',index=False)
    pd.DataFrame(matrix,index=list(LABEL2ID),columns=list(LABEL2ID)).rename_axis('gold_label').to_csv(output/'baseline_confusion.csv')
    errors=predictions[predictions.gold_id!=predictions.predicted_id].copy()
    errors.to_csv(output/'validation_errors.csv',index=False)
    queue=errors.groupby(['gold_label','predicted_label','dialect','sarcasm','has_negation'],dropna=False,sort=True).head(3).copy()
    queue['human_error_category']='';queue['human_notes']=''
    queue.to_csv(output/'error_review_queue.csv',index=False)
    errors.groupby(['gold_label','predicted_label','dialect','sarcasm','has_negation'],dropna=False).size().rename('errors').reset_index().to_csv(output/'error_patterns.csv',index=False)
    groups=subgroup_metrics(predictions)
    for field in ['has_negation','high_risk_near_duplicate']:
        for value,part in predictions.groupby(field):
            row={'field':field,'value':str(value),'n':len(part)}
            if len(part)<30: row.update(status='insufficient_sample_size')
            else:
                total,per_class,_=calculate(part.gold_id,part.predicted_id)
                row.update(status='computed',**total,per_class=per_class)
            groups.append(row)
    save(output/'baseline_subgroups.json',groups)
    rows=[]
    for group in groups:
        row={k:v for k,v in group.items() if k!='per_class'}
        for item in group.get('per_class',[]):row[item['label']+'_f1']=item['f1']
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'baseline_subgroups.csv',index=False)
    pd.DataFrame(classes).to_latex(output/'baseline_per_class.tex',index=False,float_format='%.4f')
    sensitivity=predictions[~predictions.high_risk_near_duplicate]
    if len(sensitivity):
        total,per_class,_=calculate(sensitivity.gold_id,sensitivity.predicted_id)
        save(output/'baseline_sensitivity.json',{'status':'computed',**total,'per_class':per_class})
    else: save(output/'baseline_sensitivity.json',{'status':'empty_subset'})
    intervals={}
    for name,part in [('all_validation',predictions),('exclude_high_risk',sensitivity)]:
        if len(part):
            paired=part[['source_row_id','gold_id','predicted_id']].copy()
            paired['clean_id']=paired.predicted_id
            ci=paired_bootstrap(paired,2000)
            intervals[name]={key:value for key,value in ci.items() if key.startswith('clean_')}
    save(output/'baseline_confidence_intervals.json',{'seed':42,'repetitions':2000,'method':'original-ID percentile bootstrap 95%',
        'scope':'saved clean validation predictions only; no perturbed predictions exist here','intervals':intervals})
    lines=['# Actual validation error analysis','',f'Rows: {len(predictions)}. Errors: {len(errors)}.',
           f'Accuracy: {metrics["accuracy"]:.6f}; Macro-F1: {metrics["macro_f1"]:.6f}.','',
           'The following patterns describe observed predictions; they do not establish linguistic causes.','']
    for entry in classes:
        part=errors[errors.gold_label==entry['label']]
        transitions=', '.join(f'{label}: {count}' for label,count in part.predicted_label.value_counts().items()) or 'none'
        lines += [f'- {entry["label"]}: support {entry["support"]}, recall {entry["recall"]:.6f}, F1 {entry["f1"]:.6f}; errors to {transitions}.']
    lines += ['','See baseline_subgroups.json for dialect/sarcasm/negation and high-risk flag strata,',
              'baseline_sensitivity.json for the high-risk-excluded subset, and validation_errors.csv',
              'for original-text inspection. No automatic assertion about sarcasm, negation scope,',
              'or dialect causing an error is made. Manual qualitative interpretation remains separate.']
    (output/'baseline_error_analysis.md').write_text('\n'.join(lines)+'\n')
    return matrix


def robustness_tables(predictions, repetitions, minimum=30):
    records=[]
    for subset in ['all_validation','exclude_high_risk']:
        data=predictions if subset=='all_validation' else predictions[~predictions.high_risk_near_duplicate]
        cohorts=[]
        for (rule,intensity),population in data.groupby(['rule','intensity']):
            eligible=population[~population.semantic_status.isin(['invalid','ambiguous'])]
            if len(eligible):cohorts.append((rule,intensity,'provisional_population','population_with_noop',eligible))
        for (rule,intensity,semantic),part in data.groupby(['rule','intensity','semantic_status']):
            if semantic in {'invalid','ambiguous','not_applicable'}: continue
            cohorts.append((rule,intensity,semantic,'applicable_only',part))
        for rule,intensity,semantic,coverage,part in cohorts:
            scopes=[('all','all',part)]
            for field in ['has_negation','dialect','sarcasm']:
                scopes += [(field,str(value),group) for value,group in part.groupby(field,dropna=False)]
            for field,value,group in scopes:
                common={'subset':subset,'rule':rule,'intensity':int(intensity),'semantic_status':semantic,
                        'coverage':coverage,'applicable_n':int(group.applicable.sum()),
                        'subgroup':field,'value':value,'n':len(group),'bootstrap_repetitions':repetitions,'seed':42}
                if len(group)<minimum:
                    records.append({**common,'status':'insufficient_sample_size'}); continue
                scores=paired_bootstrap(group,repetitions)
                for metric,stats in scores.items():
                    records.append({**common,'status':'computed','metric':metric,**stats})
    return pd.DataFrame(records)


def figures(matrix, tables, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(4.4,4))
    ax.imshow(matrix,cmap='Blues')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=list(LABEL2ID),yticklabels=list(LABEL2ID),
           xlabel='Predicted sentiment',ylabel='Gold sentiment',title='Validation confusion matrix')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    fig.tight_layout()
    for extension in ['pdf','svg']: fig.savefig(output/f'validation_confusion.{extension}')
    plt.close(fig)
    if tables.empty: return
    selected=tables[(tables.get('metric')=='macro_f1_drop') & (tables.subgroup=='all') & (tables.subset=='all_validation') & (tables.status=='computed')]
    for semantic,part in selected.groupby('semantic_status'):
        fig,ax=plt.subplots(figsize=(7,4)); part=part.reset_index(drop=True)
        ax.vlines(range(len(part)),part.ci_low,part.ci_high)
        ax.scatter(range(len(part)),part.estimate)
        ax.set_xticks(range(len(part)),[f'{r.rule}:{r.intensity}' for r in part.itertuples()],rotation=25)
        ax.axhline(0,color='grey',linewidth=.8)
        ax.set(ylabel='Clean minus perturbed Macro-F1',title=f'{semantic.capitalize()} candidates: paired 95% CI')
        fig.tight_layout()
        for extension in ['pdf','svg']: fig.savefig(output/f'robustness_{semantic}.{extension}')
        plt.close(fig)


def run_study(args, config):
    output=fresh(args.output); started=time.perf_counter()
    status={'status':'initializing','training_performed':False,'test_text_read':False,'test_evaluated':False,
            'seed':42,'gpu':None,'metrics':None}
    try:
        validation,near,evidence=validation_inputs(config)
        saved,hashes,baseline=verify_run(args.run,args.checkpoint,validation,evidence,config)
        validate_near_report(args.run,near)
        # Recreate deterministic candidates only; never recreate/reselect data splits.
        manifest=build_manifest(validation)
        reviews=pd.read_csv(args.reviews,keep_default_na=False) if args.reviews else None
        manifest=apply_reviews(manifest,reviews)
        if reviews is not None: reviews.to_csv(output/'human_review_decisions.csv',index=False)
        manifest.to_csv(output/'perturbation_manifest.csv',index=False)
        review_sample(manifest).to_csv(output/'human_review_queue.csv',index=False)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .marbert import seed_all, evaluate, encoded_dataset
        require(torch.cuda.is_available() and torch.version.cuda is not None,'CUDA GPU unavailable for robustness inference')
        status.update(gpu=torch.cuda.get_device_name(0),cuda_build=torch.version.cuda,
                      free_vram_bytes=torch.cuda.mem_get_info()[0])
        require(args.batch_size>=1,'Inference batch size must be positive')
        seed_all(42,4)
        tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True,trust_remote_code=False)
        model,loading=AutoModelForSequenceClassification.from_pretrained(args.checkpoint,local_files_only=True,
                    trust_remote_code=False,output_loading_info=True)
        require(not any(loading.get(k) for k in ['missing_keys','mismatched_keys','error_msgs','unexpected_keys']),
                'Trained checkpoint did not load completely; random fallback prohibited')
        require(model.config.num_labels==3 and model.config.label2id==LABEL2ID,'Trained label mapping changed')
        model.to('cuda').eval()
        def infer(frame):
            loader=torch.utils.data.DataLoader(encoded_dataset(frame,tokenizer,128),batch_size=args.batch_size,shuffle=False)
            probabilities,_=evaluate(model,loader,'cuda')
            return prediction_frame(frame,probabilities)
        clean=infer(validation)
        require(clean.predicted_id.tolist()==saved.predicted_id.tolist(),'Clean inference differs from saved baseline predictions; stop and investigate')
        require(np.allclose(clean[[f'probability_{x}' for x in LABEL2ID]],saved[[f'probability_{x}' for x in LABEL2ID]],atol=1e-5,rtol=1e-5),
                'Clean checkpoint confidence differs beyond tolerance')
        clean['has_negation']=validation.has_negation; clean['high_risk_near_duplicate']=validation.high_risk_near_duplicate
        clean['original_text']=validation.tweet
        clean.to_csv(output/'clean_predictions.csv',index=False)
        matrix=error_analysis(clean,output)
        applicable=manifest[manifest.applicable & ~manifest.semantic_status.isin(['invalid','ambiguous'])].copy()
        require(len(applicable)>0,'No applicable noninvalid variants available')
        frames=applicable.rename(columns={'gold_label':'sentiment','perturbed_text':'tweet'})
        changed=infer(frames)
        probability_columns=['predicted_id','predicted_label','confidence']+[f'probability_{x}' for x in LABEL2ID]
        for column in probability_columns: applicable[column]=changed[column].to_numpy()
        clean_lookup=clean.set_index('source_row_id')
        noop=manifest[~manifest.applicable].copy()
        for column in probability_columns:noop[column]=noop.source_row_id.map(clean_lookup[column])
        applicable=pd.concat([applicable,noop],ignore_index=True)
        applicable['clean_id']=applicable.source_row_id.map(clean_lookup.predicted_id)
        applicable['clean_confidence']=applicable.source_row_id.map(clean_lookup.confidence)
        applicable['prediction_flipped']=applicable.clean_id!=applicable.predicted_id
        applicable['clean_correct']=applicable.clean_id==applicable.gold_id
        applicable['perturbed_correct']=applicable.predicted_id==applicable.gold_id
        def tokens(text):
            ids=tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
            effective=tokenizer(text,add_special_tokens=True,truncation=True,max_length=128)['input_ids']
            return len(ids),hashlib.sha256(json.dumps(ids).encode()).hexdigest(),len(ids)>128,hashlib.sha256(json.dumps(effective).encode()).hexdigest()
        clean_tokens={r.source_row_id:tokens(r.tweet) for r in validation.itertuples()}
        change_tokens=[tokens(t) for t in applicable.perturbed_text]
        applicable['clean_token_count']=[clean_tokens[x][0] for x in applicable.source_row_id]
        applicable['perturbed_token_count']=[x[0] for x in change_tokens]
        applicable['token_count_delta']=applicable.perturbed_token_count-applicable.clean_token_count
        applicable['token_ids_changed']=[clean_tokens[x][1]!=t[1] for x,t in zip(applicable.source_row_id,change_tokens)]
        applicable['effective_token_ids_changed']=[clean_tokens[x][3]!=t[3] for x,t in zip(applicable.source_row_id,change_tokens)]
        applicable['clean_truncated']=[clean_tokens[x][2] for x in applicable.source_row_id]
        applicable['perturbed_truncated']=[t[2] for t in change_tokens]
        applicable.to_csv(output/'perturbed_predictions.csv',index=False)
        tables=robustness_tables(applicable,args.bootstrap)
        tables.to_csv(output/'robustness_metrics.csv',index=False)
        table_cols=['subset','rule','intensity','semantic_status','coverage','subgroup','value','n','applicable_n','status','metric','estimate','ci_low','ci_high']
        if not tables.empty and 'metric' in tables:
            (output/'robustness_table.tex').write_text(tables.reindex(columns=table_cols).to_latex(index=False,float_format='%.4f'))
        token_summary=applicable.groupby(['rule','intensity','semantic_status','token_ids_changed','effective_token_ids_changed']).agg(
            n=('variant_id','size'),flip_rate=('prediction_flipped','mean'),mean_token_count_delta=('token_count_delta','mean')).reset_index()
        token_summary['interpretation']='descriptive association; not causal evidence'
        token_summary.to_csv(output/'tokenization_associations.csv',index=False)
        figures(matrix,tables,output)
        require(all(sha256(Path(args.checkpoint)/name)==value for name,value in hashes.items()),'Checkpoint changed during inference')
        near.to_csv(output/'near_duplicate_audit.csv',index=False)
        save(output/'evidence.json',{**evidence,'checkpoint_sha256':hashes,'baseline_status':baseline,
             'baseline_report_sha256':{p.name:sha256(p) for p in Path(args.run).iterdir() if p.is_file()},
             'review_sha256':sha256(args.reviews) if args.reviews else None,'config':config,'versions':library_versions(),
             'inference_batch_size':args.batch_size,'bootstrap_repetitions':args.bootstrap})
        status.update(status='completed',metrics='robustness_metrics.csv',duration_seconds=time.perf_counter()-started,
            confirmed_variants=int((applicable.semantic_status=='confirmed').sum()),
            confirmed_analysis_available=bool((applicable.semantic_status=='confirmed').any()),
            provisional_variants=int((applicable.semantic_status=='provisional').sum()),
            excluded_invalid=int((manifest.semantic_status=='invalid').sum()),excluded_ambiguous=int((manifest.semantic_status=='ambiguous').sum()),
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        (output/'methodology.md').write_text('# Task 04 completed inference study\n\n'
            'No training, checkpoint reselection or official test access occurred. See evidence.json for input hashes, versions and configuration.\n\n'
            'Both applicable-only and population-with-noop summaries are provided for each rule/intensity. '
            'Inapplicable rows retain clean predictions without model re-evaluation. Mixed population summaries are conservatively labeled provisional. '
            'Clean comparator metrics use exactly the matched IDs; '
            'they need not equal whole-validation metrics. Provisional and human-confirmed variants are analyzed separately. '
            'Invalid/ambiguous reviews are excluded from both robustness analyses. Confirmation is per example, never propagated.\n\n'
            'CIs are percentile paired 95% intervals, resampling original IDs with seed 42. All three classes are retained with '
            'zero-division F1=0. Minimum reported stratum size is 30; small strata are explicitly unavailable. '
            'Intervals condition on this selected checkpoint and review status, are not simultaneous, and do not represent training-seed uncertainty.\n\n'
            'Prediction flips on provisional variants are candidate instability, not evidence of meaning-preserving robustness failure. '
            'Lexical negation cues do not establish semantic scope. Full/effective token changes and truncation are associations, not causal proof.\n\n'
            f'Confirmed variants: {status["confirmed_variants"]}; provisional variants: {status["provisional_variants"]}. '
            f'Confirmed analysis available: {status["confirmed_analysis_available"]}.\n')
    except Exception as exc:
        status.update(status='blocked_or_failed',reason=f'{type(exc).__name__}: {exc}',duration_seconds=time.perf_counter()-started)
        save(output/'run_status.json',status)
        raise
    save(output/'run_status.json',status)
    save(output/'artifact_hashes.json',{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/marbert_v1.json')
    commands=parser.add_subparsers(dest='command',required=True)
    ingest=commands.add_parser('import-reports'); ingest.add_argument('--archive',required=True); ingest.add_argument('--output',required=True)
    analysis=commands.add_parser('analyze'); analysis.add_argument('--run',required=True); analysis.add_argument('--output',required=True)
    public=commands.add_parser('publish')
    for name in ['preparation','analysis','output']:public.add_argument('--'+name,required=True)
    prep=commands.add_parser('prepare'); prep.add_argument('--output',required=True); prep.add_argument('--review-per-stratum',type=int,default=2)
    study=commands.add_parser('evaluate')
    for name in ['run','checkpoint','output']: study.add_argument('--'+name,required=True)
    study.add_argument('--reviews'); study.add_argument('--batch-size',type=int,default=4)
    study.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args(argv)
    try:
        cfg=read(args.config)
        if args.command=='prepare': result=prepare(args.output,cfg,args.review_per_stratum)
        elif args.command=='import-reports': result=import_reports(args.archive,args.output)
        elif args.command=='analyze': result=analyze_reports(args.run,args.output,cfg)
        elif args.command=='publish': result=publish(args.preparation,args.analysis,args.output)
        else: result=run_study(args,cfg)
        print(json.dumps(result,indent=2))
    except Exception as exc:
        parser.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__': main()
