"""Verify and reproduce completed external v2 results; never run or tune a model."""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import sha256
from .preflight import require
from .task04 import read,paired_bootstrap,verify_metric_report
from .task05 import fresh,save,seal
from .task06 import comparison_tables
from .task07 import METHOD_MAP,STRATEGIES,cluster_metrics,scientific_decision
from .task07_io import compare_report,verify_predictions
from .metrics import calculate

REFERENCE='304e54bd4ba0f6fa9b012244de090ee66d84b014'
EVALUATION={'development_predictions.csv.gz','rule_intensity_metrics.csv','rule_intensity_vs_RAW.csv',
    'pooled_cluster_metrics.csv','NASN2_vs_simple_paired.csv','all_clean_metrics.csv',
    'all_clean_paired_intervals.csv','scientific_decision.json','loading_info.json','run_status.json','artifact_hashes.json'}
PREPARATION={'evidence.json','run_status.json','normalization_manifest.csv.gz','normalization_rules.json','character_token_audit.csv'}
DIAGNOSIS={'pipeline_equality.csv','all_B3_B4_differences.csv','all_elongation_token_exceptions.csv',
    'guard_difference_summary.csv','annotation_quality_status.json','evidence.json'}
FILES={'evaluation/'+n for n in EVALUATION}|{'preparation/'+n for n in PREPARATION}|{'v1-diagnosis/'+n for n in DIAGNOSIS}
FILES|={'gpu_runtime.json','runtime_choices.json','dependency_check.txt','dependency_freeze.txt'}


def verified_payloads(archive):
    """Reject unknown paths/symlinks/duplicates and verify bytes before extraction."""
    with zipfile.ZipFile(archive) as z:
        members=z.infolist();names=[m.filename for m in members]
        require(len(names)==len(set(names)) and set(names)==FILES|{'report_checksums.json'},'Unexpected/unsafe/duplicate v2 ZIP members')
        require(all(not m.is_dir() and not stat.S_ISLNK(m.external_attr>>16) and m.file_size<=80*1024**2 for m in members)
                and sum(m.file_size for m in members)<=200*1024**2,'Unsafe or oversized v2 archive')
        checks=json.loads(z.read('report_checksums.json'));require(set(checks)==FILES,'Incomplete report hash coverage')
        payload={n:z.read(n) for n in names}
        for name,digest in checks.items():require(hashlib.sha256(payload[name]).hexdigest()==digest,f'Uploaded checksum differs: {name}')
    return payload,checks


def verify_evidence(root):
    status=read(root/'evaluation/run_status.json');choices=read(root/'runtime_choices.json')
    require(status['status']=='completed' and status['metrics_computed'] and status['checkpoint_reload_verified'],'Incomplete v2 run')
    for k in ['training_performed','test_text_read','test_evaluated']:
        require(status[k] is False,f'Protocol violation: {k}')
    require(status['seed']==42 and status['max_length']==128 and status['precision']=='float32'
            and status['bootstrap_repetitions']==2000,'Frozen v2 settings differ')
    require(status['new_normalization_human_validated'] is False and status['reviewer_independence_verified'] is False,
            'Archive asserts human verification without required review evidence')
    require(choices['code_ref']==REFERENCE and choices['seed']==42 and choices['max_length']==128
            and choices['precision']=='float32' and choices['training'] is False
            and choices['inference_batch_size']==status['inference_batch_size'],'Runtime choices differ from pinned protocol')
    known=read('reports/task07/v2-preparation/evidence.json');evidence=read(root/'preparation/evidence.json')
    for key in ['manifest_sha256','source_commit','source_training_sha256','task02_metadata_sha256',
                'checkpoint_sha256','v2_rules_sha256','v2_normalization_code_sha256','runner_code_sha256',
                'frozen_protocol_sha256','review_manifest_sha256','tokenizer_sha256','rows','inverse_checks_passed']:
        require(evidence[key]==known[key],f'Frozen data/rules/code identity differs: {key}')
    require(status['checkpoint_sha256']==known['checkpoint_sha256'],'Selected classifier hashes differ')
    require(sha256(root/'preparation/normalization_rules.json')==sha256('configs/nasn_v2.json'),'Frozen rule bytes differ')
    for file,key in [('nasn/task07.py','runner_code_sha256'),('nasn/normalization_v2.py','v2_normalization_code_sha256'),
                     ('docs/task07_protocol.md','frozen_protocol_sha256'),('configs/nasn_v2.json','v2_rules_sha256')]:
        require(sha256(file)==evidence[key],f'Local frozen source changed: {file}')
    loading=read(root/'evaluation/loading_info.json')
    require(not any(loading.get(k) for k in ['missing_keys','mismatched_keys','unexpected_keys','error_msgs']),'Incomplete classifier load evidence')
    gpu=read(root/'gpu_runtime.json')
    require(gpu['kernel_check_passed'] is True and gpu['gpu']==status['gpu'] and gpu['cuda_build']==status['cuda_build'],'GPU kernel/runtime evidence differs')
    for name,digest in read(root/'evaluation/artifact_hashes.json').items():
        require(Path(name).name==name and sha256(root/'evaluation'/name)==digest,'Evaluation registry differs')
    exported=pd.read_csv(root/'preparation/normalization_manifest.csv.gz',keep_default_na=False)
    registered=pd.read_csv('reports/task07/v2-preparation/normalization_manifest.csv.gz',keep_default_na=False)
    require(exported.equals(registered),'V2 input manifest differs from frozen local preparation')
    pred=pd.read_csv(root/'evaluation/development_predictions.csv.gz',keep_default_na=False)
    inv={v:k for k,v in METHOD_MAP.items()};mapped=pred.copy();mapped.strategy=mapped.strategy.map(inv)
    meta=registered.copy();meta.strategy=meta.strategy.map(inv)
    baseline=pd.read_csv('reports/task03b/marbert-v1-20261009T213143Z-be8b8811/validation_predictions.csv',keep_default_na=False)
    verify_predictions(mapped,meta,baseline)
    require(set(pred.strategy)==set(STRATEGIES),'Incomplete strategy set')
    for field in ['python','pandas','numpy','scikit-learn','transformers','tokenizers']:
        require(status['versions'][field]==known['versions'][field],f'Runtime version differs: {field}')
    # Original v1 diagnostic results remain identical, not reconstructed from v2 outcomes.
    for name in DIAGNOSIS-{'evidence.json'}:
        require((root/'v1-diagnosis'/name).read_bytes()==Path('reports/task07/v1-diagnosis',name).read_bytes(),f'V1 diagnostics changed: {name}')
    return pred,status


def reproduce(root,pred):
    mapped=pred.copy();mapped.strategy=mapped.strategy.map({v:k for k,v in METHOD_MAP.items()})
    metrics,deltas=comparison_tables(mapped,2000)
    metrics.strategy=metrics.strategy.map(METHOD_MAP);deltas.strategy=deltas.strategy.map(METHOD_MAP);deltas.reference='RAW'
    compare_report(pd.read_csv(root/'evaluation/rule_intensity_metrics.csv'),metrics)
    compare_report(pd.read_csv(root/'evaluation/rule_intensity_vs_RAW.csv'),deltas)
    print('Per-rule paired metrics and RAW contrasts reproduced',flush=True)
    pooled,contrasts=cluster_metrics(pred,2000)
    compare_report(pd.read_csv(root/'evaluation/pooled_cluster_metrics.csv'),pooled)
    compare_report(pd.read_csv(root/'evaluation/NASN2_vs_simple_paired.csv'),contrasts)
    print('Original-ID cluster intervals and simple-baseline contrasts reproduced',flush=True)
    clean=[];classes=[];clean_intervals=[]
    raw=pred[pred.kind.eq('clean')&pred.strategy.eq('RAW')].set_index('source_row_id')
    for s,g in pred[pred.kind.eq('clean')].groupby('strategy'):
        overall,perclass,_=calculate(g.gold_id,g.predicted_id);clean.append({'strategy':s,**overall})
        classes.extend({'strategy':s,**row} for row in perclass)
        pairs=g.copy();pairs['clean_id']=pairs.source_row_id.map(raw.predicted_id)
        for metric,value in paired_bootstrap(pairs,2000).items():clean_intervals.append({
            'strategy':s,'reference':'RAW_clean','condition':'normalized_clean_not_perturbed',
            'n_source_ids':len(g),'metric':metric,**value})
    clean=pd.DataFrame(clean);classes=pd.DataFrame(classes)
    compare_report(pd.read_csv(root/'evaluation/all_clean_metrics.csv'),clean)
    compare_report(pd.read_csv(root/'evaluation/all_clean_paired_intervals.csv'),pd.DataFrame(clean_intervals))
    decision=scientific_decision(pooled,contrasts,clean.set_index('strategy'))
    verify_metric_report(read(root/'evaluation/scientific_decision.json'),decision,'frozen_scientific_decision')
    return pooled,contrasts,clean,classes,decision


def summaries(pred,pooled,contrasts,clean,classes,decision,output):
    pooled.to_csv(output/'recomputed_pooled_metrics.csv',index=False)
    contrasts.to_csv(output/'recomputed_paired_contrasts.csv',index=False)
    clean.to_csv(output/'all_clean_metrics.csv',index=False);classes.to_csv(output/'all_clean_per_class.csv',index=False)
    rows=[]
    for (cohort,strategy),g in pooled.groupby(['cohort','strategy']):
        row={'source_cohort':cohort,'strategy':strategy,'n_variants':int(g.n_variants.iloc[0]),
             'n_source_ids':int(g.n_source_ids.iloc[0]),'normalization_human_validated':False}
        indexed=g.set_index('metric')
        for m in ['clean_accuracy','perturbed_accuracy','clean_macro_f1','perturbed_macro_f1','macro_f1_drop','flip_rate']:
            for key in ['estimate','ci_low','ci_high']:row[m+'_'+key]=float(indexed.at[m,key])
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(output/'primary_and_provisional_results.csv',index=False)
    table.to_latex(output/'primary_and_provisional_results.tex',index=False,float_format='%.4f')
    a=pred[pred.strategy.eq('NO_CONTEXT')].set_index('input_id')
    b=pred[pred.strategy.eq('NASN2')].set_index('input_id').loc[a.index]
    audit=a[['source_row_id','kind','source_review_status','rule','intensity','gold_id']].copy()
    audit['characters_differ']=a.normalized_sha256!=b.normalized_sha256
    audit['effective_tokens_differ']=a.normalized_effective_token_sha256!=b.normalized_effective_token_sha256
    audit['predictions_differ']=a.predicted_id!=b.predicted_id
    audit['NO_CONTEXT_prediction']=a.predicted_id;audit['NASN2_prediction']=b.predicted_id
    for method,g in [('NO_CONTEXT',a),('NASN2',b)]:
        for label in ['NEG','NEU','POS']:audit[method+'_probability_'+label]=g['probability_'+label]
    audit['max_probability_difference']=np.max(np.abs(a[[f'probability_{l}' for l in ['NEG','NEU','POS']]].to_numpy()
                                                     -b[[f'probability_{l}' for l in ['NEG','NEU','POS']]].to_numpy()),axis=1)
    audit.to_csv(output/'all_context_ablation_comparisons.csv')
    audit.groupby(['kind','source_review_status']).agg(rows=('source_row_id','size'),
        characters_differ=('characters_differ','sum'),effective_tokens_differ=('effective_tokens_differ','sum'),
        predictions_differ=('predictions_differ','sum')).reset_index().to_csv(output/'context_ablation_summary.csv',index=False)
    save(output/'scientific_decision.json',decision)


def analyze(archive,output):
    payload,checks=verified_payloads(archive)
    output=fresh(output);original=output/'original';original.mkdir()
    for name,content in payload.items():
        path=original/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    try:
        pred,status=verify_evidence(original)
        print(f'All {len(checks)} payload hashes and {len(pred)} prediction identities verified',flush=True)
        pooled,contrasts,clean,classes,decision=reproduce(original,pred)
        summaries(pred,pooled,contrasts,clean,classes,decision,output)
        save(output/'verification.json',{'status':'verified_completed_v2_results','archive_sha256':sha256(archive),
            'original_payload_sha256':checks,'payloads_verified':len(checks),'prediction_rows':len(pred),
            'all_metric_tables_and_decision_recomputed':True,'source_code_ref':REFERENCE,
            'selected_checkpoint_sha256':status['checkpoint_sha256'],'weight_bytes_verified_locally':False,
            'new_model_inference_or_training_performed_locally':False,'official_test_text_read_or_evaluated':False,
            'new_normalization_human_validated':False,'reviewer_independence_verified':False,
            'analyzer_sha256':sha256(Path(__file__)),'external_runtime':status})
        save(output/'run_status.json',{'status':'verified_results_analysis_completed','scientific_decision':decision['decision'],
            'training_performed':False,'test_evaluated':False,'new_rule_tuning':False})
        seal(output)
    except Exception as exc:
        save(output/'run_status.json',{'status':'blocked_or_invalid','reason':f'{type(exc).__name__}: {exc}',
            'results_verified':False,'no_model_or_rule_changes':True});raise
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True);p.add_argument('--output',required=True);args=p.parse_args(argv)
    try:print(analyze(args.archive,args.output))
    except Exception as exc:p.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
