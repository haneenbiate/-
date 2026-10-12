"""Strict intake for the existing Task 06 Colab report ZIP, never code execution."""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import sha256
from .preflight import require,LABEL2ID
from .task05 import fresh,save,seal,registry
from .task06 import comparison_tables

CODE_REF='95f316706e973ab3a43ae08023a1edea2502a678'
FILES={'evaluation/'+name for name in ['comparison_metrics.csv','paired_strategy_differences.csv',
    'development_predictions.csv.gz','run_status.json','loading_info.json','artifact_hashes.json']}
FILES |= {'preparation/'+name for name in ['evidence.json','character_token_audit.csv',
    'normalization_manifest.csv.gz','normalization_rules.json','B3_B4_guard_effect.csv']}
FILES |= {'gpu_runtime.json','runtime_choices.json','dependency_check.txt','dependency_freeze.txt'}


def compare_report(saved,actual):
    require(saved.shape==actual.shape and list(saved)==list(actual),'Saved metric report structure differs')
    for field in actual:
        if pd.api.types.is_numeric_dtype(actual[field]):
            require(np.allclose(pd.to_numeric(saved[field],errors='raise'),actual[field],atol=1e-12,rtol=0,equal_nan=True),f'Metric report differs: {field}')
        else:require(saved[field].fillna('').astype(str).tolist()==actual[field].fillna('').astype(str).tolist(),f'Metric metadata differs: {field}')


def verify_predictions(predictions,manifest,baseline):
    require(not predictions.duplicated(['input_id','strategy']).any(),'Duplicated strategy prediction IDs')
    require(len(predictions)==len(manifest)==62350,'Expected complete Task 06 prediction coverage')
    require(set(manifest)<=set(predictions),'Prediction metadata missing')
    # Exact source metadata includes normalized hashes, labels, review flags and token hashes.
    for field in manifest:
        require(predictions[field].astype(str).tolist()==manifest[field].astype(str).tolist(),f'Prediction identity differs: {field}')
    p=predictions[[f'probability_{label}' for label in LABEL2ID]].to_numpy(dtype=float)
    require(np.isfinite(p).all() and ((0<=p)&(p<=1)).all() and np.allclose(p.sum(axis=1),1,atol=1e-5,rtol=0),'Invalid classifier probabilities')
    require(np.array_equal(p.argmax(axis=1),predictions.predicted_id.astype(int)), 'Saved argmax differs')
    require(np.allclose(p.max(axis=1),predictions.confidence.astype(float),atol=1e-8,rtol=0),'Saved confidence differs')
    raw=predictions[predictions.strategy.eq('B0')&predictions.kind.eq('clean')].set_index('source_row_id').loc[baseline.source_row_id]
    require(raw.predicted_id.tolist()==baseline.predicted_id.tolist() and np.allclose(raw[[f'probability_{label}' for label in LABEL2ID]],baseline[[f'probability_{label}' for label in LABEL2ID]],atol=1e-5,rtol=1e-5),'B0 clean results differ from selected baseline')


def import_bundle(archive,output):
    with zipfile.ZipFile(archive) as bundle:
        members=bundle.infolist();names=[m.filename for m in members]
        require(len(names)==len(set(names)) and set(names)==FILES|{'report_checksums.json'},'Unexpected/unsafe/duplicate Task 06 ZIP members')
        require(sum(m.file_size for m in members)<=200*1024**2 and all(m.file_size<=80*1024**2 for m in members),'Report archive too large')
        require(not any(stat.S_ISLNK(m.external_attr>>16) or m.is_dir() for m in members),'Archive symlinks/directories prohibited')
        checks=json.loads(bundle.read('report_checksums.json'))
        require(set(checks)==FILES,'Report registry coverage differs')
        payload={name:bundle.read(name) for name in names}
        for name,digest in checks.items():require(hashlib.sha256(payload[name]).hexdigest()==digest,f'Uploaded hash mismatch: {name}')
    # Extract verified bytes only. Any subsequent failure leaves a blocked intake record.
    output=fresh(output)
    for name,content in payload.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    try:
        status=json.loads((output/'evaluation/run_status.json').read_text())
        require(status['status']=='completed' and status['metrics_computed'] and status['checkpoint_reload_verified'], 'Task 06 inference not complete')
        require(all(status[k] is False for k in ['training_performed','test_text_read','test_evaluated']), 'Protocol violation in Task 06 status')
        require(status['seed']==42 and status['bootstrap_repetitions']==2000 and status['inference_precision']=='float32','Task 06 inference settings differ')
        expected=json.loads(Path('reports/task03b/marbert-v1-20261009T213143Z-be8b8811/checkpoint_hashes.json').read_text())
        require(status['checkpoint_sha256']==expected,'Selected trained checkpoint hashes differ')
        choices=json.loads((output/'runtime_choices.json').read_text())
        require(choices['code_ref']==CODE_REF and choices['seed']==42 and choices['max_length']==128 and choices['training'] is False,'Unpinned Task 06 runtime protocol')
        loading=json.loads((output/'evaluation/loading_info.json').read_text())
        require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']), 'Incomplete classifier loading evidence')
        evidence=json.loads((output/'preparation/evidence.json').read_text())
        known=json.loads(Path('reports/task06b/evidence.json').read_text())
        for field in ['manifest_sha256','source_commit','source_training_sha256','task02_metadata_sha256',
                      'rules_sha256','normalization_code_sha256','task06_code_sha256','review_manifest_sha256','checkpoint_sha256','tokenizer_sha256']:
            require(evidence[field]==known[field],f'Prepared source/model protocol differs: {field}')
        for name,digest in json.loads((output/'evaluation/artifact_hashes.json').read_text()).items():
            require(Path(name).name==name and sha256(output/'evaluation'/name)==digest,'Evaluation registry mismatch')
        pred=pd.read_csv(output/'evaluation/development_predictions.csv.gz',keep_default_na=False)
        exported=pd.read_csv(output/'preparation/normalization_manifest.csv.gz',keep_default_na=False)
        public=pd.read_csv('reports/task06b/normalization_manifest.csv.gz',keep_default_na=False)
        require(exported.equals(public),'Exported Task 06 input manifest differs from registered original')
        baseline=pd.read_csv('reports/task03b/marbert-v1-20261009T213143Z-be8b8811/validation_predictions.csv',keep_default_na=False)
        verify_predictions(pred,public,baseline)
        metrics,deltas=comparison_tables(pred,2000)
        compare_report(pd.read_csv(output/'evaluation/comparison_metrics.csv'),metrics)
        compare_report(pd.read_csv(output/'evaluation/paired_strategy_differences.csv'),deltas)
        save(output/'intake_status.json',{'status':'verified_completed_task06_inference','source_archive_sha256':sha256(archive),
            'code_ref':CODE_REF,'rows':len(pred),'metric_tables_recomputed':True,'uploaded_payload_hashes_verified':len(checks),
            'selected_checkpoint_bytes_verified_here':False,'checkpoint_verification':'external recorded load/hash evidence; no local weight bytes',
            'human_independence_verified':False,'test_text_read':False,'training_performed':False})
        seal(output/'evaluation')
        seal(output)
    except Exception as exc:
        save(output/'intake_status.json',{'status':'blocked_or_invalid','reason':f'{type(exc).__name__}: {exc}'})
        raise
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True);p.add_argument('--output',required=True);args=p.parse_args(argv)
    try:print(import_bundle(args.archive,args.output))
    except Exception as exc:p.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
