"""External-runtime integrity and storage helpers; no training algorithm."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REFERENCE = 'c8dc698780cb06c8838c557f159bdaed7507aeda'
MODEL = 'UBC-NLP/MARBERT'
REVISION = '88e1fa192dd723cf0b3563500aec46209762eb22'
REQUIRED_FILES = {'config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'pytorch_model.bin'}
WEIGHTS_SHA256 = '801862ace5fc9e6b135b19476dbda5c7de1e8673b5b9d55db08d79e2c30b9dca'
REPORT_FILES = {'training_config.json','resolved_training_config.json','preflight.json','environment.json',
    'run_status.json','methodology.md','epoch_history.csv','validation_metrics.json','per_class_results.csv',
    'confusion_matrix.csv','validation_predictions.csv','subgroup_metrics.json','checkpoint_hashes.json',
    'model_loading_info.json','near_duplicate_audit.csv','near_duplicate_summary.csv','manual_review_queue.csv'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def safe_error(error):
    return re.sub(r'\?[^\s\'"<>)]*', '?[query omitted]', str(error))


def validate_lock(lock):
    require(lock['repo_id'] == MODEL and lock['revision'] == REVISION, 'Snapshot revision/identity differs from baseline')
    require(set(lock['files']) == REQUIRED_FILES, 'Snapshot lock must cover all required official files')
    require(lock['files']['pytorch_model.bin']['sha256'] == WEIGHTS_SHA256, 'Untrusted pretrained weight hash')
    for name, details in lock['files'].items():
        require(Path(name).name == name, 'Unsafe snapshot filename')
        require(bool(re.fullmatch('[0-9a-f]{64}', details['sha256'])), 'Invalid snapshot checksum')


def verify_snapshot(snapshot, lock):
    validate_lock(lock)
    snapshot = Path(snapshot)
    require(snapshot.is_dir(), 'Snapshot directory missing')
    extra = {p.name for p in snapshot.iterdir()} - REQUIRED_FILES - {'snapshot_revision.json'}
    require(not extra, f'Unexpected snapshot files (could shadow verified weights/tokenizer): {sorted(extra)}')
    hashes = {}
    for name, expected in lock['files'].items():
        path = snapshot / name
        require(path.is_file(), f'Official snapshot file missing: {name}')
        require(path.stat().st_size == expected['size'], f'Snapshot size mismatch: {name}')
        hashes[name] = sha256(path)
        require(hashes[name] == expected['sha256'], f'Snapshot SHA-256 mismatch: {name}')
    return hashes


def cache_snapshot(source, cache, lock):
    """Stage verified bytes in the standard exact-revision Hub cache; no source edits."""
    hashes = verify_snapshot(source, lock)
    target = Path(cache) / 'models--UBC-NLP--MARBERT' / 'snapshots' / REVISION
    target.mkdir(parents=True, exist_ok=True)
    extra = {p.name for p in target.iterdir()} - REQUIRED_FILES - {'snapshot_revision.json'}
    require(not extra, f'Unverified files in exact-revision cache: {sorted(extra)}')
    for name, digest in hashes.items():
        destination = target / name
        if destination.exists():
            require(sha256(destination) == digest, f'Conflicting existing Hub cache file: {name}')
        else:
            shutil.copyfile(Path(source) / name, destination)
    verify_snapshot(target, lock)
    return target


def acquire_snapshot(cache, lock, report, local=None):
    validate_lock(lock)
    try:
        if local:
            # Local revision declaration is checked, but byte trust comes from the external official lock.
            declaration = json.loads((Path(local) / 'snapshot_revision.json').read_text())
            require(declaration['repo_id'] == MODEL and declaration['revision'] == REVISION,
                    'Local snapshot revision declaration mismatch')
            snapshot = cache_snapshot(local, cache, lock)
            origin = 'verified local snapshot'
        else:
            os.environ['HF_HUB_DISABLE_XET'] = '1'
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import logging
            logging.set_verbosity_error()
            snapshot = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache,
                allow_patterns=sorted(REQUIRED_FILES), max_workers=1))
            origin = 'official Hub exact revision'
        hashes = verify_snapshot(snapshot, lock)
        save(report, {'status':'verified','repo_id':MODEL,'revision':REVISION,'source':origin,
                      'snapshot':str(snapshot),'files_sha256':hashes,'encoder_load_verified':False})
        return snapshot
    except Exception as exc:
        save(report, {'status':'blocked_or_invalid','repo_id':MODEL,'revision':REVISION,
                      'error':safe_error(exc),'weights_verified':False,'encoder_load_verified':False})
        raise ValueError(safe_error(exc)) from None


def gpu_report(path):
    import torch
    info = {'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,
            'python':sys.version,'python_executable':sys.executable,
            'cuda_available':torch.cuda.is_available(),'reference_commit':REFERENCE}
    try:
        require(torch.version.cuda is not None and torch.cuda.is_available(), 'CUDA GPU unavailable; no CPU fallback')
        free, total = torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                    free_vram_bytes=free,total_vram_bytes=total)
        info['nvidia_smi'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free',
            '--format=csv,noheader'], text=True).strip()
        require(free >= 6 * 1024**3, 'Less than 6 GiB free VRAM; request a larger GPU runtime')
        value = torch.ones((8,8),device='cuda')
        require(torch.isfinite(value @ value).all().item(), 'CUDA kernel compatibility test failed')
        info['status'] = 'gpu_gate_passed'
        save(path, info)
    except Exception as exc:
        info.update(status='blocked',error=safe_error(exc))
        save(path, info)
        raise


def make_config(original, gpu, cache, micro=None):
    require(gpu['status'] == 'gpu_gate_passed', 'GPU gate has not passed')
    if micro is None:
        free = gpu['free_vram_bytes'] / 1024**3
        micro = 4 if free >= 12 else 2 if free >= 8 else 1
    require(type(micro) is int and micro in [1,2,4], 'Allowed microbatch sizes are 4, 2, 1')
    require(original['model_revision'] == REVISION and original['batch_size'] == 16, 'Unexpected baseline configuration')
    config = {**original,'device':'cuda','micro_batch_size':micro,'gradient_accumulation_steps':16 // micro,
              'cache_directory':str(cache)}
    allowed = {'device','micro_batch_size','gradient_accumulation_steps','cache_directory'}
    changes = {k:{'original':original[k],'external':v} for k,v in config.items() if original[k] != v}
    require(set(changes) <= allowed, 'Experimental parameter changed')
    return config, changes


def check_encoder(cache, report):
    # Call exactly the same Transformers loader as the original CLI, then discard this untrained head.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import AutoModelForSequenceClassification
    from nasn.marbert import seed_all
    from nasn.preflight import LABEL2ID
    seed_all(42,4)
    model, info = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, cache_dir=cache,
        local_files_only=True,num_labels=3,label2id=LABEL2ID,id2label={v:k for k,v in LABEL2ID.items()},
        trust_remote_code=False,output_loading_info=True)
    missing = [key for key in info['missing_keys'] if not key.startswith(('classifier.','bert.pooler.'))]
    require(not missing and not info.get('error_msgs') and not info.get('mismatched_keys'),
            f'Incomplete pretrained encoder: {missing}')
    require(model.config.num_labels == 3 and model.config.label2id == LABEL2ID, 'Classification label mapping changed')
    save(report, {'status':'encoder_verified','revision':REVISION,'loading_info':info,
                  'model_parameters':sum(p.numel() for p in model.parameters()),'training_performed':False})
    del model


def verify_completed_run(run):
    run = Path(run)
    required = REPORT_FILES - {'near_duplicate_audit.csv','near_duplicate_summary.csv','manual_review_queue.csv'}
    require(all((run / name).is_file() for name in required), 'Completed-run artifacts are missing')
    status = json.loads((run / 'run_status.json').read_text())
    require(status['status'] == 'completed' and status['checkpoint_reload_verified'] is True,
            'Run not completed or checkpoint reload not verified')
    require(status['test_evaluated'] is False and status['model_revision'] == REVISION,
            'Test evaluation or incorrect model revision detected')
    hashes = json.loads((run / 'checkpoint_hashes.json').read_text())
    require('model.safetensors' in hashes, 'Trained model weight hash missing')
    checkpoint = (run / 'best_checkpoint').resolve()
    for name, digest in hashes.items():
        file = (checkpoint / name).resolve()
        require(file.is_relative_to(checkpoint) and file.is_file(), 'Invalid checkpoint hash entry')
        require(sha256(file) == digest, f'Durable checkpoint hash mismatch: {name}')
    from nasn.metrics import calculate
    import pandas as pd
    predictions = pd.read_csv(run / 'validation_predictions.csv',dtype={'sarcasm':str})
    require(len(predictions) == 2509 and not predictions.source_row_id.duplicated().any(), 'Validation prediction coverage incorrect')
    config = json.loads((run / 'resolved_training_config.json').read_text())
    manifest = pd.read_csv(Path(config['split_directory']) / 'validation_manifest.csv',dtype=str)
    preflight = json.loads((run / 'preflight.json').read_text())
    require(sha256(Path(config['split_directory']) / 'validation_manifest.csv') ==
            preflight['manifest_sha256']['validation_manifest.csv'], 'Validation manifest changed')
    require(predictions.source_row_id.tolist() == manifest.source_row_id.tolist() and
            predictions.gold_label.tolist() == manifest.sentiment.tolist(), 'Predictions do not match saved validation rows/labels')
    metrics, _, _ = calculate(predictions.gold_id,predictions.predicted_id)
    saved = json.loads((run / 'validation_metrics.json').read_text())
    require(saved['status'] == 'completed', 'Validation not completed')
    for key in ['accuracy','macro_f1','weighted_f1']:
        require(abs(saved[key] - metrics[key]) < 1e-12, f'Saved validation metric differs from predictions: {key}')
    return status


def export_reports(run, target):
    """Strict allowlist excludes weights, raw data, tokenizer dumps, and arbitrary logs."""
    run, target = Path(run), Path(target)
    require(not target.exists(), 'Lightweight report export directory already exists')
    target.mkdir(parents=True)
    for name in sorted(REPORT_FILES):
        path = run / name
        if path.is_file():
            require(path.stat().st_size <= 10 * 1024**2, f'Report exceeds size limit: {name}')
            shutil.copyfile(path,target / name)
    save(target / 'report_checksums.json', {p.name:sha256(p) for p in sorted(target.iterdir()) if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    gpu = sub.add_parser('gpu'); gpu.add_argument('--report',required=True)
    snapshot = sub.add_parser('snapshot')
    for name in ['cache','lock','report']: snapshot.add_argument('--'+name,required=True)
    snapshot.add_argument('--local')
    encoder = sub.add_parser('encoder')
    encoder.add_argument('--cache',required=True); encoder.add_argument('--report',required=True)
    verified = sub.add_parser('verify-run'); verified.add_argument('--run',required=True)
    export = sub.add_parser('export'); export.add_argument('--run',required=True); export.add_argument('--target',required=True)
    args = parser.parse_args()
    try:
        if args.command == 'gpu': gpu_report(args.report)
        elif args.command == 'snapshot': acquire_snapshot(args.cache,json.loads(Path(args.lock).read_text()),args.report,args.local)
        elif args.command == 'encoder': check_encoder(args.cache,args.report)
        elif args.command == 'verify-run': print(json.dumps(verify_completed_run(args.run),indent=2))
        elif args.command == 'export': export_reports(args.run,args.target)
    except Exception as exc:
        parser.exit(2,f'ERROR: {type(exc).__name__}: {safe_error(exc)}\n')

if __name__ == '__main__': main()
