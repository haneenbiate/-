"""MARBERT v1.0 runner: existing manifests only, validation-only model selection."""
import argparse
import json
import math
import os
import random
import re
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import DataValidationError, sha256
from .preflight import LABEL2ID, library_versions, preflight, require
from .metrics import calculate, prediction_frame, subgroup_metrics


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def safe_error(exc):
    """Retain destination/error diagnostics without signed download URL queries."""
    def strip_query(match):
        url = urlsplit(match.group(0))
        return urlunsplit((url.scheme, url.netloc, url.path, '', ''))
    message = re.sub(r'https?://[^\s\'"<>)]*', strip_query, str(exc))
    return re.sub(r'\?[^\s\'"<>)]*', '?[query omitted]', message)


def cpu_probe(model, training, tokenizer, config, output):
    """Measure a backward pass without optimizer updates or changing seeded training."""
    import torch
    sample = training.iloc[:config['batch_size']]
    dataset = encoded_dataset(sample, tokenizer, config['max_length'])
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(sample))))
    state = torch.get_rng_state()
    model.train()
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for offset in range(0, len(sample), config['micro_batch_size']):
        small = {k: v[offset:offset + config['micro_batch_size']] for k,v in batch.items()}
        (model(**small).loss * len(small['labels']) / len(sample)).backward()
    seconds = time.perf_counter() - started
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(state)
    projected = seconds * math.ceil(len(training) / config['batch_size']) * config['max_epochs']
    details = {'probe_examples': len(sample), 'effective_batch_backward_seconds': seconds,
               'projected_max_epoch_backward_seconds': projected,
               'projection_excludes': 'validation, optimizer updates, checkpoint IO; not a model result',
               'cpu_budget_seconds': config['cpu_max_projected_training_seconds'],
               'optimizer_updates': 0, 'model_parameters_updated': False}
    save_json(output / 'compute_probe.json', details)
    require(projected <= config['cpu_max_projected_training_seconds'],
        f'CPU feasibility guard: projected {projected:.1f}s for 5 epochs exceeds documented '
        f"{config['cpu_max_projected_training_seconds']}s budget; no GPU available. "
        'Use a CUDA machine or explicitly increase the CPU time budget; no epoch was trained.')


def check_config(config):
    expected = {'seed': 42, 'max_length': 128, 'learning_rate': 2e-5, 'batch_size': 16, 'max_epochs': 5,
                'weight_decay': .01, 'optimizer': 'AdamW', 'early_stopping_patience': 2,
                'checkpoint_selection': 'validation_macro_f1', 'model_name': 'UBC-NLP/MARBERT'}
    for name, value in expected.items():
        require(config[name] == value, f'Baseline v1.0 parameter changed: {name}')
    require(config['label2id'] == LABEL2ID, 'Label mapping mismatch')
    micro = config['micro_batch_size']
    require(isinstance(micro, int) and micro > 0 and 16 % micro == 0, 'Micro batch must divide effective batch 16')
    require(config['gradient_accumulation_steps'] == 16 // micro, 'Gradient accumulation mismatch')
    require(config['precision'] == 'float32' and config['lr_schedule'] == 'constant' and config['warmup_steps'] == 0,
            'Unsupported precision/scheduler configuration')
    require(config['device'] in ['auto', 'cpu', 'cuda'], 'Invalid device')
    require(config['cpu_max_projected_training_seconds'] > 0, 'CPU feasibility budget must be positive')


def environment():
    import torch
    return {'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cuda_available': torch.cuda.is_available(), 'cpu_count': os.cpu_count(),
            'cpu_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
            'memory_limit_bytes': Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None,
            'library_versions': library_versions()}


def resolve_revision(config):
    revision = config['model_revision']
    if revision is None:
        url = f"https://huggingface.co/api/models/{config['model_name']}/revision/{config['resolve_revision_from']}"
        with urllib.request.urlopen(url, timeout=30) as response:
            revision = json.load(response)['sha']
    require(bool(re.fullmatch(r'[0-9a-f]{40}', revision)), 'Model revision must be an immutable 40-character SHA')
    return revision


def seed_all(seed, threads):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def encoded_dataset(frame, tokenizer, max_length):
    import torch
    require(set(frame.partition) <= {'train', 'validation'}, 'Official test/quarantine cannot enter model dataset')
    # Feed original tweet strings directly to the official tokenizer.
    tokens = tokenizer(frame.tweet.tolist(), max_length=max_length, truncation=True, padding='max_length',
                       return_tensors='pt')
    labels = torch.tensor(frame.sentiment.map(LABEL2ID).to_numpy(dtype=np.int64))
    require(all(value.shape[0] == len(frame) for value in tokens.values()), 'Tokenizer row count mismatch')
    class TokenDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)
        def __getitem__(self, index):
            return {**{key: value[index] for key, value in tokens.items()}, 'labels': labels[index]}
    return TokenDataset()


def evaluate(model, loader, device):
    import torch
    model.eval()
    probabilities, loss_sum, total = [], 0., 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(**batch)
            require(result.logits.shape == (len(batch['labels']), 3), 'Model output shape mismatch')
            probabilities.append(result.logits.softmax(dim=-1).cpu().numpy())
            loss_sum += float(result.loss.item()) * len(batch['labels'])
            total += len(batch['labels'])
    return np.concatenate(probabilities), loss_sum / total


def train_epoch(model, loader, optimizer, device, micro_batch_size, max_grad_norm):
    import torch
    model.train()
    loss_sum, total = 0., 0
    for batch in loader:
        size = len(batch['labels'])
        optimizer.zero_grad(set_to_none=True)
        # One loader batch is an effective batch (16, except final remainder).
        # Weight microbatch mean losses by sample count; handles the final partial group correctly.
        for offset in range(0, size, micro_batch_size):
            small = {key: value[offset:offset + micro_batch_size].to(device) for key, value in batch.items()}
            n = len(small['labels'])
            result = model(**small)
            require(torch.isfinite(result.loss).item(), 'Non-finite training loss')
            (result.loss * n / size).backward()
            loss_sum += float(result.loss.item()) * n
            total += n
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    return loss_sum / total


def fit(training, validation, config, output):
    os.environ.setdefault('HF_HOME', str(Path(config['cache_directory']).resolve()))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from huggingface_hub.utils import logging as hub_logging
    hub_logging.set_verbosity_error()
    seed_all(config['seed'], config['cpu_threads'])
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if config['device'] == 'auto' else config['device']
    require(device != 'cuda' or torch.cuda.is_available(), 'Requested CUDA device unavailable')
    resolved = resolve_revision(config)
    config = {**config, 'model_revision': resolved, 'resolved_device': device,
              'resolved_library_versions': library_versions()}
    save_json(output / 'resolved_training_config.json', config)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=resolved,
        cache_dir=config['cache_directory'], trust_remote_code=False)
    # Standard sequence-classification head/pooler are initialized as required by an MLM checkpoint.
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(config['model_name'], revision=resolved,
                num_labels=3, label2id=LABEL2ID, id2label={v: k for k, v in LABEL2ID.items()},
                cache_dir=config['cache_directory'], trust_remote_code=False, output_loading_info=True)
    missing_encoder = [name for name in loading_info['missing_keys']
                       if not name.startswith(('classifier.', 'bert.pooler.'))]
    require(not missing_encoder and not loading_info.get('error_msgs'),
            f'Pretrained encoder failed to load completely: {missing_encoder}')
    save_json(output / 'model_loading_info.json', loading_info)
    require(model.config.num_labels == 3 and model.config.label2id == LABEL2ID, 'Classification head mapping incorrect')
    model.to(device)
    if device == 'cpu':
        cpu_probe(model, training, tokenizer, config, output)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'],
                                 betas=tuple(config['adam_betas']), eps=config['adam_epsilon'])
    generator = torch.Generator().manual_seed(config['seed'])
    training_loader = torch.utils.data.DataLoader(encoded_dataset(training, tokenizer, config['max_length']),
        batch_size=config['batch_size'], shuffle=True, generator=generator, num_workers=config['num_workers'])
    validation_loader = torch.utils.data.DataLoader(encoded_dataset(validation, tokenizer, config['max_length']),
        batch_size=config['micro_batch_size'], shuffle=False, num_workers=config['num_workers'])
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    checkpoint = output / 'best_checkpoint'
    best, stale, best_epoch = -1., 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, config['max_epochs'] + 1):
        epoch_start = time.perf_counter()
        training_loss = train_epoch(model, training_loader, optimizer, device, config['micro_batch_size'], config['max_grad_norm'])
        probabilities, validation_loss = evaluate(model, validation_loader, device)
        predictions = prediction_frame(validation, probabilities)
        metrics, _, _ = calculate(predictions.gold_id, predictions.predicted_id)
        improved = metrics['macro_f1'] > best + config['minimum_improvement']
        if improved:
            best, stale, best_epoch = metrics['macro_f1'], 0, epoch
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
        else:
            stale += 1
        history.append({'epoch': epoch, 'training_loss': training_loss, 'validation_loss': validation_loss,
                        **metrics, 'selected_as_best_at_epoch': improved, 'duration_seconds': time.perf_counter() - epoch_start})
        pd.DataFrame(history).to_csv(output / 'epoch_history.csv', index=False)
        print(json.dumps(history[-1]), flush=True)
        if stale >= config['early_stopping_patience']:
            break
    duration = time.perf_counter() - start
    del optimizer, model
    if device == 'cuda':
        torch.cuda.empty_cache()
    # Reload BOTH saved model and tokenizer, and rebuild raw-text validation inputs.
    reloaded_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    selected = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device)
    reload_loader = torch.utils.data.DataLoader(encoded_dataset(validation, reloaded_tokenizer, config['max_length']),
        batch_size=config['micro_batch_size'], shuffle=False, num_workers=config['num_workers'])
    probabilities, validation_loss = evaluate(selected, reload_loader, device)
    predictions = prediction_frame(validation, probabilities)
    metrics, per_class, matrix = calculate(predictions.gold_id, predictions.predicted_id)
    require(abs(metrics['macro_f1'] - best) < 1e-12, 'Reloaded checkpoint Macro-F1 differs from selected checkpoint')
    save_json(output / 'validation_metrics.json', {'status': 'completed', 'checkpoint_epoch': best_epoch,
              'validation_loss': validation_loss, **metrics})
    pd.DataFrame(per_class).to_csv(output / 'per_class_results.csv', index=False)
    pd.DataFrame(matrix, index=list(LABEL2ID), columns=list(LABEL2ID)).rename_axis('gold_label').to_csv(output / 'confusion_matrix.csv')
    predictions.to_csv(output / 'validation_predictions.csv', index=False)
    save_json(output / 'subgroup_metrics.json', subgroup_metrics(predictions, config['subgroup_min_size']))
    hashes = {str(file.relative_to(checkpoint)): sha256(file) for file in sorted(checkpoint.rglob('*')) if file.is_file()}
    save_json(output / 'checkpoint_hashes.json', hashes)
    return {'status': 'completed', 'training_duration_seconds': duration, 'device': device,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None,
            'best_epoch': best_epoch, 'best_checkpoint': str(checkpoint), 'checkpoint_reload_verified': True,
            'model_revision': resolved, 'epochs_completed': len(history), 'test_evaluated': False}


def methodology(output, config, status, preflight_summary=None):
    pre = preflight_summary or {}
    text = f'''# NASN — MARBERT Baseline v1.0 (Task 03)

Status: **{status['status']}**. Official test evaluated: **false**.

## Inputs and preflight

Use existing Task 02 manifests only. Never reconstruct splits, refit split selection,
clean text, drop flags, or merge duplicates. Source commit is
`eb85513e133cfc67e30aca4968fadbe2594fcb42`. Counts: train 10,037;
validation 2,509; quarantine 2; official test 3,000. Source and every registered
Task 02 CSV checksum, metadata/methodology hashes, runtime versions, stable row
IDs, raw text hashes, group IDs, labels, exhaustive row coverage, complete quarantine,
and exact/canonical disjointness are checked. See preflight.json for measured results.
Official test text/labels are inspected only for the requested data-quality audit;
they are not provided to tokenization, training, prediction or metric computation.

Dataset sentiment strings are NEG, NEU and POS. Task 02 records no numerical
model label IDs. This baseline explicitly freezes NEG=0, NEU=1, POS=2 in config,
model label2id/id2label and prediction artifacts; it does not claim these numerical
IDs are an official dataset convention.

## Near-duplicate concerns

Audited pairs: {pre.get('near_pairs', 'unverified')}.
Active cross-split pairs: {pre.get('near_cross_split', 'unverified')}.
Sentiment-conflicting pairs across all memberships: {pre.get('near_label_conflicts', 'unverified')}.
High-risk manual review pairs: {pre.get('high_risk_manual_review_pairs', 'unverified')}.
High risk is frozen as active cross-split AND (cosine >=0.95 OR label conflict).
near_duplicate_audit.csv records membership, score, both labels and reasons;
near_duplicate_summary.csv groups memberships. manual_review_queue.csv is a queue,
not a claim that humans reviewed these candidates. No record is removed or merged.
Similarity is lexical, not semantic equivalence. Canonical disjointness alone does
not guarantee independent text. Test-label conflicts are descriptive and cannot
change model settings, stopping or checkpoint selection.

## Model and fitting

Official model/tokenizer: UBC-NLP/MARBERT; immutable revision is resolved and saved
before loading. Current resolved revision: {status.get('model_revision', 'not resolved')}.
No fallback encoder or randomly initialized full model is permitted. The standard
three-class classifier is initialized under seed 42; the standard BERT pooler may
also require initialization because the official checkpoint is BertForMaskedLM.
Actual missing/unexpected weight keys are saved after successful loading; missing
encoder parameters fail explicitly. Raw tweet strings go
directly to the official tokenizer; its own behavior is retained, with padding
and truncation to 128 tokens. No Arabic normalization, stemming, stopword removal,
negation rules or canonicalization is applied to model inputs.

Seed 42; AdamW lr 2e-5, weight decay 0.01 on all trainable parameters,
betas (0.9,0.999), epsilon 1e-8; effective batch 16, microbatch {config['micro_batch_size']},
accumulation {config['gradient_accumulation_steps']}; at most 5 epochs; float32;
CPU feasibility limit: {config.get('cpu_max_projected_training_seconds', 'not configured in this earlier attempt')} seconds projected
for the maximum epoch count, based on one backward-only effective batch with no
optimizer update and restored Torch RNG. This operational limit can be explicitly
raised; it does not change model hyperparameters. If weights cannot be downloaded,
the compute probe is unrun. The probe is not a training experiment or metric.
constant learning rate, no warmup; gradient clipping norm 1.0. A final partial
batch uses its actual example count to weight microbatch gradients. Training rows
are shuffled with a seeded generator; validation is in saved order. No class
weights, augmentation, test evaluation or new hyperparameter search.

At each epoch end, compute validation Macro-F1 over all three fixed classes.
Select strictly highest Macro-F1; ties retain the earlier checkpoint. Early stopping
after two consecutive nonimproving epochs. Zero-division precision/recall/F1 is 0.
All random generators are seeded; deterministic Torch kernels are required. Results
can still differ across hardware, library versions or official tokenizer revisions;
versions and revision are recorded. Checkpoint reload is verified on raw validation
text only after a successful fit. A failed reload prevents a completed-run claim.

## Evaluation and provenance

Successful runs save validation accuracy, Macro-F1, weighted F1, three-class
precision/recall/F1/support, a 3x3 confusion matrix (gold rows/predicted columns),
per-example gold/prediction IDs and labels, maximum softmax confidence and all
three probabilities. Confidence is uncalibrated. Epoch history records loss,
validation metrics and elapsed time. Sarcasm/dialect subgroups require n>=30;
smaller groups are explicitly uncomputed. Subgroup Macro-F1 uses fixed three-class
support; missing classes may depress it and are reported. These are descriptive,
not new checkpoint-selection criteria.

Training config, manifest/source/checkpoint hashes, software versions, device,
training duration and peak GPU allocated/reserved memory (null on CPU) are persisted.
Best checkpoint: {status.get('best_checkpoint') or 'unavailable — no completed model training'}.
Checkpoint reload verified: {status.get('checkpoint_reload_verified', False)}.
Training duration seconds: {status.get('training_duration_seconds')}.
Blocker/failure: {status.get('reason') or 'none'}.

## Reproduction and remaining work

Exact commands are in docs/task03.md. The config is configs/marbert_v1.json;
after revision resolution, use resolved_training_config.json to reproduce the
same pretrained encoder. New/empty output directories are mandatory. No automatic
split generation or overwriting of old experiment results. If model access or
compute is unavailable, validation metrics and checkpoints are explicitly unrun;
no zeros, synthetic scores or tiny-model test results substitute for MARBERT results.
Offline random tiny-BERT tests verify software shapes/save-reload only, not this
baseline's performance or actual MARBERT checkpoint reload.
'''
    (output / 'methodology.md').write_text(text, encoding='utf-8')


def run(config_path, output, preflight_only=False):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), 'Use a new/empty experiment output directory')
    config = json.loads(Path(config_path).read_text())
    check_config(config)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'training_config.json', config)
    status = {'status': 'initializing', 'best_checkpoint': None, 'checkpoint_reload_verified': False,
              'training_duration_seconds': None, 'test_evaluated': False}
    pre = None
    started = time.perf_counter()
    try:
        training, validation, pre = preflight(config, output)
        save_json(output / 'environment.json', environment())
        if preflight_only:
            status.update(status='preflight_only', reason='Training not requested in this invocation')
        else:
            status.update(fit(training, validation, config, output))
    except Exception as exc:
        status.update(status='blocked_or_failed', reason=f'{type(exc).__name__}: {safe_error(exc)}',
                      attempt_duration_seconds=time.perf_counter() - started)
        if (output / 'resolved_training_config.json').exists():
            status['model_revision'] = json.loads((output / 'resolved_training_config.json').read_text())['model_revision']
        save_json(output / 'run_status.json', status)
        if not (output / 'validation_metrics.json').exists():
            save_json(output / 'validation_metrics.json', {'status': 'not_run', 'metrics': None, 'reason': status['reason']})
        methodology(output, config, status, pre)
        raise
    save_json(output / 'run_status.json', status)
    methodology(output, config, status, pre)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'train'])
    parser.add_argument('--config', default='configs/marbert_v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.config, args.output, args.command == 'preflight'), indent=2))
    except Exception as exc:
        parser.exit(2, f'ERROR: {type(exc).__name__}: {safe_error(exc)}\n')

if __name__ == '__main__':
    main()
