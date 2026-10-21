"""Build the Study v3 Colab notebook (training, inference, LLM, analysis) pinned to one commit."""
import argparse
import json
from pathlib import Path

# SHA-256 of the decompressed variants table (platform- and zlib-independent).
VARIANTS_CONTENT_SHA256 = {'arsarcasm': '770db43a92b85c5ed40ac9795c544c58da874f262a6273f65f9e7cb90d7b2ec8',
                           'arsas': 'e9537c2491b8dca4e8a7652162e56cf15f74df3049d31c9af29dba21e2a553b6'}


def cell(kind, source):
    base = {'cell_type': kind, 'metadata': {}, 'source': source.strip('\n').splitlines(keepends=True)}
    return base if kind == 'markdown' else {**base, 'execution_count': None, 'outputs': []}


def build(reference):
    cells = [
        cell('markdown', '''# Study v3 — multi-model orthographic-noise robustness audit

Run cells in order on a GPU runtime. Every stage writes to Google Drive and **skips
work that is already complete**, so a disconnected session resumes by re-running all
cells. Encoders fit on a T4 (about 20–30 min per run; 5 models × 3 seeds × 2 datasets);
the LLM stage needs at least 24 GB (L4 or A100) and is skipped otherwise. Official
test partitions stay locked; only validation partitions are used here.

Tip: to split the work across sessions, shorten `DATASETS`, `ENCODERS` or `SEEDS` in
the next cell; finished runs are never repeated.
'''),
        cell('code', f'''REF = '{reference}'
DRIVE = '/content/drive/MyDrive/NASN/study-v3'
DATASETS = ['arsarcasm', 'arsas']
ENCODERS = ['MARBERT', 'MARBERTv2', 'AraBERTv02-twitter', 'CAMeLBERT-DA', 'XLM-R-base']
SEEDS = [42, 43, 44]
LLMS = ['Qwen2.5-7B-Instruct']
VARIANTS_CONTENT_SHA256 = {json.dumps(VARIANTS_CONTENT_SHA256)}
'''),
        cell('code', '''import gzip, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
REPO = Path('/content/nasn-study-v3')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/haneenbiate/-.git', str(REPO)], check=True)
subprocess.run(['git', 'fetch', 'origin'], cwd=REPO, check=True)
subprocess.run(['git', 'checkout', REF], cwd=REPO, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-marbert.txt', 'accelerate==1.10.1'], cwd=REPO, check=True)
# Colab preinstalls torchvision/torchaudio built for its own torch; with the pinned torch they fail to
# load ("torchvision::nms does not exist") and transformers then cannot import BERT. Text only: remove them.
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchvision', 'torchaudio'], check=True)
ROOT = Path(DRIVE); ROOT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, 'PYTHONUTF8': '1', 'HF_HUB_DISABLE_XET': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
CACHE = '/content/hf-cache'

# Everything printed below, and every error with its traceback, is also appended to a
# log on Drive so the run can be inspected without copying cell output.
LOG = ROOT / 'logs' / f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{REF[:7]}.log'
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(text):
    print(text, flush=True)
    with open(LOG, 'a', encoding='utf-8') as handle:
        handle.write(text + '\\n')

def call(*args):
    command = [sys.executable, '-m', 'nasn.study_v3', *map(str, args), '--cache-directory', CACHE]
    log('$ ' + ' '.join(map(str, args)))
    process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        log(line.rstrip('\\n'))
    if process.wait():
        raise RuntimeError(f'Command failed with exit code {process.returncode}; see {LOG}')

import traceback
def record_error(shell, kind, value, tb, tb_offset=None):
    log('ERROR ' + ''.join(traceback.format_exception(kind, value, tb)))
    shell.showtraceback((kind, value, tb), tb_offset=tb_offset)
get_ipython().set_custom_exc((Exception,), record_error)
log(f'Run started; commit {REF}; Python {sys.version.split()[0]}')

def content_sha(path):
    return hashlib.sha256(gzip.open(path).read()).hexdigest()

import torch
log(f'GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB')

import shutil
def drive_free_gib():
    return shutil.disk_usage('/content/drive/MyDrive').free / 2**30
# Each checkpoint needs 0.4-1.1 GiB. Below this, runs train on the Colab disk instead (see training cell).
MIN_FREE_GIB = 3
log(f'Google Drive free space: {drive_free_gib():.1f} GiB')
'''),
        cell('code', '''# Data and the frozen perturbation variants (each must reproduce its registered hash).
subprocess.run([sys.executable, '-m', 'arsarcasm_audit', 'fetch', '--destination', 'data/raw'], cwd=REPO, env=ENV, check=True)
subprocess.run([sys.executable, '-m', 'nasn.arsas', 'fetch'], cwd=REPO, env=ENV, check=True)
VARIANTS = {}
for dataset in DATASETS:
    folder = ROOT / dataset / 'variants'
    VARIANTS[dataset] = folder / 'variants.csv.gz'
    if VARIANTS[dataset].exists() and content_sha(VARIANTS[dataset]) != VARIANTS_CONTENT_SHA256[dataset]:
        # Created by an earlier notebook version: keep it for the record, regenerate with this commit.
        folder.rename(folder.with_name(f'variants-mismatch-{time.strftime("%Y%m%dT%H%M%S")}'))
    if not VARIANTS[dataset].exists():
        call('prepare', '--dataset', dataset, '--partition', 'validation', '--output', folder)
    got = content_sha(VARIANTS[dataset])
    if got != VARIANTS_CONTENT_SHA256[dataset]:
        evidence = json.loads((folder / 'evidence.json').read_text())
        print(json.dumps({k: evidence.get(k) for k in ['rows', 'variants', 'applicable', 'orthography_sha256', 'versions']}, indent=1))
        print((folder / 'applicability.csv').read_text())
        raise AssertionError(f'{dataset}: variants content {got} differs from the registered set; send the output above')
    print(dataset, 'variants verified')
'''),
        cell('code', '''# Fine-tune every encoder with every seed on each dataset (frozen MARBERT v1 recipe).
# When Drive lacks space for weights, a run trains on the Colab disk, is predicted immediately,
# and everything except model.safetensors is kept on Drive. Training is deterministic, so the
# same weights can be regenerated later and checked against checkpoint_hashes.json.
LOCAL = Path('/content/local-runs')

def set_aside(path):
    path.rename(path.with_name(path.name + f'-incomplete-{len(list(path.parent.glob(path.name + "-incomplete-*")))}'))

def predicted(dataset, model, seed):
    return (ROOT / dataset / 'predictions' / f'{model}-s{seed}' / 'evidence.json').exists()

def predict(dataset, model, seed, checkpoint):
    out = ROOT / dataset / 'predictions' / f'{model}-s{seed}'
    if out.exists():
        set_aside(out)  # interrupted earlier, e.g. the session disconnected
    call('infer', '--variants', VARIANTS[dataset], '--checkpoint', checkpoint, '--model', model, '--seed', seed, '--output', out)

for dataset in DATASETS:
    for model in ENCODERS:
        for seed in SEEDS:
            run = ROOT / dataset / 'train' / f'{model}-s{seed}'
            status = run / 'run_status.json'
            completed = status.exists() and json.loads(status.read_text()).get('status') == 'completed'
            # A full Drive accepts small files but silently drops the weights: require them,
            # unless the run was deliberately kept without weights after being predicted.
            if completed and ((run / 'best_checkpoint' / 'model.safetensors').exists() or
                              ((run / 'weights_not_kept.json').exists() and predicted(dataset, model, seed))):
                continue
            if run.exists():
                set_aside(run)
            if drive_free_gib() >= MIN_FREE_GIB:
                call('train', '--dataset', dataset, '--model', model, '--seed', seed, '--output', run)
                continue
            log(f'Drive has {drive_free_gib():.1f} GiB free: training {dataset}/{model}-s{seed} on the Colab disk')
            local = LOCAL / dataset / f'{model}-s{seed}'
            shutil.rmtree(local, ignore_errors=True)
            call('train', '--dataset', dataset, '--model', model, '--seed', seed, '--output', local)
            if not predicted(dataset, model, seed):
                predict(dataset, model, seed, local / 'best_checkpoint')
            shutil.copytree(local, run, ignore=shutil.ignore_patterns('model.safetensors'))
            (run / 'weights_not_kept.json').write_text(json.dumps({
                'reason': 'Google Drive full; weights used for predictions on the Colab disk, then discarded',
                'regenerate': f'train --dataset {dataset} --model {model} --seed {seed} at commit {REF} on a T4',
                'verify': 'compare best_checkpoint/model.safetensors with checkpoint_hashes.json'}, indent=1))
            shutil.rmtree(local)
'''),
        cell('code', '''# Clean + perturbed predictions under every defense (runs stored with weights on Drive).
for dataset in DATASETS:
    for model in ENCODERS:
        for seed in SEEDS:
            if predicted(dataset, model, seed):
                continue
            checkpoint = ROOT / dataset / 'train' / f'{model}-s{seed}' / 'best_checkpoint'
            if not (checkpoint / 'model.safetensors').exists():
                log(f'WARNING: no weights for {dataset}/{model}-s{seed}; re-run the training cell first')
                continue
            predict(dataset, model, seed, checkpoint)
'''),
        cell('code', '''# Zero-shot LLM (RAW and STD inputs); needs >= 24 GB GPU memory.
if torch.cuda.get_device_properties(0).total_memory >= 22 * 2**30:
    for dataset in DATASETS:
        for model in LLMS:
            out = ROOT / dataset / 'predictions' / f'{model}-zero-shot'
            if not (out / 'evidence.json').exists():
                call('llm', '--variants', VARIANTS[dataset], '--model', model, '--defenses', 'RAW', 'STD', '--output', out)
else:
    print('Skipping LLM stage: switch to an L4/A100 runtime and re-run all cells.')
'''),
        cell('code', '''# Analysis over everything available, then a ZIP of public (text-free) results.
import shutil, time
stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
public = ROOT / 'analysis' / f'{stamp}-public'
public.mkdir(parents=True)
for dataset in DATASETS:
    files = sorted(str(p) for p in (ROOT / dataset / 'predictions').glob('*/predictions.csv.gz'))
    if not files:
        continue
    for suffix, extra in [('all', []), ('cue-tweets', ['--cue-only'])]:
        out = ROOT / 'analysis' / f'{stamp}-{dataset}-{suffix}'
        call('analyze', '--variants', VARIANTS[dataset], '--predictions', *files, *extra, '--output', out)
        for name in ['robustness_cells.csv', 'contrasts.csv', 'natural_noise_cells.csv', 'evidence.json']:
            shutil.copy(out / name, public / f'{dataset}-{suffix}-{name}')
    for run in sorted((ROOT / dataset / 'train').glob('*-s*')):
        for name in ['validation_metrics.json', 'run_status.json', 'checkpoint_hashes.json', 'epoch_history.csv']:
            if (run / name).exists():
                shutil.copy(run / name, public / f'{dataset}-{run.name}-{name}')
print(shutil.make_archive(str(public), 'zip', public))
'''),
    ]
    return {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='Commit or tag to pin')
    parser.add_argument('--output', default='notebooks/study_v3_colab.ipynb')
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(build(args.reference), indent=1, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
