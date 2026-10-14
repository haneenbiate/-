"""Build the Study v3 Colab notebook (training, inference, LLM, analysis) pinned to one commit."""
import argparse
import json
from pathlib import Path

VARIANTS_SHA256 = 'd6b467f0bd615c613e90e4c8e60ece49ea248a1ffcc6a846363d34b57874bb2d'


def cell(kind, source):
    base = {'cell_type': kind, 'metadata': {}, 'source': source.strip('\n').splitlines(keepends=True)}
    return base if kind == 'markdown' else {**base, 'execution_count': None, 'outputs': []}


def build(reference):
    cells = [
        cell('markdown', '''# Study v3 — multi-model orthographic-noise robustness audit

Run cells in order on a GPU runtime. Every stage writes to Google Drive and **skips
work that is already complete**, so a disconnected session can be resumed by
re-running all cells. Encoders fit on a T4 (about 20–30 min per run); the LLM stage
needs at least 24 GB (L4 or A100) and is skipped otherwise. The official test set
stays locked; only the validation partition is used here.
'''),
        cell('code', f'''REF = '{reference}'
DRIVE = '/content/drive/MyDrive/NASN/study-v3'
ENCODERS = ['MARBERT', 'MARBERTv2', 'AraBERTv02-twitter', 'CAMeLBERT-DA', 'XLM-R-base']
SEEDS = [42, 43, 44]
LLMS = ['Qwen2.5-7B-Instruct']
VARIANTS_SHA256 = '{VARIANTS_SHA256}'
'''),
        cell('code', '''import hashlib, json, os, subprocess, sys
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
REPO = Path('/content/nasn-study-v3')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/haneenbiate/-.git', str(REPO)], check=True)
subprocess.run(['git', 'checkout', REF], cwd=REPO, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-marbert.txt', 'accelerate==1.10.1'], cwd=REPO, check=True)
ROOT = Path(DRIVE); ROOT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, 'PYTHONUTF8': '1', 'HF_HUB_DISABLE_XET': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
CACHE = '/content/hf-cache'

def call(*args):
    print('$', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, '-m', 'nasn.study_v3', *map(str, args), '--cache-directory', CACHE], cwd=REPO, env=ENV, check=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

import torch
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30, 'GiB')
'''),
        cell('code', '''# Data and the frozen perturbation variants (must reproduce the registered hash).
subprocess.run([sys.executable, '-m', 'arsarcasm_audit', 'fetch', '--destination', 'data/raw'], cwd=REPO, env=ENV, check=True)
VARIANTS = ROOT / 'variants' / 'variants.csv.gz'
if not VARIANTS.exists():
    call('prepare', '--partition', 'validation', '--output', ROOT / 'variants')
assert sha(VARIANTS) == VARIANTS_SHA256, 'Variants differ from the registered study-v3 set'
'''),
        cell('code', '''# Fine-tune every encoder with every seed (frozen MARBERT v1 recipe).
for model in ENCODERS:
    for seed in SEEDS:
        run = ROOT / 'train' / f'{model}-s{seed}'
        status = run / 'run_status.json'
        if status.exists() and json.loads(status.read_text()).get('status') == 'completed':
            continue
        if run.exists():
            run.rename(run.with_name(run.name + f'-incomplete-{len(list(run.parent.glob(run.name + "-incomplete-*")))}'))
        call('train', '--model', model, '--seed', seed, '--output', run)
'''),
        cell('code', '''# Clean + perturbed predictions under every defense.
for model in ENCODERS:
    for seed in SEEDS:
        out = ROOT / 'predictions' / f'{model}-s{seed}'
        if (out / 'evidence.json').exists():
            continue
        call('infer', '--variants', VARIANTS, '--checkpoint', ROOT / 'train' / f'{model}-s{seed}' / 'best_checkpoint',
             '--model', model, '--seed', seed, '--output', out)
'''),
        cell('code', '''# Zero-shot LLM (RAW inputs only); needs >= 24 GB GPU memory.
if torch.cuda.get_device_properties(0).total_memory >= 22 * 2**30:
    for model in LLMS:
        out = ROOT / 'predictions' / f'{model}-zero-shot'
        if not (out / 'evidence.json').exists():
            call('llm', '--variants', VARIANTS, '--model', model, '--output', out)
else:
    print('Skipping LLM stage: switch to an L4/A100 runtime and re-run all cells.')
'''),
        cell('code', '''# Analysis over everything available, then a ZIP of public (text-free) results.
import shutil, time
files = sorted(str(p) for p in (ROOT / 'predictions').glob('*/predictions.csv.gz'))
stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
for suffix, extra in [('all', []), ('cue-tweets', ['--cue-only'])]:
    call('analyze', '--variants', VARIANTS, '--predictions', *files, *extra, '--output', ROOT / 'analysis' / f'{stamp}-{suffix}')
public = ROOT / 'analysis' / f'{stamp}-public'
public.mkdir()
for suffix in ['all', 'cue-tweets']:
    for name in ['robustness_cells.csv', 'contrasts.csv', 'evidence.json']:
        shutil.copy(ROOT / 'analysis' / f'{stamp}-{suffix}' / name, public / f'{suffix}-{name}')
for run in sorted((ROOT / 'train').glob('*-s*')):
    for name in ['validation_metrics.json', 'run_status.json', 'checkpoint_hashes.json']:
        if (run / name).exists():
            shutil.copy(run / name, public / f'{run.name}-{name}')
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
    Path(args.output).write_text(json.dumps(build(args.reference), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
