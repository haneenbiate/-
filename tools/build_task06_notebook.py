"""Generate a Colab notebook pinned to an immutable Task 06 implementation commit."""
import argparse
import json
import re
from pathlib import Path


def build(reference):
    if not re.fullmatch('[0-9a-f]{40}',reference):raise ValueError('Full immutable Task 06 code commit required')
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'execution_count':None,'outputs':[]})
    md('''# Task 06B — Fixed NASN prototype, validation development only

This notebook is **unexecuted preparation**, not new performance evidence.
Use Runtime → Change runtime type → GPU. Run cells in order and stop on errors.
It reuses the selected Task 03B epoch-2 checkpoint already on Drive. It never
downloads another encoder, trains, selects a checkpoint, regenerates data splits
or opens official test text. Outputs go directly to a new durable Drive folder.

B0–B4 are five strategies, following the supplied enumeration. Their rules are
fixed before this new comparison. The development partition is the previously
used validation set; this post-hoc study cannot serve as independent final proof.
394 source perturbations have user-declared human confirmation, with unverified
reviewer independence. New normalized outputs are **not human-validated**.
''')
    code(f'''from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, uuid
CODE_REF={reference!r}
REPO=Path('/content/nasn-task06')
RUN_ID='nasn-prototype-v1-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
def call(command, **kwargs):
    result=subprocess.run([str(x) for x in command],text=True,capture_output=True,**kwargs)
    print(result.stdout, end=''); print(result.stderr, end='',file=sys.stderr)
    result.check_returncode()
    return result
if not REPO.exists():
    call(['git','clone','--no-checkout','https://github.com/haneenbiate/-.git',REPO])
    call(['git','-C',REPO,'checkout','--detach',CODE_REF])
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==CODE_REF, 'Existing checkout has another revision: choose a fresh runtime'
assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True).strip(), 'Pinned tracked files changed'
''')
    code('''from google.colab import drive
drive.mount('/content/drive')
assert os.path.ismount('/content/drive'), 'A real mounted Drive is required'
ROOT=Path('/content/drive/MyDrive/NASN/task06')/RUN_ID
ROOT.mkdir(parents=True,exist_ok=False)
probe=ROOT/'.storage_probe';probe.write_text('durable write/read probe');assert probe.read_text()=='durable write/read probe';probe.unlink()
BASELINE=REPO/'reports/task03b/marbert-v1-20261009T213143Z-be8b8811'
CHECKPOINT=Path(json.loads((BASELINE/'run_status.json').read_text())['best_checkpoint'])
# Change this path only if the same selected checkpoint was moved on Drive.
# Every model/tokenizer/config byte must match its original registry.
assert CHECKPOINT.is_dir(), f'Selected trained checkpoint missing: {CHECKPOINT}'
assert CHECKPOINT.resolve().is_relative_to(Path('/content/drive').resolve()), 'Use verified durable checkpoint storage'
print('Durable experiment:',ROOT);print('Selected checkpoint:',CHECKPOINT)
''')
    md('''## Exact managed environment

The Colab kernel can use another Python, but all research commands run in pinned
Python 3.12.14. CUDA PyTorch 2.8.0+cu126 must pass a real kernel check. Matplotlib
uses Agg explicitly to avoid inherited Colab inline-backend errors. Global Colab
packages remain separate. No new model download occurs.
''')
    code('''BOOTSTRAP=Path('/content/nasn-task06-bootstrap')
call([sys.executable,'-m','pip','install','--target',BOOTSTRAP,'uv==0.13.0'])
UV=BOOTSTRAP/'bin/uv'
BOOT_ENV={**os.environ,'UV_PYTHON_INSTALL_DIR':'/content/nasn-task06-python','UV_CACHE_DIR':'/content/nasn-task06-uv-cache','UV_PYTHON_BIN_DIR':'/content/nasn-task06-python-bin'}
call([UV,'python','install','--system-certs','3.12.14'],env=BOOT_ENV)
MANAGED=call([UV,'python','find','--managed-python','3.12.14'],env=BOOT_ENV).stdout.strip()
VENV=Path('/content/nasn-task06-venv');assert not VENV.exists(), 'Use a fresh runtime for environment setup'
call([MANAGED,'-m','venv',VENV]);PY=VENV/'bin/python'
call([PY,'-m','pip','install','torch==2.8.0+cu126','--index-url','https://download.pytorch.org/whl/cu126'])
call([PY,'-m','pip','install','-r',REPO/'requirements-task04.txt'])
ENV={**os.environ,'PYTHONPATH':str(REPO),'MPLBACKEND':'Agg','MPLCONFIGDIR':'/content/nasn-task06-matplotlib','XDG_CACHE_HOME':'/content/nasn-task06-cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
check=call([PY,'-m','pip','check'],cwd=REPO,env=ENV)
(ROOT/'dependency_check.txt').write_text(check.stdout+check.stderr)
(ROOT/'dependency_freeze.txt').write_text(call([PY,'-m','pip','freeze'],cwd=REPO,env=ENV).stdout)
''')
    code('''GPU_SCRIPT="""import json,platform,torch
assert platform.python_version()=='3.12.14'
assert torch.cuda.is_available() and torch.version.cuda, 'CUDA GPU unavailable'
free,total=torch.cuda.mem_get_info()
assert free>=2*1024**3, 'Less than 2 GiB free VRAM before model loading'
x=torch.ones((16,16),device='cuda');assert (x@x)[0,0].item()==16;torch.cuda.synchronize()
print(json.dumps({'gpu':torch.cuda.get_device_name(0),'free_vram_bytes':free,'total_vram_bytes':total,'cuda_build':torch.version.cuda,'torch':torch.__version__,'kernel_check_passed':True}))
"""
GPU=json.loads(call([PY,'-c',GPU_SCRIPT],cwd=REPO,env=ENV).stdout)
(ROOT/'gpu_runtime.json').write_text(json.dumps(GPU,indent=2)+'\\n')
# Inference only: choose batch from free memory, not from performance metrics.
BATCH_SIZE=8 if GPU['free_vram_bytes']>=6*1024**3 else 4 if GPU['free_vram_bytes']>=3*1024**3 else 1
(ROOT/'runtime_choices.json').write_text(json.dumps({'code_ref':CODE_REF,'seed':42,'max_length':128,'inference_batch_size':BATCH_SIZE,'precision':'float32','training':False,'selection':'fixed methods, no parameter search'},indent=2)+'\\n')
call([PY,'-m','pytest','-q'],cwd=REPO,env=ENV)
''')
    md('''## Pinned training text and original Task 04 manifest

Only the official **training CSV** is restored. The existing Task 02 validation
manifest chooses its original rows; no split-generation command is used. Task 04
text restoration is explicit and verifies every existing registered ID, source/
target text hash and metadata field. It does not generate human decisions.
If the host or checksum fails, stop and retain the printed error. You can supply
the original pinned training file manually; never substitute another revision.
''')
    code('''FETCH_SCRIPT="""from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit
from arsarcasm_audit.audit import COMMIT,HASHES,digest,sha256
p=Path('data/raw/training_data.csv');p.parent.mkdir(parents=True,exist_ok=True)
url=f'https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/{COMMIT}/ArSarcasm-v2/training_data.csv'
if not p.exists():
 try:
  with urlopen(url,timeout=60) as response: content=response.read()
 except Exception as exc:
  raise RuntimeError(f'Dataset host {urlsplit(url).hostname}: {type(exc).__name__}: {exc}') from exc
 assert digest(content)==HASHES['training_data.csv'], 'Pinned training download checksum differs'
 p.write_bytes(content)
assert sha256(p)==HASHES['training_data.csv'], 'Existing pinned training checksum differs'
print('Verified official training bytes only; test CSV not downloaded or opened')
"""
call([PY,'-c',FETCH_SCRIPT],cwd=REPO,env=ENV)
call([PY,'-m','nasn.task06','restore-task04','--output',ROOT/'task04-restored'],cwd=REPO,env=ENV)
# Extract ONLY exact selected tokenizer files to a separate verified directory.
TOKENIZER=ROOT/'selected-tokenizer';TOKENIZER.mkdir()
TOKENIZER_SCRIPT="""import shutil,sys,json
from pathlib import Path
from nasn.verified_tokenizer import FILES,verify
source,target,registry_path=map(Path,sys.argv[1:])
for name in FILES:shutil.copyfile(source/name,target/name)
verify(target,json.loads(registry_path.read_text()))
print('All selected tokenizer hashes verified')
"""
call([PY,'-c',TOKENIZER_SCRIPT,CHECKPOINT,TOKENIZER,BASELINE/'checkpoint_hashes.json'],cwd=REPO,env=ENV)
call([PY,'-m','nasn.task06','prepare','--preparation',ROOT/'task04-restored','--tokenizer-directory',TOKENIZER,'--output',ROOT/'preparation'],cwd=REPO,env=ENV)
''')
    md('''## Validation development comparison — no training

The CLI verifies **every selected checkpoint file**, loads the classifier with
no missing/mismatched/unexpected weights, and checks reloaded B0 clean predictions
against the completed Task 03B baseline before any metric claim. It compares
matched clean/perturbed inputs for B0–B4. Output metrics separate the 394 reviewed
source variants from 9,557 unreviewed applicable variants. Ten reviewed exclusions
stay outside robustness metrics. Neither included cohort certifies
the meaning of the **new normalized text**. Paired seed-42 intervals use 2,000
original-ID resamples within each rule/intensity; minimum n=30.

On GPU OOM, preserve the failed directory, reduce only inference `BATCH_SIZE` and
use a new output path. No checkpoint selection or parameter search is allowed.
Small hardware float differences must remain within the declared clean baseline
probability tolerance 1e-5 and exactly preserve its predicted labels; otherwise
stop and investigate. Do not silently change the tolerance.
''')
    code('''EVALUATION=ROOT/'evaluation'
call([PY,'-m','nasn.task06','run','--prepared',ROOT/'preparation','--checkpoint',CHECKPOINT,'--batch-size',BATCH_SIZE,'--bootstrap','2000','--output',EVALUATION],cwd=REPO,env=ENV)
status=json.loads((EVALUATION/'run_status.json').read_text())
assert status['status']=='completed' and status['checkpoint_reload_verified'] and status['metrics_computed']
assert status['training_performed'] is False and status['test_evaluated'] is False and status['test_text_read'] is False
print(json.dumps(status,indent=2))
''')
    md('''## Durable readback and lightweight reports

The full normalized text/edit logs/review queue stay private in Drive. The report
ZIP contains metrics, numeric paired differences, text-free IDs/probabilities,
rules/hashes, runtime and dependency records. Do not commit model weights, raw
tweets, token IDs or review identities/rationales. Import and verify the results
before describing success. New normalization review is still needed for semantic
claims even after inference completes.
''')
    code('''VERIFY_SCRIPT="""import sys
from nasn.task05 import registry
for path in sys.argv[1:]:registry(path)
print('Durable Drive artifact registries verified')
"""
call([PY,'-c',VERIFY_SCRIPT,ROOT/'preparation',EVALUATION],cwd=REPO,env=ENV)
import zipfile
REPORT_ZIP=ROOT/(RUN_ID+'-reports.zip')
files=['evaluation/comparison_metrics.csv','evaluation/paired_strategy_differences.csv','evaluation/development_predictions.csv.gz','evaluation/run_status.json','evaluation/loading_info.json','evaluation/artifact_hashes.json','preparation/evidence.json','preparation/character_token_audit.csv','preparation/normalization_manifest.csv.gz','preparation/normalization_rules.json','preparation/B3_B4_guard_effect.csv','gpu_runtime.json','runtime_choices.json','dependency_check.txt','dependency_freeze.txt']
checks={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
with zipfile.ZipFile(REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
    for name in files:bundle.write(ROOT/name,arcname=name)
    bundle.writestr('report_checksums.json',json.dumps(checks,indent=2)+'\\n')
with zipfile.ZipFile(REPORT_ZIP) as bundle:
    assert set(bundle.namelist())==set(checks)|{'report_checksums.json'}
    for name,digest in checks.items():assert hashlib.sha256(bundle.read(name)).hexdigest()==digest
print('Durable report ZIP:',REPORT_ZIP)
from google.colab import files
files.download(str(REPORT_ZIP))
''')
    for i,c in enumerate(cells):c['id']=f'task06-{i:02d}'
    return {'cells':cells,'metadata':{'colab':{'name':'task06_nasn_colab.ipynb'},
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code-ref',required=True)
    p.add_argument('--output',default='notebooks/task06_nasn_colab.ipynb');args=p.parse_args()
    Path(args.output).write_text(json.dumps(build(args.code_ref),indent=1,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
