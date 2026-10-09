"""Build an unexecuted Colab inference notebook pinned to a Task 04 code commit."""
import argparse
import json
import re
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--code-ref',required=True)
args=parser.parse_args()
if not re.fullmatch('[0-9a-f]{40}',args.code_ref):parser.error('Full immutable Task 04 commit required')
cells=[]
def md(value):cells.append({'cell_type':'markdown','metadata':{},'source':value.splitlines(keepends=True)})
def code(value):cells.append({'cell_type':'code','metadata':{},'source':value.splitlines(keepends=True),'outputs':[],'execution_count':None})
md('''# NASN Task 04 — existing-checkpoint orthographic robustness

**Unexecuted notebook: no perturbed metrics are claimed.** Select a GPU runtime and mount the same Google Drive used for the completed Task 03B run. This notebook invokes the checked-in Task 04 CLI and never trains a model or accesses official test text. It requires the exact selected trained checkpoint, not pretrained MARBERT weights. All initial semantic candidates are provisional.
''')
code(f'''from pathlib import Path
import datetime, json, os, shutil, subprocess, sys, uuid, hashlib, urllib.request
CODE_REF = {args.code_ref!r}
REPO = Path('/content/nasn-task04')
BASELINE_RUN_ID = 'marbert-v1-20261009T213143Z-be8b8811'
RUN_ID = 'orthography-v1-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
REVIEWS = None  # optional completed full-text human review CSV in Drive; never auto-approved
INFERENCE_BATCH_SIZE = 4  # lower to 2 or 1 in a NEW run after OOM
def call(args, **kwargs):return subprocess.run([str(x) for x in args],check=True,**kwargs)
assert not REPO.exists(), 'Use a fresh runtime instead of overwriting a checkout'
call(['git','clone','--no-checkout','https://github.com/haneenbiate/-.git',REPO],env={{**os.environ,'GIT_TERMINAL_PROMPT':'0'}})
call(['git','-C',REPO,'checkout','--detach',CODE_REF])
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==CODE_REF
''')
code('''from google.colab import drive
drive.mount('/content/drive')
assert os.path.ismount('/content/drive'), 'Durable Drive mount required'
ROOT = Path('/content/drive/MyDrive/NASN/task04') / RUN_ID
ROOT.mkdir(parents=True,exist_ok=False)
probe=ROOT/'.storage_probe';probe.write_bytes(b'NASN Task04');assert probe.read_bytes()==b'NASN Task04';probe.unlink()
CHECKPOINT=Path('/content/drive/MyDrive/NASN/task03b')/BASELINE_RUN_ID/'experiment/best_checkpoint'
access={'checkpoint':str(CHECKPOINT),'directory_available':CHECKPOINT.is_dir(),'training_performed':False}
(ROOT/'checkpoint_access.json').write_text(json.dumps(access,indent=2)+'\\n')
assert CHECKPOINT.is_dir(), 'Selected Task03B checkpoint unavailable in this Drive; never substitute base/random weights'
print('Durable Task04 run:',ROOT)
''')
md('''Use the original isolated Python **3.12.14** and CUDA torch **2.8.0+cu126**. In a fresh Colab runtime, the next cell recreates that venv. If an existing venv is present it must pass the exact version checks. The CUDA/VRAM/kernel gate runs before model loading. No training algorithm is defined here.
''')
code('''VENV=Path('/content/nasn-marbert-venv');PY=VENV/'bin/python'
if not PY.exists():
    bootstrap=Path('/content/nasn-task04-bootstrap')
    call([sys.executable,'-m','pip','install','--no-cache-dir','--target',bootstrap,'uv==0.13.0'])
    uv=bootstrap/'bin/uv'
    bootenv={**os.environ,'UV_PYTHON_INSTALL_DIR':'/content/nasn-python','UV_CACHE_DIR':'/content/nasn-uv-cache','UV_PYTHON_BIN_DIR':'/content/nasn-python-bin'}
    call([uv,'python','install','--system-certs','3.12.14'],env=bootenv)
    managed=subprocess.check_output([str(uv),'python','find','--managed-python','3.12.14'],env=bootenv,text=True).strip()
    call([managed,'-m','venv',VENV])
    call([PY,'-m','pip','install','--no-cache-dir','torch==2.8.0+cu126','--index-url','https://download.pytorch.org/whl/cu126'])
assert subprocess.check_output([str(PY),'-c','import platform;print(platform.python_version())'],text=True).strip()=='3.12.14'
assert subprocess.check_output([str(PY),'-c','import torch;print(torch.__version__)'],text=True).strip()=='2.8.0+cu126'
call([PY,'-m','pip','install','--no-cache-dir','-r',REPO/'requirements-task04.txt'])
check=call([PY,'-m','pip','check'],capture_output=True,text=True)
(ROOT/'dependency_check.txt').write_text(check.stdout+check.stderr)
(ROOT/'dependency_freeze.txt').write_text(subprocess.check_output([str(PY),'-m','pip','freeze'],text=True))
ENV={**os.environ,'PYTHONPATH':str(REPO),'CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MPLCONFIGDIR':'/tmp/nasn-matplotlib','XDG_CACHE_HOME':'/tmp/nasn-cache'}
call([PY,REPO/'tools/colab_runtime.py','gpu','--report',ROOT/'gpu_runtime.json'],cwd=REPO,env=ENV)
''')
code('''# Restore ONLY pinned official training CSV; no official test CSV fetch/read.
SOURCE_REVISION='eb85513e133cfc67e30aca4968fadbe2594fcb42'
TRAIN=REPO/'data/raw/training_data.csv'
EXPECTED_SHA='1da727eb763459f436c4e61f52c6a0bc63d5300af8fcc99b5675ae4616dae04a'
if not TRAIN.exists():
    TRAIN.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/{SOURCE_REVISION}/ArSarcasm-v2/training_data.csv'
    TRAIN.write_bytes(urllib.request.urlopen(url).read())
assert hashlib.sha256(TRAIN.read_bytes()).hexdigest()==EXPECTED_SHA
BASELINE_REPORTS=REPO/'reports/task03b'/BASELINE_RUN_ID
call([PY,'-m','nasn.task04','prepare','--output',ROOT/'benchmark'],cwd=REPO,env=ENV)
call([PY,'-m','nasn.task04','analyze','--run',BASELINE_REPORTS,'--output',ROOT/'saved-prediction-analysis'],cwd=REPO,env=ENV)
''')
md('''The full-text semantic review queue is in the durable `benchmark/` folder. It is initially pending and stratifies class, dialect, sarcasm, rule/intensity and lexical negation. Review must check polarity, scope, entities and emphasis. Invalid/ambiguous examples are excluded; individual valid decisions never approve unreviewed examples. A run without `REVIEWS` measures **provisional candidate instability**, not confirmed meaning-preserving robustness.
''')
code('''INFERENCE=ROOT/'inference'
assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
command=[PY,'-m','nasn.task04','evaluate','--run',BASELINE_REPORTS,'--checkpoint',CHECKPOINT,'--output',INFERENCE,'--batch-size',str(INFERENCE_BATCH_SIZE),'--bootstrap','2000']
if REVIEWS is not None:command+=['--reviews',REVIEWS]
(ROOT/'execution_command.json').write_text(json.dumps([str(x) for x in command],indent=2)+'\\n')
with (ROOT/'execution.log').open('w') as log:
    call(command,cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
status=json.loads((INFERENCE/'run_status.json').read_text())
assert status['status']=='completed' and status['training_performed'] is False and status['test_evaluated'] is False
(ROOT/'runtime_provenance.json').write_text(json.dumps({'code_ref':CODE_REF,'run_id':RUN_ID,'checkpoint':str(CHECKPOINT),'inference_batch_size':INFERENCE_BATCH_SIZE,'baseline_run_id':BASELINE_RUN_ID,'status':status},indent=2)+'\\n')
print(json.dumps(status,indent=2))
''')
md('''Checkpoint/model input hashes, full original/perturbed texts, predictions and review decisions stay in Drive. Export only reviewed lightweight reports to GitHub. The following allowlist excludes weights and raw/full tweet copies; CSV prediction/manifest exports retain IDs, probabilities and logged spans while dropping full text. Population-with-noop and applicable-only summaries are separate; mixed population scores remain provisional. Bootstrap CIs use paired original IDs and tokenization associations are descriptive.
''')
code('''LIGHTWEIGHT=ROOT/'lightweight_reports'
call([PY,'-m','nasn.task04','publish','--preparation',ROOT/'benchmark','--analysis',ROOT/'saved-prediction-analysis','--output',LIGHTWEIGHT],cwd=REPO,env=ENV)
allowed=['run_status.json','evidence.json','methodology.md','robustness_metrics.csv','robustness_table.tex','tokenization_associations.csv','perturbed_predictions.csv','perturbation_manifest.csv','clean_predictions.csv']
csv_cleaner="import pandas as pd,sys; f=pd.read_csv(sys.argv[1],dtype={'sarcasm':str,'row_id':str},keep_default_na=False); f.drop(columns=['original_text','perturbed_text'],errors='ignore').to_csv(sys.argv[2],index=False)"
inference_reports=LIGHTWEIGHT/'inference';inference_reports.mkdir()
for name in allowed:
    source=INFERENCE/name
    if not source.is_file():continue
    if source.suffix=='.csv':call([PY,'-c',csv_cleaner,source,inference_reports/name],env=ENV)
    else:shutil.copyfile(source,inference_reports/name)
for semantic in ['provisional','confirmed','provisional_population']:
    for extension in ['pdf','svg']:
        name=f'robustness_{semantic}.{extension}'
        if (INFERENCE/name).is_file():shutil.copyfile(INFERENCE/name,inference_reports/name)
for name in ['gpu_runtime.json','dependency_check.txt','dependency_freeze.txt','runtime_provenance.json','checkpoint_access.json','execution_command.json']:
    shutil.copyfile(ROOT/name,LIGHTWEIGHT/name)
registry={str(p.relative_to(LIGHTWEIGHT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in LIGHTWEIGHT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
(LIGHTWEIGHT/'artifact_hashes.json').write_text(json.dumps(registry,indent=2)+'\\n')
archive=shutil.make_archive(str(ROOT/(RUN_ID+'-reports')),'zip',LIGHTWEIGHT)
from google.colab import files
files.download(archive)
print('Full evidence remains in Drive:',ROOT)
''')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'accelerator':'GPU'},'cells':cells}
for i,cell in enumerate(cells):cell['id']=f'task04-{i:02d}'
root=Path(__file__).resolve().parents[1]
(root/'notebooks/task04_robustness_colab.ipynb').write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
