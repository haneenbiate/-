"""Generate the auditable notebook with helpers embedded outside the pinned checkout."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helper = (ROOT / 'tools/colab_runtime.py').read_text()
lock = (ROOT / 'configs/marbert_snapshot_lock.json').read_text()
cells = []
def markdown(text): cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
def code(text): cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'execution_count':None,'outputs':[]})

markdown('''# NASN Task 03B — MARBERT Baseline v1.0 on Colab GPU

This notebook is **unexecuted preparation**, not training evidence. Choose **Runtime → Change runtime type → GPU** first. It clones baseline commit `c8dc698780cb06c8838c557f159bdaed7507aeda`, preserves its CLI and protocol, and writes experiments directly to your mounted Google Drive. No official test evaluation, split reconstruction or random-encoder fallback.

Run cells in order. Stop on any failure. A completed run requires `run_status.status == completed`, checkpoint hashes, and verified validation reload. An interrupted run is incomplete; the baseline does not implement resumable optimizer state. Start a fresh seeded run after changing only microbatch/accumulation for an OOM.
''')
code('''from pathlib import Path
import datetime, hashlib, importlib.util, json, os, shutil, subprocess, sys, uuid
REFERENCE = "c8dc698780cb06c8838c557f159bdaed7507aeda"
REPOSITORY = "https://github.com/haneenbiate/-.git"
REPO = Path("/content/nasn-task03b-baseline")
CACHE = Path("/content/nasn-task03b-cache")
RUN_ID = "marbert-v1-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
# Optional verified local snapshot directory in Drive (never a trained classifier):
LOCAL_SNAPSHOT = None  # e.g. "/content/drive/MyDrive/NASN/pretrained/MARBERT-88e1fa"
MICRO_BATCH_SIZE = None  # auto conservative VRAM heuristic, or explicitly 4, 2, 1
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
def call(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)
if not REPO.exists():
    call(["git","clone","--no-checkout",REPOSITORY,REPO],env={**os.environ,"GIT_TERMINAL_PROMPT":"0"})
    call(["git","-C",REPO,"checkout","--detach",REFERENCE])
assert subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip() == REFERENCE
assert not subprocess.check_output(["git","-C",str(REPO),"status","--porcelain","--untracked-files=no"],text=True).strip(), "Pinned tracked files were modified"
print("Pinned baseline checkout:", REFERENCE)
''')
markdown('''## Durable output gate

Google Drive authorization happens interactively in Colab. Ensure at least 10 GB of Drive quota and local disk space. Each run gets a new directory; previous Task 03 evidence is untouched. Checkpoints are written to Drive during training, not copied there only at the end. Drive can still disconnect or exhaust quota; do not claim durability until final readback/hash verification passes.
''')
code('''from google.colab import drive
drive.mount("/content/drive")
assert os.path.ismount("/content/drive"), "Drive must be mounted; do not use an ordinary temporary directory"
RUN_ROOT = Path("/content/drive/MyDrive/NASN/task03b") / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=False)
probe = RUN_ROOT / ".storage_probe"
probe.write_bytes(b"NASN durable storage write/read probe")
assert probe.read_bytes() == b"NASN durable storage write/read probe"
probe.unlink()
assert shutil.disk_usage("/content").free >= 10 * 1024**3, "Insufficient local disk"
EXPERIMENT = RUN_ROOT / "experiment"  # leave absent/empty for original CLI
print("Durable run directory:", RUN_ROOT)
''')
markdown('''## Exact interpreter and isolated dependencies

Task 02 enforces Python **3.12.14**, not merely 3.12. Use a managed, isolated interpreter and venv so Colab's global libraries remain untouched. All TLS/artifact verification stays enabled. The CUDA 12.6 PyTorch 2.8.0 build must pass an actual CUDA kernel test on the assigned driver; wheel availability alone is not GPU compatibility evidence.
''')
code('''BOOTSTRAP = Path("/content/nasn-bootstrap")
call([sys.executable,"-m","pip","install","--no-cache-dir","--target",BOOTSTRAP,"uv==0.13.0"])
UV = BOOTSTRAP / "bin/uv"
BOOT_ENV = {**os.environ,"UV_PYTHON_INSTALL_DIR":"/content/nasn-python","UV_CACHE_DIR":"/content/nasn-uv-cache","UV_PYTHON_BIN_DIR":"/content/nasn-python-bin"}
call([UV,"python","install","--system-certs","3.12.14"],env=BOOT_ENV)
MANAGED_PYTHON = subprocess.check_output([str(UV),"python","find","--managed-python","3.12.14"],env=BOOT_ENV,text=True).strip()
VENV = Path("/content/nasn-marbert-venv")
assert not VENV.exists(), "Use a fresh runtime rather than overwriting an existing venv"
call([MANAGED_PYTHON,"-m","venv",VENV])
PYTHON = VENV / "bin/python"
assert subprocess.check_output([str(PYTHON),"-c","import platform; print(platform.python_version())"],text=True).strip() == "3.12.14"
call([PYTHON,"-m","pip","install","--no-cache-dir","torch==2.8.0+cu126","--index-url",CUDA_INDEX])
call([PYTHON,"-m","pip","install","--no-cache-dir","-r",REPO / "requirements-marbert.txt"])
check = call([PYTHON,"-m","pip","check"],capture_output=True,text=True)
(RUN_ROOT / "dependency_check.txt").write_text(check.stdout + check.stderr)
(RUN_ROOT / "dependency_freeze.txt").write_text(subprocess.check_output([str(PYTHON),"-m","pip","freeze"],text=True))
ENV = {**os.environ,"PYTHONPATH":str(REPO),"HF_HOME":str(CACHE),"HF_HUB_DISABLE_XET":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8"}
''')
markdown('''## Embedded integrity helper and official snapshot lock

These preparation/storage helpers are materialized **outside** the baseline checkout. They contain no training algorithm. The weight hash comes from official Hub LFS metadata at the exact model revision, not a checksum invented from a local file. The four tokenizer/config files also have verified Git blob identities. The embedded helper bytes and lock are saved with the run for inspection.
''')
code(f'''HELPER_SOURCE = {helper!r}
HELPER_SHA256 = "{hashlib.sha256(helper.encode()).hexdigest()}"
LOCK = json.loads({lock!r})
HELPER = Path("/content/nasn_external_runtime.py")
HELPER.write_text(HELPER_SOURCE)
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA256
LOCK_PATH = RUN_ROOT / "model_snapshot_lock.json"
LOCK_PATH.write_text(json.dumps(LOCK,indent=2)+"\\n")
(RUN_ROOT / "external_runtime_helper.py").write_text(HELPER_SOURCE)
spec = importlib.util.spec_from_file_location("nasn_external_runtime", HELPER)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
''')
code('''# Verify CUDA build, GPU model/driver, total/free VRAM and a small actual CUDA matmul BEFORE model loading.
call([PYTHON,HELPER,"gpu","--report",RUN_ROOT / "gpu_runtime.json"],env=ENV,cwd=REPO)
GPU = json.loads((RUN_ROOT / "gpu_runtime.json").read_text())
print(json.dumps(GPU,indent=2))
ORIGINAL = json.loads((REPO / "configs/marbert_v1.json").read_text())
CONFIG, DEVIATIONS = runtime.make_config(ORIGINAL,GPU,CACHE,MICRO_BATCH_SIZE)
CONFIG_PATH = RUN_ROOT / "colab_training_config.json"
CONFIG_PATH.write_text(json.dumps(CONFIG,indent=2)+"\\n")
(RUN_ROOT / "runtime_provenance.json").write_text(json.dumps({
    "reference_commit":REFERENCE,"helper_sha256":HELPER_SHA256,"cuda_index":CUDA_INDEX,
    "deviations":DEVIATIONS,"reason":"CUDA device/cache paths; microbatch by available VRAM, effective batch stays 16",
    "colab_kernel_python":sys.version,"managed_python":MANAGED_PYTHON,
    "run_id":RUN_ID,"durable_storage":str(RUN_ROOT),"training_executed":False},indent=2)+"\\n")
print("Documented runtime changes:", json.dumps(DEVIATIONS,indent=2))
''')
markdown('''Free VRAM heuristic: microbatch 4 at ≥12 GiB, 2 at ≥8 GiB, otherwise 1 at ≥6 GiB. This is a conservative setup heuristic, not a capacity guarantee. On actual OOM, preserve the failed run, select a smaller microbatch and start a **new** run. Do not shorten epochs, change precision, truncate data, or tune using test results.
''')
code('''# Fetch original pinned CSV bytes only. No split-generation command appears in this notebook.
call([PYTHON,"-m","arsarcasm_audit","fetch","--destination","data/raw"],cwd=REPO,env=ENV)
# Existing CLI checks all source/manifest hashes, 10037/2509/2/3000 counts and zero overlap.
call([PYTHON,"-m","nasn.marbert","preflight","--config",CONFIG_PATH,"--output",RUN_ROOT / "preflight"],cwd=REPO,env=ENV)
# Original reference-commit software tests; tiny random models are tests, never baseline results.
call([PYTHON,"-m","pytest","-q"],cwd=REPO,env=ENV)
''')
markdown('''## Exact pretrained snapshot (online or verified local)

If online download fails, the helper saves the host/error (signed query strings omitted) and stops. Do not repeat a failed command unchanged. For offline recovery, obtain the exact official snapshot on an accessible machine, verify it against this notebook's trusted lock, and upload it to Drive with `snapshot_revision.json` containing the exact repo/revision. Then set `LOCAL_SNAPSHOT`, rerun this cell and subsequent cells. A trained checkpoint, LFS pointer, mismatching revision, missing tokenizer, or altered weights fails verification.
''')
code('''args = [PYTHON,HELPER,"snapshot","--cache",CACHE,"--lock",LOCK_PATH,"--report",RUN_ROOT / "snapshot_verification.json"]
if LOCAL_SNAPSHOT is not None:
    args += ["--local",LOCAL_SNAPSHOT]
call(args,cwd=REPO,env=ENV)
# Make the ORIGINAL model-name/revision CLI resolve exclusively to the verified cache.
# No changing config.model_name to a random/local model and no network fetch during training.
OFFLINE_ENV = {**ENV,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
call([PYTHON,HELPER,"encoder","--cache",CACHE,"--report",RUN_ROOT / "encoder_verification.json"],cwd=REPO,env=OFFLINE_ENV)
''')
markdown('''## Run the unchanged documented baseline CLI

This is the only training invocation. Output/checkpoint directories are on Drive. The existing implementation saves improved checkpoints each epoch and reloads the best model **and tokenizer** for complete validation inference. It does not evaluate official test. A browser/runtime interruption is not successful completion and is not resumable by the original runner.
''')
code('''assert not EXPERIMENT.exists(), "Never overwrite an experiment"
assert subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip() == REFERENCE
assert not subprocess.check_output(["git","-C",str(REPO),"status","--porcelain","--untracked-files=no"],text=True).strip()
command = [str(PYTHON),"-m","nasn.marbert","train","--config",str(CONFIG_PATH),"--output",str(EXPERIMENT)]
(RUN_ROOT / "execution_command.json").write_text(json.dumps(command,indent=2)+"\\n")
with (RUN_ROOT / "execution.log").open("w") as log:
    process = subprocess.Popen(command,cwd=REPO,env=OFFLINE_ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        clean = runtime.safe_error(line)
        print(clean,end="")
        log.write(clean); log.flush()
    code = process.wait()
if code:
    (RUN_ROOT / "external_blocker.json").write_text(json.dumps({"status":"failed_or_blocked","exit_code":code,
        "run_status_file":str(EXPERIMENT / "run_status.json"),"training_completed":False},indent=2)+"\\n")
    raise RuntimeError(f"Original CLI failed (exit {code}); inspect durable run_status.json and execution.log")
''')
code('''# Require complete results and read back every checkpoint file from DURABLE storage.
call([PYTHON,HELPER,"verify-run","--run",EXPERIMENT],cwd=REPO,env=OFFLINE_ENV)
STATUS = json.loads((EXPERIMENT / "run_status.json").read_text())
assert STATUS["status"] == "completed" and STATUS["checkpoint_reload_verified"] and not STATUS["test_evaluated"]
provenance = json.loads((RUN_ROOT / "runtime_provenance.json").read_text())
provenance.update(training_executed=True,training_completed=True,run_status=STATUS)
(RUN_ROOT / "runtime_provenance.json").write_text(json.dumps(provenance,indent=2)+"\\n")
print("Complete durable run verified:", EXPERIMENT)
print(json.dumps(json.loads((EXPERIMENT / "validation_metrics.json").read_text()),indent=2))
''')
markdown('''## Return lightweight reports to GitHub

The next cell exports an allowlist of CSV/JSON/Markdown experiment reports. It excludes `best_checkpoint/`, pretrained snapshots, raw CSVs, arbitrary logs and notebook outputs. Download the ZIP and review it before committing on a new report branch. Keep the full checkpoint and provenance in Drive; never put a token in this notebook. Exact Git commands are in `docs/task03b_colab.md` in the notebook's repository version.
''')
code('''LIGHTWEIGHT = RUN_ROOT / "lightweight_reports"
call([PYTHON,HELPER,"export","--run",EXPERIMENT,"--target",LIGHTWEIGHT],cwd=REPO,env=OFFLINE_ENV)
# Include explicit helper/GPU/runtime provenance with the report bundle, not raw logs or files.
for name in ["gpu_runtime.json","runtime_provenance.json","snapshot_verification.json","encoder_verification.json","dependency_freeze.txt","dependency_check.txt"]:
    shutil.copyfile(RUN_ROOT / name,LIGHTWEIGHT / name)
(LIGHTWEIGHT / "report_checksums.json").write_text(json.dumps({
    file.name:runtime.sha256(file) for file in sorted(LIGHTWEIGHT.iterdir()) if file.name != "report_checksums.json"},indent=2)+"\\n")
ZIP = shutil.make_archive(str(RUN_ROOT / (RUN_ID + "-reports")),"zip",LIGHTWEIGHT)
from google.colab import files
files.download(ZIP)
print("Trained weights stay in:", EXPERIMENT / "best_checkpoint")
''')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'marbert_baseline_colab.ipynb','provenance':[]},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},'cells':cells}
for i, cell in enumerate(cells): cell['id']=f'task03b-{i:02d}'
(ROOT / 'notebooks/marbert_baseline_colab.ipynb').write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
