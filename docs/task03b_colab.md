# Task 03B — Recover the exact MARBERT baseline on Colab GPU

Open `notebooks/marbert_baseline_colab.ipynb` in Google Colab, select a GPU runtime
(T4/L4/A100 or another compatible NVIDIA GPU), then run cells in order. The
notebook is delivered without executed cells or invented outputs. A person with
access to a Colab GPU allocation and Google Drive must execute it; this API session
has neither an attached GPU nor a connected Colab runtime.

## Immutable references and protocol

- Training repository: `haneenbiate/-`, exact commit
  `c8dc698780cb06c8838c557f159bdaed7507aeda` (detached checkout).
- MARBERT: `UBC-NLP/MARBERT`, exact revision
  `88e1fa192dd723cf0b3563500aec46209762eb22`.
- Official data revision: `eb85513e133cfc67e30aca4968fadbe2594fcb42`.
- Task 02 manifests remain byte-for-byte unchanged; the original CLI verifies
  their registered checksums, source hashes, row labels, quarantine completeness
  and zero exact/canonical overlap. Expected train/validation/quarantine/test
  counts are **10037/2509/2/3000**. Missing/changed files fail; no split creation.
- Seed 42, length 128, effective batch 16, AdamW 2e-5, weight decay .01, float32,
  maximum five epochs, patience two, highest validation Macro-F1, tie to earlier
  checkpoint. No test evaluation or data-quality-based row removal.

The notebook's own integrity/storage helper is embedded and written **outside**
the cloned checkout. It never patches `nasn` or replaces its training loop. The
training invocation is the existing documented `python -m nasn.marbert train`.
The notebook also checks that the pinned checkout has no tracked modifications
immediately before training. Baseline tests run from that reference, not from a
moving `main` branch.

## Dependency and GPU compatibility

Use **Python 3.12.14**, required by the original preflight. The notebook installs
`uv==0.13.0`, obtains that exact managed interpreter with system-certificate TLS
verification, and creates an isolated `/content/nasn-marbert-venv`. It does not
replace Colab's kernel or globally change Colab's preinstalled dependencies.

Install **torch 2.8.0+cu126**, then the original `requirements-marbert.txt`:

```bash
/content/nasn-marbert-venv/bin/python -m pip install --no-cache-dir \
  torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
/content/nasn-marbert-venv/bin/python -m pip install --no-cache-dir \
  -r /content/nasn-task03b-baseline/requirements-marbert.txt
/content/nasn-marbert-venv/bin/python -m pip check
```

The project pin `torch==2.8.0` permits the `+cu126` local build. An official
CPython 3.12/x86_64/manylinux_2_28 CUDA wheel was found during development.
That is packaging evidence, not proof of driver or kernel compatibility. The
notebook requires `torch.version.cuda`, `torch.cuda.is_available()`, NVIDIA
GPU/driver information, a successful actual CUDA matrix multiplication, and at
least 6 GiB free VRAM **before loading any model**. Failure stops with a saved
GPU blocker. Actual total/free VRAM, GPU type, driver, capabilities, Python build,
CUDA build and package freeze are recorded in Drive. The assigned GPU and driver
must support the wheel; do not turn a failed gate into CPU fallback.

Local development verified the managed Python 3.12.14 bootstrap with uv 0.13.0,
notebook schema/code compilation and integrity/storage tests. The local environment
has torch 2.8.0+cpu and no CUDA GPU; external CUDA installation/kernel compatibility
and full MARBERT encoder loading remain unexecuted. `reports/task03b/` distinguishes
these checks explicitly. The original Task 03 evidence is untouched.

## GPU memory and documented runtime changes

The only permitted generated-config changes are device (`cuda`), cache path,
microbatch size and gradient accumulation. All are recorded in
`runtime_provenance.json`; experimental values remain unchanged.

| Free VRAM at setup | Initial microbatch | Accumulation | Effective batch |
| --- | ---: | ---: | ---: |
| ≥12 GiB | 4 | 4 | 16 |
| ≥8 and <12 GiB | 2 | 8 | 16 |
| ≥6 and <8 GiB | 1 | 16 | 16 |

These thresholds are an operational heuristic, not a proven memory bound.
`MICRO_BATCH_SIZE` can explicitly select 4, 2 or 1. On OOM, preserve the failed
run and start a fresh run with a smaller microbatch. Do not change precision,
learning rate, text processing, epoch count, data or checkpoint selection. The
original runner has no optimizer-state resume support, so an interrupted run
cannot be silently resumed from `best_checkpoint` as the same experiment.
Python's managed build/compiler, CUDA build and hardware differ from the prior
CPU environment; exact versions/builds are recorded, and numerical identity
across hardware is not promised. The existing CPU feasibility guard is bypassed
by requiring CUDA, not edited.

## Pinned snapshot download and verified local fallback

`configs/marbert_snapshot_lock.json` covers precisely five official files:
config, tokenizer config, special-token map, vocabulary and PyTorch weights.
The 654,186,400-byte `pytorch_model.bin` must have SHA-256:

`801862ace5fc9e6b135b19476dbda5c7de1e8673b5b9d55db08d79e2c30b9dca`

This hash comes from the official pinned revision's LFS metadata. The other four
file hashes were measured after their official immutable download, with Git blob
identity and size checked against Hub metadata. No model-weight download or
training was claimed during notebook development. `snapshot_verification.json`
will attest actual file-byte verification only when execution succeeds.

The helper downloads only those files, checks sizes/hashes, and prepares the
standard Hub cache under the **same repo name and exact revision**. Training then
sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so the original CLI cannot
fetch or substitute another model. The encoder verification step uses the same
Transformers classification loader as the baseline and rejects any missing
encoder, mismatched or erroneous weights. Newly initialized standard classifier
and BERT pooler are allowed, as already defined by the MLM-to-classification
baseline. The CLI independently repeats its original encoder-loading check.

If download fails, inspect `snapshot_verification.json` for the host and error;
signed URL query strings are omitted. The earlier saved Task 03 blocker was
`us.aws.cdn.hf.co: 403 Forbidden`; Colab's actual response must be measured rather
than assumed. Do not disable TLS verification, alter expected hashes, or fall
back to random weights.

On a machine that can reach the official Hub, using the **Task 03B helper version**:

```bash
python tools/colab_runtime.py snapshot \
  --cache /tmp/nasn-hub-cache \
  --lock configs/marbert_snapshot_lock.json \
  --report model-download.json
```

Copy the five files from the verified snapshot path in `model-download.json`
into a curated directory, with only this additional file:

```json
{"repo_id":"UBC-NLP/MARBERT","revision":"88e1fa192dd723cf0b3563500aec46209762eb22"}
```

Name it `snapshot_revision.json`. Upload this directory to Drive and set the
notebook's `LOCAL_SNAPSHOT` to it. A declaration alone is insufficient: every
byte must match the trusted lock. Alternative `model.safetensors`, `tokenizer.json`,
other files or nested directories are rejected to prevent shadowing verified
files. Export only the five required files plus the declaration, not an entire
mixed-framework Hub repository. Existing conflicting cache bytes are never
silently overwritten. An LFS pointer or trained classifier will fail.

## Durable execution and exact CLI commands

Mount your own Google Drive in the notebook. A write/read probe must pass; ensure
at least 10 GB account quota and local disk space. The notebook makes a unique
run root under `/content/drive/MyDrive/NASN/task03b/<run-id>/`. Its `experiment/`
output is absent/empty before CLI execution. Checkpoints are saved directly to
Drive at each selected epoch. The original evidence directories under
`reports/task03/` are never overwritten.

After notebook setup has generated `colab_training_config.json` and the verified
cache, the exact original commands are (replace the run ID with the printed ID):

```bash
cd /content/nasn-task03b-baseline
export PYTHONPATH=/content/nasn-task03b-baseline
export HF_HOME=/content/nasn-task03b-cache
export HF_HUB_DISABLE_XET=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
NASN_PY=/content/nasn-marbert-venv/bin/python
NASN_RUN_ROOT=/content/drive/MyDrive/NASN/task03b/PRINTED-RUN-ID
"$NASN_PY" -m arsarcasm_audit fetch --destination data/raw
"$NASN_PY" -m nasn.marbert preflight \
  --config "$NASN_RUN_ROOT/colab_training_config.json" --output "$NASN_RUN_ROOT/preflight"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$NASN_PY" -m nasn.marbert train \
  --config "$NASN_RUN_ROOT/colab_training_config.json" --output "$NASN_RUN_ROOT/experiment"
"$NASN_PY" /content/nasn_external_runtime.py verify-run --run "$NASN_RUN_ROOT/experiment"
```

Do not run these against already populated outputs; use a new run directory.
The notebook streams CLI output to a durable sanitized execution log and checks
its actual exit code. It will not claim completion on a runtime interruption,
OOM, download failure, checkpoint readback error or nonzero CLI exit.

Successful runs must contain epoch history, validation metrics, per-class
precision/recall/F1, confusion matrix, all 2,509 validation predictions, subgroup
metrics, configs, preflight, environment/device/duration, loading info, checkpoint
hashes and model/tokenizer checkpoint. The original CLI reloads both model and
tokenizer and runs full validation inference, requiring the selected Macro-F1 to
match. The final helper then reads checkpoint files back from Drive, checks every
saved hash, matches predictions to the unchanged validation manifest and labels,
and recomputes saved aggregate validation metrics from predictions. These are
necessary checks before marking `training_completed=true` in runtime provenance.
No official test inference or evaluation is invoked.

## Return reports safely to GitHub

The final cell exports an allowlist of lightweight CSV/JSON/Markdown files and
runtime provenance, creates a ZIP in Drive and offers a browser download. Model
weights, raw dataset CSVs, tokenizer files, arbitrary logs and executed-notebook
outputs are excluded. The full checkpoint remains in Drive; do not remove it
until durable storage and its hashes are confirmed. Drive availability/quota and
persistence still require checking by the account owner.

On your normal authenticated development machine (never paste tokens in the
notebook or put them in clone URLs), extract the ZIP and review its contents:

```bash
git clone https://github.com/haneenbiate/-.git nasn-reports
cd nasn-reports
git switch -c reports/task03b-PRINTED-RUN-ID
mkdir -p reports/task03b/PRINTED-RUN-ID
# Copy reviewed lightweight ZIP contents to the new report folder above.
git add reports/task03b/PRINTED-RUN-ID
git diff --cached --stat
git diff --cached --check
git commit -m "Record Task 03B external MARBERT validation run"
git push -u origin reports/task03b-PRINTED-RUN-ID
```

Open a PR for the new report branch. Do not use `git add .`, force-push, commit
Drive checkpoints or replace old evidence. If the repository is private, authenticate
via a secure Git credential helper before cloning; the notebook clone fails
noninteractively rather than asking for or displaying tokens. Blocked-run reports
can also be exported with the helper's `export` command, retaining failure status;
do not call them successful experiments.

## Notebook development checks

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m arsarcasm_audit fetch --destination data/raw
python tools/build_colab_notebook.py
.venv/bin/python -m pytest -q
```

The builder embeds exactly the checked-in helper and lock. Tests validate notebook
schema and cell syntax, pinned references, unchanged CLI invocation, no split
command, snapshot corruption/conflicts/alternative-file rejection, protocol
preservation, failed-run reporting and allowlisted export. They do not fabricate
Colab execution, GPU compatibility or baseline validation metrics.
