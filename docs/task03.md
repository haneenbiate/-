# Reproduce MARBERT Baseline v1.0

Use Python **3.12.14** and the existing repository checkout. Task 03 requires
unchanged Task 02 artifacts; it never generates/reconstructs splits. Pinned
source CSVs may be fetched, but altered/missing manifests must be restored from
Git, not recreated. Task 02 Python/pandas/numpy/sklearn versions must match.

## Install and validate

```bash
python -m venv .venv
# CPU installation; a compatible CUDA build is required for GPU training.
.venv/bin/python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-cache-dir -r requirements-marbert.txt
.venv/bin/python -m arsarcasm_audit fetch --destination data/raw
.venv/bin/python -m pytest -q
.venv/bin/python -m nasn.marbert preflight \
  --config configs/marbert_v1.json --output reports/task03/preflight-new
```

For an NVIDIA CUDA 12.8 machine, install the PyTorch 2.8.0 CUDA build instead of
the CPU command above (check driver compatibility first):

```bash
.venv/bin/python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Record hardware and the installed torch local build in environment.json. The
committed attempted run used **torch 2.8.0+cpu**, with no GPU. CPU and CUDA builds
are distinct runtime environments and do not promise bit-identical scores.

## Train

The official MARBERT model revision is pinned to
`88e1fa192dd723cf0b3563500aec46209762eb22` (resolved from the official Hub).
Use a fresh output directory; do not overwrite experiment evidence:

```bash
.venv/bin/python -m nasn.marbert train \
  --config configs/marbert_v1.json \
  --output reports/task03/marbert-v1-new
```

The config defaults to device `auto`, effective batch 16, microbatch 4 with
four accumulation steps, float32, seed 42, maximum length 128, AdamW lr 2e-5,
weight decay .01, maximum five epochs, patience two. Highest validation Macro-F1
selects the checkpoint; ties retain the earlier epoch. No test evaluation exists
in this runner. Model features receive original tweet strings, not canonical text.

The CPU feasibility guard probes one backward-only effective batch without any
optimizer update, restores Torch RNG, and projects maximum-epoch compute. Its
explicit operational limit is 1,800 seconds (30 minutes), excluding validation
and optimizer/checkpoint overhead. It is an estimate, not a model result. If it
fails, use a CUDA machine or copy the config and explicitly increase
`cpu_max_projected_training_seconds` for a long CPU job. This alters the operational
time allowance, not initial model hyperparameters; save the exact used config.
No epoch is shortened, subset trained, or synthetic metric substituted.

## Current execution evidence and blocker

- `reports/task03/preflight/` contains the actual passed data preflight and
  near-duplicate audit. No model is required for its checks.
- `reports/task03/marbert-v1-attempt/` records the initial actual training-command
  attempt, which resolved the official model revision but failed downloading weights.
- `reports/task03/marbert-v1-retry/` records a retry after the network draft changed.
- **Neither attempt trained an epoch**: the proxy returned `403 Forbidden` on
  MARBERT's weight CDN, `us.aws.cdn.hf.co`. The model/tokenizer configuration and
  vocabulary are accessible, but complete pretrained weights are not available.
  The CPU probe therefore remains unrun. Absence of a GPU is an additional
  resource constraint, not a measured training-time result.
- The network draft now includes `huggingface.co`, `cdn-lfs.huggingface.co`,
  `cdn-lfs-us-1.hf.co`, `cas-bridge.xethub.hf.co`, and `us.aws.cdn.hf.co`, preserving
  package-manager presets. Review and save the draft in environment settings,
  then publish it. Saving the draft does not prove runtime access; retry the
  training command after applying it. This public model does not require a new
  secret/token on the evidence currently available.

No validation accuracy/F1, confusion matrix, per-class results, epoch losses,
predictions or best trained checkpoint are available. `validation_metrics.json`
explicitly records `not_run`, and `run_status.json` records the actual blocker.
Signed CDN query strings are omitted from stored diagnostics. Tiny random-BERT
unit tests check the software path only; their outputs are temporary and are
not MARBERT performance evidence.

## Successful-run outputs

| File/directory | Meaning |
| --- | --- |
| `training_config.json`, `resolved_training_config.json` | Exact configuration and immutable model revision |
| `preflight.json`, near-duplicate CSVs | Counts, complete source/manifest hashes and data-quality review queue |
| `environment.json`, `run_status.json` | Device, versions, training duration, peak GPU memory and outcome |
| `epoch_history.csv` | Actual per-epoch training loss, validation loss/metrics and duration |
| `validation_metrics.json`, `per_class_results.csv` | Selected-checkpoint validation metrics |
| `confusion_matrix.csv` | Gold rows/predicted columns in NEG, NEU, POS order |
| `validation_predictions.csv` | Stable IDs, gold/predicted labels, confidence and all three probabilities |
| `subgroup_metrics.json` | Sarcasm/dialect metrics for n>=30; smaller groups explicitly uncomputed |
| `best_checkpoint/` | Selected model.safetensors, model config, official tokenizer and vocabulary |
| `checkpoint_hashes.json` | Every saved checkpoint/tokenizer file's SHA-256 |
| `methodology.md` | Inputs, policies, findings and measured run outcome |

Successful fitting reloads the saved model **and tokenizer**, re-tokenizes raw
validation text, recomputes metrics and requires Macro-F1 to match checkpoint
selection before declaring completion. Confidence is uncalibrated. Subgroups use
fixed three-class Macro-F1 with missing-class counts/support; small dialect
groups (Task 02 validation magreb n=8) will be uncomputed. Cross-split near-duplicate
flags remain a research limitation; no human review is claimed.

Checkpoints, pretrained downloads and future run outputs should not be placed
in Git. Preserve checkpoints in durable model/artifact storage, retaining the
committed checksums and config. No official test predictions or metrics are
computed in Task 03.

The accessible official model config identifies **BertForMaskedLM**, hidden size
768, 12 layers, 12 heads and vocabulary 100,000. Standard sequence classification
adds a three-class head and may initialize the standard BERT pooler absent from
an MLM checkpoint. Successful loading records `model_loading_info.json` and
rejects missing encoder weights; actual MARBERT weight compatibility remains
unverified until the download succeeds. The official fast tokenizer's normalizer
uses `lowercase=true` and `clean_text=true`, with `strip_accents=null` (its native
BERT default). This is official tokenizer processing, not added Arabic
normalization. `reports/task03/tokenizer_audit.json` records the observed settings.
