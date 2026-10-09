# Task 03B execution status

The Colab notebook is prepared but has **not been executed on an external GPU**.
No training epoch, validation metric, confusion matrix, prediction or trained
checkpoint is claimed for Task 03B. Existing Task 03 evidence is preserved.

## Actual checks in this workspace

- Baseline reference: `c8dc698780cb06c8838c557f159bdaed7507aeda`.
- The original preflight passed, verifying source and existing manifest hashes,
  counts 10037/2509/2/3000, and zero exact/canonical overlap between active splits.
  Its audit retains 753 near-duplicate candidates, including 323 active cross-split
  pairs and 264 high-risk review flags; nothing was removed or merged. See
  `local_preflight/`. The copied methodology is original CLI output for a
  preflight-only invocation, not evidence of training or model loading.
- All 71 development tests passed in 9.09 seconds; dependency `pip check`
  reported no broken requirements. See `development_validation.json` for
  versions, artifact hashes and the distinction between local checks and
  external checks still pending.
- Managed Python bootstrap: `uv==0.13.0` successfully installed and located
  CPython `3.12.14` in an isolated temporary directory.
- The official PyTorch CUDA 12.6 index lists the required Python 3.12 Linux wheel,
  `torch==2.8.0+cu126`. Wheel availability is checked; CUDA installation and driver
  compatibility remain unverified until the notebook's GPU kernel gate passes.
- The pinned MARBERT snapshot downloaded successfully from the official Hub.
  All five required file sizes and SHA-256 hashes passed verification; see
  `local_snapshot_attempt.json`. The 654,186,400-byte encoder weight file has
  SHA-256 `801862ace5fc9e6b135b19476dbda5c7de1e8673b5b9d55db08d79e2c30b9dca`.
  This is a download integrity check, not an encoder-load or training result.
- The historical Task 03 download rejection from `us.aws.cdn.hf.co` is retained
  in its original logs. It did not recur in this download-only attempt. A future
  Colab failure must report its actual host/error, rather than assume this one.
- The local GPU gate exited with code 2: `CUDA GPU unavailable; no CPU fallback`.
  Installed torch is `2.8.0+cpu`, `torch.version.cuda` is null, and
  `torch.cuda.is_available()` is false. See `local_gpu_gate.json`.

## Remaining blocker and recovery

This workspace has no CUDA GPU and no connected external Colab runtime to execute
the notebook. GPU type, VRAM, training duration, checkpoint path and all training
metrics therefore remain unavailable. Download recovery alone does not constitute
a successful baseline run.

Open `notebooks/marbert_baseline_colab.ipynb` in Google Colab, select a GPU runtime,
mount your Google Drive and run the cells in order. Exact commands and the verified
local-snapshot fallback are in `docs/task03b_colab.md`. The notebook uses the pinned
baseline's original CLI, saves outputs directly to Drive, checks durable checkpoint
hashes and requires successful checkpoint reload/validation inference before
marking the run completed. It never evaluates the official test set.

Only device/cache paths and, if required, microbatch/gradient accumulation change.
Every such change is recorded, with effective batch size fixed at 16. No actual
GPU configuration deviation has been executed here. Drive authorization, quota,
CUDA kernel compatibility, full encoder loading and complete training must still
be verified in the external runtime. Return only the allowlisted lightweight
report bundle to a new GitHub report directory; retain weights in Drive.
