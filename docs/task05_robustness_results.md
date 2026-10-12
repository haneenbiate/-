# Task 05 confirmed robustness: actual completed results

The uploaded `orthography-v1-20261010T083120Z-3d929f7b-reports.zip` contains a
completed Task 04 Tesla T4 inference run. All payload checksums match. Source
splits, selected epoch-2 checkpoint hashes and original validation predictions
match Task 03B. The final human review covers 404 assignments, of which 394 are
eligible and 10 excluded. Analysis is complete using these actual saved model
outputs. No model retraining, checkpoint selection or official test evaluation
occurred. Publication tables/figures and full methodological limits are in
[reports/task05-robustness](../reports/task05-robustness).

| Confirmed cohort | n | Clean Macro-F1 | Perturbed Macro-F1 | Drop (points) | Flip rate |
|---|---:|---:|---:|---:|---:|
| Elongation 1 | 89 | 0.6178 | 0.6342 | -1.64 | 4.49% |
| Elongation 2 | 94 | 0.6545 | 0.6230 | +3.15 | 5.32% |
| Tatweel 1 | 94 | 0.6489 | 0.6366 | +1.23 | 8.51% |
| Tatweel 2 | 93 | 0.6158 | 0.6025 | +1.33 | 3.23% |
| Conservative Alef | 24 | unavailable | unavailable | unavailable | unavailable |

Drop is clean minus perturbed; negative means observed improvement. All four
primary 95% paired-bootstrap drop intervals include zero. No population-level
direction or significant intensity effect is established. The predeclared
minimum sample size is 30; the Alef cohort is below it. Confidence intervals,
Accuracy, per-class F1, negation/effect strata and high-risk sensitivity are in
`confirmed_robustness.csv`. Seed is 42; bootstrap repetitions are 2,000.

The different cohort rows are independently sampled subsets, so their rates
cannot establish a dose response. On 35 original IDs with **both** tatweel
intensities human-confirmed (31 after high-risk exclusion), intensity 1 and 2
have identical predictions and Macro-F1. Across all 2,474 applicable original
IDs, both full/effective token sequences and predictions match between the two
intensities; one probability vector differs at tiny exact numeric precision,
none above 1e-6. These full-population counts remain provisional. Relative to
clean text, each intensity changes tokens/probabilities for 2,474 IDs and final
predictions for 176. The selected tokenizer files are independently byte-verified,
and all 12,545 exported token count/change/truncation flags reproduce locally.
Token associations remain descriptive, not causal proof.

## Exact reproduction commands

Use the pinned Python 3.12.14 Task 04 environment and existing source/manifests.
The official pinned small MARBERT files must exist locally; checkpoint weights
are not required for this saved-prediction computation. Do not download or read
official test text. Use new output directories; commands refuse to overwrite.

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/nasn-matplotlib
export XDG_CACHE_HOME=/tmp/nasn-cache
.venv/bin/python -m nasn.task04_bundle \
  --archive /path/to/orthography-v1-20261010T083120Z-3d929f7b-reports.zip \
  --output runs/task04-imported-new
.venv/bin/python -m nasn.verified_tokenizer \
  --source data/huggingface/models--UBC-NLP--MARBERT/snapshots/88e1fa192dd723cf0b3563500aec46209762eb22 \
  --lock configs/marbert_snapshot_lock.json \
  --checkpoint-registry reports/task03b/marbert-v1-20261009T213143Z-be8b8811/checkpoint_hashes.json \
  --output runs/task05-tokenizer-verified-new
.venv/bin/python -m nasn.task05 analyze \
  --preparation runs/task04-final-v2-preparation \
  --baseline reports/task03b/marbert-v1-20261009T213143Z-be8b8811 \
  --review-package runs/task05-semantic-review-v1 \
  --decisions runs/task05-adjudication-final-baghdad/decisions \
  --inference runs/task04-imported-new/inference \
  --tokenizer-directory runs/task05-tokenizer-verified-new \
  --bootstrap 2000 --output runs/task05-confirmed-robustness-new
.venv/bin/python -m tools.publish_task05_robustness \
  --analysis runs/task05-confirmed-robustness-new \
  --import-directory runs/task04-imported-new \
  --tokenizer-directory runs/task05-tokenizer-verified-new \
  --output runs/task05-robustness-public-new
.venv/bin/python -m pytest -q
```

Private review/preparation paths refer to artifacts generated in the documented
earlier tasks; if unavailable, follow those explicit restoration commands rather
than silently reconstructing splits or inventing human decisions. The analysis
retains original provisional metrics byte-for-byte. Public files omit full tweet
text, token-ID sequences, human identities/rationales and weights. Detailed
per-ID token/probability comparisons remain in private ignored `runs/` storage;
their checksums are recorded. The 394 reviewed variants are not all validation
rows and are not a random population sample. Human roles/blinding are user-
declared; local model weight bytes and a local classifier reload are not claimed.
Full external execution provenance and recorded checkpoint hashes are preserved.
