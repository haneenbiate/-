# Task 07 v2: completed experiment and scientific decision

The supplied `nasn-v2-controlled-20261014T054250Z-2a3747bf-reports.zip` is verified.
All **26 payload hashes**, **62,350 prediction identities**, source/code/rule hashes
and selected checkpoint evidence match the frozen experiment. Every per-rule,
pooled-cluster and all-clean metric/interval, and the scientific decision, is
independently recomputed. [Original payloads](../reports/task07-v2-completed/original)
remain byte-identical. Prior preparation/blocker reports remain historical evidence.

Archive SHA256: `6306331ae954f9778bec80d9ee2e1d55fa41744b29221b4162402ab52a0f286f`.
Pinned implementation: `304e54bd4ba0f6fa9b012244de090ee66d84b014`.
Selected epoch-2 model SHA256:
`5c9fd87e2639a967d2c7517d5bb4189a6cbcb3bb477f9998e105682fcc249840`.
Recorded external runtime: Tesla T4, CUDA 12.6, Torch 2.8.0+cu126, float32,
inference batch 8, seed 42, sequence length 128 and 510.8631 seconds total run
duration. This analysis reuses real saved predictions; it does not load weights
or perform local model inference. No model training or official test evaluation
occurred, and no rule, threshold, checkpoint selection or data split is changed.

## Primary comparison

The primary cohort is **394 user-declared eligible source variants over 274 original
IDs**. New normalized outputs have not been human-validated, and independent human
review provenance remains unverified. Thus these are conditional development scores
against source sentiment labels, not certified meaning-preserving robustness.

| Method | All-clean Macro-F1, n=2,509 | Primary perturbed Macro-F1, n=394 | Paired prediction flip rate |
|---|---:|---:|---:|
| RAW | 0.750428 | 0.640226 | 5.08% |
| TAT | 0.750883 | 0.646232 | 2.28% |
| REP2 | 0.750449 | 0.649179 | 4.57% |
| NO_CONTEXT | 0.750904 | **0.655111** | **1.78%** |
| NASN2 | 0.751225 | **0.655111** | **1.78%** |

Flip rate compares each method's normalized clean and perturbed inputs on matched
IDs. All-clean scores use 2,509 unique original rows; paired pooled clean metrics
use the same variant weighting as the perturbed cohort and are saved separately.
Accuracy, per-class results and complete intervals are in the
[tables](../reports/task07-v2-completed/primary_comparison_table.csv) and preserved
original reports. Both NO_CONTEXT and NASN2 primary Accuracy are 0.667513. Their
primary perturbed Macro-F1 interval is [0.581849, 0.731046], and flip rate is 7/394,
with interval [0.004784, 0.036145].

## Incremental contribution of context awareness

NO_CONTEXT is the strongest simple comparator under the frozen primary ranking.
It shares marker/critical-word protection and combined tatweel/run-cap rules;
the ablation isolates NASN2's additional bounded negation-context abstention.

| Reference | NASN2 minus reference perturbed Macro-F1 | Paired 95% interval |
|---|---:|---:|
| TAT | +0.008878 | [-0.002942, +0.023442] |
| REP2 | +0.005932 | [-0.002208, +0.016312] |
| NO_CONTEXT | **0.000000** | **[0.000000, 0.000000]** |

NASN2 and NO_CONTEXT have identical predicted classes on **all 394** primary rows,
despite **11 different effective input sequences**. The empirical zero interval
reflects identical observed classes; it does not prove zero effect in unseen data
or that all negation protection is useless. On all-clean data, four effective
inputs and one predicted class differ, giving NASN2 a small point Macro-F1 advantage
of 0.000320 over NO_CONTEXT. This cannot replace the required positive perturbed
gain. The 9,557 unreviewed variants have 241 differing effective inputs and 15
class differences; their metrics are reported separately, not promoted to confirmed
semantic evidence. Ten source-review exclusions receive no robustness metrics.

NASN2's paired point gain over RAW is +0.014885, with interval [0.000414, 0.032061]
on this cohort. The combined simple method delivers the same primary predictions
and gain. This result cannot be attributed uniquely to context awareness or used
to support a NASN2 novelty/superiority claim. It remains exploratory and conditional
on unverified semantic/provenance assumptions.

## Scientific recommendation

The [predetermined decision gate](task07_protocol.md) requires positive lower
paired gain bounds against every simple comparator and nonnegative all-clean
point differences. **It fails.** The strongest comparator ties NASN2, while gain
intervals against TAT and REP2 include zero. This is the same decision reported
by the external run and reproduced independently here.

**Drop the claim that NASN2 improves on the strongest simple normalization baseline.**
Reframe the project as a transparent Arabic orthographic preprocessing and ablation
study showing the limited observed incremental value of the context guard. Preserve
this negative result; do not add favorable word exceptions, change repetition caps
or rerank cohorts after seeing the outcome. No corrected-significance or novelty
claim is made. This finding applies to this frozen MARBERT development experiment,
not every Arabic model or all possible negation-aware methods.

Human validation remains necessary for meaning-preservation claims: new output
review packages from Task 07 are still blank, and source reviewer independence has
not been authenticated. It is not necessary to invent or wait for human decisions
to report failure of the present performance gate on the recorded labels.

## Reproduce the audit

Use the existing pinned Python 3.12.14 environment (`requirements-task04.txt`),
pandas 2.2.3, NumPy 2.3.5 and scikit-learn 1.8.0. No GPU, trained weight files or
raw official dataset files are needed for this saved-prediction reproduction.

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/nasn-matplotlib
export XDG_CACHE_HOME=/tmp/nasn-cache
.venv/bin/python -m nasn.task07_v2_results \
  --archive /path/to/nasn-v2-controlled-20261014T054250Z-2a3747bf-reports.zip \
  --output runs/task07-v2-verified-new
.venv/bin/python -m tools.publish_task07_v2_results \
  --analysis runs/task07-v2-verified-new --output runs/task07-v2-public-new
.venv/bin/python -m pytest -q
```

New output paths are required to preserve evidence. The publication helper is
guarded to this supplied archive; another experiment needs its own report. The
analyzer verifies frozen model/data/rule identities, probability sums/argmax,
original RAW clean baseline, exclusion/eligibility metadata and all uploaded tables.
Original files remain unchanged; derived numeric tables, PDF/SVG contrast figures,
per-example ablation diagnostics and checksums are published. Raw text, token IDs,
annotator identities/notes and model weights are not committed.

## Limits

Intervals use 2,000 seed-42 paired percentile resamples. Pooled variants move as
original-ID clusters; per-rule strata contain one variant per source ID and retain
minimum n=30. They condition on one selected checkpoint and a nonrandom previously
analyzed development sample, omit training/reviewer uncertainty, and are not
simultaneous significance tests. Duplicate texts under different IDs, conflicting
source labels and high-risk near-duplicates remain limitations. New normalization
can alter emphasis or meaning, and finite lexical/context guards can misclassify
scope. No genuine independent-human agreement or semantic certification is inferred
from model outputs, hashes, inverse tests or a completed GPU run.
