# Task 07 — verified v1 failure diagnosis and controlled v2 preparation

**Latest stage:** the completed October 14 v2 report ZIP has now been verified and
all metrics/intervals recomputed. See [actual v2 results and the scientific
decision](task07_v2_results.md). The preparation/blocker sections below retain
the earlier stage; frozen rules and decision criteria are unchanged.

The supplied `nasn-prototype-v1-20261012T111339Z-4e08cdbf-reports.zip` is a completed
Task 06 run, verified from all 15 payload checksums and recomputed metric tables.
Archive SHA256: `f72c24eae1ebd58fcc755fbc0461d040b0293151e70b76f3d11a70365ba6c837`.
It used Tesla T4, Torch 2.8.0+cu126, float32, seed 42, length 128 and inference
batch 8, with recorded total duration 575.6704 seconds. The selected epoch-2
classifier hash is `5c9fd87e2639a967d2c7517d5bb4189a6cbcb3bb477f9998e105682fcc249840`.
It reports successful reload, no training, no test access/evaluation. Local
verification checks saved hashes/probabilities/IDs; weight bytes are not uploaded.
All 62,350 strategy/input metadata records exactly match the registered Task 06
manifest. Original reports remain unchanged; verified completed evidence is now
preserved separately in [Task 07](../reports/task07/v1-reproduction).

## What v1 actually did

The premise that every B1–B4 elongation prediction is identical is too broad:

| Original perturbation | n | Identical B1–B4 effective inputs | Identical B1–B4 predictions |
|---|---:|---:|---:|
| Elongation 1 | 2,474 | 2,441 | 2,473 |
| Elongation 2 | 2,474 | 2,423 | 2,472 |
| Total | 4,948 | 4,864 | 4,945 |

All methods do agree on the reviewed elongation cohorts: 89 intensity-1 and 94
intensity-2 variants, including B0. Their perturbed Macro-F1 values remain 0.634235
and 0.622950 respectively. These are **user-declared source-eligible reviews**,
not independently verified human-confirmed transformations.

B1 only removes tatweel, so it has no rule addressing repetition. B2's added
NFC/whitespace changes usually map to the same official-tokenizer inputs. B3/B4
require a run of at least three letters and collapse-to-one must produce one of
only twelve allowlisted neutral words. They also abstain on protected sentiment/
negation/identifier tokens. This sharply limits elongation coverage. Most intensity-1
edits create just two letters; they are outside v1's repetition rule. For most
eligible inputs no differential effective token sequence reaches MARBERT. Among
84 inputs with different effective sequences, only three show a different final
class: changed tokens do not imply changed argmax, and unchanged argmax does not
establish semantic safety. Different text/whitespace beyond tokenization and the
128-token context may also be invisible to the classifier.

Exact source/token replay and all exceptions are in
[pipeline equality](../reports/task07/v1-diagnosis/pipeline_equality.csv) and
[all 84 token exceptions](../reports/task07/v1-diagnosis/all_elongation_token_exceptions.csv).
The private exception export contains original, perturbed and all five normalized
texts, rules, probabilities and predictions for every exception, including the
three class differences. It is saved in `runs/task07-extra-examples` and the new
Colab private archive, not committed as raw dataset text.

## B4 versus B3: every differing input

There are **643** text differences: 20 clean originals, 20 elongation-1, 23
elongation-2, 290 tatweel-1 and 290 tatweel-2 inputs. All have different effective
token sequences. Every case is recorded, not a selected illustrative subset.
The private CSV includes original/input/normalized texts, applied edits, B4 guard
spans, probabilities, classes, correctness and source eligibility. Public hashes/
IDs/numeric evidence are in
[all differences](../reports/task07/v1-diagnosis/all_B3_B4_differences.csv).

For each tatweel intensity, 287 of the 290 differences involve B3 extra edits that
only remove tatweel outside even the broad possible-marker spans. These are
**structural typography candidates**, not human-validated safe transformations.
The three remaining cases should not be classified as safe by assumption. Across
the 290 differing cases B3 is correct in 220 and B4 in 213, conditional on source
gold labels; this descriptive count neither proves causality nor semantic validity.

In the 94 source-eligible tatweel-1 cases, B3 perturbed Macro-F1 is 0.648890 and
B4 0.647271. The paired B3/B4 diagnostic intervals are provided separately with
the same perturbed input on both sides. In that file, `clean_*` means the left
condition B3-perturbed and `perturbed_*` the right B4-perturbed, as recorded in
its explicit left/right columns; neither is a clean tweet. The observed B3-minus-
B4 Macro-F1 difference is 0.001619 with a 95% paired interval [0, 0.006153].
These small cohort point differences do
not justify a significance or causal claim. Tatweel-1 and tatweel-2 reviewed cohorts
differ (94 versus 93); do not interpret their difference as an intensity effect.
Broad prefix/conjunction matching and the three-token window can protect ordinary
words or typography near a cue. V1's protection can therefore suppress useful
typography normalization. Linguistic safety of a case still requires review.

## Annotation evidence

Original A/B decisions and adjudications exist and Task 06A retains their categorical
records/free-text hashes. Distinct recorded IDs and completed files do **not**
establish independent people, collection, blinding or actual assistance. Historical
raw-agreement/Kappa values in Task 06A are descriptive statistics of user-declared
records, not authenticated human reliability. Task 07 computes **no new genuine
independent-human agreement statistic**, because the provenance prerequisite fails.
No identities, independent collection evidence or human confirmations are invented.
The 394 source-eligible cases remain conditional; the 9,557 unreviewed provisional
and 10 reviewed exclusions stay separate. The latter receive no robustness metrics.

## Controlled v2

Rules were written before inspecting new v2 effects, in the
[frozen redesign protocol](task07_protocol.md) and [configuration](../configs/nasn_v2.json).
This is a post-hoc exploratory redesign, not preregistration or a novelty claim.

| Method | Fixed behavior |
|---|---|
| RAW | Original text byte-for-byte. |
| TAT | Tatweel removal outside literal possible negator/identifier spans. |
| REP2 | Cap eligible identical consonant runs of length >=3 at two; preserve marked/sensitive/marker/identifier tokens. |
| NO_CONTEXT | Combined TAT+REP2; simple baseline and context-awareness ablation. |
| NASN2 | Same combined rules, but the three-token negation-context guard abstains from **repetition** edits only. It does not block tatweel merely because it is near a cue. |

Shared safety checks preserve recognized negation-marker bytes in every method.
Do not delete/insert negator words, substitute letters, fold hamza/Alef/ya/ta,
strip diacritics, normalize whitespace/NFC, stem or remove stopwords. Repetition
is limited to the original Task 04 repeatable consonants, avoiding fragile character
forms; diacritically marked words abstain. The original v1 sentiment/phrase safety
list is unchanged. No dataset-specific word exceptions are added. A two-letter
run is never assumed to be a spelling error. For example, capping a stretched
word can leave a nonstandard spelling or change emphasis; reversible edits and
token equality cannot certify sentiment preservation. All new outputs are provisional.

Rules receive only text, fixed strategy and configuration, never labels,
predictions, review outcomes, source IDs, training frequencies or test data.
Unit semantic counterexamples show retained double letters, protected cues and
expressions, mixed identifiers, marked words and a possible false positive for
`فلممم` (the shared broad matcher may parse a conjunction plus لم). The test
records abstention instead of adding a favorable exception after outcomes.

Actual selected-tokenizer/inverse preparation covers all 62,350 strategy/input
records. NASN2 has 33 repetition edit events on clean inputs and 2,418 on perturbed
inputs; NO_CONTEXT has 37 and 2,670 respectively. Event counts are not unique
originals, semantic approvals or performance gains. All changes restore exactly
with private logs. The [character/token audit](../reports/task07/v2-preparation/character_token_audit.csv)
does not contain model metrics.

## New human review

188 provisional new-output assignments have blank human decision fields, sampled
deterministically by method, clean/perturbed condition and repetition-edit category.
No sampling approval transfers to unsampled rows or an entire method. Send ONLY
`new_human_reviewer_a.zip` and `new_human_reviewer_b.zip` separately; each contains
its blank UTF-8 Arabic CSV, RTL offline HTML and protocol, without model predictions
or gold labels. Do not send the analysis registry to reviewers.

Actual independent reviewers assess input versus normalized text for lexical
meaning, emphasis, sentiment and negation, record cues/scopes/rationale/identity/
timezone-aware timestamps, and retain independent original submissions. Use a
third reviewer for disagreement; never an automatic LLM adjudicator. Collect
independence/blinding/assistance documentation. Neither earlier source review nor
distinct spreadsheet identifiers proves independence. Both normalized clean and
perturbed sides require semantic validity before a trustworthy paired analysis.
The existing strict import/adjudication tooling is reusable on this new package;
no completed human decisions have been supplied or generated in Task 07.

## V2 evaluation and scientific decision

V2 GPU inference is **not executed here**. The actual local attempt fails because
the selected Drive classifier directory is unavailable. CUDA is also unavailable
locally. The Task 06 ZIP supplies model outputs/hashes, not trained weight bytes.
No substitute pretrained or random classifier is used. Every new-method metric
is unavailable; old B1/B3 scores are not proxies for new TAT/NO_CONTEXT.

The prepared CLI/notebook loads the same selected classifier, verifies every
weight/tokenizer/config hash and loading completeness, and reproduces RAW clean
Task 03B predictions before reporting anything. It saves all-clean scores/paired
intervals, per-rule Accuracy/Macro-F1/per-class F1/flips, near-duplicate sensitivity
and negation strata. Pooled variants use paired **original-ID cluster bootstrap**,
retaining all variants of a sampled original together. Seed 42, 2,000 draws, fixed
NEG/NEU/POS and minimum n=30 are frozen. Source-row clustering does not account
for separate IDs with identical text; corpus duplicates remain a limitation.

Strongest simple comparator is highest perturbed Macro-F1 on the 394 source-eligible
cohort, ties TAT→REP2→NO_CONTEXT. The decision gate requires positive lower 95% paired
gain bounds against **every** simple comparator, with nonnegative all-clean Macro-F1
point differences. It is an exploratory project decision, not corrected significance
or novelty proof. Even a passed performance gate retains unresolved semantic and
human-provenance requirements. A failed gate explicitly recommends **abandoning
the improvement claim or reframing as a transparent preprocessing audit**, not
adding rules. Current decision: **undetermined — missing v2 model evaluation**.

## Exact reproduction commands

Use pinned Python 3.12.14 and existing `.venv`; export a headless backend:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/nasn-matplotlib
export XDG_CACHE_HOME=/tmp/nasn-cache
.venv/bin/python -m nasn.task07_io \
  --archive /path/to/nasn-prototype-v1-20261012T111339Z-4e08cdbf-reports.zip \
  --output runs/task07-intake-new
.venv/bin/python -m nasn.task07 diagnose \
  --v1-prepared runs/task06b-development-frozen \
  --tokenizer-directory runs/task05-selected-tokenizer-verified \
  --task06-intake runs/task07-intake-new --output runs/task07-diagnosis-new
.venv/bin/python -m tools.export_task07_examples \
  --v1-prepared runs/task06b-development-frozen --intake runs/task07-intake-new \
  --diagnosis runs/task07-diagnosis-new --output runs/task07-examples-new
.venv/bin/python -m nasn.task07 prepare \
  --v1-prepared runs/task06b-development-frozen \
  --tokenizer-directory runs/task05-selected-tokenizer-verified \
  --output runs/task07-v2-new
.venv/bin/python -m nasn.task07 run \
  --prepared runs/task07-v2-new \
  --checkpoint /path/to/the/verified/experiment/best_checkpoint \
  --batch-size 8 --output runs/task07-v2-gpu-new
.venv/bin/python -m pytest -q
```

If private full v1 preparation is absent, explicitly restore original text using
the Task 06 documented `nasn.task06 restore-task04` and `prepare` commands on the
pinned **training-only** CSV and existing manifests. Never resplit or replace
source hashes. Selected-tokenizer-only recovery is documented in Task 05. A
missing/changing source/config/code/hash causes a failure, not silent reconstruction.
The [new Colab notebook](../notebooks/task07_nasn_v2_colab.ipynb) performs this
explicit restoration and exact report/weight verification, writes to a new Drive
folder, and exports lightweight reports plus a separate **private** example/review
archive. Set the original Task 06 ZIP path if moved. Run cells in order; preserve
failure evidence and reduce only inference batch size if OOM.

The preparation-only publication uses the recorded blocked local attempt:

```bash
.venv/bin/python -m tools.publish_task07 \
  --intake runs/task07-intake-new --diagnosis runs/task07-diagnosis-new \
  --prepared runs/task07-v2-new --attempt runs/task07-v2-blocked-new \
  --extra-examples runs/task07-examples-new --output runs/task07-public-new
```

The blocked attempt is created by the same `run` command with an unavailable
selected Drive checkpoint path and `--output runs/task07-v2-blocked-new` (exit 2).
A completed future GPU result must not be passed to the preparation-only publisher.
All Task 06 uploaded payloads are preserved in the public v1 reproduction folder;
their original outer checksum registry can replay integrity verification. To
reconstruct a light ZIP, zip exactly its registry keys plus `report_checksums.json`;
container timestamps change ZIP SHA, while registered payload hashes stay unchanged.

## Remaining limits

V1 scope findings are descriptive and conditional on official source sentiment
and unverified annotation provenance. A structural typography-only flag is not
human semantic certification. The reviewed sample is nonrandom, sparse for Alef,
and includes repeated original IDs; semantic exclusions can bias comparisons.
Development data was previously analyzed, so this cannot be an independent final
claim. Prefix guards and finite lexical lists cause false positives/negatives;
generic repeat caps may change spelling, morphology or sentiment emphasis.
New semantic review and the frozen GPU experiment are still required. No official
test examples/labels are evaluated, altered or used to select rules; no retraining,
invented human evidence, fabricated performance, or automatic novelty claim occurs.
