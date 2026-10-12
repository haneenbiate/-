# ArSarcasm-v2: leakage-safe splitting (Task 02)

Task 03, MARBERT Baseline v1.0, is documented in [docs/task03.md](docs/task03.md).
Its actual preflight and blocked training attempts are saved in `reports/task03/`.
No MARBERT training results are claimed while pretrained-weight access remains blocked.

This module prepares reproducible dataset/evaluation manifests, not models.
Official source: <https://github.com/iabufarha/ArSarcasm-v2>, pinned commit
`eb85513e133cfc67e30aca4968fadbe2594fcb42`. The official CSV bytes are verified
against fixed SHA-256 checksums before use. No source CSVs are rewritten.
See the upstream LICENSE and README for terms and attribution. Cite Abu Farha,
Zaghouani and Magdy, *Overview of the WANLP 2021 Shared Task on Sarcasm and
Sentiment Detection in Arabic*.

## Reproduce

From the repository root, using Python 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m arsarcasm_audit fetch --destination data/raw
.venv/bin/python -m arsarcasm_audit split \
  --training data/raw/training_data.csv \
  --testing data/raw/testing_data.csv \
  --output reports/reproduced-task02 \
  --seed 42 --validation-fraction 0.2
```

Use a new/empty output directory for each run. The CLI refuses nonempty output
paths and invalid/unpinned data, and exits with status 2. Installation and fetching
require Internet access; splitting and tests run locally afterward. Dependencies
(including transitives) are pinned in requirements.txt. Use the existing checkout;
no Git worktree is needed. The data and virtual environment are ignored by Git.
The checked-in report artifacts contain IDs/hashes/labels, not tweet text.

## Recovery disclosure

The restarted workspace contained only the initial README, with no earlier Task
01 source, tests, audit reports or manifests. This implementation recovers the
previously pinned dataset/checksums from the session and reconstructs the
previously documented baseline: sentiment-stratified `train_test_split`, seed
42, validation fraction 0.2. `baseline_manifest.csv` is a newly generated
reconstruction, not a recovered original artifact. All reported overlaps are
measured from this reconstruction and the downloaded source files. The old
full audit CLI and its original test suite were not recovered; the strict source
loader and new tests here cover the Task 02 workflow.

## Policies

Canonical text removes U+0640 tatweel, applies **NFC**, collapses Unicode whitespace
and strips boundary whitespace. Removing tatweel before NFC keeps normalization
idempotent when tatweel separates a letter from a combining mark. It does not
apply NFKC or casefold, strip diacritics, fold hamza/alef/ya/ta marbuta, or remove
negation, sentiment terms, punctuation or URLs. Raw tweets are never changed.

Every training group present in official test text is quarantined from fitting
and validation. Test **text only** controls quarantine. Remaining training groups
are indivisible: 512 seeded `GroupShuffleSplit` candidates are ranked using only
eligible training sentiment labels and row counts. Balance is approximate;
the methodology describes the objective and limitations. Test labels, metrics,
and near-duplicate findings do not influence split selection. Conflicting labels
are reported and retained unchanged; no majority vote or relabeling is applied.

The official test manifest retains every one of the 3,000 source rows in order.
The secondary test evaluation manifest flags original-training overlap and
within-test canonical repeats. `unique_nonoverlap` selects the first row of each
canonical test group absent from original training. These are predeclared audit
slices, not model-selection data. Use validation alone for model selection;
freeze choices before any full or secondary test evaluation.

Near-duplicates are char TF-IDF 3–5 gram cosine candidates at threshold 0.9,
flagged for inspection only. They are never merged or used to change eligibility
or assignments. The audit compares all source text, but no test-derived features
are supplied to a model. Canonical disjointness does not imply semantic
independence. Short texts and semantic paraphrases may evade candidate detection.

For future fitting and validation, use the row IDs in the active manifests:

```python
import pandas as pd
from arsarcasm_audit.audit import load_csv

raw = load_csv('data/raw/training_data.csv')
fit_ids = pd.read_csv('reports/task02/train_manifest.csv').row_id.to_numpy()
val_ids = pd.read_csv('reports/task02/validation_manifest.csv').row_id.to_numpy()
fitting_data = raw.iloc[fit_ids].copy()
validation_data = raw.iloc[val_ids].copy()
```

Only those selected records are eligible for fitting/validation. Reading the
whole official training CSV does not apply quarantine. Read annotations as
literal strings (the provided loader does this); pandas may otherwise infer
`FALSE`/`TRUE` as booleans.

## Outputs

The actual official-data run is in `reports/task02`; start with
[the methodology report](reports/task02/methodology.md).

| Artifact | Purpose |
| --- | --- |
| `metadata.json` | Source revision/checksums, seed, method, balance objective, versions and CSV artifact checksums |
| `split_manifest.csv` | All source rows: active train, validation, immutable test, or quarantine |
| `train_manifest.csv`, `validation_manifest.csv`, `test_manifest.csv` | Partition-specific manifests |
| `baseline_manifest.csv` | Reconstructed Task 01 assignments for comparison |
| `quarantine_manifest.csv` | Every excluded training record and exact/canonical-only reason |
| `training_deviations.csv` | Every official training row's baseline/new assignment and reason |
| `test_evaluation_manifest.csv` | All official test rows with overlap, canonical representative and unique-slice flags |
| `group_members.csv` | Every row's stable group ID and assignment |
| `duplicate_groups.csv` | Exact/canonical duplicate groups, member IDs, label sets and conflicts |
| `overlap_statistics.csv`, `overlap_members.csv` | Exact/canonical train-validation, train-test, validation-test overlaps before/after, with conflicting-label counts |
| `distributions.csv` | Sentiment, dialect and sarcasm counts/proportions, including quarantine |
| `near_duplicate_review.csv` | Review-only candidates with row/group IDs and cosine scores |
| `methodology.md` | Methods, actual counts, distributions, conflicts, limitations and deviations |

Source row IDs combine official source partition with a zero-based CSV record
position (not a physical line number). File checksums anchor those IDs. Group IDs
are `ct1_` plus full SHA-256 of canonical UTF-8 text. Text hashes refer to raw UTF-8
text. All duplicate training records are retained unless their canonical text is
in test; group assignment prevents splitting those records across train/validation.
No source rows are deleted; exclusions and reassignment exist only in manifests.
The API supports small unpinned test fixtures with `require_official=False`; the
CLI always requires pinned official bytes and exactly 12,548/3,000 source rows.

## Task 03B: external GPU recovery

Open [the pinned MARBERT Colab notebook](https://colab.research.google.com/github/haneenbiate/-/blob/main/notebooks/marbert_baseline_colab.ipynb)
and follow [the GPU runtime instructions](docs/task03b_colab.md). It clones the
exact Task 03 baseline commit, verifies the existing splits and pretrained bytes,
uses the original training CLI and saves checkpoints directly to Google Drive.
See [the historical execution blocker report](reports/task03b/blocker_report.md)
for the original local preparation checks. A completed external Tesla T4 run was
subsequently supplied in a report ZIP and its hashes and validation metrics were
verified: [Task 03B run evidence](reports/task03b/marbert-v1-20261009T213143Z-be8b8811).
The selected checkpoint remains in the user's Drive and is not committed.

## Task 04: validation errors and orthographic feasibility

[Actual results and limitations](reports/task04/methodology.md) cover validation
errors, dialect/sarcasm strata and near-duplicate sensitivity. Deterministic
perturbation manifests and pending review queues are prepared; no perturbed
inference results are claimed while the checkpoint/GPU are unavailable locally.
Follow [the reproduction and semantic-review instructions](docs/task04.md).
The [Task 04 Colab notebook](https://colab.research.google.com/github/haneenbiate/-/blob/main/notebooks/task04_robustness_colab.ipynb)
loads the existing Drive checkpoint and runs inference through the documented CLI.
The official test text is never opened by the Task 04 implementation.

## Task 05: semantic validation and negation scope

[Review and reproduction instructions](docs/task05.md) describe the blinded RTL
offline interface, the fixed 404-case full-text queue, independent review and
adjudication, integrity-checked imports and saved-prediction analysis. Actual
[Task 05 evidence](reports/task05) is at **prepared review**: no human decisions
or confirmed robustness results are claimed. Full-text exports remain in private
`runs/` storage; model training and official test evaluation are prohibited.

Corrected human reviews were subsequently supplied: see
[agreement and adjudication instructions](docs/task05_reviewed.md) and
[text-free reviewed evidence](reports/task05-reviewed). Human agreement covers
all 404 assignments; 191 have eligible consensus and 213 require third-human
adjudication. The LLM is analyzed separately. Confirmed robustness evaluation
remains blocked by incomplete adjudication and unavailable Task 04 predictions.

An adjudication workbook was subsequently supplied. Its 213 IDs match, but Excel
removed edge spaces from 20 text pairs and stored all timestamps without timezone.
[The intake protocol](docs/task05_adjudication_intake.md) documents explicit,
hash-checked formatting repairs and the required reviewer timezone; the current
[intake audit](reports/task05-adjudication-intake) preserves this pending status.

The user then confirmed Baghdad time. [Final review evidence](reports/task05-final-review)
now records all 404 resolved assignments: 394 eligible meaning-preserving variants
(184 with human-identified negation, 210 without), and 10 exclusions. Human review
is complete; model robustness analysis still needs the Task 04 saved predictions.
Earlier reports remain unchanged as historical stage evidence.

The completed Task 04 inference archive was then supplied. [Actual confirmed
robustness results](docs/task05_robustness_results.md) and
[publication artifacts](reports/task05-robustness) now include paired-bootstrap
metrics, human-negation strata, near-duplicate sensitivity and matched tatweel
intensity/token comparisons. No primary Macro-F1 drop interval excludes zero.
All predictions come from the existing selected baseline; no retraining or
official test evaluation occurs. NASN had not been implemented at Task 05 completion.

[Task 06](docs/task06.md) now audits the original independent-review records and
all 213 adjudications. Reviewer identity/independence remains explicitly unverified.
The deterministic B0–B4 NASN prototype has reversible logs, negation/sentiment
guards and actual selected-tokenizer audits. [Annotation quality](reports/task06a)
and [development preparation/comparison tables](reports/task06b) preserve all prior
evidence. Historical B0 scores are available; new B1–B4 classifier metrics remain
blocked by missing selected checkpoint/GPU locally. The pinned
[Colab GPU notebook](notebooks/task06_nasn_colab.ipynb) performs inference with the
existing trained checkpoint only. No NASN robustness improvement is claimed.

[Task 07](docs/task07.md) subsequently verifies the completed Task 06 GPU ZIP and
recomputes its real B0–B4 tables. V1 elongation predictions agree in 4,945 of 4,948
applicable variants (all 183 source-eligible reviewed elongations agree), rather
than every input. All 643 B3/B4 guard differences and 84 elongation token exceptions
are audited. [Completed v1 evidence and frozen v2 preparation](reports/task07) are
preserved separately; historical Task 06 blocker reports remain untouched.
The v2 prototype uses explicit conservative run caps, simple baselines and a
context ablation, with new blank independent-review packages. Its model comparison
is not executed locally because selected weights/CUDA remain unavailable. The
[Task 07 Colab notebook](notebooks/task07_nasn_v2_colab.ipynb) performs that exact
selected-checkpoint experiment; the scientific decision is undetermined until
verified v2 results arrive. No new authenticated-human agreement or novelty claim
is made.
