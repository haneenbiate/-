# Task 05: semantic validation and negation scope

**Actual stage: prepared review.** The original Task 04 queue has 404 variants,
including 200 lexical-negation and 204 non-negation examples. These are heuristic
strata, not human scope findings. The 404 variants span 279 original text IDs.
Full original/perturbed texts were verified
against the pinned official training partition and deterministic Task 04
manifest. The official test CSV is never opened. No human decisions or completed
perturbed inference artifacts have been supplied locally. Confirmed metrics,
agreement and measured tatweel token/probability differences are unavailable.

Public evidence: [reports/task05](../reports/task05). Private full-text export:
`runs/task05-semantic-review-v1/`; do not commit its tweets or reviewer identities. Copy it
to durable storage such as the existing NASN Google Drive before leaving a
temporary runtime. The trained checkpoint is unchanged and remains on Drive.

## Download without workspace file links

Workspace file links may not download in the chat client. Open the
[review-download Colab notebook](https://colab.research.google.com/github/haneenbiate/-/blob/main/notebooks/task05_review_download.ipynb),
use a normal CPU runtime, choose **Run all**, and authorize your own Drive mount.
Click the two download buttons at the end, or download the ZIPs directly from
the printed Drive path. No checkpoint or model dependencies are needed.
The notebook uses Python 3.12.14 and minimal pinned review dependencies, explicitly
prepares deterministic candidates from existing Task 02 manifests, then compares
all 404 IDs/text hashes/rules/intensities with the original published Task 04
queue. It never regenerates splits, loads models or fetches official test text.
It uses a new directory for each run so existing checkouts are preserved.
The notebook is unexecuted in Colab here; local CPU dependency/CLI checks are
recorded separately. Public download links do not contain raw tweets; full-text
ZIPs are generated and saved in the user's private Drive.

## Reproduce preparation

Use Python **3.12.14** and the existing checkout. Install the unchanged Task 04
dependencies; Task 05 adds no dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-task04.txt -r requirements-notebook.txt
.venv/bin/python -m pip check
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/nasn-matplotlib
export XDG_CACHE_HOME=/tmp/nasn-cache
```

Restore the pinned official **training CSV only** using the checksum-verified
command in [Task 04 instructions](task04.md). Use the existing Task 04 full-text
`benchmark/` directory, containing its original hash registry, manifest and
404-item queue. A published text-free manifest alone is insufficient. Do not
reconstruct missing Task 02 splits. If the original private preparation is
absent in a fresh checkout, explicitly prepare deterministic Task 04 candidates
from the *existing pinned splits*, not new splits:

```bash
.venv/bin/python -m nasn.task04 prepare --output runs/task04-prepared-for-review
.venv/bin/python -m nasn.task05 prepare \
  --preparation runs/task04-prepared-for-review \
  --baseline reports/task03b/marbert-v1-20261009T213143Z-be8b8811 \
  --output runs/task05-review-new
.venv/bin/python -m pytest -q
```

Exact original preparation command executed in this workspace:

```bash
.venv/bin/python -m nasn.task05 prepare \
  --preparation runs/task04-final-v2-preparation \
  --baseline reports/task03b/marbert-v1-20261009T213143Z-be8b8811 \
  --output runs/task05-semantic-review-v1
```

Every output must be a new directory. Preparation fails on missing evidence,
changed manifest bytes, wrong source/labels/protocol or changed queue membership.
Checkpoint/tokenizer hashes are verified **as recorded metadata** at preparation;
physical checkpoint access is not claimed.

To create independent downloadable private ZIPs with per-file checksums:

```bash
.venv/bin/python -m tools.package_task05_review \
  --review-package runs/task05-semantic-review-v1 \
  --output runs/task05-review-downloads-v1
```

Each ZIP contains only its own blank CSV/RTL HTML, Arabic reviewer protocol
and checksums. Transfer these privately and keep durable copies outside this
workspace. The text-free public reports include the archive SHA-256 values.

## Independent review protocol

Send each reviewer only their corresponding `reviewer_a.html`/`reviewer_a.csv`
or `reviewer_b.html`/`reviewer_b.csv` and this protocol. Do not send analytical
metadata, gold labels, probabilities, predictions or the other reviewer's work.
Open HTML locally in a browser; it needs no server or internet. It presents RTL
Arabic and never sends data elsewhere. Enter a distinct reviewer ID, complete
each pair and click **حفظ القرارات CSV** frequently. Closing the page loses
unexported work. **استئناف ملف CSV سابق** restores a previous export, verifying
IDs and texts. Files include exact full Arabic text with a UTF-8 BOM.
For spreadsheet editing, import UTF-8 and set all columns to **Text** (including
tweet content), use RTL view, and never save rewritten IDs or strings. Prefer
the HTML interface to avoid spreadsheet formulas and automatic conversions.

Required decisions:

| Column | Allowed values | Question |
|---|---|---|
| semantic_validity | preserved / changed / ambiguous | Is meaning preserved, including entities, scope and emphasis? |
| sentiment_preserved | yes / no / uncertain | Is polarity/intensity of sentiment preserved? |
| negation_present | yes / no / uncertain | Does either text contain actual linguistic negation? |
| negation_affects_sentiment | yes / no / uncertain | Does its scope affect expressed sentiment? |
| linguistic_notes | nonempty free text | Explain the judgment, dialect, irony and ambiguity. |
| reviewer_id | distinct nonempty ID | Who made this independent decision? |
| review_timestamp | ISO 8601 with timezone | When was it last recorded? |

When negation is present, also complete `negation_cue`,
`original_negation_scope` and `perturbed_negation_scope` in natural language.
Describe words/propositions under negation, sentiment targets, negative concord,
idioms and uncertainty. Absence of a lexical cue does not prove absence of
negation. If scope cannot be determined, use uncertainty and explain it. Do not
use model predictions to resolve semantics. A meaning change excludes a case
even if sentiment stays the same; extra emphasis is not automatically harmless.

Two reviewers must finish independently before discussing cases. Import both
files to produce agreement measured **before adjudication** and a separate
disagreement queue:

```bash
.venv/bin/python -m nasn.task05 import-reviews \
  --review-package runs/task05-semantic-review-v1 \
  --reviewer-a /path/to/completed-reviewer-a.csv \
  --reviewer-b /path/to/completed-reviewer-b.csv \
  --output runs/task05-independent-review-audit
```

Blank unstarted assignments are permitted for coverage reporting; started but
incomplete assignments fail. All assigned IDs must remain in each CSV exactly
once. Altered text/hashes, extra model columns, invalid decisions, absent notes,
naive timestamps, contradictory negation judgments and same-reviewer pairs fail.
Actual reviewer independence cannot be proven by a CSV.

Disagreements in any categorical field **or free-text cue/scope** require a third
independent reviewer. Use `adjudication.html`/`adjudication_queue.csv`; consult
both independent rationale records while keeping predictions hidden. Exact
wording differences conservatively trigger adjudication too. Complete all
disagreement rows with an explicit rationale and distinct adjudicator ID. Repeat
the import with the original independent CSVs, the completed adjudication CSV,
`--adjudication /path/to/completed-adjudication.csv` and a new output directory.
The adjudication CSV must retain exactly the disagreement assignment IDs.

Agreement is nominal Cohen kappa and exact agreement per categorical field,
with joint sample count. Constant identical decisions give undefined kappa,
not 1.0. Coverage uses 404 assigned variants; aggregate and original stratified
coverage tables show both reviewers, resolved decisions and confirmed examples.
Free-text scope notes have no automatic numerical agreement score.

## After human review and Task 04 inference finish

**Do not execute this stage while review is incomplete.** The CLI enforces both
reviews on all 404 cases and complete disagreement adjudication. Eligibility is
derived only from explicit human decisions: preserved meaning, yes sentiment,
and non-uncertain negation presence/effect. No approval transfers to unreviewed
variants or entire perturbation rules.

Use the complete original Task 04 `inference/` directory from Drive, including
full registry/evidence and all original provisional predictions; a loose
prediction CSV or incomplete lightweight export is insufficient. Task 05 loads
no model and runs no inference or training:

```bash
MPLBACKEND=Agg .venv/bin/python -m nasn.task05 analyze \
  --preparation /path/to/task04/benchmark \
  --baseline reports/task03b/marbert-v1-20261009T213143Z-be8b8811 \
  --review-package /path/to/task05-review-package \
  --decisions /path/to/completed-import-audit \
  --inference /path/to/task04/inference \
  --checkpoint /path/to/task03b/experiment/best_checkpoint \
  --bootstrap 2000 --output runs/task05-confirmed-new
```

`--checkpoint` verifies every selected checkpoint file against Task 03B hashes
and loads **only its tokenizer** to record full/effective token IDs. Without
that argument, probability and final-prediction comparisons still run, while
token sequence comparison is explicitly unavailable. No random/base tokenizer
fallback exists. CPU suffices for saved-prediction analysis and tokenization.
The original inference probability vectors and selected checkpoint are unchanged.

Outputs include `confirmed_robustness.csv/.tex`, paired tatweel audit and summary
CSV/LaTeX, preserved original Task 04 metrics in a separate file, methodology,
coverage/agreement audit, input/decision/checkpoint/code checksums and run status.
Metrics retain all three classes, report Accuracy/Macro-F1/per-class F1, flips
and drops, and use seed-42 paired original-ID bootstrap CIs. Strata below 30
report counts with unavailable scores. Human negation presence and effects are
separate from lexical heuristics. High-risk near-duplicate sensitivity is retained.
Tatweel clean-versus-1, clean-versus-2 and 1-versus-2 counts distinguish changed
full/effective token sequences, exact probability changes, changes above 1e-6
and prediction flips. Class probability differences are saved per original ID.
Full-population tatweel comparisons remain provisional candidate instability.

The reviewed stratified sample is not a random population sample. Bootstrap
intervals exclude reviewer uncertainty and training-seed variation and are not
simultaneous. Group differences and tokenization associations are descriptive.
Keep private texts/reviewer IDs on Drive; return only vetted aggregate tables,
methodology, coverage and hashes to GitHub. Never commit weights or raw tweets.

The prepared-review status is the actual endpoint until human work is supplied.
