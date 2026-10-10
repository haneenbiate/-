# Task 04: validation error analysis and orthographic feasibility

The supplied Task 03B report archive contains a completed external baseline run.
Its report hashes and prediction-derived metrics have been verified locally.
The archive excludes model weights. Local robustness inference is blocked by the
unavailable selected checkpoint and CUDA GPU; no perturbed metrics are claimed.
Read `reports/task04/methodology.md` for actual results and evidence limitations.
No NASN method is implemented. No training or checkpoint reselection occurs.

For Colab `CalledProcessError` and failed analysis retries, follow
[the diagnostic and recovery instructions](task04_troubleshooting.md).

## Reproduce local preparation and actual saved-prediction analysis

Use Python **3.12.14**, as required by Task 02. In the project checkout:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-task04.txt
.venv/bin/python -m pip install -r requirements-notebook.txt  # existing notebook schema tests
.venv/bin/python -m pip check
# Obtain only the training CSV if absent. Do not fetch, analyze or tokenize test text.
.venv/bin/python - <<'PY'
from pathlib import Path
from urllib.request import urlopen
from arsarcasm_audit.audit import COMMIT, HASHES, sha256
p = Path('data/raw/training_data.csv')
if not p.exists():
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(urlopen(f'https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/{COMMIT}/ArSarcasm-v2/training_data.csv').read())
assert sha256(p) == HASHES[p.name]
PY
.venv/bin/python -m nasn.task04 prepare --output runs/task04-prepared-new
MPLCONFIGDIR=/tmp/nasn-matplotlib XDG_CACHE_HOME=/tmp/nasn-cache \
  .venv/bin/python -m nasn.task04 analyze \
  --run reports/task03b/marbert-v1-20261009T213143Z-be8b8811 \
  --output runs/task04-analyzed-new
.venv/bin/python -m pytest -q
```

Use a fresh output directory on every command; existing evidence is never
overwritten. The pinned Task 02 metadata, methodology and all registered manifest
hashes are verified, and only official training text is loaded. Validation IDs,
labels and text/group hashes are joined using the original loader. Train/test
overlap checks use registered manifest hashes; the official test CSV is never
opened by Task 04. Expected counts remain 10037/2509/2/3000. The test source hash
is an existing recorded checksum, not a new verification of test CSV bytes.

To import the original uploaded report ZIP into a fresh directory:

```bash
.venv/bin/python -m nasn.task04 import-reports \
  --archive /path/to/marbert-v1-20261009T213143Z-be8b8811-reports.zip \
  --output runs/imported-task03b
```

Only flat allowlisted report filenames are accepted. All archived report bytes
must match `report_checksums.json` before any extraction. Extraction does not
execute content or establish possession of the trained weights. Analysis checks
baseline protocol, numeric/string labels, confidence/probability consistency,
row metadata, manifest/source hashes, per-class results, confusion matrix,
subgroup metrics, near-pair audit and highest-Macro-F1/earliest-tie selection.

## Rules and applicability

`nasn/orthography.py` freezes seed 42 and `orthography-v1`. Every source row has
five manifest entries, including explicitly inapplicable entries:

| Rule | Intensity | Candidate operation | Limitation |
| --- | --- | --- | --- |
| elongation | 1, 2 | Repeat one eligible consonant once or twice | May alter emphasis or lexical meaning |
| tatweel | 1, 2 | Insert one or two U+0640 after a joining letter | Typography is not automatic semantic certification |
| alef_allowlist | 1 | Omit hamza only in explicit whole-word pairs | May introduce ambiguity; contextual review required |

Alef pairs are `إلى→الى`, `أيضا→ايضا`, `أيضاً→ايضاً`, `أحيانا→احيانا`,
`أحياناً→احياناً`. There is no global Alef/hamza mapping, ya/alef-maqsura mapping,
ta-marbuta substitution, stemming or stopword removal. No additional variant has
been validated, so no other transformation is enabled. Future rules require a
new explicit version and semantic audit; do not silently expand this benchmark.

Selection is SHA-256-ranked by seed, original ID, rule and version, independent
of dataframe order. Intensities use the same location for a given row/rule.
Positions are zero-based Python Unicode code-point offsets, with end-exclusive
source/target spans. The manifest logs full original/changed text locally, exact
spans, ambiguity, intensity, source/group/variant IDs, labels and both text hashes.
URLs, handles, hashtags, Latin-attached tokens and marked words are protected.

Lexical negation cues include `لا لم لن ليس ليست لست لسنا ليسوا ما مو مش مب مهو
موش مافي مافيه ماني بدون غير`, common conjunction prefixes and `ما...ش`.
Diacritics/tatweel are ignored only for cue detection, never removed from model
input. Known cues and the following token are protected. A broader edit guard
overprotects words beginning with likely attached negators, including `ما...`
without final `ش`; this guard is not the negation-subgroup definition. Before
accepting a candidate, original cue spans/strings and cue sequence must survive,
and a newly detected cue invalidates the candidate. No negation is inserted,
deleted or reversed intentionally. Heuristics cannot guarantee recognition of
every dialect spelling or infer scope. Human review must reject any unnoticed
negation change. False positives and missed cues remain documented limitations.

## Human semantic review

Full-text `human_review_queue.csv` is generated under ignored `runs/`; it contains
two examples per nonempty rule/intensity × gold class × dialect × sarcasm ×
lexical-negation stratum, or all examples for smaller strata. The public queue
retains IDs/hashes/spans but omits tweet text. Regenerate the full queue to review
it; the public file alone is not review evidence. Review without model predictions.

Retain IDs and hashes, edit only these decision fields, and save a new CSV:

- `review_status`: `pending`, `valid`, `invalid`, or `ambiguous`.
- `reviewer`: attributable reviewer ID for every nonpending decision.
- `rationale`: contextual explanation, including any negation/emphasis concern.

Check original and variant in context for polarity, negation, entities,
intensity/emphasis and lexical interpretation. Use `invalid` for an actual
meaning change and `ambiguous` when equivalence cannot be established. Only a
per-example explicit `valid` decision becomes `confirmed`; decisions never
propagate from a sample to other examples or rules. Duplicate IDs, stale hashes,
unknown statuses or unattributed decisions fail. Human inter-reviewer agreement
and adjudication are future work, not claimed here. All 404 initial sampled
examples remain pending, and all generated changes remain provisional.

## GPU inference using the selected checkpoint

Open [the Task 04 GPU notebook](https://colab.research.google.com/github/haneenbiate/-/blob/main/notebooks/task04_robustness_colab.ipynb).
It is unexecuted and pins Task 04 code to
`bde827f741a625e5b8b4ceaca7c6dd6bc36901a6`. It restores only training text,
mounts the same Drive, checks the existing checkpoint and CUDA, invokes the CLI,
and exports text-free lightweight results after completion. The current notebook
uses the exact Task 03B run ID from the uploaded report; do not switch checkpoints.

The external checkpoint reported by the uploaded run is:

```text
/content/drive/MyDrive/NASN/task03b/marbert-v1-20261009T213143Z-be8b8811/experiment/best_checkpoint
```

Mount the same Drive in a GPU Colab runtime. Use the Task 03B-managed Python
3.12.14 venv and torch 2.8.0+cu126, or repeat its isolated interpreter/dependency
setup. Use this Task 04 checkout, not the older baseline commit, for the CLI;
the original training implementation and Task 02 splits remain unchanged.
For an exact code checkout:

```bash
git clone https://github.com/haneenbiate/-.git nasn-task04
cd nasn-task04
git checkout --detach bde827f741a625e5b8b4ceaca7c6dd6bc36901a6
```

Record and retain the exact Task 04 commit (`git rev-parse HEAD`) and code hashes.
The selected weights must remain exactly those registered in the uploaded report;
never download the base encoder as a replacement for the trained classifier.

```bash
# In the Task 04 repository checkout, with only pinned training CSV restored:
NASN_PY=/content/nasn-marbert-venv/bin/python
NASN_CHECKPOINT=/content/drive/MyDrive/NASN/task03b/marbert-v1-20261009T213143Z-be8b8811/experiment/best_checkpoint
NASN_OUTPUT=/content/drive/MyDrive/NASN/task04/CHOOSE-A-NEW-RUN-ID
"$NASN_PY" -m pip install -r requirements-task04.txt
"$NASN_PY" -m pip check
MPLCONFIGDIR=/tmp/nasn-matplotlib XDG_CACHE_HOME=/tmp/nasn-cache \
  "$NASN_PY" -m nasn.task04 evaluate \
  --run reports/task03b/marbert-v1-20261009T213143Z-be8b8811 \
  --checkpoint "$NASN_CHECKPOINT" --output "$NASN_OUTPUT" \
  --batch-size 4 --bootstrap 2000
# For a separate reviewed run, add --reviews /path/to/completed-human-review.csv
```

The CLI requires the exact registered set of checkpoint/tokenizer files and all
SHA-256 hashes, no missing/unexpected encoder keys, the unchanged three-class
mapping and a CUDA GPU. Clean validation predictions must match the saved baseline
classes exactly and saved probabilities within 1e-5 absolute/relative tolerance.
Any mismatch stops instead of redefining the baseline. No CPU/random/pretrained
base fallback, optimizer, fitting or model save occurs. Lower inference batch
size after OOM in a fresh directory; it does not alter sequence length or weights.

Robustness is summarized both across the validation population (inapplicable
rows use their clean predictions as no-ops) and applicable-only matched IDs.
Provisional population summaries include confirmed and unreviewed noninvalid
candidates conservatively labeled provisional; confirmed-only summaries require
individual human approval. Invalid/ambiguous rows are excluded and counted.
Both scopes are repeated after excluding 112 flagged validation IDs. This is a
sensitivity analysis, not a split change or new checkpoint selection.

Accuracy/Macro-F1/per-class F1 drops are clean minus changed on matched IDs.
Flip, correct-to-wrong, wrong-to-correct and wrong-to-wrong flip rates distinguish
instability from error induction/correction. None establishes semantic equivalence
on provisional candidates. Dialect, sarcasm and lexical-negation strata use a
minimum of 30 examples; absent classes use F1=0 with all three labels retained.
Paired percentile 95% CIs use 2000 bootstrap draws with seed 42, sampling original
IDs jointly for clean and changed outcomes. They condition on this checkpoint,
do not measure training-seed uncertainty or adjust for multiple comparisons,
and may be optimistic for correlated same-text/group records. Token counts,
full/effective token-ID changes and truncation correlate with flips; no causal
interpretation is warranted. CSV/LaTeX tables and PDF/SVG figures are generated
only from actual supplied predictions or completed inference.

## Persist reports without weights or raw tweets

```bash
.venv/bin/python -m nasn.task04 publish \
  --preparation runs/task04-prepared-new --analysis runs/task04-analyzed-new \
  --output runs/task04-public-new
```

This verifies generated artifact registries and exports text-free manifests and
allowlisted aggregate baseline reports. Keep full-text error/review queues and
inference predictions in durable storage. Commit reviewed lightweight outputs to
a new report directory, preserving earlier evidence. Never commit the selected
checkpoint, raw CSVs or an executed notebook containing full tweets/credentials.
