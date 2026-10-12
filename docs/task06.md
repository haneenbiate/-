# Task 06: annotation quality and a fixed NASN prototype

Task 06A is completed from the **original** A/B decisions, not final adjudication.
Task 06B character/token preparation is completed using the exact selected MARBERT
tokenizer. New B1–B4 classifier comparisons are **blocked by unavailable selected
trained weights and a CUDA GPU locally**. The prepared Colab notebook uses the
existing epoch-2 checkpoint on Drive and performs inference only. No improvement
claim or invented new classifier metric is made.

## Annotation quality

| Original dimension | Paired n | Observed agreement | Cohen's Kappa |
|---|---:|---:|---:|
| Semantic validity | 404 | 94.55% | -0.0091 |
| Sentiment preserved | 404 | 94.80% | 0.2529 |
| Negation present | 404 | 91.83% | 0.8367 |
| Negation affects sentiment | 404 | 66.83% | 0.4590 |

The categorical disagreements number 22, 21, 33 and 134 respectively; these
counts overlap. Literal cue differences occur in 118 assignments and original/
perturbed scope wording differences in 203 each. These free-text equality
statistics are **not semantic scope agreement**; no Kappa is applied to free text.
Independent rationale wording has zero literal equality and is not a correctness
measure. Administrative identity/time fields are not annotation dimensions.

The 213 conflict assignments comprise 150 categorical and 63 additional literal
cue/scope-only conflicts. All have an explicit third-role adjudication record;
142 categorical-conflict and 61 wording-only cases are eligible, while eight and
two are excluded respectively. The 191 nonconflicting assignments retain the
original categorical/cue/scope consensus. Each adjudicated dimension is audited
as matching A, B, both, or neither. For the **134 negation-effect disagreements**,
110 final decisions match A, 19 B, and five neither. Adjudication can also revise
an originally agreed dimension in a case conflicted on another dimension. The
unaltered rationale hashes and exact per-case resolution record are published;
the original Arabic private notes remain preserved, not machine-interpreted.

Both A/B files contain 404 completed assignments and distinct recorded identifiers.
Each has one shared timestamp. No authenticated distinct people, isolated
submission history, blinding evidence, qualifications or verified assistance
disclosure is available. **Human independence and identity provenance remain
unverified.** The user declared the corrected files to be human reviews. The
AI-related A worksheet title and shared creator metadata are inspection flags,
not proof of either origin. No identity is invented or LLM decision used to
adjudicate. The 394 eligible variants over 274 originals are therefore conditional
on user-declared human decisions. High raw semantic agreement masks extreme
category prevalence imbalance and near-zero Kappa. Final agreement must never be
reported as independent-review reliability.

Full results: [Task 06A report](../reports/task06a/methodology.md),
[agreement](../reports/task06a/agreement.csv),
[213 resolution records](../reports/task06a/disagreement_resolutions.csv),
[source provenance](../reports/task06a/source_provenance.json).

## Five fixed pipelines

The request calls these four strategies but enumerates B0–B4; all **five** are
implemented. Rules are in [the versioned configuration](../configs/nasn_prototype_v1.json).
The prototype is a heuristic research implementation, not a trained NASN model.

| Strategy | Exact behavior |
|---|---|
| B0 | Byte-identical original string to the official tokenizer; max length 128. |
| B1 | Remove U+0640 tatweel globally, and nothing else. |
| B2 | B1 plus Unicode NFC and collapse Unicode whitespace to one ASCII space, stripping edge whitespace. No letter mapping. |
| B3 | For eligible standalone Arabic tokens only, remove tatweel and use NFC; collapse runs of three or more identical Arabic letters to one **only if the complete resulting word is in the fixed neutral-word allowlist**. Preserve other tokens. |
| B4 | B3 plus conservative lexical negation guards covering the cue/possible attached cue and the next three Arabic tokens, stopping at a listed punctuation/newline boundary. Guarded token spans remain byte-identical. |

B3/B4 share exact lexical protection for negation words, a small fixed sentiment
lexicon and explicit multiword expressions, URLs, mentions, hashtags, and Arabic
substrings attached to Latin/digits/underscore. This common safety list is not
scope inference: **only B4** detects possible attached cues and protects following
context. Guard matching tolerates tatweel/marks, repeated inspection letters and
up to two conjunction prefixes without modifying source text. Broad `ما/مو/مش/
مب/ليس/لم/لن/لا/غير/بدون` prefixes intentionally overprotect possible dialect cues.
This causes false positives and misses other negation/sentiment forms. Punctuation
and a three-token window are a scope proxy, not a validated syntactic parser.

Fixed neutral repeat outcomes: اليوم، الوقت، الناس، البيت، الشارع، كتاب، مكان،
صورة، موقع، برنامج، فيلم، كلام. Runs of two letters remain unchanged. No Alef/
hamza folding, ya/alef-maqsura substitution, ta-marbuta mapping, diacritic deletion,
stemming, stopword removal, case rules or negator insertion/deletion is permitted.
NFC composes canonically equivalent representations; it does not collapse hamza
distinctions. B1/B2 can remove decorative tatweel inside cues, while their lexical
negator sequence is asserted unchanged. B3/B4 preserve recognized cue spellings
and sensitive expressions verbatim. An incomplete lexical list is not universal
sentiment protection. New normalized outputs remain provisional.

Every decision receives **only the text, strategy and fixed configuration**.
Gold classes, saved predictions, confidence, review outcomes, corpus frequencies,
test data and metrics cannot select an edit. Human review determines evaluation
membership afterward, not normalization. Threshold 3, repeat target 1, guard
window 3, allowlists, seed 42, NFC, sequence length 128 and fixed-label metrics
are recorded. There is no parameter search, retraining or checkpoint selection.
The design is post-hoc and informed by the earlier study; it is not preregistered
or a blinded held-out experiment.

Each private edit log stores source/target offsets and spans, rule, input/output
hashes, guards and versioned method. Every output can be restored exactly **with
its edit log**; deleted characters cannot be recovered from output text alone.
All 62,350 strategy/input records passed character-level inverse checks and
actual selected-tokenizer auditing. Full/effective token hashes and counts show
changes outside/inside the 128-token context. Token equality/differences are
descriptive and cannot prove sentiment equivalence or causal errors.

Actual guard-ablation counts: B3 and B4 differ in 20 clean originals and 623
applicable perturbed records, including effective token sequences. These are
input differences, not performance changes. The clean partition retains 12 exact
duplicate groups covering 24 rows, including four conflicting-label groups, in
each strategy; no records are merged. No exact or canonical overlap with active
training hashes was found in any strategy/condition. This post-hoc hash audit
does not choose normalization edits. Original-ID bootstrap does not cluster
separate IDs carrying the same text, which can understate uncertainty.

## Development comparison and status

Only Task 02's 2,509 validation originals and 9,961 applicable original Task 04
perturbations are considered. No-ops are excluded consistently from per-rule
perturbation comparisons. The reviewed subset is **394 source variants**, and
the disjoint unreviewed provisional subset contains **9,557**. The **10 reviewed
exclusions** remain a separate excluded cohort with no robustness metrics. These memberships
do not certify any new normalized text. A new private review queue covers changed
normalized clean and perturbed examples with empty decisions. Nothing is marked
human-validated automatically.

The [comparison table](../reports/task06b/comparison_table.csv) includes actual
historical B0 saved predictions, recalculated on these disjoint cohorts with
paired intervals. **Every B1–B4 metric is unavailable**, rather than zero or a
copied B0 score. Character/token counts are available independently of weights.
The [blocker](../reports/task06b/blocker_report.json) records the real failed local
run. The selected trained classifier is not an official pretrained encoder and
cannot be replaced by a newly downloaded/random model.

The GPU CLI verifies every selected model/tokenizer/config file against Task 03B,
requires no missing/mismatched/unexpected weights, seed 42, float32 and 128 tokens,
then checks B0 clean labels/probabilities against the original baseline (exact
labels, probabilities within atol=rtol=1e-5). Only then are B0–B4 metrics computed.
Exact normalized strings are deduplicated for inference; logits do not depend on
the unused supervised loss computed by the unchanged baseline evaluation helper.
Gold labels are used for reporting only. No fitted model or normalization choice
is made from them.

Per strategy/rule/intensity, matched clean/perturbed Accuracy, fixed NEG/NEU/POS
Macro-F1 and per-class F1, drops and flip rates receive seed-42 95% percentile
intervals from 2,000 paired original-ID resamples. One perturbation per source
exists in each rule/intensity; variants are never pooled as independent observations.
Paired B1–B4 minus B0 uncertainty also resamples the same source IDs for all four
conditions. Tables separate human-negation/lexical-heuristic strata and high-risk
near-duplicate sensitivity; minimum n=30 is enforced (reviewed Alef n=24 unavailable).
There is no automatic winner or multiplicity-adjusted confirmatory significance
claim. Clean performance and perturbation robustness must both be considered.

## Reproduction

Use the existing Python 3.12.14 `.venv` and pinned dependencies. Set the backend
for subprocesses to avoid inherited Colab inline-backend configuration.

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/nasn-matplotlib
export XDG_CACHE_HOME=/tmp/nasn-cache

# Public annotation counts/Kappa/resolution replay; original decisions, no model.
.venv/bin/python -m nasn.annotation_quality audit \
  --inputs reports/task06a --output runs/task06a-recomputed

# Optional private re-export from preserved original full Arabic decisions.
.venv/bin/python -m nasn.annotation_quality export \
  --decisions runs/task05-adjudication-final-baghdad/decisions \
  --review-package runs/task05-semantic-review-v1 \
  --output runs/task06a-originals-new

# Explicit text restoration compares existing Task 04 hashes/IDs; never resplits.
.venv/bin/python -m nasn.task06 restore-task04 \
  --output runs/task06-task04-restored-new
.venv/bin/python -m nasn.task06 prepare \
  --preparation runs/task06-task04-restored-new \
  --tokenizer-directory runs/task05-selected-tokenizer-verified \
  --output runs/task06b-prepared-new

# Requires the SAME selected trained checkpoint and CUDA. Inference only.
.venv/bin/python -m nasn.task06 run \
  --prepared runs/task06b-prepared-new \
  --checkpoint /path/to/verified/experiment/best_checkpoint \
  --batch-size 8 --bootstrap 2000 --output runs/task06b-gpu-new
.venv/bin/python -m pytest -q
```

Tokenizer-only restoration is documented in [Task 05 reproduction](task05_robustness_results.md).
The required pinned official training CSV may be explicitly fetched from
`https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/eb85513e133cfc67e30aca4968fadbe2594fcb42/ArSarcasm-v2/training_data.csv`
and must match SHA256
`1da727eb763459f436c4e61f52c6a0bc63d5300af8fcc99b5675ae4616dae04a`.
Do not download/open test text for this task. Missing manifests/checksums cause
an explicit failure; no silent source replacement or split reconstruction occurs.

The prepared [Colab notebook](../notebooks/task06_nasn_colab.ipynb) pins the exact
Task 06 implementation commit in its first code cell. Select GPU, run all cells
in order, retain the default recorded Drive checkpoint path or point to an exact
verified copy, and upload its final lightweight report ZIP for verification. It
records GPU/VRAM, actual CUDA kernel check, dependencies, duration, hashes and
inference batch size. On OOM use a new output directory and reduce **inference**
batch size only; preserve failed evidence. The notebook is generated and syntax-
tested locally, **not executed in Colab by this environment**.

The public preparation-only publication can be reproduced after importing the
original Task 04 report ZIP as documented in Task 05, and making an explicitly
blocked local attempt (it returns exit 2):

```bash
.venv/bin/python -m nasn.task06 run \
  --prepared runs/task06b-prepared-new \
  --checkpoint /content/drive/MyDrive/NASN/task03b/marbert-v1-20261009T213143Z-be8b8811/experiment/best_checkpoint \
  --output runs/task06b-blocked-new
# On this local machine the verified Drive checkpoint path is unavailable.
# This preparation-only publisher requires that specific recorded blocker.
.venv/bin/python -m tools.publish_task06 \
  --prepared runs/task06b-prepared-new \
  --preparation runs/task06-task04-restored-new \
  --inference runs/task04-imported-20261010T083120Z/inference \
  --attempt runs/task06b-blocked-new --output runs/task06b-public-new
```

The publisher retains original evidence and publishes text-free manifests,
tables and hashes. Full raw/normalized tweets, reversible logs, blank review
queues, annotator notes/identities and model weights stay private in ignored
`runs/` or Drive. Upload the notebook ZIP, not checkpoint files. Before returning
reports to GitHub, verify all archive checksums and source/model identities,
then publish only metrics/provenance and numeric predictions.

## Limitations and remaining work

Unverified reviewer independence, marginal-prevalence effects, subjective scope,
small/nonrandom reviewed strata, repeated originals, uncertain-case exclusions,
and flagged near-duplicate risk limit the 394-case reference. A new normalization
review is needed for both clean and perturbed outputs; rule transparency, inverse
tests and token identity are not human semantic validation. B3/B4's lexicons and
bounded guards miss dialects, implicit negation, sarcasm and sentiment-sensitive
forms, and can overprotect positive uses of ambiguous `ما/غير`.

The same previously analyzed validation data now serves development; future rule
revisions must be recorded as new protocols, with no retroactive selection claim.
No new B1–B4 classifier run has finished locally. Results cannot establish NASN
improvement until the verified checkpoint run completes, paired uncertainty is
examined, and new normalized examples receive sufficient semantic review. No
official test data or test labels are used or modified, no NASN model is trained,
and no numerical result is fabricated.
