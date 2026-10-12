# Task 05 confirmed orthographic robustness results

## Evidence and execution

The supplied Task 04 ZIP reports a completed Tesla T4 inference run lasting
264.5467 seconds. All 39 registered payload hashes match; 40 original files
(including the checksum registry) are retained unchanged. A derived flat
inference registry references the exact exported bytes. No CSV is reserialized
during import. Task 02 source/manifests, all 2,509 clean prediction IDs/golds,
the selected epoch-2 checkpoint registry and paired probabilities are verified
against the completed Task 03B evidence. All 12,545 variant IDs/text hashes map
to deterministic original candidates: 9,961 applicable and 2,584 no-ops.
The recorded test checksum is metadata only; official test text is never read.

The model weight bytes are not in the archive or local workspace. Weight
verification/reload belongs to the supplied completed external run; this
analysis checks its recorded selected hashes and clean predictions. No local
model reload, new inference, fitting, checkpoint selection or test evaluation
is claimed. Selected model SHA-256:
`5c9fd87e2639a967d2c7517d5bb4189a6cbcb3bb477f9998e105682fcc249840`.

Four selected tokenizer files are independently byte-verified. The pinned
official small snapshot files were already available locally. Loading its
official tokenizer and applying the baseline's declared truncation=128 and
max-length padding restores its saved backend serialization state. The
synthetic probe sets that state only and is not research data. Every saved file,
including tokenizer.json, must match the selected checkpoint hashes exactly;
no alternative tokenizer is accepted. No model weights are loaded. All recorded
token counts, full/effective change flags and truncation flags reproduce across
all 12,545 candidates, independently checking exported tokenization evidence.

## Confirmed versus provisional populations

All 404 human assignments are resolved. Only 394 eligible reviewed variants are
confirmed (184 human-negation, 210 non-negation); 10 changed/ambiguous/uncertain
variants remain excluded. Human identities/blinding are user-declared, not
independently authenticated. Eligibility never propagates to other variants.
Each rule/intensity compares clean and perturbed predictions on its own matched
original IDs. Different rows have different review cohorts and cannot be read
as a causal intensity trend. The Alef cohort has n=24, below the predeclared
minimum of 30: metrics and intervals are explicitly unavailable. Original Task
04 provisional metrics remain byte-identical in task04_original_metrics.csv.

Seed-42 percentile 95% intervals use 2,000 paired bootstrap resamples by original
row ID within each stratum. Accuracy, fixed NEG/NEU/POS Macro-F1 and per-class F1,
clean-minus-perturbed drops and flip/error-transition rates are saved. Fixed
labels and absent-class F1=0 are preserved. Intervals condition on this checkpoint
and nonrandom stratified review sample, exclude training/reviewer uncertainty,
and are not simultaneous. Empirical zero-event bootstrap intervals can degenerate
to [0,0]; they do not prove zero risk in unseen text. Minimum n=30 is enforced
for reported metric strata. No primary Macro-F1 drop interval excludes zero;
the signed point changes do not establish a reliable direction of population
performance change. Individual prediction flips are directly observed.

## Negation and near-duplicate sensitivity

Human negation presence and whether it affects sentiment are separate subgroup
fields. The lexical cue heuristic is separately labeled and is not semantic
ground truth. Subgroup tables show counts and paired intervals; class/dialect
composition and small samples confound causal interpretations of group
differences. The frozen high-risk cross-split near-duplicate flags are used only
for a separately reported sensitivity subset; no source row, split or model is
changed, deleted or merged. Flagged pairs still require inspection. Ambiguous
semantic transformations are excluded from confirmed results, so provisional
instability remains distinct from confirmed meaning-preserving robustness.

## Tatweel intensity feasibility

Within all 2,474 applicable original IDs, intensity 1 and 2 have identical full
and effective token sequences and identical final predictions. One probability
vector differs at exact numeric precision, none differs above 1e-6. Clean-versus-
either intensity changes token sequences/probability vectors for 2,474 IDs and
final predictions for 176 IDs. These full-population counts remain provisional
because most candidates lack human review. Source token IDs/probability details
remain in ignored private analysis artifacts, not public raw-tweet exports.

The valid intensity comparison on human-confirmed examples uses the intersection
of both reviewed intensities: 35 original IDs, or 31 after high-risk exclusion.
Both comparisons observe zero prediction flips and zero Macro-F1 difference
between intensity 1 and 2. The separately sampled tatweel-1 and tatweel-2 cohorts
(94 and 93 variants) therefore must not be used to claim a dose effect. On this
benchmark the two character-level intensities do not yield different token-level
inputs. Tokenization associations are descriptive, not causal proof of errors.

All human/model evidence and original reports are preserved. No NASN method is
implemented. The remaining limits are small/nonrandom review cohorts, subjective
human scope judgments, unresolved near-duplicate risks and unavailable local
weight bytes. Results cover validation only and are not official test results.
