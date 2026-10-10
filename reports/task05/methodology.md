# Task 05 semantic validation and negation scope

The original Task 04 404-item stratified queue is fixed. Original and perturbed
Arabic strings, stable IDs and SHA-256 checksums are retained exactly. Review
packages omit gold labels, model predictions, confidences and lexical negation
flags. All decision cells start blank. The offline interface presents text in
RTL, saves a UTF-8 BOM CSV and supports restoring a previous CSV. CSVs are the
authoritative records; do not edit identifiers or text. Store full-text packages
and human identities in durable private storage, separate from public reports.

Two Arabic-speaking reviewers independently assess each pair before discussing
decisions. Judge meaning and sentiment polarity, emphasis, entities, irony and
dialect. Identify negation cues and describe their scope and sentiment effects
in both texts; a lexical cue is not proof of scope. Record uncertainty rather
than guessing. Notes and timezone-aware timestamps are required. Present
negation requires cue and both scope descriptions. Disagreements in any
categorical decision or scope description go to a third independent reviewer.
The adjudicator records an explicit decision and rationale after inspecting
both original judgments in the separate adjudication audit files. No model
results are revealed during review. Do not use the model as a semantic judge.

Agreement is measured before adjudication for each categorical field: exact
agreement and nominal Cohen kappa on jointly completed rows. Kappa is unavailable
with no pairs or expected agreement of one. Coverage denominators are always
404 assigned variants, with stratum coverage separately reported. Free-text
scope differences require adjudication but do not receive numerical kappa.
No automated sentence or token heuristic creates human decisions. Eligibility
requires two agreeing explicit human decisions, or third-reviewer adjudication:
meaning preserved, sentiment yes, negation presence and effects not uncertain.
Changed, ambiguous, unresolved and unreviewed cases are excluded from confirmed
meaning-preserving analysis. Even sentiment-stable meaning changes are excluded.

Analysis is gated until all 404 assignments have both independent reviews and
all disagreements are resolved. It reads saved predictions only; no fitting,
checkpoint selection or official test text access occurs. Original Task 04
metrics are copied byte-for-byte into a separate file with their original status.
Confirmed metrics are matched clean/perturbed scores by rule and intensity,
including a sensitivity subset excluding Task 04 high-risk near duplicates.
Negation presence and sentiment effects use human judgments; lexical heuristic
groups are separately labeled. Subgroups below 30 original IDs report sample
size and unavailable metrics. Macro-F1 uses fixed NEG/NEU/POS, absent-class F1=0.
95% percentile CIs use 2,000 paired resamples by original text ID, seed 42;
one variant per ID in each stratum. Intervals condition on this checkpoint,
selected review sample and decisions, exclude training-seed uncertainty and
are not simultaneous. Group comparisons are descriptive; overlapping intervals
or point differences alone do not establish a difference between populations.

Tatweel comparisons pair intensities 1 and 2 on identical original IDs and
report clean-versus-1, clean-versus-2 and 1-versus-2. Probability differences
include exact numeric inequality and maximum absolute difference above 1e-6
to separate tiny roundoff from larger changes. Full token-ID sequences cannot
be inferred from lengths or changed-versus-clean flags. With a byte-verified
selected checkpoint tokenizer, effective IDs (special tokens, truncation at
128) are recomputed without loading or changing the model and saved for audit.
Without tokenizer access that comparison is explicitly unavailable. Counts on
the full candidate population remain provisional instability, not confirmed
semantic robustness. Tokenization associations do not establish causality.

This nonrandom stratified 404-variant sample covers at most those reviewed
examples; approval is never propagated to other variants or whole rules.
Dialect and irony may defeat simple scope judgments. Review identities are
self-reported audit records, not cryptographic proof of independent reviewers.
Checksums detect altered bytes but are not a signature of provenance.

## Actual execution status

```json
{
  "status": "prepared_review",
  "queue_n": 404,
  "candidate_n": 12545,
  "applicable_n": 9961,
  "reviewer_a_completed": 0,
  "reviewer_b_completed": 0,
  "confirmed_n": 0,
  "agreement": null,
  "robustness_metrics": null,
  "tatweel_comparison": null,
  "human_review_completed": false,
  "training_performed": false,
  "test_evaluated": false,
  "remaining": [
    "Two independent human reviews and adjudication",
    "Completed Task 04 inference evidence and paired predictions",
    "Verified tokenizer/checkpoint access for intensity-1 versus intensity-2 token comparison"
  ]
}
```
