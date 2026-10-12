# Task 05 completed human semantic review

The user confirmed Baghdad time. All 213 native Excel timestamps are converted
to ISO 8601 with Asia/Baghdad offset +03:00, preserving local clock values.
40 text cells in 20 variants lost only ASCII edge spaces (19 trailing pairs and
one leading pair); the explicit repair restores hash-registered original text.
The source XLSX, raw cells, auxiliary diffs and every repair are retained privately.
No words, internal spaces, negation or human judgments are changed. Final decisions
are recomputed from both independent review files plus explicit third-reviewer
adjudication. A pandas row-index portability bug in expected queue comparison was
fixed by resetting its index; an end-to-end test covers a nonzero conflict index.

All 404 assignments are resolved: 394 eligible, including 184 human-negation
and 210 non-negation variants; 10 excluded (3 ambiguous meaning, 1 changed meaning
with uncertain sentiment, 6 preserved meaning with uncertain negation effects).
Confirmation is per reviewed variant only; no approval transfers to other tweets
or entire perturbation rules. Counts and tables concern the stratified review
sample, not the entire validation partition. Repeated original IDs require paired
original-ID bootstrap per rule/intensity when inference results are available.
Pre-adjudication agreement is retained rather than replaced with final consensus.
Human identities, independence and blinding are user-declared, not authenticated.

LLM agreement with the finalized human reference is reported separately on all
404 assignments, retaining uncertainty and changed/ambiguous categories. Nominal
kappa and raw agreement depend strongly on marginal label prevalence; high raw
agreement need not indicate reliable identification of changed/ambiguous cases.
The LLM's model revision, prompt and settings remain unverified. These are
post-hoc comparisons of provided annotations, not a new LLM annotation experiment
or MARBERT validation/robustness metrics. LLM judgments never adjudicate humans.

The supplied Task 03B ZIP contains original validation predictions, but Task 04
perturbed predictions and full inference evidence have not been supplied locally.
Consequently Accuracy/Macro-F1 drops, flip rates, paired bootstrap intervals and
tatweel token/probability comparisons remain unavailable. No model is trained,
reselected or evaluated on official test text. Model analysis resumes only when
the saved clean/perturbed probabilities, stable IDs and checkpoint/input provenance
are verified. The selected tokenizer is needed for full/effective token-ID
comparison; it is not needed merely to explain annotation counts.
