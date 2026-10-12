# Task 07 frozen redesign protocol — before new v2 effect analysis

This is an exploratory, post-hoc redesign, not a preregistered novel method.
Task 06 GPU inference evidence is **not yet available locally**. On 2026-10-12
(Baghdad date) the user first reported no execution, then reported completion
and supplied the source notebook URL; the result ZIP is awaiting verification.
No identical-prediction premise is accepted without actual model outputs.
Existing Task 06 character/token evidence and original source rules are inspected.
No gold sentiment, model probabilities or test information selects a transformation.

## Changes from v1, fixed before measuring new v2 effects

1. Replace the twelve-word repetition outcome allowlist and collapse-to-one with
   an explicitly provisional, generic **run length >=3 capped at 2** rule. Two
   characters are retained because double letters can be morphological/spelling
   content. No attempt is made to recover the original word or edit two-character
   runs. This may still change emphasis or meaning and requires human review.
2. Restrict eligible runs to the original Task 04 repeatable consonants. Do not
   change hamza/Alef/ya/ta-marbuta or vowel marks. Abstain from repetition edits
   in words containing diacritics. This deliberately limits coverage, including
   most intensity-1 perturbations; low coverage must not trigger post-hoc rules.
3. Remove v1's NFC/whitespace operations from the new methods to isolate actual
   character edits. Preserve URLs, handles, hashtags, mixed-script/numeric
   substrings and literal recognized negation-marker spans in all new methods.
4. Keep the exact v1 sentiment/phrase safety lexicon, with no dataset-specific
   additions. Protect it from repetition edits. Tatweel outside marker/identifier
   spans can be removed even within a negation context: a broad context window
   should not block a typography operation merely because it is near a cue.
   This is a typography hypothesis, **not human-certified semantic safety**.
5. Only NASN2's repetition operation adds the original bounded lexical context
   guard: cue/possible attached cue plus three Arabic tokens before punctuation.
   NO_CONTEXT applies identical combined rules and marker/word safety lists with
   this context guard disabled. Their difference isolates context abstention,
   not detection of every negator. The guard is not a syntactic scope parser.
6. Compare RAW, TAT (tatweel), REP2 (repetition), NO_CONTEXT (both) and NASN2.
   TAT/REP2/NO_CONTEXT are simple baselines; NO_CONTEXT is also the context ablation.
   All choices are fixed in `configs/nasn_v2.json`; no rule search or retraining.

## Populations, metrics and decision

Use the unchanged Task 02 development/validation partition and original Task 04
applicable variants. Retain 394 user-declared eligible source variants, 9,557
unreviewed provisional variants and 10 reviewed exclusions separately. No new
normalization output inherits semantic approval. Exclusions receive no robustness
metrics. Human identity/independence remains unverified. No fresh inter-annotator
agreement is computed or claimed as independently human-validated.

Report all unique clean development originals as well as matched clean/perturbed
Accuracy, Macro-F1, per-class F1, drops and flips per rule/intensity. Report
near-duplicate sensitivity and negation strata. Use seed 42, 2,000 paired 95%
percentile bootstrap resamples, fixed NEG/NEU/POS and n>=30. For pooled results,
resample original source IDs as clusters containing all their reviewed variants,
not variants as independent records. Repeated exact texts with different source
IDs remain a further dependence limitation. Conditions share bootstrap draws.

The primary exploratory cohort is the 394 user-declared eligible source variants.
Strongest simple baseline means highest perturbed Macro-F1 on this cohort; ties
use TAT, then REP2, then NO_CONTEXT. Publish paired differences against **all**
simple methods and the context ablation. Evidence for improvement requires NASN2's
paired gain CI lower bound above zero against every simple method and a nonnegative
clean Macro-F1 point difference on all 2,509 unique clean development IDs. Pooled
clean/perturbed metrics use the matched variant weighting and are separately
labeled; the clean safety gate uses unique clean IDs. This conservative gate is a project decision,
not a multiplicity-adjusted claim of statistical significance. Semantic review
and independently documented annotation provenance remain separate prerequisites
for trustworthy sentiment-robustness claims.

If the gate fails, recommend abandoning the NASN improvement claim or reframing
the work as a transparent preprocessing/abstention and tokenization audit. Do not
add rules after seeing outcomes. If weights/results are unavailable, the scientific
decision is **undetermined**, never success or failure inferred from token counts.
Do not keep promoting NASN while the evidence is absent.
