# Task 06A annotation quality audit

Agreement is recalculated from the original A/B decisions, never from consensus
or adjudicated labels. All 404 paired assignments are complete. Four categorical
dimensions use unweighted nominal Cohen's Kappa, observed agreement and explicit
uncertain categories. Full contingency tables expose prevalence imbalance.
Kappa is undefined for identical constant marginals; free-text cues/scopes and
rationales use literal equality only. Wording differences do not establish
semantic disagreement. Administrative IDs/timestamps are not annotation labels.

213 assignments have a conflict: 150 categorical and 63 additional cue/scope
wording-only cases. Categories are exclusive; dimension counts overlap. Explicit
third-role decisions resolve all 213; 191 nonconflicting cases retain the original
categorical and cue/scope consensus. The per-case resolution table records A/B/
final categories, literal cue/scope hashes, which reviewer (if either) the final
decision matches, and a hash of the unedited adjudicator rationale. Private
original rationale/cue/scope cells remain available in preserved intake records;
their meaning is not guessed from hashes. Adjudicators can choose different
categories or wording from both reviewers. This describes recorded resolutions,
not independent evidence that every resolution is linguistically correct.

## Provenance and limits of the 394 eligible variants

The user declared corrected A/B and adjudication files to be human reviews.
Distinct recorded identifiers are present, but two independent people, isolated
collection, blinding, qualifications and actual assistance cannot be established.
Provenance is explicitly UNVERIFIED. Each A/B file has one shared timestamp across
404 rows, compatible with bulk export rather than individual activity evidence.
The earlier AI-related worksheet title and shared creator metadata are recorded
as inspection flags, not proof of automated or human origin. No identities are
fabricated and LLM decisions are never used to adjudicate this audit.

394 eligible variants over 274 original IDs are conditional on user-declared
human decisions, not independently certified meaning preservation. Ten exclusions
remain unchanged. High raw semantic/sentiment agreement is dominated by preserved/
yes marginals; semantic-validity Kappa is approximately -0.009, sentiment Kappa
0.253, negation presence 0.837 and sentiment effect 0.459. Rare changed/ambiguous
cases and substantial scope/effect disagreements warrant caution. The stratified
nonrandom sample, repeated source IDs, uncertainty exclusions, subjective scope
judgments and cross-split near-duplicate flags limit population inference. No
retroactive reclassification or automatic confirmation is performed. More human
review is needed to authenticate provenance and validate new normalization outputs.

Reproduce from the published original categorical records and exact free-text
hashes with `python -m nasn.annotation_quality audit --inputs reports/task06a
--output runs/task06a-recomputed`. Full original Arabic review cells are private;
this public reproduction verifies counts, Kappa and literal resolution integrity,
not the semantic content or independence of private human judgments.
