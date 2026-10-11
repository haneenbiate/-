# Task 05 submitted workbook audit

Three uploaded workbooks contain the same 404 assignments. Each original and
perturbed Arabic string, ID, text checksum, rule and intensity is compared with
the verified blank Task 05 template. Excel formulas/macros/multiple sheets are
rejected; typed data is converted to literal string CSV without editing any
decisions, rationale or timestamps. Original workbook bytes are retained in
private storage. Input and output hashes record the conversion. All three files
pass the existing decision-integrity checks. These checks establish structural
consistency, not semantic truth or human review provenance.

The corrected A/B uploads are accepted as human reviews on the user's explicit
declaration that the earlier labels/disclaimer were a typographical/upload error.
This declaration is tied to exact corrected workbook hashes. It is not
independent verification of identities, blinding or actual annotation process.
A retains an `ai_review` worksheet title; this remaining metadata flag is
recorded without overriding the user's declaration. Only two explicit matching
human judgments can establish eligible consensus. The LLM is never an adjudicator.

Filename, reviewer ID, common spreadsheet creator and timestamps cannot
establish or disprove actual human involvement. These are source evidence,
not instructions to execute. The LLM file remains separate; its declared
identifier is not a verified provider/model revision. Prompt and sampling
settings are absent. No automated decision is promoted to human confirmation.

Exact agreement and nominal Cohen kappa are computed on all 404 paired
categorical decisions for A/B, A/LLM and B/LLM, with full contingency counts.
The A/B table is human agreement on the user's declared source roles, without
independent identity/blinding verification. Without a source declaration, it
is only submitted-file agreement. Neither LLM comparison is LLM accuracy against
adjudicated gold. The dominant preserved category can
produce high raw agreement and low/negative kappa: consult contingency tables
and marginal distributions. Repeated variants span 279 original text IDs;
agreement is descriptive at variant level, without independence-based CIs.

Any difference in the four categorical fields or literal cue/scope strings is
flagged for inspection. Scope wording differences may describe the same scope;
they are not automatic proof of linguistic disagreement. No LLM or heuristic
merges them, and no original judgments are overwritten. The blank 213-item
inspection queue supports later human adjudication if A/B provenance is
established. When independent human review has not occurred, the full original
404 assignments still require it; matching A/B judgments do not replace it.
Other submitted source rationales are separate from the blank decision queue,
and LLM suggestions are excluded from the adjudication interface.

Negation presence and sentiment effects are separately tabulated against the
frozen Task 04 lexical heuristic. The heuristic is not gold truth. Uncertainty
is retained, not recoded as no. Saved confidence vectors, tokenizer IDs and
paired perturbed MARBERT predictions are absent locally, so no confirmed
robustness, bootstrap intervals, probability changes or tatweel prediction
statistics are computed. Original baseline and Task 04/05 evidence is preserved.
No model training, checkpoint selection or official test text read/evaluation
occurs. No protocol is relaxed to force completion.

## Actual status

```json
{
  "status": "user_declared_human_reviews_audited_pending_adjudication",
  "queue_n": 404,
  "submitted_a_n": 404,
  "submitted_b_n": 404,
  "submitted_llm_n": 404,
  "categorical_disagreement_n": 150,
  "literal_scope_difference_n": 203,
  "inspection_queue_n": 213,
  "fully_matching_records_n": 191,
  "human_provenance_verified": false,
  "human_confirmed_n": 191,
  "human_review_completed": false,
  "llm_used_for_adjudication": false,
  "robustness_metrics_computed": false,
  "checkpoint_loaded": false,
  "training_performed": false,
  "test_text_read": false,
  "test_evaluated": false,
  "remaining": [
    "Complete explicit third-human adjudication for all flagged cases",
    "Supply LLM model/version/prompt/settings provenance",
    "Supply completed Task 04 perturbed predictions and verified tokenizer access"
  ],
  "human_source_user_declared": true,
  "human_consensus_eligible_n": 191,
  "source_provenance_basis": "user_declaration_not_independent_identity_verification"
}
```

Source flags:

```json
{
  "a": {
    "explicit_ai_assistance_disclaimer_n": 0,
    "explicit_not_independent_human_disclaimer_n": 0,
    "unique_timestamp_n": 1,
    "reviewer_identifier_n": 1,
    "negation_uncertain_n": 1,
    "negation_effect_uncertain_n": 19
  },
  "b": {
    "explicit_ai_assistance_disclaimer_n": 0,
    "explicit_not_independent_human_disclaimer_n": 0,
    "unique_timestamp_n": 1,
    "reviewer_identifier_n": 1,
    "negation_uncertain_n": 0,
    "negation_effect_uncertain_n": 102
  },
  "llm": {
    "explicit_ai_assistance_disclaimer_n": 0,
    "explicit_not_independent_human_disclaimer_n": 0,
    "unique_timestamp_n": 404,
    "reviewer_identifier_n": 1,
    "negation_uncertain_n": 5,
    "negation_effect_uncertain_n": 7
  }
}
```
