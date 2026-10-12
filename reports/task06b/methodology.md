# Task 06B NASN prototype feasibility — preparation complete, inference blocked

Five fixed text-only methods B0–B4 are implemented. B0 retains raw input; B1
removes tatweel; B2 additionally applies NFC/whitespace normalization; B3 selectively
removes tatweel/NFC and allowlisted repetition, with shared critical-word protection;
B4 adds a bounded lexical negation-context guard. Gold labels, predictions, review
decisions and test information never choose an edit. Full exact rules, limits and
reproduction commands are in [the Task 06 methodology](../../docs/task06.md).

The exact selected tokenizer bytes are verified. Actual character/token auditing
covers 2,509 clean validation originals and 9,961 applicable Task 04 variants for
all five strategies: 62,350 records. Reversible private source/target logs recover
every input exactly. Public IDs/hashes/token counts are retained without full
text or token IDs. Post-normalization clean duplicate groups/label conflicts are
reported as an audit, never merged or used to select transformations.

394 reviewed source variants remain conditionally eligible; 9,557 source variants
are unreviewed provisional, and 10 reviewed exclusions are kept separate with no
robustness metrics. Reviewer independence is unverified in Task 06A. New normalized
clean/perturbed strings have no human semantic validation, and a private blank
review queue is prepared. Original review approval never propagates to new text.

Comparison tables show actual historical B0 saved predictions on the two eligible
reporting cohorts. Paired seed-42 intervals use 2,000 original-ID resamples within
rule/intensity, fixed NEG/NEU/POS labels and minimum n=30. B1–B4 scores remain
unavailable. The actual local CLI failed because the selected trained checkpoint
directory is unavailable; CUDA is also unavailable. No substitute encoder or
random classifier is used. The recorded epoch-2 model hash and durable external
path are in blocker_report.json. No NASN improvement is claimed.

The generated Colab notebook pins the immutable implementation, verifies GPU/VRAM,
CUDA kernels, every selected model/tokenizer/config file and B0 clean baseline
predictions before metric reporting. It performs no training or checkpoint
selection. Outputs/checksums remain durable on Drive, with a lightweight report
ZIP. The notebook is prepared and syntax-tested locally, not executed in Colab
here. A completed verified run and semantic review are remaining external steps.

Limits: post-hoc development on previously analyzed validation data, unverified
human provenance, prevalence-sensitive Kappa, nonrandom/small review strata,
subjective negation scope, uncertain-case exclusions, duplicate/near-duplicate
risks, finite lexicons and scope-window false positives/negatives. Tokenization
changes are descriptive correlations, not proof of semantic or causal effects.
There is no automatic winning strategy, parameter tuning, official test access,
new trained model or fabricated performance result. Prior evidence is unchanged.
