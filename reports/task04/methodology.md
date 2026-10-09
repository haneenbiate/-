# Task 04: actual validation errors and orthographic feasibility

Saved clean-prediction analysis and benchmark preparation are complete.
Perturbed inference is **blocked**, not completed: the uploaded report bundle
contains no selected checkpoint bytes, and this workspace has no CUDA GPU.
All orthographic changes are provisional. No human semantic approval, perturbed
metric, robustness drop or prediction flip result is claimed.

## Evidence and checkpoint

The user supplied `marbert-v1-20261009T213143Z-be8b8811-reports.zip`, SHA-256
`8b2b294400df6f80813f2af61b10f8cf053edcaa81cc06553ac6fe638cf0426a`.
All 23 report payload hashes were verified before extraction; the checksum
registry itself is anchored by the ZIP hash. Imported report bytes are preserved
under `reports/task03b/marbert-v1-20261009T213143Z-be8b8811/`, beside the earlier
blocked/preparation evidence, which has not been overwritten.

The external run records baseline commit
`c8dc698780cb06c8838c557f159bdaed7507aeda`, pretrained revision
`88e1fa192dd723cf0b3563500aec46209762eb22`, Tesla T4, torch `2.8.0+cu126`,
CUDA 12.6, Python 3.12.14 and training duration 1212.7425 seconds.
Four epochs ran; epoch 2 has the highest validation Macro-F1, followed by two
nonimproving epochs. All recorded experimental settings match the baseline;
external changes are device and cache location, with microbatch 4/accumulation 4.
The report records checkpoint reload and durable hash checks as successful.
Those are external run evidence, not a local reloading or GPU verification.

The selected classifier's registered `model.safetensors` SHA-256 is
`5c9fd87e2639a967d2c7517d5bb4189a6cbcb3bb477f9998e105682fcc249840`.
Its tokenizer and other five file hashes are preserved in the supplied
`checkpoint_hashes.json` and `evidence.json`. Task 04 has verified the registry's
format and report integrity, **not these checkpoint bytes locally**. The exact
selected checkpoint is reported at:

```text
/content/drive/MyDrive/NASN/task03b/marbert-v1-20261009T213143Z-be8b8811/experiment/best_checkpoint
```

Task 02 metadata/methodology/all registered manifest hashes and official training
CSV checksum were verified. The pinned dataset revision remains
`eb85513e133cfc67e30aca4968fadbe2594fcb42`. Counts remain 10037 train, 2509
validation, 2 quarantined, 3000 test. Registered exact-text/group hashes are
disjoint across active splits. Only original training CSV text is loaded;
Task 04 does not open, tokenize, predict or evaluate the official test CSV.
Its existing source checksum is recorded without rereading its bytes. No source
data, split, checkpoint selection or training implementation was changed.

Validation prediction IDs, string/numeric labels, dialect/sarcasm annotations,
probabilities/confidence, source/manifest hashes, confusion matrix, per-class and
subgroup metrics, near-duplicate audit and epoch selection were cross-checked.
Accuracy, Macro-F1 and weighted F1 were independently recomputed from the supplied
2509 prediction rows. Local code/environment and model/data hash evidence are
machine readable in `evidence.json`, `preparation.json` and `development_validation.json`.

## Actual clean-validation results

| Measure | All 2509 rows | Excluding 112 high-risk rows (2397) |
| --- | ---: | ---: |
| Accuracy | 0.765245 | 0.768878 |
| Macro-F1 | 0.750428 | 0.754229 |
| Weighted F1 | 0.764598 | 0.767988 |

There are **589 classification errors**. The confusion matrix has gold rows and
predicted columns in NEG, NEU, POS order:

| Gold | NEG | NEU | POS |
| --- | ---: | ---: | ---: |
| NEG | 798 | 92 | 33 |
| NEU | 214 | 815 | 120 |
| POS | 58 | 72 | 307 |

NEG support is 923, recall 0.864572, F1 0.800803; most NEG errors go to NEU (92).
NEU support is 1149, recall 0.709312, F1 0.765977; NEU→NEG (214) is the largest
off-diagonal cell. POS support is 437, recall 0.702517, F1 0.684504, with 72
errors to NEU and 58 to NEG. POS has the weakest observed class F1. These are
prediction patterns, not established linguistic or model-internal causes.

| Subgroup | N | Macro-F1 |
| --- | ---: | ---: |
| Sarcasm FALSE | 2049 | 0.732024 |
| Sarcasm TRUE | 460 | 0.506968 |
| Dialect egypt | 537 | 0.703763 |
| Dialect gulf | 124 | 0.729153 |
| Dialect levant | 112 | 0.781653 |
| Dialect msa | 1728 | 0.754069 |
| Dialect magreb | 8 | Unavailable: fewer than 30 rows |
| Lexical negation detected | 594 | 0.683820 |
| No lexical negation detected | 1915 | 0.757411 |

Subgroup Macro-F1 always retains all three classes, including absent classes with
F1=0. Class mixtures and supports differ; differences are descriptive and do not
prove that sarcasm, dialect or negation caused the errors. See
`baseline_subgroups.json` for per-class support and `error_patterns.csv` for
observed gold/predicted/dialect/sarcasm/negation combinations. Full-text error
records and an unannotated qualitative review queue are generated privately by
`analyze`; no human qualitative coding is asserted.

The existing 753 near-pairs include 323 active cross-split pairs, 291 label
conflicts across all memberships and 264 high-risk pairs. Task 03's frozen high
risk rule is active cross-split AND (cosine ≥0.95 OR different sentiment labels).
112 validation rows occur in at least one such pair. Their descriptive Macro-F1
is 0.644597. Removing these IDs only for sensitivity reporting leaves 2397 rows;
no candidate was deleted/merged, no model was retrained, and no checkpoint was
reselected. The modest sensitivity difference is not proof of independence or
absence of leakage. Existing test-related flags are used only as frozen metadata.

Original-ID percentile bootstrap (2000 draws, seed 42) gives all-validation 95%
CIs of [0.748904, 0.781188] for Accuracy and [0.731658, 0.767874] for Macro-F1.
Full per-class/all/sensitivity intervals are in `baseline_confidence_intervals.json`.
These quantify saved clean predictions only. They condition on the checkpoint,
do not include training-seed uncertainty or multiple-comparison correction, and
may understate uncertainty for correlated duplicate/group records.

## Controlled benchmark preparation

`orthography-v1` uses seed 42, source IDs/group IDs, deterministic hash-ranked
single edits and nested intensity locations. Each row has five logged entries;
source/target spans use zero-based, end-exclusive Unicode code-point offsets.
Original text and labels remain intact. Public manifests omit full tweets and
retain IDs/hashes/spans; full-text manifests are reproducibly generated under
ignored `runs/`. Text-free public review files are not semantic-review evidence.

| Rule | Intensity | Validation rows | Applicable changed rows |
| --- | ---: | ---: | ---: |
| elongation | 1 | 2509 | 2474 |
| elongation | 2 | 2509 | 2474 |
| tatweel | 1 | 2509 | 2474 |
| tatweel | 2 | 2509 | 2474 |
| alef_allowlist | 1 | 2509 | 65 |

There are 12545 entries: **9961 provisional changes** and 2584 inapplicable
no-ops. `failure_cases.csv` separates missing Arabic tokens, absent allowlisted
words, protected/attached candidates, unsuitable marked/short tokens and cue-guard
rejections when present. Applicability does not establish meaning preservation.

Elongation repeats one eligible consonant; tatweel inserts U+0640 after a joining
letter. Alef variation is restricted to the five whole-word pairs documented in
`docs/task04.md`; no blanket Arabic substitution is used. No additional unvalidated
variant is enabled. Each change logs an explicit ambiguity: repetition may change
emphasis/lexical meaning; hamza omission may be contextually ambiguous; typography
still requires review.

Known lexical negation cues and the following token are protected. A broader
prefix guard also protects potential attached dialect negators. URLs, handles,
hashtags, Latin-attached strings and marked words are protected. Original cue
strings/spans must survive and detected cues must not be introduced or removed.
These lexical safeguards cannot identify every dialect spelling or establish
scope. Unknown negation or meaning changes must be rejected in human review;
no claim of universal automatic semantic preservation is made.

The review sample contains **404 pending examples**, stratified by rule/intensity,
gold class, dialect, sarcasm and lexical-negation detection: 200 with detected
negation, 204 without. None has been human validated. A per-example `valid`
decision with unchanged identity/text hashes, reviewer and rationale permits
confirmed analysis. Invalid and ambiguous examples are excluded; pending examples
stay provisional. Sample decisions never propagate to unreviewed variants.

## Inference and remaining blocker

The `evaluate` CLI verifies all selected checkpoint/tokenizer bytes before
loading, rejects missing/unexpected weights, requires CUDA, and reproduces saved
clean validation classes and probabilities before perturbation inference.
It uses the original model loader/evaluation helpers and saved official tokenizer,
128-token truncation and unchanged three-class mapping. It never calls a training
loop or evaluates official test. Neither random weights nor the unfinetuned
pretrained MARBERT snapshot can replace the selected classifier.

The local attempt exited with code 2: **Selected trained checkpoint directory
unavailable**. `inference-attempt/run_status.json` preserves this actual failure;
`local_hardware.json` records no CUDA GPU. Thus no perturbed prediction, robustness
drop, flip rate, tokenization association or confirmed meaning-preserving score
has been measured here. Local baseline analysis is successful despite this
separate inference blocker. See `docs/task04.md` for exact commands to run on the
same mounted Drive/GPU, including human-review decisions when available.

Once run, outputs distinguish population-with-noop and matched applicable-only
comparisons; mixed population scores remain conservatively provisional. They
also distinguish confirmed/provisional examples, full/high-risk-excluded subsets,
and negation/dialect/sarcasm strata. Accuracy, Macro-F1, per-class F1, flips,
correct-to-wrong/correction rates and paired original-ID 95% bootstrap CIs are
reported for eligible matched rows. Invalid/ambiguous reviews are excluded and
counted. Full/effective token-ID changes, token-count changes and truncation are
descriptive associations, not causal proof. Small strata are explicitly unavailable.

Actual baseline tables are supplied in CSV/LaTeX and the confusion figure in
vector PDF/SVG. Perturbation tables/figures will be generated only after complete
actual inference. All source code, public manifests, report checksums and exact
reproduction commands are retained; earlier Task 02/03/03B evidence is preserved.
