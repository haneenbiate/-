# NASN — MARBERT Baseline v1.0 (Task 03)

Status: **blocked_or_failed**. Official test evaluated: **false**.

## Inputs and preflight

Use existing Task 02 manifests only. Never reconstruct splits, refit split selection,
clean text, drop flags, or merge duplicates. Source commit is
`eb85513e133cfc67e30aca4968fadbe2594fcb42`. Counts: train 10,037;
validation 2,509; quarantine 2; official test 3,000. Source and every registered
Task 02 CSV checksum, metadata/methodology hashes, runtime versions, stable row
IDs, raw text hashes, group IDs, labels, exhaustive row coverage, complete quarantine,
and exact/canonical disjointness are checked. See preflight.json for measured results.
Official test text/labels are inspected only for the requested data-quality audit;
they are not provided to tokenization, training, prediction or metric computation.

Dataset sentiment strings are NEG, NEU and POS. Task 02 records no numerical
model label IDs. This baseline explicitly freezes NEG=0, NEU=1, POS=2 in config,
model label2id/id2label and prediction artifacts; it does not claim these numerical
IDs are an official dataset convention.

## Near-duplicate concerns

Audited pairs: 753.
Active cross-split pairs: 323.
Sentiment-conflicting pairs across all memberships: 291.
High-risk manual review pairs: 264.
High risk is frozen as active cross-split AND (cosine >=0.95 OR label conflict).
near_duplicate_audit.csv records membership, score, both labels and reasons;
near_duplicate_summary.csv groups memberships. manual_review_queue.csv is a queue,
not a claim that humans reviewed these candidates. No record is removed or merged.
Similarity is lexical, not semantic equivalence. Canonical disjointness alone does
not guarantee independent text. Test-label conflicts are descriptive and cannot
change model settings, stopping or checkpoint selection.

## Model and fitting

Official model/tokenizer: UBC-NLP/MARBERT; immutable revision is resolved and saved
before loading. Current resolved revision: 88e1fa192dd723cf0b3563500aec46209762eb22.
No fallback encoder or randomly initialized full model is permitted. The standard
three-class classifier is initialized under seed 42; the standard BERT pooler may
also require initialization because the official checkpoint is BertForMaskedLM.
Actual missing/unexpected weight keys are saved after successful loading; missing
encoder parameters fail explicitly. Raw tweet strings go
directly to the official tokenizer; its own behavior is retained, with padding
and truncation to 128 tokens. No Arabic normalization, stemming, stopword removal,
negation rules or canonicalization is applied to model inputs.

Seed 42; AdamW lr 2e-5, weight decay 0.01 on all trainable parameters,
betas (0.9,0.999), epsilon 1e-8; effective batch 16, microbatch 4,
accumulation 4; at most 5 epochs; float32;
CPU feasibility limit: 1800 seconds projected
for the maximum epoch count, based on one backward-only effective batch with no
optimizer update and restored Torch RNG. This operational limit can be explicitly
raised; it does not change model hyperparameters. If weights cannot be downloaded,
the compute probe is unrun. The probe is not a training experiment or metric.
constant learning rate, no warmup; gradient clipping norm 1.0. A final partial
batch uses its actual example count to weight microbatch gradients. Training rows
are shuffled with a seeded generator; validation is in saved order. No class
weights, augmentation, test evaluation or new hyperparameter search.

At each epoch end, compute validation Macro-F1 over all three fixed classes.
Select strictly highest Macro-F1; ties retain the earlier checkpoint. Early stopping
after two consecutive nonimproving epochs. Zero-division precision/recall/F1 is 0.
All random generators are seeded; deterministic Torch kernels are required. Results
can still differ across hardware, library versions or official tokenizer revisions;
versions and revision are recorded. Checkpoint reload is verified on raw validation
text only after a successful fit. A failed reload prevents a completed-run claim.

## Evaluation and provenance

Successful runs save validation accuracy, Macro-F1, weighted F1, three-class
precision/recall/F1/support, a 3x3 confusion matrix (gold rows/predicted columns),
per-example gold/prediction IDs and labels, maximum softmax confidence and all
three probabilities. Confidence is uncalibrated. Epoch history records loss,
validation metrics and elapsed time. Sarcasm/dialect subgroups require n>=30;
smaller groups are explicitly uncomputed. Subgroup Macro-F1 uses fixed three-class
support; missing classes may depress it and are reported. These are descriptive,
not new checkpoint-selection criteria.

Training config, manifest/source/checkpoint hashes, software versions, device,
training duration and peak GPU allocated/reserved memory (null on CPU) are persisted.
Best checkpoint: unavailable — no completed model training.
Checkpoint reload verified: False.
Training duration seconds: None.
Blocker/failure: ProxyError: (MaxRetryError("HTTPSConnectionPool(host='us.aws.cdn.hf.co', port=443): Max retries exceeded with url: /xet-bridge-us/621ffdc136468d709f1779f0/5eab7d5e5eb4c28a79f981a2a49e7e293b6a8bfc1e0a43111e62b3eb9650b820?[query omitted] (Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403 Forbidden')))"), '(Request ID: db520784-a828-4b1c-b1fc-3f86a252d5a8)').

## Reproduction and remaining work

Exact commands are in docs/task03.md. The config is configs/marbert_v1.json;
after revision resolution, use resolved_training_config.json to reproduce the
same pretrained encoder. New/empty output directories are mandatory. No automatic
split generation or overwriting of old experiment results. If model access or
compute is unavailable, validation metrics and checkpoints are explicitly unrun;
no zeros, synthetic scores or tiny-model test results substitute for MARBERT results.
Offline random tiny-BERT tests verify software shapes/save-reload only, not this
baseline's performance or actual MARBERT checkpoint reload.
