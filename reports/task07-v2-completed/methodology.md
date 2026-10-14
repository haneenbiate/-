# Task 07 completed v2 results

All 26 supplied light-report payload hashes and 62,350 prediction identities match
the frozen Task 07 code, source manifests, rules and selected checkpoint registry.
Every per-rule metric, contrast, original-ID cluster interval, all-clean score and
paired interval, and scientific decision is recomputed from actual saved outputs.
Original uploaded bytes are preserved under original/. The recorded external run
used Tesla T4, CUDA 12.6, Torch 2.8.0+cu126, seed 42, float32, length 128 and batch
8; total recorded run duration is 510.8631 seconds. No local model inference or
weight reload is claimed. Selected checkpoint verification is external recorded
load/hash evidence corroborated by RAW clean baseline reproduction.

The primary cohort contains 394 user-declared eligible source variants across
274 original IDs. New normalized text is not human-validated and reviewer
independence remains unverified. These are conditional exploratory scores against
original sentiment labels, not a certified meaning-preserving robustness benchmark.
9,557 unreviewed variants are reported separately; 10 reviewed exclusions have no
robustness metrics. All-clean table columns use 2,509 unique development rows;
paired pooled clean metrics use matched variant weighting, explicitly separately.

NASN2 and strongest simple baseline NO_CONTEXT both have primary perturbed Macro-F1
0.655111 and clean/perturbed flip rate 7/394 (1.7766%). Their perturbed predictions
are identical across all 394 primary rows, despite 11 different effective input
sequences. Paired NASN2 gain over NO_CONTEXT is 0, with empirical interval [0,0].
This is no observed incremental classification benefit from the bounded context
guard in this cohort; a degenerate empirical interval does not prove zero effect
in unseen text. The baseline shares marker/critical-word safety; this ablation
tests the added context guard, not the value of all negation protection.

Against TAT and REP2, NASN2 point gains are 0.008878 and 0.005932; their paired
95% intervals [-0.002942,0.023442] and [-0.002208,0.016312] include zero. NASN2's
point all-clean Macro-F1 is slightly higher than NO_CONTEXT (0.751225 vs 0.750904),
corresponding to one changed clean prediction. This does not satisfy the frozen
positive perturbed-gain requirement. RAW primary Macro-F1 is 0.640226; the combined
simple method already provides the same primary gain as NASN2. It cannot be
attributed uniquely to context awareness or treated as a novelty claim.

The predetermined gate fails. Recommend abandoning the NASN2 superiority claim or
reframing as a transparent Arabic preprocessing/ablation audit. Do not add post-hoc
rules, alter the selected checkpoint or reinterpret provisional results as verified
semantics. Retain old preparation/blocker reports as historical stages. New semantic
reviews and provenance evidence remain required for meaning-preservation claims,
but are not needed to observe failure of this performance gate on current labels.

Intervals use 2,000 seed-42 paired resamples. Pooled observations move together by
original source ID; per-rule strata use one variant per ID and minimum n=30.
They condition on one trained checkpoint and a nonrandom development review sample,
omit training/reviewer uncertainty, and are not simultaneous significance tests.
Exact-text duplicates with separate IDs and prior validation reuse further limit
inference. No official test text/labels are read/evaluated, no model is trained,
and no normalization rule or decision threshold is changed.

Reproduce using `python -m nasn.task07_v2_results --archive /path/to/original.zip
--output runs/task07-v2-verified-new`, then `python -m tools.publish_task07_v2_results
--analysis runs/task07-v2-verified-new --output runs/task07-v2-public-new`.
Use the pinned environment and MPLBACKEND=Agg as documented in
[the complete results report](../../docs/task07_v2_results.md).
