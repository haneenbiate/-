# Study v3: orthographic noise, tokenizers and negation in Arabic sentiment models

Status: **preregistered draft** (not yet frozen). The study is frozen when
`configs/study_v3.json`, `nasn/orthography_v3.py` and `nasn/study_v3.py` are
committed and tagged `study-v3-frozen`. After that tag, no condition, metric,
model, seed or decision rule changes. Anything added later is labelled exploratory.

## Why a new study

Tasks 05–07 found no Macro-F1 drop whose interval excluded zero, and NASN2 tied
the simple NO_CONTEXT baseline on all 394 eligible variants. Three design limits
explain why those results are uninformative rather than negative:

1. **One model.** All conclusions were conditional on one MARBERT checkpoint.
2. **Weak noise.** One edited word per tweet, and v1 elongation excluded the long
   vowels ا/و/ي, which carry most real emphatic repetition (حلووو، كثيييير).
3. **Unreliable labels.** Human semantic-validity Kappa was −0.009, and reviewer
   independence could not be verified (docs/task06.md).

Study v3 therefore asks a broader question. It does not claim that NASN improves
on the baselines. NASN2 and NO_CONTEXT appear only as frozen inference-time defenses.

## Research questions

- **RQ1 (robustness).** How much do realistic orthographic variations change
  predictions of Arabic-specific encoders, a multilingual encoder and a zero-shot LLM?
- **RQ2 (mechanism).** Do tokenizer effects (unknown-token replacement,
  fragmentation) explain the differences between models?
- **RQ3 (negation).** Is perturbing the negation cue itself more damaging than
  perturbing another word in the same tweet?
- **RQ4 (defenses).** Does inference-time normalization (STD, NO_CONTEXT, NASN2)
  recover clean behaviour, and does negation-context abstention add anything?

## Observation recorded before preregistration

The CPU-only tokenizer audit (`reports/study-v3/tokenizer_summary.csv`) was run
before freezing and is disclosed as such. In MARBERT, MARBERTv2,
AraBERTv02-twitter and CAMeLBERT-DA, **every** word containing a tatweel becomes
`[UNK]`, including negation cues (404/404 cue edits). XLM-R and Qwen2.5 never produce
an unknown token but fragment edited words 1.7–2.9×. MARBERT's lower-casing strips
hamza from alef, so 35.5% of single-word folding edits leave its tokens unchanged.
The same audit on the ArSAS validation partition reproduces every one of these
figures: 100% `[UNK]` for tatweel in the four WordPiece encoders, and 30.9% of MARBERT
folding edits unchanged (`reports/study-v3/arsas_tokenizer_summary.csv`). Hypotheses
H2 and H3 below build on this observation. They are confirmatory only for model
*predictions*, which no one has observed yet.

**Natural prevalence** (`reports/study-v3/natural_noise.csv`, train and validation
partitions only):

| | ArSarcasm-v2 | ArSAS |
|---|---|---|
| Tweets with natural tatweel | 4.6% | 3.2% |
| Tweets with a letter repeated 3+ times | 4.0% | 2.8% |

Natural noise *on a negation cue* is rare (≤0.04% of tweets). The cue-targeted
conditions are therefore a **controlled stress test** of how the models represent
negation; they do not estimate how often this happens in the wild.

## Design

**Data.**
- **ArSarcasm-v2**, using the Task 02 splits (leakage-safe, pinned hashes).
  Development: the 2,509 validation tweets. Final: the official test set (3,000),
  evaluated once after the freeze tag.
- **ArSAS** (Elmadany et al., 2018; 21,064 tweets), pinned and split by
  `nasn/arsas.py`.
  - *Exclusions:* Mixed sentiment (1,302) is dropped. Tweet ID repeats and identical
    texts are merged into groups. Groups with conflicting labels (90 tweets + 44
    linked copies) and the one tweet shared with ArSarcasm-v2 are excluded.
  - *Split:* a stratified group split gives 13,738 train / 1,966 validation / 3,923
    test tweets (`reports/arsas/`).
  - *Why ArSAS rather than ASTD:* 2,473 ArSarcasm-v2 training tweets (20%) also
    appear in ASTD, so ASTD could not serve as an independent second dataset.

Every model is trained and evaluated separately on each dataset; the two datasets
are never pooled. Both test partitions stay locked until the freeze.

**Perturbations** (`nasn/orthography_v3.py`; deterministic, hash-selected, logged):

| Family | Edit | Meaning-preserving? |
|---|---|---|
| `tatweel` | insert ـ after the last joining letter of a word | presumed |
| `elongationx2`, `elongationx4` | repeat the last internal long vowel, else the final letter, 2 or 4 extra times | presumed (emphasis may change) |
| `fold` | أ/إ/آ→ا, final ة→ه, final ى→ي | presumed |
| `typo` | one adjacent-key substitution on the Arabic keyboard | **no**: damage control |

Density: `single` (one word) or `dense` (⌈30% of Arabic words⌉). Target: `non_cue`
(any word except unambiguous negation cues, URLs, mentions and hashtags) or `cue`
(the negation cue itself; tatweel and elongation only). «ما» is excluded from cues
because its non-negating uses dominate. `single` edits are a subset of `dense`
edits, and elongation levels edit the same character (a nested design).

**Models.** Five encoders fine-tuned with the unchanged MARBERT v1 recipe
(lr 2e-5, batch 16, ≤5 epochs, validation Macro-F1 selection), each with seeds
42/43/44: MARBERT, MARBERTv2, AraBERTv02-twitter, CAMeLBERT-DA, XLM-R-base.
One zero-shot LLM: Qwen2.5-7B-Instruct, using a fixed English instruction and
first-token label probabilities. Revisions are pinned in `configs/study_v3.json`.

**Defenses** applied to clean and perturbed inputs alike: RAW; STD (remove tatweel,
collapse runs of ≥3 identical letters to one); NO_CONTEXT and NASN2 (frozen v2 rules).

## Metrics and statistics

- **Primary metric:** flip rate, the share of applicable tweets whose predicted
  class differs between the clean and perturbed text. It needs no gold labels and
  isolates the effect of the perturbation.
- Secondary: clean Macro-F1, perturbed Macro-F1 and their difference (drop),
  computed on the same applicable rows.
- Natural-noise slice: tweets that already contain tatweel or a letter repeated 3+
  times are compared with their own STD-cleaned text (cleaned treated as the
  reference). This is the only real-world estimate in the study; its n is limited
  to roughly 3–5% of the validation tweets.
- Uncertainty: 2,000 cluster-bootstrap resamples over Task 02 group IDs (duplicate
  tweets move together), seed 42, percentile 95% intervals. Encoder metrics are
  averaged over the three training seeds inside every resample.
- Cells with n < 30 are not computed.
- Contrasts are paired on identical tweets:
  - cue vs non-cue: `X/single/cue` minus `X/single/non_cue` on tweets where both apply;
  - dense vs single: within each family.

## Hypotheses and decision rules

A hypothesis is **supported** for a model when its 95% interval excludes zero in the
stated direction. All results are reported regardless of direction.

- **H1:** for every model, `dense` flip rate exceeds `single` flip rate within each
  meaning-preserving family.
- **H2:** under `tatweel/dense/non_cue` with RAW inputs, each of the four Arabic
  WordPiece encoders has a higher flip rate than XLM-R-base (difference of
  seed-averaged rates; the bootstrap pairs tweets).
- **H3:** cue-targeted tatweel produces a higher flip rate than non-cue tatweel for
  the Arabic WordPiece encoders (the cue becomes `[UNK]`).
- **H4:** STD lowers the RAW flip rate for tatweel and elongation for every encoder.
- **H5 (ablation):** NASN2 minus NO_CONTEXT flip rate is reported. No superiority
  claim is made unless its interval excludes zero in favour of NASN2 for at least
  three of the five encoders.
- RQ2 (descriptive): Spearman correlation between the per-model, per-condition
  edited-word `[UNK]` rate or fragmentation and the flip rate.

## Human validation

The claim that a perturbation preserves meaning depends on the new annotation
(`docs/study_v3_annotation_protocol_ar.md`):
- 330 items, 30 per condition, one variant per tweet;
- two annotators who are verifiably independent;
- the typo items serve as an attention check;
- agreement is reported with Cohen's κ and Gwet's AC1.

Robustness results restricted to human-confirmed meaning-preserving items are
reported alongside the full applicable population. **The earlier Task 05 human
labels are not reused for any claim.**

## Reproduction

```bash
python -m nasn.study_v3 prepare --partition validation --output runs/study-v3/validation-variants
python -m nasn.study_v3 tokenizer-audit --variants runs/study-v3/validation-variants/variants.csv.gz --output runs/study-v3/tokenizer-audit
python -m nasn.study_v3 sample-annotation --variants runs/study-v3/validation-variants/variants.csv.gz --output runs/study-v3/annotation-v1
python -m nasn.study_v3 prevalence --output runs/study-v3/natural-noise
python -m nasn.arsas fetch && python -m nasn.arsas split --output <new directory>   # reproduces reports/arsas
python -m nasn.study_v3 prepare --dataset arsas --partition validation --output runs/study-v3/arsas-validation-variants
python tools/build_study_v3_notebook.py --reference <commit>   # GPU stages in Colab
python -m tools.build_annotation_workbooks --package runs/study-v3/annotation-v1 --output <folder>   # one Excel per annotator
python -m nasn.annotation_v3 --package runs/study-v3/annotation-v1 --a A.xlsx --b B.xlsx --output <new directory>
```

On Windows, set `PYTHONUTF8=1`. The registered variants hashes are:
- ArSarcasm-v2: `d6b467f0bd615c613e90e4c8e60ece49ea248a1ffcc6a846363d34b57874bb2d`
- ArSAS: `a12f747fc04813037b29c8e14e0e6710918dafa51402417b9e4cbd7443ab2693`

## Known limits

- Edit placement is rule-based. Real noise co-occurs in ways these families do not model.
- The cue lexicon is finite and excludes «ما».
- Validation tweets were used for checkpoint selection, so the development numbers
  are optimistic for clean accuracy. Flip rates are less affected, and the test
  evaluation addresses this.
- A single prompt is used for the LLM.
