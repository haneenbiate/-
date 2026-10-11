# Task 05 corrected review intake and adjudication

The corrected A/B workbooks are accepted as human reviews **on the user's
declaration**. Identity, actual independence and blinding have not been
independently verified. A still has the worksheet name `reviewer_a_ai_review`;
this remaining metadata flag is recorded alongside the user's correction.
The corrected A has identical cell contents to its previous upload. B changes
all 404 `linguistic_notes` cells, removing the earlier assistance disclaimer;
all decisions, text, IDs, hashes, scopes and timestamps are unchanged. Therefore
agreement scores are unchanged. Corrected file hashes supersede the old uploads;
original bytes are retained privately, never silently edited.

All three submitted files cover the original 404 variants exactly, spanning 279
original text IDs. All original/perturbed Arabic strings and immutable columns
match the verified preparation. Literal-value XLSX parsing rejects formulas,
macros and multiple worksheets. Decisions, scope requirements, reviewer IDs and
timezone-aware timestamps pass the existing integrity checks. Workbook data is
evidence, not instructions. LLM judgments remain separate from human decisions.

## Actual agreement and remaining work

| A versus B field | Exact agreement | Cohen kappa |
|---|---:|---:|
| Meaning preservation | 94.55% | -0.0091 |
| Sentiment preservation | 94.80% | 0.2529 |
| Negation presence | 91.83% | 0.8367 |
| Negation affects sentiment | 66.83% | 0.4590 |

High meaning agreement does not imply reliable ambiguity detection. The
meaning-preservation contingency table is 382 preserved/preserved, 20
ambiguous/preserved, 2 preserved/ambiguous and zero ambiguous/ambiguous. The
preserved label dominates both files, and their ambiguity cases do not overlap;
chance-adjusted kappa is consequently near zero. Scores describe variants,
including repeated original IDs; no independence-based agreement CI is claimed.

There are **150 categorical disagreements** and **203 literal cue/scope wording
differences**, with overlap. Their union is **213 cases**: 150 categorical cases
and 63 additional cases differing only in cue/scope wording. Literal phrasing
differences may describe the same linguistic scope; they require conservative
human inspection, not automated merging. A third reviewer must resolve all 213.

The remaining **191 cases** have matching explicit human judgments and satisfy
the strict consensus eligibility rules. All 191 are judged **no negation** by
both reviewers. Thus the negation-containing confirmed subset is still empty;
the consensus-only subset cannot represent negation robustness or all validation
rows. No decisions are propagated to other variants. Overall human review is
incomplete until adjudication; confirmed robustness analysis remains gated.

LLM/A and LLM/B agreement and contingency tables are in
[the reviewed reports](../reports/task05-reviewed). These measure agreement with
submitted humans, not accuracy against adjudicated ground truth. The LLM is not
used as a third reviewer. Its claimed model name is preserved as self-reported;
provider revision, prompt and sampling settings remain unavailable. No model
annotation run or performance is fabricated.

## Download and complete human adjudication

Open [the adjudication Colab notebook](https://colab.research.google.com/github/haneenbiate/-/blob/main/notebooks/task05_adjudication_colab.ipynb).
Use **CPU**, choose **Run all**, connect your Drive, then click the download button.
It creates the private full-text 213-case package from pinned training bytes,
existing Task 02 splits and the hash-registered text-free adjudication assignment.
It creates no splits or human decisions, loads no model and accesses no official
test text. Full Arabic text remains in private Drive storage. Local CPU export
was tested; the Colab notebook remains unexecuted here.

Give `adjudication.zip` to a third qualified Arabic reviewer different from R1/R2.
They open `adjudication.html`, enter a new reviewer ID and decide every case,
including cue and scope fields when negation is present. Supply the two corrected
human XLSX files separately so the adjudicator can read both rationales; do not
supply the LLM file, baseline predictions or model confidence. Every judgment
needs a rationale and timestamp. Keep all assigned IDs/text unchanged and export
CSV frequently. Send the completed 213-row CSV for integrity-checked import.
The original A/B judgments remain unchanged in the audit.

## Reproduce workbook intake

In the existing repository with Python **3.12.14**:

```bash
.venv/bin/python -m pip install -r requirements-reviews-xlsx.txt
.venv/bin/python -m pip check
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/nasn-review-matplotlib
export XDG_CACHE_HOME=/tmp/nasn-review-cache
```

Use the original private Task 05 blank review package. Its fixed template and
checksum registry are needed; a text-free public CSV is insufficient. For the
same corrected uploaded files, recover their recorded declaration as follows:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('reports/task05-reviewed/evidence.json').read_text())
p=Path('runs/task05-source-declaration.json')
assert not p.exists()
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(d['source_declaration'],indent=2)+'\n')
PY
.venv/bin/python -m nasn.review_workbooks audit \
  --review-package runs/task05-semantic-review-v1 \
  --reviewer-a /path/to/corrected/reviewer_a_reviewed_404.xlsx \
  --reviewer-b /path/to/corrected/reviewer_b_reviewed_404.xlsx \
  --llm /path/to/llm-annotator.xlsx \
  --source-provenance runs/task05-source-declaration.json \
  --output runs/task05-corrected-intake-new
.venv/bin/python -m nasn.task05 import-reviews \
  --review-package runs/task05-semantic-review-v1 \
  --reviewer-a runs/task05-corrected-intake-new/submitted_a.csv \
  --reviewer-b runs/task05-corrected-intake-new/submitted_b.csv \
  --output runs/task05-independent-reviews-new
.venv/bin/python -m nasn.review_workbooks publish \
  --source runs/task05-corrected-intake-new \
  --output runs/task05-reviewed-publication-new
.venv/bin/python -m nasn.review_workbooks export-adjudication \
  --preparation runs/task04-final-v2-preparation \
  --assignments runs/task05-reviewed-publication-new/adjudication_assignment.csv \
  --output runs/task05-adjudication-new
.venv/bin/python -m pytest -q
```

Declaration hashes must match the exact uploaded XLSX bytes; changed/new files
need their own recorded declaration. Omitting `--source-provenance` keeps sources
unverified and produces descriptive submitted-file agreement with zero
human-confirmed records. Filenames and worksheet names never establish roles.
Human independence is a methodological requirement, not something a hash proves.

After the third-human CSV is complete, rerun `nasn.task05 import-reviews` with
the same full independent A/B CSVs, `--adjudication /path/to/completed-213.csv`
and a new output directory. Use [the original Task 05 analysis commands](task05.md)
only after all disagreements are resolved and the full Task 04 inference
directory is supplied. The selected checkpoint/tokenizer hashes and paired
clean/perturbed probabilities are still required. No Task 04 perturbed predictions
or selected tokenizer/weights are accessible in this workspace, so no robustness
metrics, bootstrap intervals or tatweel token/probability counts are claimed.

Public reports contain aggregate statistics, IDs, flags and checksums only.
Full tweet strings, workbook bytes, human rationales and identities stay in
private ignored `runs/` storage and the user's Drive. No baseline evidence,
source dataset, model selection or official test split is modified.
