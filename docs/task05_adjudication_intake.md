# XLSX adjudication intake and formatting integrity

The submitted `adjudication_reviewed.xlsx` contains all 213 required IDs once,
with a third reviewer ID. It includes an additional `diffs` column, which is
literal auxiliary edit metadata, never evaluated as Python or used as a human
decision. The original bytes and full original cell values are retained.

There are 20 rows whose original and perturbed texts differ from the registered
queue only through deletion of ASCII spaces at the edges: 19 lose two trailing
spaces, one loses two leading spaces, in each of the two text columns (40 cells).
Every other character, immutable ID, rule, intensity and recorded text checksum
matches. A reviewed-text display deviation is documented; no Arabic letters,
negation, internal whitespace or model input preprocessing are changed.
The explicit `--restore-edge-spaces` option restores these cells to the
hash-registered original strings. It fails on internal-space changes, changed
letters, negation, IDs or hashes. This is a conversion repair with an audit log,
not an automatic semantic judgment or a modification of the source XLSX.

All 213 review timestamps are native Excel datetime cells without timezone.
No timezone is inferred from the filename, workbook metadata or chat location.
The reviewer must declare it; then `--timestamp-timezone` attaches that zone and
serializes ISO 8601 while preserving the recorded local clock time. Aware input
timestamps are preserved unchanged. All other human fields remain verbatim.

The intake fails clearly until protected fields and timezone integrity pass.
It uses the original two full independent human CSVs and recomputes the expected
adjudication assignment from their disagreements; it never silently trusts a
new/different queue. The third reviewer must be distinct from the first two.
Human identity and independence remain user-declared rather than independently
authenticated. No LLM is used as adjudicator.

Once the actual timezone is supplied, run in Python 3.12.14 (replace the zone
with the reviewer's declared value; this example is **not an inferred value**):

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/nasn-review-matplotlib XDG_CACHE_HOME=/tmp/nasn-review-cache \
  .venv/bin/python -m nasn.adjudication_intake \
  --review-package runs/task05-semantic-review-v1 \
  --independent-review runs/task05-corrected-human-review-v1 \
  --workbook /path/to/adjudication_reviewed.xlsx \
  --timestamp-timezone Asia/Baghdad \
  --restore-edge-spaces --output runs/task05-adjudication-intake-new
```

Use a new output directory. Outputs preserve the workbook, raw cells,
auxiliary `diffs`, canonical CSV and cell-level repair log separately. The
`decisions/` directory uses the original Task 05 importer to merge the explicit
judgments, retain all independent decisions and record pre-adjudication
agreement, coverage and confirmed eligibility. Every nested decision registry
is hash-bound to the root intake evidence. Changed, ambiguous, sentiment-uncertain
and negation-uncertain cases are excluded from confirmed meaning-preserving
analysis. Human verdicts are never changed to force eligibility.

Analysis additionally requires the completed Task 04 inference directory and
selected checkpoint/tokenizer provenance, which have not been supplied locally.
Missing inputs block metrics rather than reconstructing predictions or inventing
results. No training, checkpoint selection or official test text access occurs.
