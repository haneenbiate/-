# Study v3 paper draft

- `main.tex` — ACL-format draft. Sections marked `[PENDING: ...]` (red) depend on
  GPU runs or the new human annotation and contain no results yet.
- `figures/tokenizer_edits.pdf` — regenerate with
  `python tools/plot_study_v3_tokenizer.py` (reads `reports/study-v3/*tokenizer_summary.csv`).
- Build: copy `acl.sty` and `acl_natbib.bst` from https://github.com/acl-org/acl-style-files
  into this folder, install the Amiri font, then `latexmk -xelatex main.tex`.
- Verify every entry in `references.bib` against the ACL Anthology before submission.
