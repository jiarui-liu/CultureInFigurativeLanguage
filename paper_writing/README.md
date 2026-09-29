# Paper-writing support files

The Overleaf repo (`../../OverleafCultureInFigurativeLanguage/`) holds LaTeX only, without comments.
Everything else that produces or documents the paper lives here.

- `code/` — generators that write into the Overleaf `latex/` directories:
  `make_tables.py` (Table 1 from `docs/paper_stats/v2/ci_report.json`), `make_ci_appendix.py`,
  `make_idiomatlas_table.py`, `make_figures.py` (entity and CPT-effect figures),
  `make_matrix_figure.py` (2B transfer matrix), `shared_entity_rate.py` (Sec. 4.3 numbers),
  and `clean_tex.py`, which strips comments from all Overleaf sources — run it after any generator.
- `notes/` — citation and licence research (`fix_citations_licences.md`), the candidate bib file
  (already merged into the paper's `custom.bib`), and the former Overleaf README.
- `drafts/` — earlier versions of Section 5.

Compile check on babel: `~/.local/tectonic/tectonic -X compile main.tex --outdir <tmp>`
(Chinese glyphs need Overleaf's fonts).
