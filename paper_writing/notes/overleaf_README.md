# Culture-in-idioms paper draft

Layout mirrors `~/ars/OverleafMetaAutoresearch`: `main.tex` + `latex/NN_section.tex`,
`latex/tables/*.tex`, `latex/figures/*.{tex,pdf}`, `references.bib`.

- `\todo{...}` (red, defined at the top of `main.tex`) marks every claim that needs an
  experiment, a validation, or a citation that does not exist yet. `grep -rn '\\todo' .`
- `code/make_figures.py`  -> `fig_entities.pdf`, `fig_cpt_effects.pdf` (from the KB files and `docs/paper_stats/ci_report.json`)
- `code/make_tables.py`   -> `latex/tables/main_results.tex` (auto-generated, do not hand-edit)
- `code/shared_entity_rate.py` -> the 5.5% / 7.5% shared-entity numbers in Sec. 4.3
- Chinese text uses `\zh{}` (CJKutf8). It renders on Overleaf; on a TeX install without
  CJKutf8 the characters are silently dropped (pinyin/gloss remain). This machine also
  lacks `multirow.sty`; Overleaf has it.
- The Arabic KB is not on this filesystem; `make_figures.py` reads a copy downloaded from
  HF `Jerry9999/CultureInFigurativeLanguage` (set `AR_KB=/path/to/idioms_merged_llm_formatted.jsonl`).
- `references.bib` was written from memory: verify each entry before submission.
- Template: ACL/ARR (`acl.sty`, `acl_natbib.bst` from acl-style-files). `\usepackage[review]{acl}` in `main.tex`; switch to `[final]` / `[preprint]` as needed. Two-column layout, so all floats are `figure*` / `table*`.
- ARR long papers allow 8 pages of content (Limitations, ethics, references, appendix excluded). The draft currently runs ~11 pages of content including the red TODO blocks, so it needs cutting.
