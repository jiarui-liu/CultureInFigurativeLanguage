# Paper artifacts

The paper-facing statistics and figures are generated in this repository. The Overleaf repository contains only the rendered PDFs and LaTeX wrappers.

```bash
PYTHONPATH=src uv run --no-project --with numpy --with scipy python -m culture.analysis.v3.cross_language_symbolism
uv run --no-project --with matplotlib --with numpy python paper_writing/code/make_acquisition_figures.py
.venv/bin/python paper_writing/code/validate_paper_claims.py
```

The independent probe audit is cached in `docs/paper_stats/v3/symbolism_probe_audit.json`. To rebuild it with configured MetaGen credentials:

```bash
PYTHONPATH=src uv run --no-project --with numpy --with scipy --with requests python -m culture.analysis.v3.audit_symbolism_probe
```

Copy `paper_writing/figures/fig_acquisition.pdf` and `fig_symbolism.pdf` to the paper repository after regenerating them.
