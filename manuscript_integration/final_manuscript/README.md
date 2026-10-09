# Manuscript typesetting package

Primary source: `main.tex`. Bibliography: `references.bib` (all 22 supplied entries retained).
Upload the entire folder contents to Overleaf and select `main.tex` as the main document.
Use XeLaTeX; pdfLaTeX is also compatible with the supplied packages and figure formats.
With a local standard TeX distribution: XeLaTeX, BibTeX, XeLaTeX, XeLaTeX.

Tables are loaded from five local `.tex` fragments. Referenced figures are in `figures/`.
All figures have an SVG, a 400-DPI PNG, and a saved source CSV in `supporting_evidence/`.
The sensitivity figure also has a PDF, with quiescence terminology typeset from its saved grid.
No raw tweet text, model checkpoint, training output, or analysis environment is included.

`supporting_evidence/tables/` contains copied canonical derived evidence and assumptions.
These tables preserve saved cohort and method identifiers; display names in the manuscript
describe the architecture and task. No estimates are combined across cohorts.
`manuscript_manifest.json` gives SHA-256 hashes and provenance for package inputs.
The project reproducibility package contains the analysis scripts and linked output manifests;
this typesetting ZIP is not an independently executable analysis release or licence grant.
