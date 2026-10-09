# Reviewer correction reproducibility package

Run from the existing project root with its Python environment. Protected data remain in data/raw and data/processed; data are not distributed here. Canonical analysis outputs are in results/tables and results/figures/reviewer_corrected. Superseded generated validation drafts in results/reviewer_interim_20261009 are retained only for audit.

Commands (each output is exclusive; existing paths fail, so do not rerun in place):
```powershell
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py reconcile
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py layers
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py mixing
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py temporal
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py simulations
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py spectral
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py forecasting
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py figures
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py supplements
.\.venv\Scripts\python.exe -X utf8 src/reviewer_validate_evidence.py
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py report
.\.venv\Scripts\python.exe -X utf8 src/reviewer_correction_analysis.py package
```

For fresh reproduction, use a new copy of the input project containing the listed protected inputs and existing forecasting/configuration artifacts, but excluding reviewer outputs, then hash that copy before analysis. Script copies in this package are archival; execute the original new src scripts. Package bootstrap excludes its own manifest/hash ledgers from their self-referential hash contents. Hash verification is independently repeated at package creation. No model is fitted or forecast recomputed. See assumptions_and_limitations.md and individual analysis reports. No raw tweet text is copied. Username-derived metadata require access control; these files are not automatically approved for public distribution.
