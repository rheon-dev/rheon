# Study 495: an online VAR(p) with Granger-causality tests

The write-up, with every table, is `docs/records/STUDY-495-VAR.md`.

- `python scripts/study_495.py` runs the comparisons and prints the tables
  (`--drift` adds the rolling window's long-stream drift).
- `python -m pytest studies/var_495` runs the checks as asserts.

Either needs only numpy, scipy and statsmodels (and pytest), not `uv sync`.
Files: `online_var.py` (the model and its rolling window), `data.py` (seeded
synthetic data), `compare.py` (the checks), `test_online_var.py` (the asserts),
`calibration.py` (the Granger test's size under decay, Monte Carlo).
