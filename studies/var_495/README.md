# Study 495: an online VAR(p) with Granger-causality tests

A numpy prototype of PLAN task 495, checked against `statsmodels`' `VAR`. Python
only; no Rust. Nothing here is collected by the repo's pytest (`testpaths = ["tests"]`).

Files:

- `online_var.py`: the model (`OnlineVAR`) and a feeder (`run`). The math is in
  its module docstring.
- `data.py`: seeded synthetic VAR data (no data files).

Status: in progress (part 1: model and first comparison).
