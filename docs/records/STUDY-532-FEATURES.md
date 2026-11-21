# Study 532: resumable time-series features held in a state

Status: in progress. Part 1 done: lags, rolling (rows and `_by`) and EW features
as state machines that mirror Polars 2.0.0's kernels; all 47 outputs equal
Polars' to the bit, grouped, chunked with save and load, with nulls, zero-weight
rows and non-finite values. Next: tables (report.py), state bytes, cost, the
gaps of the forms that do not mirror Polars' order, the write-up.
