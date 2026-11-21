"""Each estimator against hfhd on the same ticks: the checks behind the
write-up's tables. Every hfhd call states its settings."""

from __future__ import annotations

import warnings

import numpy as np
from hfhd import hf

import data as D
import estimators as E

warnings.filterwarnings("ignore", module="numba")


def _indices_and_values(tick_series_list):
    """hfhd 0.1.4's ``_get_indeces_and_values`` with one change: it pads the
    uint64 stamp matrix with ``np.nan``, which numpy 2 refuses ("cannot
    convert float NaN to integer"), so every pairwise path and
    ``hayashi_yoshida`` raise. This pads with 0. hfhd never reads the
    padding: each row is cut at the count of non-NaN *values*, which keep
    their NaN padding as before."""
    n_max = np.max([len(x) for x in tick_series_list])
    idx = np.zeros((len(tick_series_list), n_max), dtype="uint64")
    val = np.full((len(tick_series_list), n_max), np.nan)
    for i, x in enumerate(tick_series_list):
        t = np.array(x.dropna().index, dtype="uint64")
        v = np.array(x.dropna().to_numpy(), dtype="float64")
        idx[i, : t.shape[0]] = t
        val[i, : t.shape[0]] = v
    return idx, val


hf._get_indeces_and_values = _indices_and_values


def rel(a, b) -> float:
    """Largest absolute difference over the largest absolute entry of ``b``."""
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    return float(np.abs(a - b).max() / np.abs(b).max())
