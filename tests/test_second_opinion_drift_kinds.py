"""``emit_drift``'s other detectors (``drift_kind``) against their oracles
(task 508).

Each test fits a bank, reads the residuals and the scale it reports, feeds
the oracle the rows the bank's detector reads -- a weight above 0 and a
residual, and for the two scaled kinds a scale -- and holds the flags row for
row:

- ``adwin``, ``ddm`` and ``eddm`` against river's ``drift.ADWIN``,
  ``drift.binary.DDM`` and ``drift.binary.EDDM`` on the same values (``|resid|``,
  or whether it passes ``error_above``). river's detectors take no weights,
  so a row of positive weight is one row, and a zero-weight row none.
- ``kswin`` against ``scipy.stats.ks_2samp(method="exact")`` on the two
  samples the detector compares, the older one drawn by the definition the
  docs give (SplitMix64 from the seed, Floyd's algorithm, Lemire's bounded
  draw), written out here. river's KSWIN draws with CPython's generator,
  which the library does not reproduce.
- ``cusum`` against Page's (1954) recursion, written out here from its
  definition: no maintained library computes the tabular CUSUM (``pyspc``
  fixes ``k`` at half a deviation, starts no sum again after a signal and
  needs matplotlib; R's ``qcc`` does not run on this machine's arm64 R).
  ``crates/rheon-core/src/drift/cusum.rs`` holds the scheme's published
  average run lengths.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import polars as pl
import pytest
import river.drift as river_drift
import river.drift.binary as river_binary
from scipy.stats import ks_2samp

from rheon import polars as rp

TIER = "essential"

N = 4000


def _frame(n: int = N, seed: int = 61) -> pl.DataFrame:
    """A level shift of six noise deviations at row 1,500 and a doubling of
    the noise at 2,800; uneven weights, every 25th row of weight 0 with a
    wild target, and a null target every 97th row."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    noise[2800:] *= 2.0
    y = 1.0 + 2.0 * x + noise
    y[1500:] += 3.0
    w = rng.uniform(0.5, 1.5, n)
    w[::25] = 0.0
    y[::25] += 50.0
    t = np.cumsum(rng.choice([0.5, 1.0, 3.0], n))
    return pl.DataFrame({"t": t, "x": x, "y": y, "w": w}).with_columns(
        y=pl.when(pl.int_range(pl.len()) % 97 == 3).then(None).otherwise(pl.col("y"))
    )


def _spec(**kw):
    common = dict(
        features=["x"],
        targets=["y"],
        half_life=400.0,
        weight="w",
        min_weight=0.0,
        delta=1.0,
        emit_sigma=True,
        emit_drift=True,
    )
    common.update(kw)
    return rp.spec.rls("m", **common)


def _fit(df: pl.DataFrame, **kw) -> pl.DataFrame:
    return rp.ModelBank([_spec(**kw)]).fit_predict(df)["m"].struct.unnest()


def _read(df: pl.DataFrame, out: pl.DataFrame, scaled: bool = False):
    """The residuals, the scale, the flags, and which rows the detector
    reads."""
    resid = out["resid_y"].to_numpy()
    sigma = out["sigma_y"].to_numpy()
    flags = out["drift_y"].fill_null(False).to_numpy()
    read = (df["w"].to_numpy() > 0) & np.isfinite(resid)
    if scaled:
        read &= np.isfinite(sigma) & (sigma > 0)
    return resid, sigma, flags, read


def _held(flags: np.ndarray, read: np.ndarray, oracle) -> list[int]:
    """Feed `oracle(i)` each read row and hold its flag to the bank's; a row
    not read flags nothing. The rows flagged."""
    flagged = []
    for i in range(len(flags)):
        if read[i]:
            want = oracle(i)
            assert bool(flags[i]) == want, f"row {i}: bank {flags[i]}, oracle {want}"
            if want:
                flagged.append(i)
        else:
            assert not flags[i], f"row {i} is not read but flagged"
    assert flagged == np.flatnonzero(flags).tolist()
    return flagged


ADWIN_CASES = [
    {},
    {"clock": 1, "delta": 0.05},
    {"max_buckets": 3, "min_window_length": 10, "grace_period": 50},
    # `range` at river's 1, given: river's detector, which has no range.
    {"range": 1.0},
]


@pytest.mark.parametrize("options", ADWIN_CASES, ids=["defaults", "clock1", "buckets3", "range1"])
def test_adwin_is_rivers(options):
    df = _frame()
    out = _fit(df, drift_kind="adwin", drift_options=options)
    resid, _, flags, read = _read(df, out)
    adwin = river_drift.ADWIN(**{k: v for k, v in options.items() if k != "range"})

    def oracle(i: int) -> bool:
        adwin.update(float(abs(resid[i])))
        return bool(adwin.drift_detected)

    flagged = _held(flags, read, oracle)
    assert any(f >= 1500 for f in flagged), flagged
    assert read.sum() > 3700


@pytest.mark.parametrize("kind", ["ddm", "eddm"])
@pytest.mark.parametrize("error_above", [0.75, 1.5])
def test_ddm_and_eddm_are_rivers(kind, error_above):
    df = _frame()
    options: dict = {"error_above": error_above}
    if kind == "ddm":
        options |= {"warm_start": 50, "drift_threshold": 2.5}
        det = river_binary.DDM(warm_start=50, drift_threshold=2.5)
    else:
        options |= {"warm_start": 20, "beta": 0.85}
        det = river_binary.EDDM(warm_start=20, beta=0.85)
    out = _fit(df, drift_kind=kind, drift_options=options)
    resid, _, flags, read = _read(df, out)

    def oracle(i: int) -> bool:
        det.update(1 if abs(resid[i]) > error_above else 0)
        return bool(det.drift_detected)

    flagged = _held(flags, read, oracle)
    assert flagged, "the case needs a flag"


def test_ddm_on_a_classifier_counts_misclassifications():
    """``error_above = 0.5`` on a probability of a 0/1 target: an error is a
    row whose rounded prediction is not its label."""
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(0.0, 1.0, n)
    slope = np.where(np.arange(n) < 1500, 2.0, -2.0)
    p = 1.0 / (1.0 + np.exp(-slope * x))
    y = (rng.uniform(size=n) < p).astype(float)
    df = pl.DataFrame({"x": x, "y": y, "w": np.ones(n)})
    spec = rp.spec.ftrl(
        "m",
        features=["x"],
        targets=["y"],
        half_life=float("inf"),
        min_weight=0.0,
        emit_drift=True,
        drift_kind="ddm",
        drift_options={"error_above": 0.5},
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
    pred = out["pred_y"].to_numpy()
    flags = out["drift_y"].fill_null(False).to_numpy()
    ddm = river_binary.DDM()
    seen = []
    for i in range(n):
        ddm.update(int((pred[i] > 0.5) != (y[i] == 1.0)))
        assert bool(flags[i]) == ddm.drift_detected, i
        if ddm.drift_detected:
            seen.append(i)
    assert any(i >= 1500 for i in seen), seen


# --- KSWIN, by its definition and scipy -------------------------------------

_MASK = (1 << 64) - 1


class _SplitMix64:
    def __init__(self, seed: int) -> None:
        self.s = seed & _MASK

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & _MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)


def _below(rng: _SplitMix64, n: int) -> int:
    """Lemire's bounded draw: the high word of ``x n``, redrawn while the low
    word is under ``2^64 mod n``."""
    m = rng.next() * n
    if (m & _MASK) < n:
        threshold = ((1 << 64) - n) % n
        while (m & _MASK) < threshold:
            m = rng.next() * n
    return m >> 64


def _draw(rng: _SplitMix64, ring: deque, size: int, r: int) -> list[float]:
    """Floyd's algorithm over the older ``size - r`` values, oldest first."""
    n = size - r
    taken: set[int] = set()
    old = []
    for j in range(n - r, n):
        t = _below(rng, j + 1)
        pick = j if t in taken else t
        taken.add(pick)
        old.append(ring[pick])
    return old


def _kswin_oracle(alpha, size, r, seed, restart):
    ring: deque = deque(maxlen=size)
    rng = _SplitMix64(seed)
    margins = []

    def step(x: float) -> bool:
        nonlocal rng
        ring.append(x)
        if len(ring) < size:
            return False
        old = _draw(rng, ring, size, r)
        new = list(ring)[size - r :]
        res = ks_2samp(old, new, method="exact")
        margins.append(abs(np.log(res.pvalue / alpha)))
        flag = bool(res.pvalue <= alpha and res.statistic > 0.1)
        if flag:
            if restart:
                ring.clear()
                rng = _SplitMix64(seed)
            else:
                for _ in range(size - r):
                    ring.popleft()
        return flag

    return step, margins


KSWIN_CASES = [
    {},
    {"alpha": 0.01, "window_size": 150, "stat_size": 40, "seed": 12345},
    {"window_size": 60, "stat_size": 20, "restart": False},
]


@pytest.mark.parametrize("options", KSWIN_CASES, ids=["defaults", "wider", "kept"])
def test_kswin_is_scipys_exact_test_on_the_drawn_samples(options):
    df = _frame()
    out = _fit(df, drift_kind="kswin", drift_options=options)
    resid, _, flags, read = _read(df, out)
    step, margins = _kswin_oracle(
        options.get("alpha", 0.005),
        options.get("window_size", 100),
        options.get("stat_size", 30),
        options.get("seed", 0),
        options.get("restart", True),
    )
    flagged = _held(flags, read, lambda i: step(float(abs(resid[i]))))
    assert any(f >= 1500 for f in flagged), flagged
    # No decision sat within rounding of the level: the p-values are a
    # discrete set (the closest measured, 2026-11-22, is a factor of 1.2).
    assert min(margins) > 1e-6, min(margins)


# --- CUSUM, Page's definition --------------------------------------------------


def _page(z: np.ndarray, d: np.ndarray, k: float, h: float):
    """Page's tabular two-sided CUSUM over the read rows: the flags, and the
    sums' closest approach to ``h`` (evidence no flag rests on rounding)."""
    up = down = 0.0
    flags = np.zeros(len(z), dtype=bool)
    margin = np.inf
    for i in range(len(z)):
        up = max(0.0, up + d[i] * (z[i] - k))
        down = max(0.0, down + d[i] * (-z[i] - k))
        margin = min(margin, abs(up - h), abs(down - h))
        if up > h or down > h:
            flags[i] = True
            up = down = 0.0
    return flags, margin


@pytest.mark.parametrize(
    ("k", "h", "clock"),
    [(0.5, 5.0, None), (0.5, 4.0, None), (0.25, 8.0, None), (0.5, 12.0, "t"), (1.0, 3.0, "t")],
    ids=["defaults", "h4", "k0.25", "clock", "clock-k1"],
)
def test_cusum_is_pages_recursion(k, h, clock):
    df = _frame()
    kw: dict = {"drift_delta": k, "drift_threshold": h}
    if clock:
        kw |= {"clock": clock, "gap_cap": 1e6}
    out = _fit(df, drift_kind="cusum", **kw)
    resid, sigma, flags, read = _read(df, out, scaled=True)
    idx = np.flatnonzero(read)
    z = resid[idx] / sigma[idx]
    # The clock step of each read row is its row's own step: rows not read
    # are still steps of the clock (a zero-weight row is not, and is not
    # read either, so it is left out of the steps too).
    if clock:
        t = df["t"].to_numpy()
        counted = df["w"].to_numpy() > 0
        prev = np.maximum.accumulate(np.where(counted, np.arange(len(t)), -1))
        last = np.concatenate([[-1], prev[:-1]])
        steps = np.where(last >= 0, t - t[np.maximum(last, 0)], 0.0)
        d = steps[idx]
    else:
        d = np.ones(len(idx))
    want, margin = _page(z, d, k, h)
    got = flags[idx]
    assert not flags[~read].any()
    assert np.array_equal(got, want), (idx[got].tolist(), idx[want].tolist())
    assert want.any()
    assert margin > 1e-9, margin


def test_a_cusum_default_threshold_is_five():
    df = _frame()
    a = _fit(df, drift_kind="cusum")["drift_y"]
    b = _fit(df, drift_kind="cusum", drift_threshold=5.0)["drift_y"]
    assert a.equals(b)
    assert not a.equals(_fit(df, drift_kind="cusum", drift_threshold=4.0)["drift_y"])


# --- every kind: chunks, a save mid-stream, zero-weight rows -------------------

KINDS = {
    "page_hinkley": {},
    "cusum": {"drift_threshold": 4.0},
    "adwin": {"drift_options": {"clock": 4}},
    "adwin_kept": {"drift_options": {"restart": False, "delta": 0.05}},
    "ddm": {"drift_options": {"error_above": 0.75}},
    "eddm": {"drift_options": {"error_above": 0.75}},
    "kswin": {"drift_options": {"window_size": 80, "stat_size": 25}},
    "kswin_kept": {"drift_options": {"restart": False}},
}


def _kind(name: str) -> dict:
    return {"drift_kind": name.removesuffix("_kept"), **KINDS[name]}


@pytest.mark.parametrize("name", list(KINDS))
def test_every_kind_is_chunk_invariant_and_goes_on_after_a_save(name):
    df = _frame(n=3000)
    spec = _spec(drift_action="reset", **_kind(name))
    whole = rp.ModelBank([spec])
    want = whole.fit_predict(df)
    assert want["m"].struct.field("drift_y").fill_null(False).any(), "the case needs a flag"
    rng = np.random.default_rng(3)
    cuts = np.sort(rng.choice(np.arange(1, len(df)), 12, replace=False)).tolist()
    bank = rp.ModelBank([spec])
    parts, start = [], 0
    for end in [*cuts, len(df)]:
        parts.append(bank.fit_predict(df.slice(start, end - start)))
        bank = rp.ModelBank.load_bytes(bank.save_bytes(), [spec])
        start = end

    # `coef` is filled on each chunk's last row by design; every other field
    # is held to the bit.
    def fields(f: pl.DataFrame) -> pl.DataFrame:
        u = f["m"].struct.unnest()
        return u.select(c for c in u.columns if not c.startswith("coef"))

    assert fields(pl.concat(parts)).equals(fields(want))
    assert bank.save_bytes() == whole.save_bytes()


@pytest.mark.parametrize("name", list(KINDS))
def test_a_zero_weight_row_is_not_there(name):
    """The rows of positive weight report the same with the zero-weight rows
    left out of the stream: a zero-weight row reaches no detector and counts
    in no clock (on a clock column, its step goes to the next row)."""
    df = _frame(n=3000)
    clock = {"clock": "t", "gap_cap": 1e6}
    kw = _kind(name)
    if name.startswith(("page", "cusum")):
        kw.setdefault("drift_threshold", 20.0)
    with_zero = _fit(df, **clock, **kw)
    kept = df["w"].to_numpy() > 0
    without = _fit(df.filter(pl.col("w") > 0), **clock, **kw)
    for c in ["drift_y", "pred_y", "sigma_y"]:
        assert with_zero[c].filter(pl.Series(kept)).equals(without[c]), c
