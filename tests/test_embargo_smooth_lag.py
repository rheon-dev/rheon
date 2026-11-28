"""Embargoed and delayed-learning ``smooth_lag`` for ``kalman`` and ``hmm``.

Under ``embargo`` or a formula target, a single arriving row can release zero,
one, or many held rows. Smoothed outputs therefore become parallel list columns
holding one entry per emitting release in release order, and ``null`` (never an
empty list ``[]``) on rows where zero releases emitted:

- ``kalman``: ``smoothed_coef`` (``List(List(Float64))``), ``smoothed_se_coef``
  (``List(List(Float64))`` when ``emit_smoothed_se=True``), and
  ``smoothed_for_<t>`` (``List(<clock>)``).
- ``hmm``: ``smoothed_state<j>`` (``List(Float64)``) and ``smoothed_for``
  (``List(<clock>)``).

Non-delayed specs keep their flat dtypes and byte-identical outputs.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import warnings

import numpy as np
import polars as pl
import pytest

from conftest import _toml_value, spec_toml
from rheon import polars as rp

TIER = "essential"


# ------------------------------------------------------------------ helpers


def _series_bits(s: pl.Series) -> tuple:
    """Bit-exact recursive representation of any Series (including nested Lists)."""
    dtype = s.dtype
    if dtype == pl.Float64:
        arr = s.to_numpy(allow_copy=True)
        mask = s.is_null().to_numpy()
        return ("f", mask.tobytes(), np.where(mask, 0.0, arr).view(np.uint64).tobytes())
    if isinstance(dtype, pl.List):
        return (
            "l",
            _series_bits(s.is_null()),
            _series_bits(s.list.len()),
            _series_bits(s.explode()),
        )
    if isinstance(dtype, pl.Struct):
        fields = tuple((f, _series_bits(s.struct.field(f))) for f in dtype.to_schema())
        return ("s", _series_bits(s.is_null()), fields)
    return ("v", str(dtype), tuple(s.to_list()))


def _assert_frames_bit_equal(got: pl.DataFrame, want: pl.DataFrame, label: str = "") -> None:
    assert got.columns == want.columns, f"{label}: columns {got.columns} != {want.columns}"
    assert got.height == want.height, f"{label}: height {got.height} != {want.height}"
    for col in want.columns:
        assert got[col].dtype == want[col].dtype, (
            f"{label}.{col}: dtype {got[col].dtype} != {want[col].dtype}"
        )
        assert _series_bits(got[col]) == _series_bits(want[col]), (
            f"{label}.{col}: bit mismatch\n"
            f"got={got[col].to_list()[:10]}\n"
            f"want={want[col].to_list()[:10]}"
        )


def _assert_no_empty_lists(df: pl.DataFrame, cols: list[str]) -> None:
    for c in cols:
        n_empty = int((df[c].list.len() == 0).fill_null(False).sum())
        assert n_empty == 0, f"column {c!r} has {n_empty} empty list(s) instead of null"


def _assert_parallel_list_lengths(df: pl.DataFrame, cols: list[str]) -> None:
    base = cols[0]
    base_null = df[base].is_null()
    base_len = df[base].list.len()
    for c in cols[1:]:
        assert (df[c].is_null() == base_null).all(), f"{c} null mask disagrees with {base}"
        assert (df[c].list.len().fill_null(-1) == base_len.fill_null(-1)).all(), (
            f"{c} list lengths disagree with {base}"
        )


def _embargo_released_indices(
    t: np.ndarray, embargo: float, w: np.ndarray | None = None
) -> list[int]:
    """Exact indices released under an embargo on a numeric clock (positive-weight rows only)."""
    pending: list[tuple[float, int]] = []
    released: list[int] = []
    for i, ti in enumerate(t):
        if w is not None and w[i] <= 0.0:
            continue
        while pending and (float(ti) - pending[0][0]) >= embargo:
            released.append(pending.pop(0)[1])
        pending.append((float(ti), i))
    return released


def _bursty_kalman_frame(n: int = 160, seed: int = 2750) -> pl.DataFrame:
    """Uneven clock producing 0-, 1-, and multi-release bursts under embargo=3.0."""
    rng = np.random.default_rng(seed)
    steps = np.array([0.5, 0.5, 0.5, 0.5, 3.25, 0.4, 0.4, 0.4, 3.6, 1.2], dtype=float)
    t = np.cumsum(steps[np.arange(n) % len(steps)])
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    b0 = 1.0 + np.cumsum(0.03 * rng.standard_normal(n))
    b1 = -0.7 + np.cumsum(0.02 * rng.standard_normal(n))
    y = [
        None
        if (i % 11 == 3)
        else float(0.5 + b0[i] * x0[i] + b1[i] * x1[i] + 0.2 * rng.standard_normal())
        for i in range(n)
    ]
    z = [
        None
        if (i % 9 == 4)
        else float(-0.3 - 0.5 * b0[i] * x0[i] + b1[i] * x1[i] + 0.25 * rng.standard_normal())
        for i in range(n)
    ]
    w = 0.5 + rng.random(n)
    return pl.DataFrame({"t": t, "x0": x0, "x1": x1, "y": y, "z": z, "w": w})


def _bursty_hmm_frame(n: int = 140, seed: int = 2751) -> pl.DataFrame:
    """Two-regime Gaussian stream on an uneven clock with 0-, 1-, and multi-release bursts."""
    rng = np.random.default_rng(seed)
    steps = np.array([0.5, 0.5, 0.5, 3.2, 0.4, 0.4, 0.4, 3.6, 3.1], dtype=float)
    t = np.cumsum(steps[np.arange(n) % len(steps)])
    regimes = ((np.arange(n) // 18) % 2).astype(float)
    centers = np.where(regimes[:, None] == 0.0, np.array([-1.5, -1.5]), np.array([1.5, 1.5]))
    x = centers + 0.6 * rng.standard_normal((n, 2))
    w = 0.7 + 0.6 * rng.random(n)
    return pl.DataFrame({"t": t, "x0": x[:, 0], "x1": x[:, 1], "w": w})


# --------------------------------------------------------------- Kalman tests


def test_kalman_embargo_bit_exact_against_released_rows_oracle():
    """Exploding parallel smoothed lists from embargoed multi-target ``kalman``
    (with ``emit_smoothed_se``, null targets, and per-target ``min_weight``)
    matches a non-embargoed smoother fed the exact released rows to the bit."""
    df = _bursty_kalman_frame(160, seed=2750)
    embargo = 3.0
    lag = 4
    base_kw: dict = dict(
        targets=["y", "z"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=15.0,
        half_life=80.0,
        coef_half_life=40.0,
        weight="w",
        min_weight=[10.0, 2.0],
        smooth_lag=lag,
        emit_smoothed_se=True,
    )
    s_emb = rp.spec.kalman("k", embargo=embargo, **base_kw)
    s_plain_emb = rp.spec.kalman(
        "k",
        embargo=embargo,
        **{k: v for k, v in base_kw.items() if k not in ("smooth_lag", "emit_smoothed_se")},
    )
    s_oracle = rp.spec.kalman("k", **base_kw)

    # output_index declares list[list[f64]] and list[clock].
    idx = dict(
        zip(rp.spec.output_index(s_emb)["field"], rp.spec.output_index(s_emb)["dtype"], strict=True)
    )
    assert idx["smoothed_coef"] == "list[list[f64]]"
    assert idx["smoothed_se_coef"] == "list[list[f64]]"
    assert idx["smoothed_for_y"] == "list[clock]"
    assert idx["smoothed_for_z"] == "list[clock]"

    out = rp.ModelBank([s_emb]).fit_predict(df)["k"].struct.unnest()
    plain = rp.ModelBank([s_plain_emb]).fit_predict(df)["k"].struct.unnest()

    # Smoother never perturbs the filter's predictions, residuals, or coef.
    for col in ("pred_y", "pred_z", "resid_y", "resid_z", "weight_sum", "coef"):
        assert _series_bits(out[col]) == _series_bits(plain[col]), col

    sm_cols = ["smoothed_coef", "smoothed_se_coef", "smoothed_for_y", "smoothed_for_z"]
    assert out["smoothed_coef"].dtype == pl.List(pl.List(pl.Float64))
    assert out["smoothed_se_coef"].dtype == pl.List(pl.List(pl.Float64))
    assert out["smoothed_for_y"].dtype == pl.List(pl.Float64)
    assert out["smoothed_for_z"].dtype == pl.List(pl.Float64)

    _assert_no_empty_lists(out, sm_cols)
    _assert_parallel_list_lengths(out, sm_cols)

    # Activation proof: 0-release (null), 1-release, and multi-release (>=2) rows all occur.
    lens = out["smoothed_coef"].list.len()
    assert int(out["smoothed_coef"].is_null().sum()) > 10
    assert int((lens == 1).fill_null(False).sum()) > 5
    assert int((lens >= 2).fill_null(False).sum()) > 10

    # Oracle: feed exact released rows in release order to a non-embargoed smoother.
    rel_idx = _embargo_released_indices(df["t"].to_numpy(), embargo)
    oracle = rp.ModelBank([s_oracle]).fit_predict(df[rel_idx])["k"].struct.unnest()

    exploded = out.filter(pl.col("smoothed_coef").is_not_null()).select(sm_cols).explode(sm_cols)
    want = oracle.filter(pl.col("smoothed_coef").is_not_null()).select(sm_cols)
    assert exploded.height > 100
    _assert_frames_bit_equal(exploded, want, "kalman_embargo_oracle")


@pytest.mark.parametrize(
    ("flt", "family", "link", "ukf_rule"),
    [
        ("ekf", "binomial", "logit", None),
        ("ukf", "poisson", "log", "scalar_index"),
        ("ukf", "poisson", "log", "merwe"),
    ],
)
def test_kalman_ekf_ukf_embargo_bit_exact_against_released_rows_oracle(
    flt: str, family: str, link: str, ukf_rule: str | None
):
    """EKF and UKF smoothers under ``embargo`` match their non-embargoed
    oracles on the exact released rows to the bit."""
    rng = np.random.default_rng(4970)
    n = 110
    steps = np.array([0.5, 0.5, 0.5, 3.0, 0.6, 0.6, 3.2], dtype=float)
    t = np.cumsum(steps[np.arange(n) % len(steps)])
    x0 = rng.standard_normal(n) * 0.5
    x1 = rng.standard_normal(n) * 0.5
    eta = 0.2 + 0.6 * x0 - 0.4 * x1
    if family == "binomial":
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    df = pl.DataFrame({"t": t, "x0": x0, "x1": x1, "y": y})

    embargo = 2.5
    kw: dict = dict(
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=15.0,
        half_life=60.0,
        family=family,
        link=link,
        filter=flt,
        q=[1e-3, 1e-3, 1e-3],
        p0=1.0,
        min_weight=0.0,
        smooth_lag=3,
        emit_smoothed_se=True,
    )
    if ukf_rule is not None:
        kw["ukf_rule"] = ukf_rule
    s_emb = rp.spec.kalman("k", embargo=embargo, **kw)
    s_oracle = rp.spec.kalman("k", **kw)

    out = rp.ModelBank([s_emb]).fit_predict(df)["k"].struct.unnest()
    sm_cols = ["smoothed_coef", "smoothed_se_coef", "smoothed_for_y"]
    _assert_no_empty_lists(out, sm_cols)
    _assert_parallel_list_lengths(out, sm_cols)
    assert int((out["smoothed_coef"].list.len() >= 2).fill_null(False).sum()) > 5

    rel_idx = _embargo_released_indices(t, embargo)
    oracle = rp.ModelBank([s_oracle]).fit_predict(df[rel_idx])["k"].struct.unnest()
    exploded = out.filter(pl.col("smoothed_coef").is_not_null()).select(sm_cols).explode(sm_cols)
    want = oracle.filter(pl.col("smoothed_coef").is_not_null()).select(sm_cols)
    _assert_frames_bit_equal(exploded, want, f"kalman_{flt}_{ukf_rule}")


def test_kalman_formula_target_smooth_lag():
    """A look-ahead formula target delays learning (resolving 0, 1, or many
    held rows per arriving row), so ``smooth_lag`` emits list-typed columns
    that are chunk- and save/load-invariant."""
    rng = np.random.default_rng(2755)
    n = 120
    steps = np.array([0.5, 0.5, 0.5, 4.5, 0.5, 0.5, 5.0], dtype=float)
    t = np.cumsum(steps[np.arange(n) % len(steps)])
    x0 = rng.standard_normal(n)
    mid = 100.0 + np.cumsum(0.2 * x0 + 0.5 * rng.standard_normal(n))
    df = pl.DataFrame({"t": t, "x0": x0, "mid": mid})

    fwd = (
        rp.ops.ewm_mean("mid", half_life=5.0, window_size=3.0, reverse=True) - pl.col("mid")
    ).alias("fwd")
    base_kw: dict = dict(
        targets=[fwd],
        features=["x0"],
        clock="t",
        gap_cap=20.0,
        half_life=50.0,
        coef_half_life=30.0,
        min_weight=0.0,
        smooth_lag=3,
        emit_smoothed_se=True,
    )
    s = rp.spec.kalman("k", embargo=5.0, **base_kw)
    s_no_emb = rp.spec.kalman("k", **base_kw)

    for sp in (s, s_no_emb):
        idx = dict(
            zip(rp.spec.output_index(sp)["field"], rp.spec.output_index(sp)["dtype"], strict=True)
        )
        assert idx["smoothed_coef"] == "list[list[f64]]"
        assert idx["smoothed_se_coef"] == "list[list[f64]]"
        assert idx["smoothed_for_fwd"] == "list[clock]"

    sm_cols = ["smoothed_coef", "smoothed_se_coef", "smoothed_for_fwd"]
    whole = rp.ModelBank([s]).fit_predict(df)["k"].struct.unnest()
    _assert_no_empty_lists(whole, sm_cols)
    _assert_parallel_list_lengths(whole, sm_cols)
    assert int((whole["smoothed_coef"].list.len() >= 2).fill_null(False).sum()) > 5

    # Look-ahead formula without embargo works in fit() + smoothed_tail() + save/load.
    b_fit = rp.ModelBank([s_no_emb])
    b_fit.fit(df[:60])
    b_fit_reloaded = rp.ModelBank.load_bytes(b_fit.save_bytes(), [s_no_emb])
    b_fit_reloaded.fit(df[60:])
    b_fit_whole = rp.ModelBank([s_no_emb])
    b_fit_whole.fit(df)
    assert b_fit_whole.smoothed_tail("k").height == 3
    _assert_frames_bit_equal(
        b_fit_reloaded.smoothed_tail("k"), b_fit_whole.smoothed_tail("k"), "formula_fit_tail"
    )

    # Chunk invariance + save/load mid-stream.
    for size in (1, 13):
        b = rp.ModelBank([s])
        parts = [b.fit_predict(df.slice(i, size)) for i in range(0, n, size)]
        got = pl.concat(parts)["k"].struct.unnest()
        _assert_frames_bit_equal(got.drop("coef"), whole.drop("coef"), f"formula_chunk_{size}")

    b = rp.ModelBank([s])
    head = b.fit_predict(df[:55])["k"].struct.unnest().drop("coef")
    loaded = rp.ModelBank.load_bytes(b.save_bytes(), [s])
    tail = loaded.fit_predict(df[55:])["k"].struct.unnest().drop("coef")
    _assert_frames_bit_equal(pl.concat([head, tail]), whole.drop("coef"), "formula_save_load")


# ------------------------------------------------------------------ HMM tests


def test_hmm_embargo_bit_exact_against_released_rows_oracle_and_hmmlearn():
    """Embargoed ``hmm`` emits parallel ``List(Float64)`` / ``List(<clock>)``
    columns matching both the non-embargoed released-row oracle to the bit and
    ``hmmlearn`` at fixed parameters to 1e-12."""
    from hmmlearn.hmm import GaussianHMM

    df = _bursty_hmm_frame(140, seed=2751)
    embargo = 3.0
    lag = 4
    m0 = [-1.5, -1.5, 1.5, 1.5]
    c0 = [0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5]
    pi0 = np.array([[0.9, 0.1], [0.15, 0.85]])

    # 1. Fixed parameters: bit-exact vs non-embargoed HMM AND <= 1e-12 vs hmmlearn.
    fixed_kw: dict = dict(
        features=["x0", "x1"],
        n_components=2,
        precision_prior=1e-2,
        clock="t",
        gap_cap=15.0,
        half_life=1e9,
        learn=False,
        means=m0,
        covs=c0,
        transition=pi0.ravel().tolist(),
        smooth_lag=lag,
    )
    s_emb = rp.spec.hmm("h", embargo=embargo, **fixed_kw)
    s_plain_emb = rp.spec.hmm(
        "h", embargo=embargo, **{k: v for k, v in fixed_kw.items() if k != "smooth_lag"}
    )
    s_oracle = rp.spec.hmm("h", **fixed_kw)

    idx = dict(
        zip(rp.spec.output_index(s_emb)["field"], rp.spec.output_index(s_emb)["dtype"], strict=True)
    )
    assert idx["smoothed_state0"] == "list[f64]"
    assert idx["smoothed_state1"] == "list[f64]"
    assert idx["smoothed_for"] == "list[clock]"

    out = rp.ModelBank([s_emb]).fit_predict(df)["h"].struct.unnest()
    plain = rp.ModelBank([s_plain_emb]).fit_predict(df)["h"].struct.unnest()
    for col in (
        "filtered_state0",
        "filtered_state1",
        "predicted_state0",
        "predicted_state1",
        "state",
        "loglik",
        "weight_sum",
    ):
        assert _series_bits(out[col]) == _series_bits(plain[col]), col

    sm_cols = ["smoothed_state0", "smoothed_state1", "smoothed_for"]
    _assert_no_empty_lists(out, sm_cols)
    _assert_parallel_list_lengths(out, sm_cols)
    lens = out["smoothed_state0"].list.len()
    assert int(out["smoothed_state0"].is_null().sum()) > 10
    assert int((lens == 1).fill_null(False).sum()) > 5
    assert int((lens >= 2).fill_null(False).sum()) > 10

    rel_idx = _embargo_released_indices(df["t"].to_numpy(), embargo)
    oracle = rp.ModelBank([s_oracle]).fit_predict(df[rel_idx])["h"].struct.unnest()
    exploded = out.filter(pl.col("smoothed_state0").is_not_null()).select(sm_cols).explode(sm_cols)
    want = oracle.filter(pl.col("smoothed_state0").is_not_null()).select(sm_cols)
    _assert_frames_bit_equal(exploded, want, "hmm_fixed_embargo_oracle")

    # Cross-check against hmmlearn on the released prefix.
    x_rel = df[rel_idx].select("x0", "x1").to_numpy()
    t_rel = df[rel_idx]["t"].to_numpy()
    gh = GaussianHMM(n_components=2, covariance_type="full", init_params="")
    gh.startprob_ = np.array([0.5, 0.5]) @ pi0
    gh.transmat_ = pi0
    gh.means_ = np.array(m0).reshape(2, 2)
    gh.covars_ = np.array(c0).reshape(2, 2, 2) + 1e-2 * np.eye(2)[None, :, :]
    assert exploded.height == len(rel_idx) - lag
    for step_idx in range(lag, len(rel_idx)):
        row_i = step_idx - lag
        ref = gh.predict_proba(x_rel[: step_idx + 1])[step_idx - lag]
        assert exploded["smoothed_for"][row_i] == t_rel[step_idx - lag]
        assert exploded["smoothed_state0"][row_i] == pytest.approx(ref[0], abs=1e-12)
        assert exploded["smoothed_state1"][row_i] == pytest.approx(ref[1], abs=1e-12)

    # 2. Online learning + spherical covariance + warm_rows + min_weight + weights.
    learn_kw: dict = dict(
        features=["x0", "x1"],
        n_components=2,
        covariance_type="spherical",
        precision_prior=0.1,
        warm_rows=10,
        clock="t",
        gap_cap=15.0,
        half_life=30.0,
        weight="w",
        min_weight=15.0,
        smooth_lag=3,
    )
    s_l_emb = rp.spec.hmm("h", embargo=embargo, **learn_kw)
    s_l_oracle = rp.spec.hmm("h", **learn_kw)
    out_l = rp.ModelBank([s_l_emb]).fit_predict(df)["h"].struct.unnest()
    oracle_l = rp.ModelBank([s_l_oracle]).fit_predict(df[rel_idx])["h"].struct.unnest()
    _assert_no_empty_lists(out_l, sm_cols)
    _assert_parallel_list_lengths(out_l, sm_cols)
    exploded_l = (
        out_l.filter(pl.col("smoothed_state0").is_not_null()).select(sm_cols).explode(sm_cols)
    )
    want_l = oracle_l.filter(pl.col("smoothed_state0").is_not_null()).select(sm_cols)
    assert exploded_l.height > 80
    _assert_frames_bit_equal(exploded_l, want_l, "hmm_learn_embargo_oracle")


# ----------------------------------------------------------- clock dtype tests


@pytest.mark.parametrize(
    (
        "clock_kind",
        "clock_series",
        "want_inner_dtype",
        "gap_cap",
        "half_life",
        "coef_hl",
        "embargo",
    ),
    [
        (
            "float64",
            pl.Series("t", np.cumsum([0.5, 0.5, 3.0] * 20), dtype=pl.Float64),
            pl.Float64,
            15.0,
            50.0,
            30.0,
            2.5,
        ),
        (
            "int64",
            pl.Series("t", np.cumsum([1, 1, 5] * 20), dtype=pl.Int64),
            pl.Int64,
            20,
            50.0,
            30.0,
            4,
        ),
        (
            "int32",
            pl.Series("t", np.cumsum([1, 1, 5] * 20), dtype=pl.Int32),
            pl.Int32,
            20,
            50.0,
            30.0,
            4,
        ),
        (
            "uint32",
            pl.Series("t", np.cumsum([1, 1, 5] * 20), dtype=pl.UInt32),
            pl.UInt32,
            20,
            50.0,
            30.0,
            4,
        ),
        (
            "date",
            pl.Series(
                "t",
                [
                    dt.date(2025, 1, 1) + dt.timedelta(days=int(d))
                    for d in np.cumsum([1, 1, 5] * 20)
                ],
                dtype=pl.Date,
            ),
            pl.Date,
            "30d",
            "50d",
            "30d",
            "4d",
        ),
        (
            "datetime_us",
            pl.Series(
                "t",
                [
                    dt.datetime(2025, 1, 1) + dt.timedelta(minutes=int(m))
                    for m in np.cumsum([1, 1, 5] * 20)
                ],
                dtype=pl.Datetime("us"),
            ),
            pl.Datetime("us"),
            "30m",
            "50m",
            "30m",
            "4m",
        ),
        (
            "datetime_ms_utc",
            pl.Series(
                "t",
                [
                    dt.datetime(2025, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=int(s))
                    for s in np.cumsum([1, 1, 5] * 20)
                ],
                dtype=pl.Datetime("ms", "UTC"),
            ),
            pl.Datetime("ms", "UTC"),
            "30s",
            "50s",
            "30s",
            "4s",
        ),
        (
            "duration_ms",
            pl.Series(
                "t",
                [dt.timedelta(milliseconds=int(ms)) for ms in np.cumsum([100, 100, 500] * 20)],
                dtype=pl.Duration("ms"),
            ),
            pl.Duration("ms"),
            "3s",
            "5s",
            "3s",
            "400ms",
        ),
        (
            "row_index",
            None,
            pl.Int64,
            None,
            50.0,
            30.0,
            3,
        ),
    ],
)
def test_all_clock_dtypes_preserved_including_empty_and_lazy(
    clock_kind: str,
    clock_series: pl.Series | None,
    want_inner_dtype: pl.DataType,
    gap_cap: object,
    half_life: object,
    coef_hl: object,
    embargo: object,
):
    """Every clock dtype (and row index without a clock) is preserved inside
    ``List(<clock>)`` across eager, empty 0-row, and lazy execution for both
    ``kalman`` and ``hmm``."""
    rng = np.random.default_rng(2760)
    n = 60
    cols: dict[str, object] = {
        "x0": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "y": rng.standard_normal(n),
    }
    if clock_series is not None:
        cols["t"] = clock_series
    df = pl.DataFrame(cols)

    clock_kw = {"clock": "t", "gap_cap": gap_cap} if clock_series is not None else {}
    s_kal = rp.spec.kalman(
        "k",
        targets=["y"],
        features=["x0", "x1"],
        half_life=half_life,
        coef_half_life=coef_hl,
        embargo=embargo,
        smooth_lag=2,
        emit_smoothed_se=True,
        min_weight=0.0,
        **clock_kw,  # type: ignore[arg-type]
    )
    s_hmm = rp.spec.hmm(
        "h",
        features=["x0", "x1"],
        n_components=2,
        precision_prior=0.1,
        half_life=half_life,
        means=[-1.0, -1.0, 1.0, 1.0],
        covs=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        embargo=embargo,
        smooth_lag=2,
        **clock_kw,  # type: ignore[arg-type]
    )

    want_for_dtype = pl.List(want_inner_dtype)
    for spec, name, for_field in ((s_kal, "k", "smoothed_for_y"), (s_hmm, "h", "smoothed_for")):
        out = rp.ModelBank([spec]).fit_predict(df)[name].struct.unnest()
        assert out[for_field].dtype == want_for_dtype, f"{clock_kind}/{name}"
        assert out[for_field].drop_nulls().len() >= 15
        assert int(out[for_field].list.len().sum()) >= 50

        # Empty 0-row frame keeps exact nested schema.
        empty = rp.ModelBank([spec]).fit_predict(df.head(0))[name].struct.unnest()
        assert empty[for_field].dtype == want_for_dtype, f"empty {clock_kind}/{name}"

        # LazyFrame IO plugin matches eager bit-for-bit.
        lazy_out = df.lazy().rheon.fit_predict([spec]).collect()[name].struct.unnest()
        _assert_frames_bit_equal(
            lazy_out.drop("coef"), out.drop("coef"), f"lazy_{clock_kind}_{name}"
        )


def test_non_embargo_specs_remain_flat_and_unchanged():
    """Without ``embargo``, ``label_clock``, or formula targets, smoothed
    fields stay completely flat."""
    df = _bursty_kalman_frame(80, seed=2761)
    s_kal = rp.spec.kalman(
        "k",
        targets=["y", "z"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=15.0,
        half_life=50.0,
        coef_half_life=30.0,
        smooth_lag=3,
        emit_smoothed_se=True,
    )
    s_hmm = rp.spec.hmm(
        "h",
        features=["x0", "x1"],
        n_components=2,
        precision_prior=0.1,
        clock="t",
        gap_cap=15.0,
        half_life=50.0,
        means=[-1.0, -1.0, 1.0, 1.0],
        covs=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        smooth_lag=3,
    )
    k_out = rp.ModelBank([s_kal]).fit_predict(df)["k"].struct.unnest()
    assert k_out["smoothed_coef"].dtype == pl.List(pl.Float64)
    assert k_out["smoothed_se_coef"].dtype == pl.List(pl.Float64)
    assert k_out["smoothed_for_y"].dtype == pl.Float64
    h_out = rp.ModelBank([s_hmm]).fit_predict(df)["h"].struct.unnest()
    assert h_out["smoothed_state0"].dtype == pl.Float64
    assert h_out["smoothed_for"].dtype == pl.Float64


# ------------------------------------ chunking, save/load, last_row, predict


def test_chunk_invariance_save_load_last_row_predict_and_write_json(tmp_path):
    """Chunk invariance (1 chunk == 1-row == random chunks) and mid-stream
    save/load (msgpack + JSON) at 0-release, 1-release, and >1-release rows,
    plus ``last_row()`` and ``predict()``."""
    df_k = _bursty_kalman_frame(120, seed=2762)
    s_k = rp.spec.kalman(
        "k",
        targets=["y", "z"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=15.0,
        embargo=3.0,
        half_life=60.0,
        coef_half_life=35.0,
        weight="w",
        min_weight=[10.0, 1.0],
        smooth_lag=3,
        emit_smoothed_se=True,
    )
    df_h = _bursty_hmm_frame(120, seed=2763)
    s_h = rp.spec.hmm(
        "h",
        features=["x0", "x1"],
        n_components=2,
        covariance_type="spherical",
        precision_prior=0.1,
        warm_rows=10,
        clock="t",
        gap_cap=15.0,
        embargo=3.0,
        half_life=30.0,
        weight="w",
        min_weight=12.0,
        smooth_lag=3,
    )

    for label, df, spec, name, probe_col, sm_cols in (
        (
            "kalman",
            df_k,
            s_k,
            "k",
            "smoothed_coef",
            ["smoothed_coef", "smoothed_se_coef", "smoothed_for_y", "smoothed_for_z"],
        ),
        (
            "hmm",
            df_h,
            s_h,
            "h",
            "smoothed_state0",
            ["smoothed_state0", "smoothed_state1", "smoothed_for"],
        ),
    ):
        whole = rp.ModelBank([spec]).fit_predict(df)[name].struct.unnest()

        # 1-row chunks and random chunk sizes match whole bit-for-bit.
        rng = np.random.default_rng(2764)
        for mode in ("ones", "random"):
            b = rp.ModelBank([spec])
            parts = []
            pos = 0
            while pos < df.height:
                step = 1 if mode == "ones" else int(rng.integers(1, 19))
                parts.append(b.fit_predict(df.slice(pos, step)))
                pos += step
            got = pl.concat(parts)[name].struct.unnest()
            _assert_frames_bit_equal(got.drop("coef"), whole.drop("coef"), f"{label}_{mode}")

        # Pick explicit split boundaries landing on 0-, 1-, and >=2-emission rows (after row 30).
        lens = whole[probe_col].list.len().to_list()
        row0 = next(i for i in range(30, df.height - 10) if lens[i] is None)
        row1 = next(i for i in range(30, df.height - 10) if lens[i] == 1)
        row_many = next(
            i for i in range(30, df.height - 10) if lens[i] is not None and lens[i] >= 2
        )

        for tag, at in (("zero_rel", row0), ("one_rel", row1), ("many_rel", row_many)):
            b = rp.ModelBank([spec])
            head = b.fit_predict(df[: at + 1])[name].struct.unnest()

            # last_row() matches row `at` of head bit-for-bit (including smoothed lists!).
            lr_before = b.last_row(name).drop("spec", "group")
            _assert_frames_bit_equal(lr_before, head.slice(at, 1), f"{label}_{tag}_last_row_before")

            # Msgpack save/load preserves last_row(), predict(), and continuation bit-for-bit.
            raw = b.save_bytes()
            loaded = rp.ModelBank.load_bytes(raw, [spec])
            lr_after = loaded.last_row(name).drop("spec", "group")
            _assert_frames_bit_equal(lr_after, head.slice(at, 1), f"{label}_{tag}_last_row_after")

            # write_json() self-validates lossless JSON round-trip of LastRow (including NaNs!).
            json_path = tmp_path / f"{label}_{tag}.json"
            b.write_json(json_path)
            text = json_path.read_text(encoding="utf-8")
            assert text == b.write_json()
            assert text.startswith("{")

            # predict() emits null (never []) in all smoothed list columns.
            pred = loaded.predict(df[at + 1 : at + 12])[name].struct.unnest()
            for c in sm_cols:
                assert pred[c].null_count() == pred.height, f"{label}_{tag}_predict_{c}"

            # Continuation matches unbroken whole frame bit-for-bit.
            tail = loaded.fit_predict(df[at + 1 :])[name].struct.unnest()
            joined = pl.concat([head.drop("coef"), tail.drop("coef")])
            _assert_frames_bit_equal(joined, whole.drop("coef"), f"{label}_{tag}_continue")


# ---------------------------------------------- Rule 9 zero-weight isolation


def test_rule9_zero_weight_rows_under_embargo():
    """Hard Rule 9 under embargo (score as predict): zero-weight rows do not
    release held rows and are neither learned nor buffered, so their smoothed
    list columns are ``null``, and the positive-weight rows match the stream
    with zero-weight rows removed to the bit."""
    df_k = _bursty_kalman_frame(120, seed=2765)
    w_k = df_k["w"].to_numpy().copy()
    zero_rows = [5, 14, 29, 44, 67, 89, 104]
    w_k[zero_rows] = 0.0
    df_kz = df_k.with_columns(pl.Series("w", w_k))
    df_knz = df_kz.filter(pl.col("w") > 0)

    s_k = rp.spec.kalman(
        "k",
        targets=["y", "z"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=15.0,
        embargo=3.0,
        half_life=60.0,
        coef_half_life=35.0,
        weight="w",
        min_weight=0.0,
        smooth_lag=3,
        emit_smoothed_se=True,
    )
    bk_z = rp.ModelBank([s_k])
    out_kz = bk_z.fit_predict(df_kz)["k"].struct.unnest()
    bk_nz = rp.ModelBank([s_k])
    out_knz = bk_nz.fit_predict(df_knz)["k"].struct.unnest()

    k_sm_cols = ["smoothed_coef", "smoothed_se_coef", "smoothed_for_y", "smoothed_for_z"]
    for zr in zero_rows:
        for c in k_sm_cols:
            assert out_kz[c][zr] is None, f"kalman weight-0 row {zr} column {c} not null"
    _assert_frames_bit_equal(
        out_kz.filter(pl.Series(w_k > 0)).drop("coef"),
        out_knz.drop("coef"),
        "kalman_rule9",
    )
    _assert_frames_bit_equal(bk_z.smoothed_tail("k"), bk_nz.smoothed_tail("k"), "kalman_rule9_tail")

    # Same for HMM.
    df_h = _bursty_hmm_frame(120, seed=2766)
    w_h = df_h["w"].to_numpy().copy()
    w_h[zero_rows] = 0.0
    df_hz = df_h.with_columns(pl.Series("w", w_h))
    df_hnz = df_hz.filter(pl.col("w") > 0)

    s_h = rp.spec.hmm(
        "h",
        features=["x0", "x1"],
        n_components=2,
        precision_prior=0.1,
        clock="t",
        gap_cap=15.0,
        embargo=3.0,
        half_life=30.0,
        weight="w",
        means=[-1.5, -1.5, 1.5, 1.5],
        covs=[0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5],
        smooth_lag=3,
    )
    out_hz = rp.ModelBank([s_h]).fit_predict(df_hz)["h"].struct.unnest()
    out_hnz = rp.ModelBank([s_h]).fit_predict(df_hnz)["h"].struct.unnest()
    h_sm_cols = ["smoothed_state0", "smoothed_state1", "smoothed_for"]
    for zr in zero_rows:
        for c in h_sm_cols:
            assert out_hz[c][zr] is None, f"hmm weight-0 row {zr} column {c} not null"
    _assert_frames_bit_equal(
        out_hz.filter(pl.Series(w_h > 0)).drop("coef"),
        out_hnz.drop("coef"),
        "hmm_rule9",
    )


# ----------------- multi-group, multi-half-life, unnest, tail, session reset


def test_multi_group_multi_half_life_unnest_smoothed_tail_and_session_reset():
    """Interleaved groups, half-life grids (``@h20``, ``@h100``), ``output_fields``,
    ``output_index``, ``coef_fields``, ``coef_index``, ``unnest``,
    ``smoothed_tail`` (including a group with 0 releases), and session reset."""
    g0 = _bursty_kalman_frame(80, seed=2770).with_columns(pl.lit("A").alias("g"))
    g1 = _bursty_kalman_frame(70, seed=2771).with_columns(pl.lit("B").alias("g"))
    # Group C has only 2 rows spaced 0.5 apart -> 0 rows released under embargo=3.0.
    g2 = _bursty_kalman_frame(2, seed=2772).with_columns(pl.lit("C").alias("g"))
    rows = []
    for i in range(max(g0.height, g1.height, g2.height)):
        for g in (g0, g1, g2):
            if i < g.height:
                rows.append(g.slice(i, 1))
    df = pl.concat(rows)

    s_k = rp.spec.kalman(
        "k",
        targets=["y", "z"],
        features=["x0", "x1"],
        group="g",
        clock="t",
        gap_cap=15.0,
        embargo=3.0,
        half_life=[20.0, 100.0],
        coef_half_life=35.0,
        weight="w",
        min_weight=0.0,
        smooth_lag=3,
        emit_smoothed_se=True,
    )
    s_h = rp.spec.hmm(
        "h",
        features=["x0", "x1"],
        group="g",
        n_components=2,
        precision_prior=0.1,
        clock="t",
        gap_cap=15.0,
        embargo=3.0,
        half_life=[20.0, 100.0],
        weight="w",
        means=[-1.5, -1.5, 1.5, 1.5],
        covs=[0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5],
        smooth_lag=3,
    )

    # Verify output_fields, output_index, coef_fields, coef_index.
    k_idx = rp.spec.output_index(s_k)
    k_dtypes = dict(zip(k_idx["field"], k_idx["dtype"], strict=True))
    for suf in ("@h20", "@h100"):
        assert k_dtypes[f"smoothed_coef{suf}"] == "list[list[f64]]"
        assert k_dtypes[f"smoothed_se_coef{suf}"] == "list[list[f64]]"
        assert k_dtypes[f"smoothed_for_y{suf}"] == "list[clock]"
        assert k_dtypes[f"smoothed_for_z{suf}"] == "list[clock]"
    assert set(rp.spec.output_fields(s_k)) == set(k_idx["field"].to_list())
    assert rp.spec.coef_fields(s_k).height == 2 * rp.spec.coef_index(s_k).height

    h_idx = rp.spec.output_index(s_h)
    h_dtypes = dict(zip(h_idx["field"], h_idx["dtype"], strict=True))
    for suf in ("@h20", "@h100"):
        assert h_dtypes[f"smoothed_state0{suf}"] == "list[f64]"
        assert h_dtypes[f"smoothed_state1{suf}"] == "list[f64]"
        assert h_dtypes[f"smoothed_for{suf}"] == "list[clock]"

    bank = rp.ModelBank([s_k, s_h])
    out = bank.fit_predict(df)

    # Multi-group isolation: each group matches running that group alone bit-for-bit.
    for gname, gdf in (("A", g0), ("B", g1), ("C", g2)):
        alone = rp.ModelBank([s_k, s_h]).fit_predict(gdf)
        part = out.filter(pl.col("g") == gname)
        for name in ("k", "h"):
            _assert_frames_bit_equal(
                part[name].struct.unnest().drop("coef@h20", "coef@h100"),
                alone[name].struct.unnest().drop("coef@h20", "coef@h100"),
                f"group_{gname}_{name}",
            )

    # Group C had 0 released rows -> all smoothed columns are null, smoothed_tail is empty.
    c_k = out.filter(pl.col("g") == "C")["k"].struct.unnest()
    assert c_k["smoothed_coef@h20"].null_count() == 2
    assert bank.smoothed_tail("k", group="C").height == 0

    # smoothed_tail for A matches the non-embargoed oracle on A's released rows.
    rel_a = _embargo_released_indices(g0["t"].to_numpy(), 3.0)
    s_k_oracle = {**s_k}
    s_k_oracle.pop("embargo")
    b_oracle = rp.ModelBank([s_k_oracle])
    b_oracle.fit(g0[rel_a])
    _assert_frames_bit_equal(
        bank.smoothed_tail("k", group="A"), b_oracle.smoothed_tail("k", group="A"), "tail_A"
    )

    # unnest() works cleanly both with and without separator.
    flat_k = rp.unnest(out.select("t", "g", "k"), [s_k])
    assert flat_k["smoothed_coef@h20"].dtype == pl.List(pl.List(pl.Float64))
    assert flat_k["smoothed_for_y@h100"].dtype == pl.List(pl.Float64)
    flat_both = rp.unnest(out, [s_k, s_h], separator=".")
    assert flat_both["k.smoothed_coef@h20"].dtype == pl.List(pl.List(pl.Float64))
    assert flat_both["h.smoothed_state0@h100"].dtype == pl.List(pl.Float64)
    assert flat_both["h.smoothed_for@h20"].dtype == pl.List(pl.Float64)

    # Session reset drops pending buffer and clears smoother rings cleanly.
    n_sess = 100
    df_s = _bursty_hmm_frame(n_sess, seed=2773).with_columns(
        pl.Series("s", [0] * 50 + [1] * 50),
        pl.Series("y", np.arange(n_sess, dtype=float) * 0.1),
    )
    s_ks = rp.spec.kalman(
        "k",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=15.0,
        embargo=3.0,
        session="s",
        session_gap="reset",
        half_life=50.0,
        coef_half_life=30.0,
        min_weight=0.0,
        smooth_lag=3,
    )
    s_hs = rp.spec.hmm(
        "h",
        features=["x0", "x1"],
        n_components=2,
        precision_prior=0.1,
        clock="t",
        gap_cap=15.0,
        embargo=3.0,
        session="s",
        session_gap="reset",
        half_life=50.0,
        means=[-1.5, -1.5, 1.5, 1.5],
        covs=[0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5],
        smooth_lag=3,
    )
    out_s = rp.ModelBank([s_ks, s_hs]).fit_predict(df_s)
    # Session 1 (rows 50..100) must match a fresh bank fed only rows 50..100 bit-for-bit!
    fresh_s = rp.ModelBank([s_ks, s_hs]).fit_predict(df_s[50:])
    for name in ("k", "h"):
        _assert_frames_bit_equal(
            out_s[50:][name].struct.unnest().drop("coef"),
            fresh_s[name].struct.unnest().drop("coef"),
            f"session_reset_{name}",
        )


# ------------------------------------------------------------ CLI TOML parity


@pytest.mark.parametrize("fmt", ["parquet", "ipc"])
def test_cli_toml_parity_for_embargo_smooth_lag(rheon_cli, tmp_path, fmt: str):
    """CLI TOML runner matches ``ModelBank.fit_predict`` and ``save_bytes``
    bit-for-bit for both embargoed ``kalman`` and embargoed ``hmm``."""
    df = _bursty_kalman_frame(110, seed=2775)
    specs = [
        rp.spec.kalman(
            "k",
            targets=["y", "z"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=15.0,
            embargo=3.0,
            half_life=60.0,
            coef_half_life=35.0,
            weight="w",
            min_weight=2.0,
            smooth_lag=3,
            emit_smoothed_se=True,
        ),
        rp.spec.hmm(
            "h",
            features=["x0", "x1"],
            n_components=2,
            covariance_type="spherical",
            precision_prior=0.1,
            warm_rows=10,
            clock="t",
            gap_cap=15.0,
            embargo=3.0,
            half_life=30.0,
            weight="w",
            smooth_lag=3,
        ),
    ]
    bank = rp.ModelBank(specs, threads=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = bank.fit_predict(df)

    src = tmp_path / "in.parquet"
    dst = tmp_path / f"out.{fmt}"
    state = tmp_path / "bank.state"
    df.write_parquet(src)
    top = {
        "input": src.as_posix(),
        "output": dst.as_posix(),
        "save_state": state.as_posix(),
        "chunk_size": 23,
        "threads": 2,
        "output_layout": "struct",
    }
    cfg = tmp_path / "bank.toml"
    cfg.write_text(
        "\n".join(
            [f"{k} = {_toml_value(v)}" for k, v in top.items()] + [spec_toml(s) for s in specs]
        )
        + "\n",
        encoding="utf-8",
    )
    res = subprocess.run(
        [str(rheon_cli), "--config", str(cfg), "--quiet"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    got = pl.read_parquet(dst) if fmt == "parquet" else pl.read_ipc(dst)
    for name in ("k", "h"):
        _assert_frames_bit_equal(
            got[name].struct.unnest().drop("coef"),
            want[name].struct.unnest().drop("coef"),
            f"cli_{fmt}_{name}",
        )
    assert state.read_bytes() == bank.save_bytes()
