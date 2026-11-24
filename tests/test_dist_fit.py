"""``dist_fit`` through the bank: each field against its definition, row by
row, and what the bank owes every model -- out-of-sample reads, chunk
invariance, a save mid-stream, zero weights and nulls, weights as relative
evidence, its refusals and its output schema.

The oracles: ``loc`` and ``scale`` are the normal's weighted
maximum-likelihood fit over the rows before each row, at their decayed
weights, written out in numpy two-pass (statsmodels' ``DescrStatsW`` holds
the same on a clock in ``tests/test_second_opinion_dist.py``); the
quantile, ``pit`` and ``log_score`` are ``scipy.stats.norm``'s ``ppf``,
``cdf`` and ``logpdf`` at the reported ``loc`` and ``scale``; and the
expected shortfall is ``scipy.stats.norm.expect`` of ``y`` below the
quantile, by numerical integration, over the level.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from scipy import stats

from rheon import polars as rp

TIER = "essential"

HALF_LIFE = 30.0


def _frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = 3.0 + 1.5 * rng.standard_normal(n)
    z = -2.0 + 0.5 * rng.standard_t(5, n)
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "g": np.where(np.arange(n) % 3 == 0, "a", "b"),
            "y": y,
            "z": np.where(rng.uniform(size=n) < 0.1, np.nan, z),
            "w": np.where(rng.uniform(size=n) < 0.1, 0.0, rng.uniform(0.5, 2.0, n)),
        }
    ).with_columns(pl.col("z").fill_nan(None))


def _spec(**kw) -> dict:
    return rp.spec.dist_fit(
        "m",
        targets=kw.pop("targets", ["y"]),
        quantiles=kw.pop("quantiles", [0.05, 0.5, 0.99]),
        es_levels=kw.pop("es_levels", [0.025, 0.1]),
        half_life=kw.pop("half_life", HALF_LIFE),
        **kw,
    )


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("m")


def _longhand(y: np.ndarray, w: np.ndarray, floc: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Each row's fit from the rows before it, two-pass: row ``i``'s weight
    ``w_i`` times ``0.5 ** (k / HALF_LIFE)``, ``k`` the rows of positive
    weight after it (a row of weight 0 moves no clock)."""
    n = len(y)
    loc, scale = np.full(n, np.nan), np.full(n, np.nan)
    learned = w > 0
    for r in range(n):
        idx = np.flatnonzero(learned[:r])
        if len(idx) == 0:
            continue
        age = np.array([learned[i + 1 : r].sum() for i in idx], dtype=float)
        om = w[idx] * 0.5 ** (age / HALF_LIFE)
        m = floc if floc is not None else np.sum(om * y[idx]) / np.sum(om)
        var = np.sum(om * (y[idx] - m) ** 2) / np.sum(om)
        if var > 0:
            loc[r], scale[r] = m, np.sqrt(var)
    return loc, scale


@pytest.mark.parametrize("floc", [None, 2.5])
def test_each_field_is_its_definition_at_the_fit_before_the_row(floc):
    df = _frame(300, 1).drop("g", "z")
    out = _out(_spec(weight="w", floc=floc), df)
    y, w = df["y"].to_numpy(), df["w"].to_numpy()
    loc, scale = _longhand(y, w, floc)
    got_loc, got_scale = out["loc_y"].to_numpy(), out["scale_y"].to_numpy()
    reported = ~np.isnan(loc)
    assert reported.sum() > 280
    assert np.array_equal(reported, out["loc_y"].is_not_null().to_numpy())
    np.testing.assert_allclose(got_loc[reported], loc[reported], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(got_scale[reported], scale[reported], rtol=1e-12)
    rows = np.flatnonzero(reported)
    m, s = got_loc[rows], got_scale[rows]
    for p in (0.05, 0.5, 0.99):
        np.testing.assert_allclose(
            out[f"q{p}_y"].to_numpy()[rows], stats.norm.ppf(p, m, s), rtol=1e-13, atol=1e-13
        )
    for a in (0.025, 0.1):
        want = [
            stats.norm.expect(lambda x: x, loc=mi, scale=si, ub=stats.norm.ppf(a, mi, si)) / a
            for mi, si in zip(m[::25], s[::25], strict=True)
        ]
        np.testing.assert_allclose(out[f"es{a}_y"].to_numpy()[rows][::25], want, rtol=1e-7)
    np.testing.assert_allclose(
        out["pit_y"].to_numpy()[rows], stats.norm.cdf(y[rows], m, s), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        out["log_score_y"].to_numpy()[rows], stats.norm.logpdf(y[rows], m, s), rtol=1e-12
    )


def test_a_rows_fit_is_read_before_the_row():
    """Changing one row's value moves its own ``pit`` and ``log_score`` and
    the next row's fit, never its own fit."""
    df = _frame(200, 2).drop("g")
    s = _spec()
    a = _out(s, df)
    b = _out(
        s,
        df.with_columns(
            pl.when(pl.int_range(200) == 150).then(40.0).otherwise(pl.col("y")).alias("y")
        ),
    )
    for f in ("loc_y", "scale_y", "q0.05_y", "es0.1_y"):
        assert a[f][:151].equals(b[f][:151]), f
        assert a[f][151] != b[f][151], f
    assert a["pit_y"][150] != b["pit_y"][150]
    assert a["log_score_y"][150] != b["log_score_y"][150]


def test_chunk_invariance():
    df = _frame(500, 3)
    s = _spec(targets=["y", "z"], group="g", weight="w", clock="t", gap_cap=5.0)
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 13, 211):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 500, size)])
        assert one.equals(many, null_equal=True), size


@pytest.mark.parametrize("floc", [None, 0.0])
def test_a_save_goes_on_to_the_bit(floc):
    df = _frame(400, 4)
    s = _spec(targets=["y", "z"], group="g", weight="w", floc=floc)
    for cut in (1, 2, 150, 399):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_a_zero_weight_row_learns_nothing_and_moves_no_clock():
    """The stream with rows of weight 0 gives every other row, to the bit,
    the fit of the stream without them; such a row is still read."""
    df = _frame(400, 5).drop("g", "z")
    s = _spec(weight="w")
    full = _out(s, df)
    keep = (df["w"] > 0).to_numpy()
    thin = _out(s, df.filter(pl.Series(keep)))
    for f in ("loc_y", "scale_y", "q0.99_y", "es0.025_y", "pit_y", "log_score_y"):
        assert full[f].filter(pl.Series(keep)).equals(thin[f], null_equal=True), f
    read = full.filter(pl.Series(~keep))
    assert read["pit_y"].drop_nulls().len() > 20, "such a row is still read"


def test_a_null_target_learns_nothing_and_reports_no_score():
    """A row whose ``z`` is null gives ``z`` no pit and no log score, and
    leaves its fit where it was for the next row; ``y`` is learned as usual."""
    df = _frame(300, 6).drop("g")
    out = _out(_spec(targets=["y", "z"]), df)
    null = df["z"].is_null().to_numpy()
    rows = np.flatnonzero(null[20:-1]) + 20
    assert len(rows) > 10
    assert out["pit_z"][rows].is_null().all() and out["log_score_z"][rows].is_null().all()
    # The fit read after a null row is the one read before it: the decay
    # scales every weight alike, which moves neither the mean nor the spread.
    for f in ("loc_z", "scale_z"):
        np.testing.assert_allclose(out[f][rows + 1].to_numpy(), out[f][rows].to_numpy(), rtol=1e-15)
    assert out["loc_y"][rows].is_not_null().all()
    assert out["loc_y"][rows + 1].to_numpy().tolist() != out["loc_y"][rows].to_numpy().tolist()


def test_a_weight_scales_nothing_but_the_weight():
    df = _frame(300, 7).drop("g")
    s = _spec(targets=["y", "z"], weight="w")
    a = _out(s, df)
    b = _out(s, df.with_columns(pl.col("w") * 4.0))
    for f in ("loc_y", "scale_z", "q0.05_z", "es0.1_y", "pit_y", "log_score_z"):
        np.testing.assert_allclose(a[f].to_numpy(), b[f].to_numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(b["weight_sum"].to_numpy(), 4 * a["weight_sum"].to_numpy())


def test_a_constant_series_is_withheld_never_divided_by():
    df = pl.DataFrame({"y": [2.0] * 50, "z": np.linspace(0.0, 1.0, 50)})
    out = _out(_spec(targets=["y", "z"]), df)
    reasons = out["withheld_reason_y"].cast(pl.String).to_list()
    assert reasons == ["no_weight"] + ["degenerate_variance"] * 49
    assert out["loc_y"].is_null().all() and out["pit_y"].is_null().all()
    assert out["withheld_reason_z"].cast(pl.String).to_list()[:3] == [
        "no_weight",
        "degenerate_variance",
        None,
    ]
    assert out["scale_z"][2:].is_not_null().all()


def test_names_match_the_realized_struct():
    df = _frame(100, 8)
    for kw in ({}, {"quantiles": [], "es_levels": []}, {"floc": 0.0, "es_levels": [0.01]}):
        spec = _spec(targets=["y", "z"], **kw)
        out = rp.ModelBank([spec]).fit_predict(df)
        assert [f.name for f in out.schema["m"].fields] == rp.spec.output_fields(spec)


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"quantiles": [0.0]}, "quantiles"),
        ({"quantiles": [0.5, 1.0]}, "quantiles"),
        ({"es_levels": [-0.1]}, "es_levels"),
        ({"floc": float("nan")}, "floc"),
        ({"emit_sigma": True}, "emit_sigma"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.ModelBank([_spec(**kw)])


def test_family_and_features_are_refused():
    with pytest.raises(ValueError, match="family"):
        rp.spec.dist_fit("m", targets=["y"], half_life=10.0, family="skewt")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="features"):
        rp.spec.dist_fit("m", targets=["y"], features=["x"])  # type: ignore[call-arg]


# Student's t: what the bank owes it, as the normal's tests above hold the
# normal. Its fit against scipy's weighted maximum is in
# ``tests/test_second_opinion_t.py``.


def _t_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "t": np.cumsum(rng.exponential(1.0, n)),
            "g": np.where(np.arange(n) % 3 == 0, "a", "b"),
            "y": 3.0 + 1.5 * rng.standard_t(3, n),
            "z": np.where(rng.uniform(size=n) < 0.1, np.nan, -2.0 + 0.5 * rng.standard_t(5, n)),
            "w": np.where(rng.uniform(size=n) < 0.1, 0.0, rng.uniform(0.5, 2.0, n)),
        }
    ).with_columns(pl.col("z").fill_nan(None))


def _t_spec(**kw) -> dict:
    return _spec(family="t", half_life=kw.pop("half_life", 300.0), **kw)


def test_t_chunk_invariance_and_a_save_to_the_bit():
    df = _t_frame(1500, 11)
    s = _t_spec(targets=["y", "z"], group="g", weight="w", clock="t", gap_cap=5.0)
    one = rp.ModelBank([s]).fit_predict(df)
    reasons = one.unnest("m")["withheld_reason_y"].cast(pl.String).to_list()
    assert {None, "not_converged", "shape_unidentified"} <= set(reasons)
    for size in (1, 37, 500):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 1500, size)])
        assert one.equals(many, null_equal=True), size
    for floc in (None, 3.0):
        s = _t_spec(targets=["y", "z"], group="g", weight="w", floc=floc)
        for cut in (1, 40, 700):
            a = rp.ModelBank([s])
            a.fit_predict(df.slice(0, cut))
            buf = io.BytesIO()
            a.save(buf)
            b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
            rest = df.slice(cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_t_zero_weight_rows_learn_nothing_and_move_no_clock():
    df = _t_frame(1200, 12).drop("g", "z")
    s = _t_spec(weight="w", clock="t", gap_cap=1e6)
    full = _out(s, df)
    keep = pl.Series((df["w"] > 0).to_numpy())
    thin = _out(s, df.filter(keep))
    for f in ("loc_y", "scale_y", "df_y", "q0.99_y", "es0.025_y", "pit_y", "log_score_y"):
        assert full[f].filter(keep).equals(thin[f], null_equal=True), f
    assert full.filter(~keep)["pit_y"].drop_nulls().len() > 20, "such a row is still read"


def test_t_a_rows_fit_is_read_before_the_row():
    df = _t_frame(800, 13).drop("g")
    s = _t_spec()
    a = _out(s, df)
    b = _out(
        s,
        df.with_columns(
            pl.when(pl.int_range(800) == 600).then(400.0).otherwise(pl.col("y")).alias("y")
        ),
    )
    for f in ("loc_y", "scale_y", "df_y", "q0.05_y", "es0.1_y"):
        assert a[f][:601].equals(b[f][:601]), f
        assert a[f][601] != b[f][601], f
    assert a["pit_y"][600] != b["pit_y"][600]


def test_t_withholds_the_tail_until_it_is_told_from_the_normals():
    """A normal stream: ``loc`` and ``scale`` report, and the tail's fields
    are null with ``shape_unidentified`` on most rows; a heavy-tailed one
    reports them."""
    rng = np.random.default_rng(14)
    df = pl.DataFrame({"y": rng.standard_normal(3000), "z": rng.standard_t(2, 3000)})
    out = _out(_t_spec(targets=["y", "z"], half_life=1000.0), df)[1000:]
    late = out["withheld_reason_y"].cast(pl.String)
    assert (late == "shape_unidentified").mean() > 0.8
    unidentified = out.filter(pl.col("withheld_reason_y") == "shape_unidentified")
    assert unidentified["loc_y"].is_not_null().all() and unidentified["scale_y"].is_not_null().all()
    for f in ("df_y", "q0.05_y", "es0.1_y", "pit_y", "log_score_y"):
        assert unidentified[f].is_null().all(), f
    assert out["withheld_reason_z"].is_null().all()
    assert out["df_z"].median() < 3.0


def test_t_names_match_the_realized_struct():
    df = _t_frame(100, 15)
    for kw in ({}, {"quantiles": [], "es_levels": []}, {"floc": 0.0, "es_levels": [0.01]}):
        spec = _t_spec(targets=["y", "z"], **kw)
        out = rp.ModelBank([spec]).fit_predict(df)
        assert [f.name for f in out.schema["m"].fields] == rp.spec.output_fields(spec)
    reasons = out.schema["m"].to_schema()["withheld_reason_y"]
    assert list(reasons.categories) == [
        "no_weight",
        "min_weight",
        "degenerate_variance",
        "not_converged",
        "shape_unidentified",
    ]
