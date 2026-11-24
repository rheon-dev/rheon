"""`isotonic` against scikit-learn's ``IsotonicRegression`` (PLAN 573).

The oracle is ``IsotonicRegression(increasing, y_min, y_max,
out_of_bounds).fit(x, y, sample_weight)`` on the rows each row is predicted
from, read with ``predict`` at the row's feature; under a decay the weights
are the decay factors ``w_i 0.5^((t - t_i) / h)``.

Beside it, a longhand transcription of what scikit-learn runs, in plain
Python floats: ``_make_unique`` (ties pooled from 0 in sorted order),
scipy's ``optimize.isotonic_regression`` (Busing's PAVA, line for line),
``np.clip`` and ``numpy.interp``'s rules. Twice: unfused, as the sources
read, and with each ``a * b + c`` fused into one rounding (``math.fma``),
as scipy's, scikit-learn's and numpy's arm64 wheels compile them.

What is held, and why:

- **The bank against the unfused transcription, to the bit**, wherever the
  arithmetic order is scikit-learn's: no decay and no repeated feature
  without a window, and every window (whose rows at one feature are pooled
  in scikit-learn's sorted order).
- **scikit-learn against one of the two transcriptions, to the bit**, on
  every fit: so the only gap between the bank and scikit-learn is the
  fused multiply-add of the wheel's build, and the worst gap is measured
  and bounded (``GAP``).
- **Ties without a window and a decay**: pooled as they arrive, and a
  weight from one ``exp2`` of the clock; within ``1e-14`` of scikit-learn
  on targets of order 1 (measured 2.1e-15 at worst over ten seeds of each
  case, about 10 ulps of 1).

Then the kind through the bank: one chunk against many, a save going on to
the bit, rows of weight 0 and null targets, the withheld reason, the
refusals, and the state's cap as the rows go in, on both sides.
"""

from __future__ import annotations

import io
import math
import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.isotonic import IsotonicRegression

from rheon import polars as rp

TIER = "mixed"

#: The most a prediction parts from scikit-learn's where the bank computes
#: scikit-learn's order but unfused, on targets of order 1: measured on
#: arm64 at 8.3e-16 at worst over ten seeds of each form (about 4 ulps of 1;
#: more ulps of a prediction near 0, whose ulp is smaller).
GAP = 2e-15


def _same(a: float | None, b: float) -> bool:
    if a is None:
        return math.isnan(b)
    return np.float64(a).view(np.int64) == np.float64(b).view(np.int64) or (
        math.isnan(a) and math.isnan(b)
    )


def _mul_add(a: float, b: float, c: float, fused: bool) -> float:
    return math.fma(a, b, c) if fused else a * b + c


def _pava(x: list[float], w: list[float], fused: bool) -> list[float]:
    """scipy's ``pava`` (Busing 2022, Algorithm 1), line for line."""
    x, w = list(x), list(w)
    n = len(x)
    r = [-1] * (n + 1)
    r[0] = 0
    if n > 1:
        r[1] = 1
    b = 0
    xb_prev, wb_prev = x[0], w[0]
    i = 1
    while i < n:
        b += 1
        xb, wb = x[i], w[i]
        if xb_prev >= xb:
            b -= 1
            sb = _mul_add(wb_prev, xb_prev, wb * xb, fused)
            wb += wb_prev
            xb = sb / wb
            while i < n - 1 and xb >= x[i + 1]:
                i += 1
                sb = _mul_add(w[i], x[i], sb, fused)
                wb += w[i]
                xb = sb / wb
            while b > 0 and x[b - 1] >= xb:
                b -= 1
                sb = _mul_add(w[b], x[b], sb, fused)
                wb += w[b]
                xb = sb / wb
        x[b] = xb_prev = xb
        w[b] = wb_prev = wb
        r[b + 1] = i + 1
        i += 1
    f = n - 1
    for k in range(b, -1, -1):
        t = r[k]
        for j in range(f, t - 1, -1):
            x[j] = x[k]
        f = t - 1
    return x


def _fit(
    rows: list[tuple[float, float, float]], cfg: dict[str, Any], fused: bool
) -> tuple[list[float], list[float]]:
    """scikit-learn's ``_build_y`` in longhand: the distinct ``x`` and the
    fit there."""
    rows = sorted((r for r in rows if r[2] > 0), key=lambda r: (r[0], r[1]))
    ux: list[float] = []
    uy: list[float] = []
    uw: list[float] = []
    if not rows:
        return ux, uy
    cx, cy, cw = rows[0][0], 0.0, 0.0
    for x, y, w in rows:
        if x != cx:
            ux.append(cx)
            uw.append(cw)
            uy.append(cy / cw)
            cx, cw, cy = x, w, y * w
        else:
            cw += w
            cy = _mul_add(y, w, cy, fused)
    ux.append(cx)
    uw.append(cw)
    uy.append(cy / cw)
    if cfg.get("increasing", True):
        fy = _pava(uy, uw, fused)
    else:
        fy = _pava(uy[::-1], uw[::-1], fused)[::-1]
    lo = cfg.get("y_min")
    hi = cfg.get("y_max")
    lo = -math.inf if lo is None else lo
    hi = math.inf if hi is None else hi
    return ux, [min(max(v, lo), hi) for v in fy]


def _read(fx: list[float], fy: list[float], x: float, oob: str, fused: bool) -> float:
    """scikit-learn's ``predict``: ``interp1d`` handing ``numpy.interp``
    a one-dimensional fit, ``nan`` or ``clip`` beyond it."""
    n = len(fx)
    if n == 0:
        return math.nan
    if n == 1:
        return fy[0]
    if oob == "clip":
        x = min(max(x, fx[0]), fx[-1])
    elif x < fx[0] or x > fx[-1]:
        return math.nan
    if x == fx[-1]:
        return fy[-1]
    j = int(np.searchsorted(fx, x, side="right")) - 1
    if fx[j] == x:
        return fy[j]
    slope = (fy[j + 1] - fy[j]) / (fx[j + 1] - fx[j])
    r = _mul_add(slope, x - fx[j], fy[j], fused)
    if math.isnan(r):
        r = _mul_add(slope, x - fx[j + 1], fy[j + 1], fused)
        if math.isnan(r) and fy[j] == fy[j + 1]:
            r = fy[j]
    return r


def _sklearn(rows: list[tuple[float, float, float]], cfg: dict[str, Any], x: float) -> float:
    rows = [r for r in rows if r[2] > 0]
    if not rows:
        return math.nan
    a = np.array(rows)
    ir = IsotonicRegression(
        increasing=cfg.get("increasing", True),
        y_min=cfg.get("y_min"),
        y_max=cfg.get("y_max"),
        out_of_bounds=cfg.get("out_of_bounds", "nan"),
    ).fit(a[:, 0], a[:, 1], sample_weight=a[:, 2])
    return float(ir.predict(np.array([x]))[0])


def _frame(n: int, seed: int, *, levels: int = 0, weighted: bool = True) -> pl.DataFrame:
    """A score with a monotone trend in its target, continuous or on
    ``levels`` values (repeated), weights 1 or varied, a clock with uneven
    steps."""
    rng = np.random.default_rng(573_000 + seed)
    u = rng.uniform(0.0, 1.0, n)
    x = np.floor(u * levels) * 0.37 if levels else u * 10.0
    y = np.tanh(x - 3.0) + rng.normal(0.0, 0.4, n)
    w = rng.uniform(0.25, 1.5, n) if weighted else np.ones(n)
    t = np.cumsum(rng.integers(1, 4, n)).astype(float)
    return pl.DataFrame({"t": t, "x": x, "y": y, "w": w})


def _out(spec: dict[str, Any], df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest(spec["name"])


def _past(
    df: pl.DataFrame, i: int, cfg: dict[str, Any], half_life: float | None = None
) -> list[tuple[float, float, float]]:
    """The rows row ``i`` is predicted from, with their weights: every row
    before it, or a window's, each weight times its decay to the last."""
    t = df["t"].to_list()
    x, y, w = df["x"].to_list(), df["y"].to_list(), df["w"].to_list()
    keep = [k for k in range(i) if w[k] > 0 and y[k] is not None]
    if cfg.get("window_rows"):
        keep = keep[-cfg["window_rows"] :]
    if cfg.get("window_size"):
        span = cfg["window_size"]
        both = cfg.get("closed") == "both"
        keep = [k for k in keep if (t[i] - t[k] <= span if both else t[i] - t[k] < span)]
    last = max(keep) if keep else 0
    out = []
    for k in keep:
        d = 1.0 if half_life is None else 0.5 ** ((t[last] - t[k]) / half_life)
        out.append((x[k], y[k], w[k] * d))
    return out


FORMS: dict[str, dict[str, Any]] = {
    "plain": {},
    "decreasing_clip": {"increasing": False, "out_of_bounds": "clip"},
    "bounded": {"y_min": -0.6, "y_max": 0.5, "out_of_bounds": "clip"},
    "rows": {"window_rows": 40},
    "clock": {"window_size": 30.0},
    "clock_both": {"window_size": 30.0, "closed": "both", "increasing": False},
}


class TestAgainstScikitLearn:
    @pytest.mark.parametrize("form", list(FORMS))
    def test_each_row_is_the_longhand_fit_of_the_rows_before_it_to_the_bit(self, form: str) -> None:
        """No repeated feature without a window; a window repeats them."""
        cfg = FORMS[form]
        windowed = "window_rows" in cfg or "window_size" in cfg
        df = _frame(400, 1, levels=17 if windowed else 0)
        s = rp.spec.isotonic(
            "i", targets=["y"], features=["x"], weight="w", clock="t", gap_cap=10.0, **cfg
        )
        out = _out(s, df)
        oob = cfg.get("out_of_bounds", "nan")
        x = df["x"].to_list()
        for i, p in enumerate(out["pred_y"].to_list()):
            fx, fy = _fit(_past(df, i, cfg), cfg, fused=False)
            want = _read(fx, fy, x[i], oob, fused=False)
            assert _same(p, want), (form, i, p, want)

    @pytest.mark.parametrize("form", list(FORMS))
    def test_scikit_learn_is_a_transcription_and_the_gap_is_its_fused_rounding(
        self, form: str
    ) -> None:
        cfg = FORMS[form]
        windowed = "window_rows" in cfg or "window_size" in cfg
        df = _frame(300, 2, levels=11 if windowed else 0)
        s = rp.spec.isotonic(
            "i", targets=["y"], features=["x"], weight="w", clock="t", gap_cap=10.0, **cfg
        )
        out = _out(s, df)
        oob = cfg.get("out_of_bounds", "nan")
        x = df["x"].to_list()
        worst = 0.0
        for i, p in enumerate(out["pred_y"].to_list()):
            past = _past(df, i, cfg)
            sk = _sklearn(past, cfg, x[i])
            reads = []
            for fused in (False, True):
                fx, fy = _fit(past, cfg, fused)
                reads.append(_read(fx, fy, x[i], oob, fused))
            assert any(_same(r, sk) for r in reads), (form, i, sk, reads)
            if math.isnan(sk):
                assert p is None, (form, i)
            else:
                assert p is not None, (form, i)
                worst = max(worst, abs(p - sk))
        assert worst <= GAP, (form, worst)

    @pytest.mark.parametrize(
        ("levels", "half_life"),
        [(9, None), (0, 25.0), (9, 25.0), (9, 3.0)],
    )
    def test_ties_and_a_decay_agree_within_their_rounding(
        self, levels: int, half_life: float | None
    ) -> None:
        """A tie pooled as it arrives and a weight from one ``exp2`` of the
        clock: scikit-learn's fit on the decayed weights within 1e-14."""
        df = _frame(500, 3, levels=levels)
        common: dict[str, Any] = {} if half_life is None else {"half_life": half_life}
        s = rp.spec.isotonic(
            "i", targets=["y"], features=["x"], weight="w", clock="t", gap_cap=10.0, **common
        )
        out = _out(s, df)
        x = df["x"].to_list()
        worst = 0.0
        for i, p in enumerate(out["pred_y"].to_list()):
            sk = _sklearn(_past(df, i, {}, half_life), {}, x[i])
            if math.isnan(sk):
                assert p is None, i
                continue
            assert p is not None, i
            worst = max(worst, abs(p - sk))
        assert worst < 1e-14, worst

    @pytest.mark.parametrize("closed", ["right", "both"])
    def test_a_decayed_window_is_scikit_learn_on_its_rows_weighted(self, closed: str) -> None:
        cfg = {"window_size": 25.0, "closed": closed}
        df = _frame(400, 4, levels=13)
        s = rp.spec.isotonic(
            "i",
            targets=["y"],
            features=["x"],
            weight="w",
            clock="t",
            gap_cap=10.0,
            half_life=10.0,
            **cfg,
        )
        out = _out(s, df)
        x = df["x"].to_list()
        for i, p in enumerate(out["pred_y"].to_list()):
            sk = _sklearn(_past(df, i, cfg, 10.0), cfg, x[i])
            if math.isnan(sk):
                assert p is None, i
            else:
                assert p is not None and abs(p - sk) <= 1e-14, (i, p, sk)

    def test_two_targets_are_fitted_each_on_its_own_rows(self) -> None:
        df = _frame(300, 5).with_columns(
            z=pl.when(pl.int_range(pl.len()) % 3 == 0).then(None).otherwise(-pl.col("y"))
        )
        s = rp.spec.isotonic(
            "i", targets=["y", "z"], features=["x"], increasing=False, out_of_bounds="clip"
        )
        out = _out(s, df)
        x, z = df["x"].to_list(), df["z"].to_list()
        for i, p in enumerate(out["pred_z"].to_list()):
            past = [(x[k], z[k], 1.0) for k in range(i) if z[k] is not None]
            fx, fy = _fit(past, {"increasing": False}, fused=False)
            assert _same(p, _read(fx, fy, x[i], "clip", fused=False)), i


class TestTheKind:
    def test_the_outputs(self) -> None:
        df = _frame(50, 6)
        out = _out(rp.spec.isotonic("i", targets=["y"], features=["x"]), df)
        fields = [c for c in out.columns if c not in df.columns]
        assert fields == ["pred_y", "resid_y", "weight_sum", "settled_frac", "withheld_reason"]
        assert out["weight_sum"].to_list() == list(map(float, range(50)))

    def test_a_feature_beyond_the_rows_is_withheld_as_not_identified(self) -> None:
        df = pl.DataFrame(
            {"x": [1.0, 2.0, 0.5, 1.5, 3.0, 1.2], "y": [0.0, 1.0, 0.2, 0.8, 1.0, 0.1]}
        )
        out = _out(rp.spec.isotonic("i", targets=["y"], features=["x"]), df)
        # Row 0 has no rows; row 1 one point, read everywhere; row 2 at 0.5
        # is below 1 and 2; row 4 at 3 is above every row.
        assert out["pred_y"].is_null().to_list() == [True, False, True, False, True, False]
        reasons = out["withheld_reason"].cast(pl.String).to_list()
        assert reasons == ["not_identified", None, "not_identified", None, "not_identified", None]
        clip = _out(rp.spec.isotonic("i", targets=["y"], features=["x"], out_of_bounds="clip"), df)
        assert clip["pred_y"].is_null().to_list() == [True] + [False] * 5

    @pytest.mark.parametrize("form", ["plain", "rows", "clock_both"])
    def test_one_chunk_and_many_give_the_same_output(self, form: str) -> None:
        df = _frame(1_200, 7, levels=40).with_columns(g=pl.col("t").cast(pl.Int64) % 3)
        s = rp.spec.isotonic(
            "i",
            targets=["y"],
            features=["x"],
            weight="w",
            clock="t",
            gap_cap=10.0,
            group="g",
            **FORMS[form],
        )
        one = rp.ModelBank([s]).fit_predict(df)
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, 7)) for i in range(0, 1_200, 7)])
        assert one.equals(many, null_equal=True)

    @pytest.mark.parametrize("form", ["plain", "rows", "clock"])
    def test_a_save_goes_on_to_the_bit(self, form: str) -> None:
        df = _frame(900, 8, levels=30)
        s = rp.spec.isotonic(
            "i",
            targets=["y"],
            features=["x"],
            weight="w",
            clock="t",
            gap_cap=10.0,
            half_life=40.0,
            **FORMS[form],
        )
        for cut in (1, 333, 899):
            a = rp.ModelBank([s])
            a.fit_predict(df.slice(0, cut))
            buf = io.BytesIO()
            a.save(buf)
            b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
            rest = df.slice(cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut

    @pytest.mark.parametrize("form", ["plain", "rows", "clock"])
    def test_a_zero_weight_row_learns_nothing_and_is_still_read(self, form: str) -> None:
        df = _frame(600, 9, levels=20).with_columns(
            w=pl.when(pl.int_range(pl.len()) % 5 == 2).then(0.0).otherwise(pl.col("w"))
        )
        s = rp.spec.isotonic("i", targets=["y"], features=["x"], weight="w", **FORMS[form])
        full = _out(s, df)
        keep = (df["w"] > 0).to_numpy()
        thin = _out(s, df.filter(pl.Series(keep)))
        assert full.filter(pl.Series(keep)).equals(thin, null_equal=True)
        assert full.filter(pl.Series(~keep))["pred_y"].drop_nulls().len() > 100

    def test_the_scale_of_the_weights_moves_only_weight_sum(self) -> None:
        df = _frame(300, 10, levels=12)
        s = rp.spec.isotonic("i", targets=["y"], features=["x"], weight="w", half_life=30.0)
        a = _out(s, df)
        b = _out(s, df.with_columns(pl.col("w") * 1024.0))
        assert a["pred_y"].equals(b["pred_y"], null_equal=True)


class TestRefusals:
    @pytest.mark.parametrize("features", [[], ["x", "z"]])
    def test_one_feature_exactly(self, features: list[str]) -> None:
        with pytest.raises((ValueError, TypeError), match="one"):
            rp.spec.isotonic("i", targets=["y"], features=features)

    @pytest.mark.parametrize(
        ("kw", "says"),
        [
            ({"y_min": 1.0, "y_max": 0.0}, "y_min"),
            ({"y_max": math.inf}, "y_max"),
            ({"out_of_bounds": "raise"}, "out_of_bounds"),
            ({"window_rows": 0}, "window_rows"),
            ({"window_rows": 5, "window_size": 5.0}, "window_rows"),
            ({"closed": "both"}, "window_size"),
            ({"window_size": 0.0}, "window_size"),
            ({"max_state_bytes": 0}, "max_state_bytes"),
        ],
    )
    def test_each_bad_setting_is_refused_by_name(self, kw: dict[str, Any], says: str) -> None:
        with pytest.raises(ValueError, match=says):
            rp.spec.isotonic("i", targets=["y"], features=["x"], **kw)

    @pytest.mark.parametrize("rows", [6_000, 6_600])
    def test_a_row_window_against_the_cap_at_the_call(self, rows: int) -> None:
        """A row window holds 64 bytes a row for one target: 6,000 rows sit
        under 400,000 bytes with the header, 6,600 do not."""
        build = lambda: rp.spec.isotonic(  # noqa: E731
            "i", targets=["y"], features=["x"], window_rows=rows, max_state_bytes=400_000
        )
        if rows == 6_000:
            build()
        else:
            with pytest.raises(ValueError, match="max_state_bytes"):
                build()


class TestTheCap:
    def test_distinct_features_past_the_cap_stop_the_run_naming_it(self) -> None:
        """2,000 distinct scores are 80,000 bytes of points: past a cap of
        64 KB the run is refused; under one of 128 KB it runs, said once
        past half of it."""
        df = pl.DataFrame({"x": np.arange(2_000.0), "y": np.arange(2_000.0) % 7})
        tight = rp.spec.isotonic("i", targets=["y"], features=["x"], max_state_bytes=65_536)
        with pytest.raises(Exception, match="max_state_bytes"):
            rp.ModelBank([tight]).fit_predict(df)
        roomy = rp.spec.isotonic(
            "i", targets=["y"], features=["x"], out_of_bounds="clip", max_state_bytes=131_072
        )
        with pytest.warns(rp.ReadinessWarning, match="past half of max_state_bytes") as said:
            out = _out(roomy, df)
        assert len([w for w in said if "max_state_bytes" in str(w.message)]) == 1
        assert out["pred_y"].drop_nulls().len() == 1_999

    def test_a_rounded_score_stays_under_half_the_cap(self) -> None:
        """The other side: 50,000 rows of a score rounded to 3 places hold
        at most 1,001 points, 40,040 bytes, under half of 128 KB: nothing is
        said, and the fit calibrates."""
        rng = np.random.default_rng(11)
        x = rng.uniform(0.0, 1.0, 50_000).round(3)
        df = pl.DataFrame({"x": x, "y": (rng.uniform(0.0, 1.0, 50_000) < x).astype(float)})
        s = rp.spec.isotonic(
            "i",
            targets=["y"],
            features=["x"],
            half_life=2_000.0,
            out_of_bounds="clip",
            max_state_bytes=131_072,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", rp.ReadinessWarning)
            out = _out(s, df)
        assert out["pred_y"].drop_nulls().len() == 49_999
        # The calibrated probability tracks the score it was drawn from.
        tail = out.tail(10_000).with_columns(x=pl.Series(x[-10_000:]))
        assert (tail["pred_y"] - tail["x"]).abs().mean() < 0.05
