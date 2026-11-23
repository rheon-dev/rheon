"""The DDSketch against the `ddsketch` package, `crick` and exact reads.

PLAN 544, part 7. The core's sketch (``_rheon.ddsketch_buckets`` and
``_rheon.ddsketch_reads``: Masson, Rim and Lee's DDSketch with the
``ddsketch`` package's logarithmic mapping) and the ``ddsketch`` package
3.0.1 (Datadog, Apache-2.0; ``DDSketch(alpha)``, a ``LogarithmicMapping``
and two unbounded ``DenseStore``) read the same seeded streams: normal,
lognormal, Student-t(3) and Pareto(2.5).

What is held, and why each tolerance is what it is:

- **The buckets**: every key and every count the package keeps, exactly.
  Our keys come from ``log2`` summed in plain arithmetic, the package's
  from the platform's ``log``; the two differ only for a value within a few
  ulps of a bucket's edge, which no stream here has.
- **The representatives**: each bucket's value, within 2 ulps of the
  package's (``2^(k/mu)`` from a series against ``math.pow``).
- **The quantiles**: the package reads rank ``p (n - 1)`` (the first
  bucket whose count passes it), ours the first bucket whose weight reaches
  ``p n``; at ``n = 100,001`` and ``p n`` not whole the two rules pick the
  same order statistic, so the reads agree within 2 ulps.
- **Against exact**: each quantile within ``alpha`` of numpy's
  ``inverted_cdf`` quantile, relatively (the paper's guarantee, not a
  measurement); each tail mean within ``alpha`` times the tail's mean
  magnitude of the exact tail mean, the partial-mass rule over the rows
  themselves (the pairing argument of the study's section 5).
- **crick**: its t-digest at compression 1,000, a sketch of another kind,
  reads the central quantiles of the smooth streams within ``2 alpha``
  plus its own rank error turned into value.

Below them, the ``ddsketch`` kind through the bank: the spec and its
refusals, the outputs, each row's reads against the core sketch to the bit,
a group's own sketch, the row window and the clock window against numpy on
exactly their rows, one chunk against a thousand, a save going on to the
bit, rows of weight 0, the withheld levels, and the state's cap as the
rows go in.
"""

from __future__ import annotations

import io
import math

import crick
import numpy as np
import polars as pl
import pytest
from ddsketch import DDSketch

from rheon import _rheon as native
from rheon import polars as rp
from rheon.spec import ddsketch as ddsketch_spec

TIER = "mixed"

ALPHA = math.tanh(1.0 / 128.0)
N = 100_001
LEVELS = (0.01, 0.05, 0.5, 0.95, 0.99)
DISTS = ("normal", "lognormal", "t3", "pareto")
SEEDS = [0] + [pytest.param(s, marks=pytest.mark.extended(reason="20 seeds")) for s in range(1, 20)]


def _stream(dist: str, seed: int, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(544_700 + seed)
    return {
        "normal": lambda: rng.normal(0.0, 1.0, n),
        "lognormal": lambda: rng.lognormal(0.0, 1.0, n),
        "t3": lambda: rng.standard_t(3, n),
        "pareto": lambda: rng.pareto(2.5, n) + 1.0,
    }[dist]()


def _ulps(a: float, b: float) -> int:
    return abs(int(np.float64(a).view(np.int64)) - int(np.float64(b).view(np.int64)))


def _package(xs: np.ndarray, alpha: float) -> DDSketch:
    sk = DDSketch(relative_accuracy=alpha)
    for x in xs.tolist():
        sk.add(x)
    return sk


def _store_dict(store: object) -> dict[int, float]:
    bins, offset = store.bins, store.offset  # type: ignore[attr-defined]
    return {offset + i: c for i, c in enumerate(bins) if c > 0}


def _exact(xs: np.ndarray, ws: np.ndarray, p: float) -> tuple[float, float]:
    """numpy's ``inverted_cdf`` quantile over the weighted rows, and the
    partial-mass tail mean: the mass ``p W`` from the smallest at
    ``p < 0.5``, ``(1 - p) W`` from the largest above it."""
    order = np.argsort(xs, kind="stable")
    x, w = xs[order], ws[order]
    total = w.sum()
    cum = np.cumsum(w)
    q = float(x[np.searchsorted(cum, p * total, side="left")])

    def tail(xv: np.ndarray, wv: np.ndarray, m: float) -> float:
        c = np.cumsum(wv)
        j = int(np.searchsorted(c, m, side="left"))
        before = c[j - 1] if j > 0 else 0.0
        return float(((xv[:j] * wv[:j]).sum() + (m - before) * xv[j]) / m)

    es = tail(x, w, p * total) if p < 0.5 else tail(x[::-1], w[::-1], (1.0 - p) * total)
    return q, es


def _tail_magnitude(xs: np.ndarray, p: float) -> float:
    s = np.sort(xs)
    k = max(1, int(len(s) * min(p, 1.0 - p)))
    part = s[:k] if p < 0.5 else s[-k:]
    return float(np.abs(part).mean())


class TestTheBucketsAreThePackages:
    @pytest.mark.parametrize("dist", DISTS)
    @pytest.mark.parametrize("alpha", [ALPHA, 0.01, 0.05])
    def test_every_key_and_count(self, dist: str, alpha: float) -> None:
        xs = _stream(dist, 0, 20_000)
        if dist == "normal":
            xs[::97] = 0.0
        pos, neg, zero = native.ddsketch_buckets(xs.tolist(), alpha)
        sk = _package(xs, alpha)
        assert dict(pos) == _store_dict(sk._store)
        assert dict(neg) == _store_dict(sk._negative_store)
        assert zero == sk._zero_count

    @pytest.mark.parametrize("alpha", [ALPHA, 0.01, 1e-4])
    def test_each_representative_within_two_ulps(self, alpha: float) -> None:
        sk = DDSketch(relative_accuracy=alpha)
        keys = list(range(-5_000, 5_001, 7))
        ours = native.ddsketch_values(keys, alpha)
        for k, v in zip(keys, ours, strict=True):
            assert _ulps(v, sk._mapping.value(k)) <= 2, k

    @pytest.mark.parametrize("dist", DISTS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_quantiles_agree_with_the_package(self, dist: str, seed: int) -> None:
        xs = _stream(dist, seed)
        reads = native.ddsketch_reads(xs.tolist(), [1.0] * N, ALPHA, list(LEVELS))
        sk = _package(xs, ALPHA)
        for p, (q, _) in zip(LEVELS, reads, strict=True):
            assert _ulps(q, sk.get_quantile_value(p)) <= 2, (p, q, sk.get_quantile_value(p))


class TestAgainstExact:
    @pytest.mark.parametrize("dist", DISTS)
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_quantile_and_tail_mean_within_alpha(
        self, dist: str, weighted: bool, seed: int
    ) -> None:
        xs = _stream(dist, seed)
        rng = np.random.default_rng(seed)
        ws = rng.uniform(0.1, 3.0, N) if weighted else np.ones(N)
        reads = native.ddsketch_reads(xs.tolist(), ws.tolist(), ALPHA, list(LEVELS))
        for p, (q, es) in zip(LEVELS, reads, strict=True):
            eq, ees = _exact(xs, ws, p)
            assert abs(q - eq) <= ALPHA * abs(eq) * (1 + 1e-12), (p, q, eq)
            bound = ALPHA * _tail_magnitude(xs, p) * 1.05
            assert abs(es - ees) <= bound, (p, es, ees, bound)
            assert (es <= q) if p < 0.5 else (es >= q)

    def test_a_coarser_alpha_is_coarser(self) -> None:
        """The other side: at alpha 0.05 a quantile strays past the default's
        bound, so the bound is the sketch's own and not a loose one."""
        xs = _stream("lognormal", 0)
        ones = [1.0] * N
        coarse = native.ddsketch_reads(xs.tolist(), ones, 0.05, list(LEVELS))
        errors = [
            abs(q - _exact(xs, np.ones(N), p)[0]) / abs(_exact(xs, np.ones(N), p)[0])
            for p, (q, _) in zip(LEVELS, coarse, strict=True)
        ]
        assert max(errors) > ALPHA


class TestCrick:
    @pytest.mark.parametrize("dist", ["normal", "lognormal"])
    def test_central_quantiles_agree_with_cricks_digest(self, dist: str) -> None:
        xs = _stream(dist, 0)
        reads = native.ddsketch_reads(xs.tolist(), [1.0] * N, ALPHA, [0.25, 0.5, 0.75])
        td = crick.TDigest(compression=1_000)
        td.update(xs)
        s = np.sort(xs)
        for p, (q, _) in zip((0.25, 0.5, 0.75), reads, strict=True):
            theirs = float(td.quantile(p))
            # Crick's rank error at compression 1,000, turned into value by
            # the empirical quantile's slope, beside our relative alpha.
            lo, hi = np.quantile(s, [p - 1e-3, p + 1e-3])
            assert abs(q - theirs) <= 2 * ALPHA * abs(theirs) + (hi - lo), (p, q, theirs)


# --- The ddsketch kind through the bank. ---


def _frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "g": np.where(np.arange(n) % 3 == 0, "a", "b"),
            "t": np.cumsum(rng.integers(0, 3, n)).astype(float),
            "y": rng.standard_t(3, n),
            "z": np.where(rng.uniform(size=n) < 0.1, np.nan, rng.lognormal(0.0, 1.0, n)),
            "w": np.where(rng.uniform(size=n) < 0.1, 0.0, 1.0 + rng.integers(0, 3, n)),
        }
    ).with_columns(pl.col("z").fill_nan(None))


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df)[spec["name"]].struct.unnest()


FORMS = {
    "plain": {},
    "rows": {"window_rows": 40},
    "clock": {"window_size": 25.0, "clock": "t", "gap_cap": 10.0},
    "clock_both": {"window_size": 25.0, "clock": "t", "gap_cap": 10.0, "closed": "both"},
}


class TestTheSpec:
    def test_the_defaults_build(self) -> None:
        s = rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4])
        assert s["model"] == {"type": "ddsketch", "quantiles": [0.4]}
        rp.ModelBank([s])

    @pytest.mark.parametrize("alpha", [1e-6, 0.01, 0.5])
    def test_alpha_builds(self, alpha: float) -> None:
        rp.ModelBank([rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4], alpha=alpha)])

    @pytest.mark.parametrize("alpha", [0.0, 9.9e-7, 1.0, -0.1])
    def test_alpha_out_of_range_is_refused(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha must be at least"):
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4], alpha=alpha)

    @pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.1])
    def test_a_level_outside_zero_and_one_is_refused(self, level: float) -> None:
        with pytest.raises(ValueError, match="above 0 and below 1"):
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4, level])

    def test_the_median_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="quantiles lists 0.5"):
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.05, 0.5])
        rp.ModelBank([rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4999, 0.5001])])

    def test_a_level_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4, 0.4])

    @pytest.mark.parametrize("decay", [{"half_life": 50.0}, {"lam": 0.99}])
    def test_a_decay_is_refused_naming_the_windowed_ddsketch(self, decay: dict) -> None:
        key = next(iter(decay))
        with pytest.raises(ValueError, match=rf"{key} does not apply to ddsketch") as e:
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4], **decay)
        assert "windowed DDSketch" in str(e.value)

    def test_both_windows_are_refused(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4], window_rows=5, window_size=5.0)

    def test_closed_both_without_window_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="closed"):
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4], window_rows=5, closed="both")

    @pytest.mark.parametrize("rows, good", [(0, False), (1, True)])
    def test_window_rows_edge(self, rows: int, good: bool) -> None:
        if good:
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4], window_rows=rows)
        else:
            with pytest.raises(ValueError, match="at least 1"):
                rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4], window_rows=rows)

    def test_a_row_window_past_the_cap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_state_bytes"):
            rp.spec.ddsketch(
                "d", targets=["y"], quantiles=[0.4], window_rows=100_000, max_state_bytes=100_000
            )

    def test_features_are_refused(self) -> None:
        with pytest.raises(TypeError, match="features"):
            rp.spec.ddsketch("d", targets=["y"], quantiles=[0.4], features=["x"])  # type: ignore[call-arg]


class TestThroughTheBank:
    def test_the_outputs(self) -> None:
        s = ddsketch_spec("d", targets=["y", "z"], quantiles=[0.1, 0.9])
        out = _out(s, _frame(300, 2))
        assert out.columns == [
            "q0.1_y",
            "q0.9_y",
            "es0.1_y",
            "es0.9_y",
            "q0.1_z",
            "q0.9_z",
            "es0.1_z",
            "es0.9_z",
            "weight_sum",
            "settled_frac",
            "withheld_reason",
        ]
        assert rp.spec.output_fields(s) == out.columns

    def test_each_row_reads_the_rows_before_it(self) -> None:
        df = _frame(400, 3).drop("g")
        out = _out(ddsketch_spec("d", targets=["z"], quantiles=[0.2, 0.4, 0.8]), df)
        z = df["z"].to_list()
        for i in (57, 250, 399):
            before = [v for v in z[:i] if v is not None]
            want = native.ddsketch_reads(before, [1.0] * len(before), ALPHA, [0.2, 0.4, 0.8])
            got = [(out[f"q{p}_z"][i], out[f"es{p}_z"][i]) for p in ("0.2", "0.4", "0.8")]
            assert got == [tuple(r) for r in want], i

    def test_each_group_has_its_own_sketch(self) -> None:
        df = _frame(600, 4)
        out = _out(ddsketch_spec("d", targets=["y"], quantiles=[0.4], group="g"), df)
        ys, gs = df["y"].to_list(), df["g"].to_list()
        i = 599
        before = [v for v, g in zip(ys[:i], gs[:i], strict=True) if g == gs[i]]
        ((q, _),) = native.ddsketch_reads(before, [1.0] * len(before), ALPHA, [0.4])
        assert out["q0.4_y"][i] == q

    def test_a_row_window_is_its_last_rows(self) -> None:
        """Every row's reads over the last 40 rows of positive weight,
        against numpy on exactly those rows: the quantile within alpha, the
        tail mean within alpha of the tail's magnitude."""
        df = _frame(2_000, 5).drop("g")
        s = ddsketch_spec("d", targets=["y"], quantiles=[0.1, 0.4, 0.9], window_rows=40, weight="w")
        out = _out(s, df)
        y, w = df["y"].to_numpy(), df["w"].to_numpy()
        kept = np.flatnonzero(w > 0)
        for i in range(100, 2_000, 37):
            rows = kept[kept < i][-40:]
            assert out["weight_sum"][i] == w[rows].sum()
            for p in (0.1, 0.4, 0.9):
                eq, ees = _exact(y[rows], w[rows], p)
                q, es = out[f"q{p}_y"][i], out[f"es{p}_y"][i]
                assert abs(q - eq) <= ALPHA * abs(eq) * (1 + 1e-12), (i, p)
                assert abs(es - ees) <= ALPHA * np.abs(y[rows]).max() * 1.05, (i, p)

    @pytest.mark.parametrize("closed", ["right", "both"])
    def test_a_clock_window_is_its_span(self, closed: str) -> None:
        df = _frame(1_500, 6).drop("g")
        s = ddsketch_spec(
            "d",
            targets=["y"],
            quantiles=[0.4],
            window_size=25.0,
            clock="t",
            gap_cap=10.0,
            closed=closed,
        )
        out = _out(s, df)
        y, t = df["y"].to_numpy(), df["t"].to_numpy()
        for i in range(200, 1_500, 41):
            age = t[i] - t[:i]
            rows = np.flatnonzero(age < 25.0 if closed == "right" else age <= 25.0)
            eq, _ = _exact(y[rows], np.ones(len(rows)), 0.4)
            assert out["weight_sum"][i] == len(rows), i
            assert abs(out["q0.4_y"][i] - eq) <= ALPHA * abs(eq) * (1 + 1e-12), i


class TestStreaming:
    @pytest.mark.parametrize("form", FORMS)
    def test_one_chunk_and_a_thousand_give_the_same_output(self, form: str) -> None:
        df = _frame(3_000, 7)
        s = ddsketch_spec(
            "d",
            targets=["y", "z"],
            quantiles=[0.01, 0.4, 0.95],
            weight="w",
            group="g",
            **FORMS[form],
        )
        one = rp.ModelBank([s]).fit_predict(df)
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, 3)) for i in range(0, 3_000, 3)])
        assert one.equals(many, null_equal=True)

    @pytest.mark.parametrize("form", FORMS)
    def test_a_save_goes_on_to_the_bit(self, form: str) -> None:
        df = _frame(2_500, 8)
        s = ddsketch_spec(
            "d",
            targets=["y", "z"],
            quantiles=[0.05, 0.4, 0.95],
            weight="w",
            group="g",
            **FORMS[form],
        )
        for cut in (1, 30, 1_234, 2_499):
            a = rp.ModelBank([s])
            a.fit_predict(df.slice(0, cut))
            buf = io.BytesIO()
            a.save(buf)
            b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
            rest = df.slice(cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut

    @pytest.mark.parametrize("form", ["plain", "rows"])
    def test_a_zero_weight_row_counts_in_no_clock(self, form: str) -> None:
        df = _frame(800, 9).drop("g")
        s = ddsketch_spec("d", targets=["y"], quantiles=[0.05, 0.4], weight="w", **FORMS[form])
        full = _out(s, df)
        keep = (df["w"] > 0).to_numpy()
        thin = _out(s, df.filter(pl.Series(keep)))
        assert full.filter(pl.Series(keep)).equals(thin, null_equal=True)
        assert full.filter(pl.Series(~keep))["q0.4_y"].drop_nulls().len() > 20


class TestWithheld:
    def test_a_level_its_rows_do_not_identify_is_withheld_with_a_reason(self) -> None:
        df = pl.DataFrame({"y": np.arange(1, 41, dtype=float)})
        out = _out(ddsketch_spec("d", targets=["y"], quantiles=[0.05, 0.4]), df)
        assert out["q0.05_y"][:19].is_null().all()
        assert out["q0.05_y"][19:].is_not_null().all()
        assert out["es0.05_y"].is_null().equals(out["q0.05_y"].is_null())
        assert out["q0.4_y"][:2].is_null().all() and out["q0.4_y"][2:].is_not_null().all()
        reasons = out["withheld_reason"].cast(pl.String).to_list()
        assert reasons[:19] == ["level_beyond_sample"] * 19
        assert reasons[19:] == [None] * 21

    def test_a_window_withholds_as_its_rows_leave(self) -> None:
        """A row window of 10 never holds the 19 rows 0.05 needs."""
        df = pl.DataFrame({"y": np.arange(1, 61, dtype=float)})
        out = _out(ddsketch_spec("d", targets=["y"], quantiles=[0.05], window_rows=10), df)
        assert out["q0.05_y"].is_null().all()

    def test_the_rule_reads_kishs_size_not_the_weights_units(self) -> None:
        df = _frame(200, 10).drop("g")
        s = ddsketch_spec("d", targets=["y"], quantiles=[0.05, 0.4], weight="w")
        a = _out(s, df)
        b = _out(s, df.with_columns(pl.col("w") * 1_000.0))
        for f in ("q0.05_y", "es0.05_y", "q0.4_y", "withheld_reason"):
            assert a[f].is_null().equals(b[f].is_null()), f


class TestTheCap:
    def test_a_range_past_the_cap_stops_the_run_naming_the_group(self) -> None:
        """20,000 values spread over 300 decades are 20,000 live buckets, 12
        bytes each: past a cap of 64 KB the run is refused naming the cap,
        and at 1 MB it runs."""
        x = 10.0 ** np.linspace(-150, 150, 20_000)
        df = pl.DataFrame({"y": x})
        tight = ddsketch_spec("d", targets=["y"], quantiles=[0.4], max_state_bytes=65_536)
        with pytest.raises(Exception, match="max_state_bytes"):
            rp.ModelBank([tight]).fit_predict(df)
        roomy = ddsketch_spec("d", targets=["y"], quantiles=[0.4], max_state_bytes=1 << 20)
        assert _out(roomy, df)["q0.4_y"].drop_nulls().len() > 19_000

    def test_a_narrow_range_stays_far_under_the_cap(self) -> None:
        """The other side: 100,000 rows of one decade hold a few hundred
        buckets and say nothing."""
        rng = np.random.default_rng(1)
        df = pl.DataFrame({"y": rng.uniform(1.0, 10.0, 100_000)})
        s = ddsketch_spec("d", targets=["y"], quantiles=[0.4], max_state_bytes=65_536)
        assert _out(s, df)["q0.4_y"].drop_nulls().len() > 99_000
