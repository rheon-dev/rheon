"""The merging t-digest against the `tdigest` package and exact ranks.

PLAN 544, part 3. The core's digest (``_rheon.tdigest_quantiles``, Dunning
and Ertl's 2019 merging digest on the ``k_1`` scale) and the ``tdigest``
package (Dunning's 2013 t-digest, MIT) read the same seeded streams: normal,
lognormal, Student-t(3) and Pareto(2.5), N = 100,000, at compression 100 and
200, at the levels 0.01, 0.05, 0.5, 0.95 and 0.99.

The rank error of an estimate ``q`` at level ``p`` is ``|F(q) - p|``, ``F``
the stream's empirical CDF (the share of values at most ``q``). The two
digests are compared in rank, ``|F(q_ours) - F(q_theirs)|``, not in value:
a quantile's value moves with the distribution's scale, its rank does not.

The package's compression. ``tdigest.TDigest(delta, K)`` bounds a centroid
at quantile ``q`` to ``4 n delta q (1 - q)`` (its ``_threshold``) and
compresses once it holds more than ``K / delta`` centroids: ``delta`` is the
reciprocal of the compression (the paper's ``delta = 1 / compression``), so
compression 100 is ``delta = 0.01``, its default. It is a finer digest than
the merging one at the same compression (about 835 centroids at 100, 1,550
at 200, against the merging digest's 58 and 116), so its own rank error is
the smaller, and the difference between the two is the merging digest's.

The tolerances are the merging digest's against exact ranks, measured over
20 seeds at its worst (2.41e-3 at compression 100, 1.20e-3 at 200, at the
median) with a margin of 1.5: 3.8e-3 and 1.8e-3. The difference from the
package is held to the same, and the package's own error to the same.
Seed 0 runs in the essentials; seeds 1 to 19 in the extended tier.

Below them, the ``tdigest`` kind through the bank (part 5): the spec and
its refusals, the outputs named ``q<level>_<column>`` and
``es<level>_<column>``, each row's levels
read from the rows before it against the core digest to the bit, a group's
own digest, one chunk against a thousand, a save going on to the bit, rows
of weight 0, and a level withheld with ``level_beyond_sample`` until the
rows identify it.

The expected shortfall (part 6): ``es<level>_<column>`` is the partial-mass
rule over the digest's own centroids past the level (the lower tail below
0.5, the upper above; 0.5 itself is refused). It is checked against that
definition, worked out in fractions from the centroids; against the
``tdigest`` package's ``trimmed_mean`` (the same rule over its finer
centroids); and against numpy's exact tail mean of the rows.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
import tdigest

from rheon import _rheon as native
from rheon import polars as rp
from rheon.spec import tdigest as tdigest_spec

TIER = "mixed"

N = 100_000
LEVELS = (0.01, 0.05, 0.5, 0.95, 0.99)
DISTS = ("normal", "lognormal", "t3", "pareto")
TOLERANCE = {100: 3.8e-3, 200: 1.8e-3}
SEEDS = [0] + [pytest.param(s, marks=pytest.mark.extended(reason="20 seeds")) for s in range(1, 20)]


def _stream(dist: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(544_000 + seed)
    return {
        "normal": lambda: rng.normal(0.0, 1.0, N),
        "lognormal": lambda: rng.lognormal(0.0, 1.0, N),
        "t3": lambda: rng.standard_t(3, N),
        "pareto": lambda: rng.pareto(2.5, N) + 1.0,
    }[dist]()


def _ranks(xs_sorted: np.ndarray, values: list[float]) -> np.ndarray:
    return np.searchsorted(xs_sorted, np.asarray(values), side="right") / len(xs_sorted)


def rank_errors(dist: str, seed: int, compression: int) -> tuple[float, float, float]:
    """The largest, over the levels: our rank error, the package's, and
    the rank difference between the two."""
    xs = _stream(dist, seed)
    ours = native.tdigest_quantiles(xs.tolist(), float(compression), list(LEVELS))
    assert all(q is not None for q in ours)
    theirs_digest = tdigest.TDigest(delta=1.0 / compression)
    theirs_digest.batch_update(xs)
    theirs = [theirs_digest.percentile(100.0 * p) for p in LEVELS]
    s = np.sort(xs)
    f_ours = _ranks(s, ours)  # type: ignore[arg-type]
    f_theirs = _ranks(s, theirs)
    p = np.asarray(LEVELS)
    return (
        float(np.max(np.abs(f_ours - p))),
        float(np.max(np.abs(f_theirs - p))),
        float(np.max(np.abs(f_ours - f_theirs))),
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("compression", [100, 200])
@pytest.mark.parametrize("dist", DISTS)
def test_quantiles_agree_with_the_tdigest_package(dist: str, compression: int, seed: int) -> None:
    ours, theirs, between = rank_errors(dist, seed, compression)
    tol = TOLERANCE[compression]
    assert ours <= tol, f"{dist}, compression {compression}: our rank error {ours:.2e}"
    assert theirs <= tol, f"{dist}, compression {compression}: tdigest's rank error {theirs:.2e}"
    assert between <= tol, (
        f"{dist}, compression {compression}: ours and tdigest {between:.2e} apart"
    )


def test_a_coarser_digest_moves_away_from_the_package() -> None:
    """The other side of the tolerance: at compression 20 the merging digest
    (about 12 centroids) reads the median more than 3.8e-3 in rank from the
    package's digest at 100."""
    xs = _stream("normal", 0)
    (ours,) = native.tdigest_quantiles(xs.tolist(), 20.0, [0.5])
    theirs_digest = tdigest.TDigest(delta=0.01)
    theirs_digest.batch_update(xs)
    s = np.sort(xs)
    between = abs(float(_ranks(s, [ours])[0] - _ranks(s, [theirs_digest.percentile(50.0)])[0]))  # type: ignore[list-item]
    assert between > TOLERANCE[100], between


# --- The `tdigest` kind through the bank (PLAN 544, part 5) ---------------


def _frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "g": np.where(np.arange(n) % 3 == 0, "a", "b"),
            "y": rng.standard_t(3, n),
            "z": np.where(rng.uniform(size=n) < 0.1, np.nan, rng.lognormal(0.0, 1.0, n)),
            "w": np.where(rng.uniform(size=n) < 0.1, 0.0, 1.0 + rng.integers(0, 3, n)),
        }
    ).with_columns(pl.col("z").fill_nan(None))


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df)[spec["name"]].struct.unnest()


class TestTheSpec:
    def test_the_defaults_build(self) -> None:
        s = rp.spec.tdigest("d", targets=["y"], quantiles=[0.25])
        assert s["model"] == {"type": "tdigest", "quantiles": [0.25]}
        rp.ModelBank([s])

    @pytest.mark.parametrize("compression", [100, 200])
    def test_compression_100_and_200_build(self, compression: int) -> None:
        s = rp.spec.tdigest("d", targets=["y"], quantiles=[0.01, 0.99], compression=compression)
        rp.ModelBank([s])

    @pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.1])
    def test_a_level_outside_zero_and_one_is_refused(self, level: float) -> None:
        with pytest.raises(ValueError, match="above 0 and below 1"):
            rp.spec.tdigest("d", targets=["y"], quantiles=[0.25, level])

    def test_a_level_that_is_not_a_number_is_refused(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            rp.spec.tdigest("d", targets=["y"], quantiles=[0.25, float("nan")])

    def test_a_level_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            rp.spec.tdigest("d", targets=["y"], quantiles=[0.25, 0.25])

    def test_the_median_is_refused_by_name(self) -> None:
        """0.5 has no tail of its own, so its expected shortfall no side;
        the levels either side of it are taken."""
        with pytest.raises(ValueError, match="quantiles lists 0.5"):
            rp.spec.tdigest("d", targets=["y"], quantiles=[0.05, 0.5])
        rp.ModelBank([rp.spec.tdigest("d", targets=["y"], quantiles=[0.4999, 0.5001])])

    @pytest.mark.parametrize("decay", [{"half_life": 50.0}, {"lam": 0.99}])
    def test_a_decayed_spec_is_refused_naming_the_windowed_ddsketch(self, decay: dict) -> None:
        key = next(iter(decay))
        with pytest.raises(
            ValueError, match=rf"{key} does not apply to tdigest, whose merging digest"
        ) as e:
            rp.spec.tdigest("d", targets=["y"], quantiles=[0.25], **decay)
        assert "the windowed DDSketch (ewm_quantile with a window_size)" in str(e.value)

    @pytest.mark.parametrize("window", ["window_size", "window_rows"])
    def test_a_windowed_spec_is_refused_naming_the_windowed_ddsketch(self, window: str) -> None:
        s = rp.spec.tdigest("d", targets=["y"], quantiles=[0.25])
        s["model"][window] = 20
        with pytest.raises(ValueError, match=rf"{window} does not apply to tdigest") as e:
            rp.ModelBank([s])
        assert "the windowed DDSketch" in str(e.value)

    def test_features_are_refused(self) -> None:
        with pytest.raises(TypeError, match="features"):
            rp.spec.tdigest("d", targets=["y"], quantiles=[0.25], features=["x"])  # type: ignore[call-arg]


class TestThroughTheBank:
    def test_one_column(self) -> None:
        out = _out(tdigest_spec("d", targets=["y"], quantiles=[0.05, 0.25]), _frame(300, 1))
        assert out.columns == [
            "q0.05_y",
            "q0.25_y",
            "es0.05_y",
            "es0.25_y",
            "weight_sum",
            "settled_frac",
            "withheld_reason",
        ]

    def test_two_columns_name_each_level_and_column(self) -> None:
        s = tdigest_spec("d", targets=["y", "z"], quantiles=[0.1, 0.9])
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
        assert not any(c.startswith(("pred", "resid", "coef")) for c in out.columns)
        assert rp.spec.output_fields(s) == out.columns

    def test_each_row_reads_the_rows_before_it(self) -> None:
        """The core digest over the rows before each row, to the bit: the
        row's own value is never in its own quantile (out of sample), and a
        null value adds nothing."""
        df = _frame(400, 3).drop("g")
        out = _out(tdigest_spec("d", targets=["z"], quantiles=[0.25]), df)
        z = df["z"].to_list()
        for i in (5, 57, 250, 399):
            before = [v for v in z[:i] if v is not None]
            (want,) = native.tdigest_quantiles(before, 100.0, [0.25])
            assert out["q0.25_z"][i] == want, i

    def test_against_the_tdigest_package_and_exact_ranks(self) -> None:
        """The last row's levels over 20,000 rows, against the package and
        the exact ranks, within the tolerance above; the median's
        neighbours in place of the median, which a spec refuses."""
        levels = (0.01, 0.05, 0.4999, 0.5001, 0.95, 0.99)
        xs = _stream("t3", 0)[:20_001]
        out = _out(
            tdigest_spec("d", targets=["y"], quantiles=list(levels)), pl.DataFrame({"y": xs})
        )
        last = [out[f"q{p}_y"][-1] for p in levels]
        theirs_digest = tdigest.TDigest(delta=0.01)
        theirs_digest.batch_update(xs[:-1])
        theirs = [theirs_digest.percentile(100.0 * p) for p in levels]
        s = np.sort(xs[:-1])
        assert np.max(np.abs(_ranks(s, last) - np.asarray(levels))) <= TOLERANCE[100]
        assert np.max(np.abs(_ranks(s, last) - _ranks(s, theirs))) <= TOLERANCE[100]

    def test_each_group_has_its_own_digest(self) -> None:
        df = _frame(600, 4)
        out = _out(tdigest_spec("d", targets=["y"], quantiles=[0.25], group="g"), df)
        ys, gs = df["y"].to_list(), df["g"].to_list()
        i = 599
        before = [v for v, g in zip(ys[:i], gs[:i], strict=True) if g == gs[i]]
        (want,) = native.tdigest_quantiles(before, 100.0, [0.25])
        assert out["q0.25_y"][i] == want


class TestStreaming:
    def test_one_chunk_and_a_thousand_give_the_same_output(self) -> None:
        df = _frame(3_000, 5)
        s = tdigest_spec(
            "d", targets=["y", "z"], quantiles=[0.01, 0.25, 0.95], weight="w", group="g"
        )
        one = rp.ModelBank([s]).fit_predict(df)
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, 3)) for i in range(0, 3_000, 3)])
        assert one.equals(many, null_equal=True)

    def test_a_save_goes_on_to_the_bit(self) -> None:
        df = _frame(2_500, 6)
        s = tdigest_spec(
            "d", targets=["y", "z"], quantiles=[0.05, 0.25, 0.95], weight="w", group="g"
        )
        for cut in (1, 30, 1_234, 2_499):
            a = rp.ModelBank([s])
            a.fit_predict(df.slice(0, cut))
            buf = io.BytesIO()
            a.save(buf)
            b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
            rest = df.slice(cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut

    def test_a_zero_weight_row_counts_in_no_clock(self) -> None:
        """The stream with rows of weight 0 gives every other row the
        output of the stream without them, to the bit; they are still read."""
        df = _frame(800, 7).drop("g")
        s = tdigest_spec("d", targets=["y"], quantiles=[0.05, 0.25], weight="w")
        full = _out(s, df)
        keep = (df["w"] > 0).to_numpy()
        thin = _out(s, df.filter(pl.Series(keep)))
        assert full.filter(pl.Series(keep)).equals(thin, null_equal=True)
        assert full.filter(pl.Series(~keep))["q0.25_y"].drop_nulls().len() > 20


class TestWithheld:
    def test_a_level_its_rows_do_not_identify_is_withheld_with_a_reason(self) -> None:
        """0.05 needs 19 rows before it (``p (n + 1) >= 1``), 0.6 two: on
        the rows before, the level is null and ``withheld_reason`` says why,
        not a number."""
        df = pl.DataFrame({"y": np.arange(40, dtype=float)})
        out = _out(tdigest_spec("d", targets=["y"], quantiles=[0.05, 0.6]), df)
        assert out["q0.05_y"][:19].is_null().all()
        assert out["q0.05_y"][19:].is_not_null().all()
        for p in ("0.05", "0.6"):
            assert out[f"es{p}_y"].is_null().equals(out[f"q{p}_y"].is_null()), p
        assert out["q0.6_y"][:2].is_null().all() and out["q0.6_y"][2:].is_not_null().all()
        reasons = out["withheld_reason"].cast(pl.String).to_list()
        assert reasons[:19] == ["level_beyond_sample"] * 19
        assert reasons[19:] == [None] * 21

    def test_a_group_with_too_little_weight_is_withheld(self) -> None:
        """A group of three rows reads no 0.05 level; a group of thirty does,
        each by its own rows."""
        df = pl.DataFrame({"g": ["small"] * 3 + ["big"] * 30, "y": np.arange(33, dtype=float)})
        out = _out(tdigest_spec("d", targets=["y"], quantiles=[0.05], group="g"), df)
        small, big = out[:3], out[3:]
        assert small["q0.05_y"].is_null().all()
        assert small["withheld_reason"].cast(pl.String).to_list() == ["level_beyond_sample"] * 3
        assert big["q0.05_y"][19:].is_not_null().all()
        assert big["withheld_reason"][19:].is_null().all()

    def test_the_rule_reads_kishs_size_not_the_weights_units(self) -> None:
        """Every weight scaled by 1,000 withholds the same rows."""
        df = _frame(200, 8).drop("g")
        s = tdigest_spec("d", targets=["y"], quantiles=[0.05, 0.25], weight="w")
        a = _out(s, df)
        b = _out(s, df.with_columns(pl.col("w") * 1_000.0))
        for f in ("q0.05_y", "q0.25_y", "es0.05_y", "es0.25_y", "withheld_reason"):
            assert a[f].is_null().equals(b[f].is_null()), f


ES_LEVELS = (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)


def _exact_tail_mean(xs_sorted: np.ndarray, p: float) -> float:
    """The mean of the ``p n`` smallest values (``p < 0.5``) or the
    ``(1 - p) n`` largest, a fractional row at the edge counted in part."""
    n = len(xs_sorted)
    if p < 0.5:
        m, vals = p * n, xs_sorted
    else:
        m, vals = (1.0 - p) * n, xs_sorted[::-1]
    whole = int(np.floor(m))
    total = vals[:whole].sum() + (m - whole) * (vals[whole] if whole < n else 0.0)
    return float(total / m)


def _partial_mass_tail_mean(xs: list[float], compression: float, p: float) -> float:
    """The definition: the partial-mass rule over the digest's own centroids
    (``_rheon.tdigest_centroids``), worked out here with ``fractions`` so no
    rounding of ours is in the reference: the mass ``p N`` from the smallest
    mean below 0.5, ``(1 - p) N`` from the largest above it, the boundary
    centroid counted at only the weight the tail still needs."""
    from fractions import Fraction

    cs = native.tdigest_centroids(xs, compression)
    total = sum(Fraction(w) for _, w in cs)
    order, mass = (cs, Fraction(p) * total) if p < 0.5 else (cs[::-1], (1 - Fraction(p)) * total)
    before, acc = Fraction(0), Fraction(0)
    for c, w in order:
        take = min(Fraction(w), mass - before)
        acc += take * Fraction(c)
        before += take
        if before >= mass:
            break
    return float(acc / mass)


#: Our tail mean against the exact one and against the package's trimmed
#: mean, relative to the exact tail mean, at N = 100,000: the largest over
#: 20 seeds and the six levels, with a margin of 1.5. Measured (ours vs
#: exact; the package's own gap to exact is under 6.5e-5 everywhere, its
#: digest about 835 centroids at compression 100 against our 58):
#: compression 100, normal 1.15e-2, lognormal 4.32e-2, t3 2.13e-2, Pareto
#: 1.94e-2; compression 200, normal 3.28e-3, lognormal 1.19e-2, t3 5.13e-3,
#: Pareto 4.32e-3. The gap is the boundary centroid's and the tail's
#: centroids' own spread, largest in a 1% tail of two or three centroids.
ES_TOLERANCE = {
    (100, "normal"): 1.8e-2,
    (100, "lognormal"): 6.5e-2,
    (100, "t3"): 3.2e-2,
    (100, "pareto"): 2.9e-2,
    (200, "normal"): 4.9e-3,
    (200, "lognormal"): 1.8e-2,
    (200, "t3"): 7.7e-3,
    (200, "pareto"): 6.5e-3,
}


class TestExpectedShortfall:
    @pytest.mark.parametrize("p", [0.01, 0.2, 0.45, 0.55, 0.8, 0.99])
    def test_it_is_the_partial_mass_rule_over_the_centroids(self, p: float) -> None:
        xs = _stream("lognormal", 3)[:5_000].tolist()
        (got,) = native.tdigest_tail_means(xs, 100.0, [p])
        assert got is not None
        want = _partial_mass_tail_mean(xs, 100.0, p)
        assert abs(got - want) <= 1e-12 * abs(want), (got, want)

    def test_the_lower_tail_sits_below_the_mean_and_the_upper_above(self) -> None:
        xs = _stream("t3", 1)[:20_000].tolist()
        es = native.tdigest_tail_means(xs, 100.0, list(ES_LEVELS))
        mean = float(np.mean(xs))
        for p, e in zip(ES_LEVELS, es, strict=True):
            assert e is not None
            assert (e <= mean) if p < 0.5 else (e >= mean), (p, mean, e)

    @pytest.mark.parametrize("dist", DISTS)
    @pytest.mark.parametrize("compression", [100, 200])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_it_agrees_with_the_package_and_the_exact_tail_mean(
        self, dist: str, compression: int, seed: int
    ) -> None:
        xs = _stream(dist, seed)
        ours = native.tdigest_tail_means(xs.tolist(), float(compression), list(ES_LEVELS))
        pkg = tdigest.TDigest(delta=1.0 / compression)
        pkg.batch_update(xs)
        s = np.sort(xs)
        for p, e in zip(ES_LEVELS, ours, strict=True):
            assert e is not None
            exact = _exact_tail_mean(s, p)
            theirs = (
                pkg.trimmed_mean(0.0, 100.0 * p) if p < 0.5 else pkg.trimmed_mean(100.0 * p, 100.0)
            )
            tol = ES_TOLERANCE[(compression, dist)] * abs(exact)
            assert abs(e - exact) <= tol, (p, e, exact)
            assert abs(e - theirs) <= tol, (p, e, theirs)

    def test_through_the_bank_it_is_the_core_digests(self) -> None:
        """Each row's tail means are the core digest's over the rows before
        it, to the bit."""
        df = _frame(400, 9).drop("g")
        out = _out(tdigest_spec("d", targets=["y"], quantiles=[0.1, 0.4, 0.9]), df)
        y = df["y"].to_list()
        for i in (30, 211, 399):
            want = native.tdigest_tail_means(y[:i], 100.0, [0.1, 0.4, 0.9])
            got = [out[f"es{p}_y"][i] for p in ("0.1", "0.4", "0.9")]
            assert got == want, i

    def test_one_value_reads_itself_at_every_level(self) -> None:
        """Every row alike: each tail's mean is that value, at both tails."""
        es = native.tdigest_tail_means([2.5] * 500, 100.0, [0.01, 0.4, 0.99])
        assert es == [2.5, 2.5, 2.5]
