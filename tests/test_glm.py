"""`glm`: generalized linear models by recursive IRLS (docs/PLAN.md task 500).

The arithmetic against statsmodels and glum is
`tests/test_second_opinion_glm.py`; this file holds the bank's side: the
refusals, chunk invariance, the zero-weight rule, a saved state's
continuation and the refit's budget.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl
import pytest

from rheon import _rheon as native
from rheon import _spec
from rheon import polars as rp

TIER = "essential"


def counts(n: int = 3000, seed: int = 0, k: int = 3) -> pl.DataFrame:
    """Poisson counts with log mean ``0.5 + x'b``, ``x`` standard normal."""
    rng = np.random.default_rng([500, seed, k])
    x = rng.standard_normal((n, k))
    b = np.linspace(0.3, -0.2, k)
    y = rng.poisson(np.exp(0.5 + x @ b)).astype(float)
    cols = {f"x{j}": x[:, j] for j in range(k)}
    return pl.DataFrame({**cols, "y": y, "w": np.where(np.arange(n) % 7 == 3, 0.0, 1.0)})


def spec(**kw: object) -> dict:
    d: dict = dict(targets=["y"], features=["x0", "x1", "x2"], half_life=float("inf"))
    d.update(kw)
    return rp.spec.glm("m", **d)


def numbers(out: pl.DataFrame) -> pl.DataFrame:
    """Every field but ``coef``, which a chunk's last row writes by design."""
    return out.select(pl.col("m").struct.unnest()).drop("coef")


def coef(bank: rp.ModelBank) -> np.ndarray:
    return np.array([r["coef"] for r in bank.coef().iter_rows(named=True)])


def test_it_recovers_a_poisson_rate():
    bank = rp.ModelBank([spec()])
    bank.fit(counts(20_000))
    assert coef(bank) == pytest.approx([0.5, 0.3, 0.05, -0.2], abs=0.04)


@pytest.mark.parametrize("chunks", [7, 300])
def test_chunking_moves_nothing(chunks):
    """The refit fires on a count of learned rows, never at a chunk's end."""
    df = counts(1200)
    s = spec(half_life=200.0, weight="w", refit_rows=100)
    whole = rp.ModelBank([s]).fit_predict(df)
    parts = rp.ModelBank([s])
    n = df.height
    edges = np.linspace(0, n, chunks + 1).astype(int)
    pieces = [parts.fit_predict(df[a:b]) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    out = pl.concat(pieces)
    assert numbers(whole).equals(numbers(out), null_equal=True)


def test_a_zero_weight_row_is_no_row_without_decay():
    """Hard rule 9: with no decay a row of weight 0 changes nothing, the
    refit's count included."""
    df = counts(800)
    s = spec(weight="w", refit_rows=100)
    a = rp.ModelBank([s])
    a.fit(df)
    b = rp.ModelBank([s])
    b.fit(df.filter(pl.col("w") > 0))
    assert np.array_equal(coef(a), coef(b))


def test_the_prediction_is_made_before_the_row_is_learned():
    """Hard rule 2: changing a row's target moves no prediction up to it."""
    df = counts(400)
    y = df["y"].to_numpy().copy()
    y[200] += 50.0
    a = rp.ModelBank([spec()]).fit_predict(df)["m"].struct.field("pred_y")
    b = (
        rp.ModelBank([spec()])
        .fit_predict(df.with_columns(y=pl.Series(y)))["m"]
        .struct.field("pred_y")
    )
    assert a[:201].equals(b[:201], null_equal=True)
    assert not a[201:].equals(b[201:], null_equal=True)


@pytest.mark.parametrize("cut", [50, 150, 399])
def test_a_saved_state_goes_on_to_the_bit(cut, tmp_path):
    """Saved with the refit's buffer part full, and after it ran."""
    df = counts(400)
    s = spec(half_life=100.0, refit_rows=120)
    whole = rp.ModelBank([s]).fit_predict(df)
    first = rp.ModelBank([s])
    head = first.fit_predict(df[:cut])
    first.save(tmp_path / "glm.state")
    tail = rp.ModelBank.load(tmp_path / "glm.state").fit_predict(df[cut:])
    assert numbers(whole).equals(numbers(pl.concat([head, tail])), null_equal=True)


@pytest.mark.parametrize(
    ("family", "bad", "what"),
    [
        ("poisson", -1.0, "a count, which is never negative"),
        ("gamma", 0.0, "a positive amount, above 0"),
        ("binomial", 1.5, "a proportion, from 0 to 1"),
        ("tweedie", -0.5, "an amount, which is never negative"),
    ],
)
def test_a_target_outside_the_family_is_refused_naming_the_row(family, bad, what):
    df = counts(30).with_columns(y=pl.col("y").clip(0.1, 0.9))
    y = df["y"].to_list()
    y[17] = bad
    bank = rp.ModelBank([spec(family=family)])
    with pytest.raises((pl.exceptions.ComputeError, ValueError)) as e:
        bank.fit(df.with_columns(y=pl.Series(y)))
    msg = str(e.value)
    assert "at row 17" in msg and what in msg, msg


@pytest.mark.parametrize(
    ("kw", "what"),
    [
        (dict(family="normal"), "family must be"),
        (dict(link="probit"), "link must be"),
        (dict(link="logit"), "the poisson family takes the link log"),
        (dict(family="binomial", link="log"), "must stay below 1"),
        (dict(var_power=1.5), "var_power is the Tweedie family's power"),
        (dict(family="tweedie", var_power=2.0), "var_power must be above 1"),
        (dict(ridge=0.0), "ridge must be above 0"),
        (dict(step_cap=0.0), "step_cap must be above 0"),
        (dict(refit_rows=2**21), "bytes a group, past the cap of 262144"),
        (dict(family="gamma", link="inverse", refit_rows=0), "needs refit_rows above 0"),
        (dict(refit_budget=0.0), "refit_budget must be above 0"),
    ],
)
def test_each_bad_setting_is_refused_by_name(kw, what):
    with pytest.raises(ValueError, match="") as e:
        rp.ModelBank([spec(**kw)]).fit(counts(30))
    assert what in str(e.value), str(e.value)


def test_the_canonical_links_fit():
    """Gamma's inverse link and Tweedie's power link start from the refit:
    nothing is predicted before it, and after it every prediction is a
    positive mean."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2000)
    mu = 1.0 / (0.5 + 0.1 * x).clip(0.2)
    df = pl.DataFrame({"x0": x, "x1": x, "x2": x, "y": rng.gamma(2.0, mu / 2.0)})
    for family, link in (("gamma", "inverse"), ("tweedie", "power")):
        s = spec(features=["x0"], family=family, link=link, refit_rows=100, min_weight=0.0)
        pred = rp.ModelBank([s]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
        assert np.isnan(pred[:100]).all(), family
        assert (pred[100:] > 0).all(), family


def test_the_refit_budget_gives_the_later_groups_up():
    """Each group reserves its buffers when it first appears; past the
    budget a group keeps the one-pass fit, and the summary counts it."""
    df = pl.concat([counts(300, seed=g).with_columns(g=pl.lit(f"g{g}")) for g in range(4)]).sort(
        "g", maintain_order=True
    )
    # 3 features, 200 rows: 200 * 5 * 8 = 8,000 bytes a group.
    s = spec(group="g", refit_rows=200, refit_budget=2.5 * 8000 / 2**20)
    bank = rp.ModelBank([s])
    # The notice names the setting `glm` takes (review 2026-11-23, BANK-5:
    # it named `huber_refit_rows`).
    with pytest.warns(rp.ReadinessWarning, match="or lower refit_rows, the rows each refit holds"):
        bank.fit(df)
    assert bank.summary()["refit_given_up"].sum() == 2
    # Under a span of the clock the rows are the data's, and the notice
    # names the bytes a refit may hold.
    s = spec(group="g", refit_span=50.0, refit_budget=2.5 * 8000 / 2**20, max_state_bytes=8000)
    with pytest.warns(rp.ReadinessWarning, match="or lower max_state_bytes, the bytes each"):
        rp.ModelBank([s]).fit(df)


def test_a_refit_past_the_groups_cap_is_refused_at_the_call():
    """The refit's buffers are the one part of the state whose size is not
    the model's shape: ``8 * targets * refit_rows * (features + 2)`` bytes a
    group, known from the spec, so a spec past 256 KB is refused as it is
    built, naming the bytes and the most rows that fit."""
    with pytest.raises(ValueError) as e:
        spec(refit_rows=6554)  # 6,554 rows of 5 numbers: 262,160 bytes
    msg = str(e.value)
    assert "262160 bytes a group, past the cap of 262144" in msg, msg
    assert "at most 6553," in msg, msg
    # At the cap it builds, and two targets halve the rows that fit.
    rp.ModelBank([spec(refit_rows=6553)]).fit(counts(50))
    with pytest.raises(ValueError, match="at most 3276,"):
        spec(targets=["y", "y"], refit_rows=6553)
    # The default uses adaptive doubling refit bounded by max_state_bytes,
    # with no k > 100 cliff (105 features fits 306 >= 2*106 rows inside 256 KB).
    many = rp.spec.glm(
        "m", targets=["y"], features=[f"x{j}" for j in range(105)], half_life=float("inf")
    )
    resolved = json.loads(native.resolved_defaults(_spec._json(many)))
    assert resolved["model"]["adaptive_refit"] is True
    assert resolved["model"]["refit_rows"] == 0
    bank_wide = rp.ModelBank([many])
    bank_wide.fit(counts(350, seed=9, k=105))
    assert np.all(np.isfinite(coef(bank_wide)))


def test_a_half_life_and_a_gap_cap_in_durations_are_the_numbers_on_a_clock_in_the_same_unit():
    """``half_life`` and ``gap_cap``, in days on a date clock, are the run on
    that clock as a number of days. ``refit_rows`` counts rows; its clock
    form is ``refit_span``, which ``test_the_refit_budget_gives_the_later_groups_up``
    runs on a number clock."""
    days = [d for d in pl.date_range(date(2024, 1, 1), date(2024, 12, 31), eager=True)]
    days = [d for d in days if d.weekday() < 5]
    rng = np.random.default_rng(5)
    x = rng.normal(size=len(days))
    df = pl.DataFrame({"d": days, "x0": x, "y": rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float)})
    df = df.with_columns(n=pl.col("d").cast(pl.Int32).cast(pl.Float64))
    kw = dict(targets=["y"], features=["x0"])
    ref = rp.ModelBank([rp.spec.glm("m", clock="n", half_life=5.0, gap_cap=10.0, **kw)])
    got = rp.ModelBank([rp.spec.glm("m", clock="d", half_life="5d", gap_cap="10d", **kw)])
    ref_out, got_out = ref.fit_predict(df)["m"], got.fit_predict(df)["m"]
    assert got_out.equals(ref_out, null_equal=True)
    # The decay is on: the weight is well below the row count.
    assert ref_out.struct.field("weight_sum").max() < 10


def test_a_zero_weight_row_is_in_no_count_and_the_weighted_rows_are_the_stream_without_it():
    """A zero-weight row is predicted and learns nothing, and adds nothing to
    ``weight_sum`` or to the refit's count of learned rows: the weighted rows'
    outputs, every field, are those of the stream with the zero-weight rows
    taken out, to the bit. With no decay, since a decay by the clock's
    step is the clock layer's to leave out for such a row. The first row and
    a run of rows at the refit's boundary are among the zeros."""
    df = counts(600, seed=3).with_columns(w=pl.lit(1.0))
    zero = np.zeros(df.height, dtype=bool)
    zero[[0, 1, 2, 40, 41, 119, 120, 121, 122, 300]] = True
    zero[::11] = True
    withzero = df.with_columns(w=pl.when(pl.Series(zero)).then(0.0).otherwise(1.0))
    kept = withzero.filter(pl.col("w") > 0)
    kw = dict(weight="w", refit_rows=60, coef_every=0)
    a = rp.ModelBank([spec(**kw)]).fit_predict(withzero)["m"].struct.unnest()
    b = rp.ModelBank([spec(**kw)]).fit_predict(kept)["m"].struct.unnest()
    mask = pl.Series(~zero)
    a = a.filter(mask)
    assert a.height == b.height > 400
    assert a.equals(b, null_equal=True), [c for c in a.columns if not a[c].equals(b[c])]
    # The zero-weight rows are predicted all the same.
    zeros = rp.ModelBank([spec(**kw)]).fit_predict(withzero)["m"].struct.unnest().filter(~mask)
    assert zeros["pred_y"].drop_nulls().len() > 0


def _family_frame(family: str, n: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng([578, len(family)])
    x = rng.standard_normal(n)
    mu = np.exp(0.2 + 0.4 * x)
    y = {
        "binomial": (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x)))).astype(float),
        "poisson": rng.poisson(mu).astype(float),
        "gamma": rng.gamma(2.0, mu / 2.0),
        "tweedie": rng.poisson(mu).astype(float),
    }[family]
    return pl.DataFrame({"x0": x, "y": y})


@pytest.mark.parametrize("family", ["binomial", "poisson", "gamma", "tweedie"])
def test_each_familys_hit_rate_is_its_own_test_in_the_bank_and_in_eval(family):
    """Review 2026-11-23, GLM-METRICS: ``emit_metrics`` and ``rp.eval`` read
    a ``glm``'s ``hit_rate`` by the sign test about zero, as for a
    regression target. A binomial ``pred`` is a probability against a 0/1
    label, so its hit is the 0.5 threshold, as for a logistic ``sgd``, and
    ``rp.eval`` reads it as a classifier (``log_loss``); a Poisson, Gamma or
    Tweedie mean is positive against a target never negative, so the sign
    test agrees on every row and ``hit_rate`` is null, as for a Poisson
    ``sgd``."""
    df = _family_frame(family)
    s = spec(features=["x0"], family=family, emit_metrics=True, min_weight=5.0)
    out = rp.ModelBank([s]).fit_predict(df)
    hit = out["m"].struct.field("hit_rate_y")
    m = rp.eval.metrics(out, s, min_samples=1)
    if family == "binomial":
        pred = out["m"].struct.field("pred_y").to_numpy()
        ok = ~np.isnan(pred)
        assert ((pred[ok] > 0.0) & (pred[ok] < 1.0)).all()
        last = hit.drop_nulls()[-1]
        # The bank's figure, read before the last row, against the
        # threshold's over the rows ahead of it, with no decay.
        y = df["y"].to_numpy()
        rows = ok.copy()
        rows[-1] = False
        want = np.mean((pred[rows] > 0.5) == (y[rows] > 0.5))
        assert last == pytest.approx(want, abs=1e-12)
        assert "log_loss" in m.columns
    else:
        assert hit.null_count() == df.height
        assert m["hit_rate"][0] is None
    # A spec without a family is a Poisson fit.
    if family == "poisson":
        raw = dict(s)
        raw["model"] = {k: v for k, v in raw["model"].items() if k != "family"}
        assert rp.eval.metrics(out, raw, min_samples=1)["hit_rate"][0] is None


@pytest.mark.parametrize("key", ["emit_specification", "emit_tails"])
def test_a_glm_refuses_the_checks_of_a_constant_normal_residual(key):
    """A glm's residuals have its family's variance, a function of the mean,
    and its family's shape: on a stream a Poisson fit fits exactly,
    Breusch-Pagan and Jarque-Bera rejected on every row, and RESET, read
    under a constant variance, on a quarter of them (review 2026-11-23,
    DIAG-11). Each switch is refused, by name."""
    with pytest.raises(ValueError, match=f"{key} tests residuals of constant variance"):
        rp.ModelBank(
            [rp.spec.glm("m", features=["x"], targets=["y"], half_life=200.0, **{key: True})]
        )


def test_a_glms_mean_checks_hold_their_level_on_an_exact_fit():
    """The calibration's Wald, a sandwich, and the CUSUM read a glm's mean:
    on a stream a Poisson fit fits exactly each holds its 5% level after
    the first thousand rows."""
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float)
    spec = rp.spec.glm(
        "m",
        features=["x"],
        targets=["y"],
        half_life=200.0,
        family="poisson",
        emit_calibration=True,
        emit_breaks=True,
    )
    out = rp.ModelBank([spec]).fit_predict(pl.DataFrame({"x": x, "y": y}))["m"].struct.unnest()
    out = out[1000:]
    assert (out["calibration_wald_y"] > 5.991).mean() < 0.1
    assert (out["cusum_y"].abs() > 1.96).mean() < 0.1
