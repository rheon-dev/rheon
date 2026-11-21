"""`glm`: generalized linear models by recursive IRLS (docs/PLAN.md task 500).

The arithmetic against statsmodels and glum is
`tests/test_second_opinion_glm.py`; this file holds the bank's side: the
refusals, chunk invariance, the zero-weight rule, a saved state's
continuation and the refit's budget.
"""

from __future__ import annotations

import json

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
    with pytest.warns(rp.ReadinessWarning):
        bank.fit(df)
    assert bank.summary()["refit_given_up"].sum() == 2


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
    # The default is held to the cap: 100 features, 321 rows.
    many = rp.spec.glm(
        "m", targets=["y"], features=[f"x{j}" for j in range(100)], half_life=float("inf")
    )
    resolved = json.loads(native.resolved_defaults(_spec._json(many)))
    assert resolved["model"]["refit_rows"] == 321
