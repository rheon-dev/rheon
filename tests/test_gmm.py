"""``gmm`` through the bank: its start's withheld reasons by name, the fit a
separated pair starts at, chunk invariance, and its refusals. The oracles
(scikit-learn's ``GaussianMixture`` from the same start, and the weighted
longhand under decay) come with the model's next part.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

INF = float("inf")


def pair(n: int, seed: int = 1) -> pl.DataFrame:
    """Two round clusters, 65 and 35 percent of the rows, at (4, 3) and the
    origin, spread 0.5."""
    rng = np.random.default_rng(seed)
    far = rng.random(n) < 0.65
    x = rng.standard_normal((n, 2)) * 0.5 + np.where(far[:, None], [4.0, 3.0], [0.0, 0.0])
    return pl.DataFrame(x, schema=["a", "b"])


def one(n: int, seed: int = 2) -> pl.DataFrame:
    """One round cluster."""
    x = np.random.default_rng(seed).standard_normal((n, 2))
    return pl.DataFrame(x, schema=["a", "b"])


def fields(df: pl.DataFrame, **kw) -> pl.DataFrame:
    spec = rp.spec.gmm("g", features=["a", "b"], half_life=INF, **kw)
    return rp.ModelBank([spec]).fit_predict(df)["g"].struct.unnest()


def test_a_separated_pair_starts_and_reads_its_components():
    out = fields(pair(2000), n_components=2)
    started = out["cluster"].is_not_null()
    first = int(started.arg_true()[0])
    # The start is tried at each doubling of the buffer, and never forced.
    assert first & (first - 1) == 0, first
    assert out["withheld_reason"][0] == "below_warm_rows"
    assert out["withheld_reason"][first] is None
    late = out[first:]
    total = late["membership_0"] + late["membership_1"]
    assert np.allclose(total.to_numpy(), 1.0)
    # The larger component first: most rows read component 0.
    assert (late["cluster"] == 0).mean() > 0.55


def test_one_cluster_never_starts_two_components_and_says_why():
    out = fields(one(1000), n_components=2)
    assert out["cluster"].is_null().all()
    # A second component on one cluster is either too light to identify or
    # not told apart from the first, by the try.
    assert out["withheld_reason"][-1] in ("not_identified", "not_separated")


def test_a_buffer_at_its_bound_is_withheld_by_its_own_reason():
    out = fields(pair(400), n_components=2, max_state_bytes=24 * 10)
    assert out["cluster"].is_null().all()
    assert out["withheld_reason"][-1] == "above_max_state_bytes"


def test_one_chunk_or_many_give_the_same_rows():
    df = pair(600)
    spec = rp.spec.gmm("g", features=["a", "b"], n_components=2, half_life=200.0)
    whole = rp.ModelBank([spec]).fit_predict(df)
    bank = rp.ModelBank([spec])
    parts = pl.concat([bank.fit_predict(df[i : i + 37]) for i in range(0, 600, 37)])
    # `coef` is written on each chunk's last row by default, so it is left
    # out: every field the rows read is the same.
    a, b = (t["g"].struct.unnest().drop("coef") for t in (whole, parts))
    assert a.equals(b, null_equal=True)


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"n_components": 0}, "n_components must be at least 1"),
        ({"n_components": 2, "covariance_type": "tied"}, "covariance_type must be"),
        ({"n_components": 2, "reg_covar": 0.0}, "reg_covar must be"),
        ({"n_components": 2, "delta_bic": -1.0}, "delta_bic must be"),
        ({"n_components": 2, "tol": 0.0}, "tol must be"),
        ({"n_components": 2, "n_init": 0}, "n_init and max_iter"),
        ({"n_components": 2, "max_state_bytes": 0}, "max_state_bytes must be"),
    ],
)
def test_each_setting_is_refused_by_name(kw, says):
    with pytest.raises(Exception, match=says):
        rp.ModelBank([rp.spec.gmm("g", features=["a", "b"], half_life=INF, **kw)])


def test_it_takes_no_targets():
    with pytest.raises(TypeError, match="targets"):
        rp.spec.gmm("g", targets=["a"], features=["a", "b"], n_components=2)  # type: ignore[call-arg]
