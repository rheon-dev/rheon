"""The five-row stream Hypothesis drew for `test_properties`' out-of-sample
property under `expectile` (docs/PLAN.md task 538, the coordinator's
point 1): a feature that is constant at -1000, collinear with the
intercept, four learned rows and a fifth of weight 0. The property needs a
row the model scored; this pins what `expectile` makes of the stream."""

from __future__ import annotations

import polars as pl

from rheon import polars as rp

TIER = "essential"


def stream() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "g": ["g0"] * 5,
            "t": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x0": [-1000.0] * 5,
            "x1": [-1742.0, -1001.0, -1742.0, -1000.0, -1000.0],
            "y0": [-1001.0, -1001.0, -1001.0, -1001.0, -1000.0],
            "w": [1.0, 1.0, 1.0, 1.0, 0.0],
            "s": ["s0"] * 5,
        }
    )


def spec(model: str, **extra: object) -> dict:
    opts: dict = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        half_life=20.0,
        weight="w",
        group="g",
        min_weight=2.0,
    )
    opts.update(extra)
    return getattr(rp.spec, model)("m", **opts)


def preds(model: str, **extra: object) -> list:
    out = rp.ModelBank([spec(model, **extra)]).fit_predict(stream())
    field = next(f.name for f in out.schema["m"].fields if f.name.startswith("pred_"))
    return out["m"].struct.field(field).to_list()


def test_expectile_scores_a_row_of_the_collinear_stream():
    """`ewridge`, `huber`, `quantile`, `rls` and `sgd` score the last warm
    row of this stream, -1001; `expectile` did not, on integration e52d7345
    as on this branch: its ridge, 1e-6 in the features' squared units, left
    the constant feature's pivot below 1e-12 of the raw Gram's largest
    diagonal, the feature's squared level of 1e6, so no solve passed. The
    system is now solved about the features' means where the raw one fails
    its floor, the same minimiser, and every stream the raw solve fitted
    keeps its bits."""
    for model, kw in [
        ("ewridge", {}),
        ("huber", {}),
        ("quantile", {"quantile": 0.8}),
        ("sgd", {}),
    ]:
        assert preds(model, **kw)[3] == -1001.0, model
    for tau in [0.2, 0.5, 0.8]:
        got = preds("expectile", expectile=tau)
        assert got[:3] == [None, None, None], got
        assert abs(got[3] + 1001.0) < 1e-9 * 1001.0, (tau, got)


def test_without_a_ridge_the_exact_fit_still_waits():
    """`ridge = 0` is the exact loss's fit, which waits for a pivot the
    collinear design never gives: nothing is scored, as documented."""
    assert preds("expectile", expectile=0.8, ridge=0.0) == [None] * 5
