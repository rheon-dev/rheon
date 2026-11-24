"""The surfaces 1.0 does not promise say so (docs/PLAN.md task 198; review
round 4, D5, N23, AP12, DA4 and YB11): a docstring admonition, and an
``UnstableWarning`` raised only under ``RHEON_WARN_UNSTABLE=1``, as
Polars raises its own under ``POLARS_WARN_UNSTABLE``.

The surfaces: the formula tree's written form in a TOML file (the command
line's, in ``crates/rheon-cli/tests/run.rs``), ``fit_predict_arrow``,
``predict_arrow`` and ``ArrowStruct``, and the modules ``rp.sim``,
``rheon.corr`` and ``rheon.polars._gbdt``. Each emits under the variable and
is silent without it. A ``with_windows`` state and a formula target inside a
bank's state were labelled too until PLAN 542 put every state file under one
envelope, versioned with a loader for each layout: they are promised now,
and :func:`test_a_promised_surface_never_warns` holds them silent.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp
from rheon._warnings import UNSTABLE_VAR
from rheon.polars import _gbdt

TIER = "essential"

pytestmark = pytest.mark.filterwarnings("ignore::rheon.ReadinessWarning")


def frame(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((n, 2))
    return pl.DataFrame(
        {
            "t": np.arange(float(n)),
            "x0": x[:, 0],
            "x1": x[:, 1],
            "y": x @ [1.0, -0.5] + 0.1 * rng.standard_normal(n),
            "mid": 100.0 + np.cumsum(0.1 * rng.standard_normal(n)),
        }
    )


def ridge() -> dict:
    return rp.spec.ewridge("m", targets=["y"], features=["x0", "x1"], half_life=20.0)


def formula_spec() -> dict:
    target = (
        rp.ewm_mean("mid", half_life=5.0, window_size=10.0, reverse=True) - pl.col("mid")
    ).alias("fwd")
    return rp.spec.ewridge(
        "edge",
        targets=[target],
        features=["x0"],
        clock="t",
        gap_cap=50.0,
        half_life=40.0,
        embargo=12.0,
    )


def unstable_calls(tmp_path) -> dict[str, Callable[[], object]]:
    """One call of each labelled surface."""
    df = frame()
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    return {
        "fit_predict_arrow": lambda: rp.ModelBank([ridge()]).fit_predict_arrow(df),
        "predict_arrow": lambda: rp.ModelBank([ridge()]).predict_arrow(df),
        "rp.sim.regimes": lambda: rp.sim.regimes(
            2, states=[np.eye(2)], transition=[[1.0]], n_blocks=2, rows_per_block=5, seed=1
        ),
        "rheon.corr.nearest": lambda: rp.corr.nearest(corr),
        "rheon.corr.shrink": lambda: rp.corr.shrink(corr, alpha=0.3),
        "rheon.polars._gbdt.xgboost_matrix": lambda: _gbdt.xgboost_matrix(df, label="y"),
        "rheon.polars._gbdt.lightgbm_dataset": lambda: _gbdt.lightgbm_dataset(
            df, label="y", params={"verbose": -1}
        ),
    }


@pytest.fixture
def calls(tmp_path, monkeypatch):
    monkeypatch.delenv(UNSTABLE_VAR, raising=False)
    return unstable_calls(tmp_path)


SURFACES = [
    "fit_predict_arrow",
    "predict_arrow",
    "rp.sim.regimes",
    "rheon.corr.nearest",
    "rheon.corr.shrink",
    "rheon.polars._gbdt.xgboost_matrix",
    "rheon.polars._gbdt.lightgbm_dataset",
]


def test_the_warning_is_exported_and_named_as_polars_names_it():
    assert issubclass(rp.UnstableWarning, Warning)
    assert "UnstableWarning" in rp.__all__
    assert UNSTABLE_VAR == "RHEON_WARN_UNSTABLE"


@pytest.mark.parametrize("surface", SURFACES)
def test_each_surface_warns_under_the_variable(surface, calls, monkeypatch):
    monkeypatch.setenv(UNSTABLE_VAR, "1")
    with pytest.warns(rp.UnstableWarning, match="this is unstable, so it may change") as caught:
        calls[surface]()
    assert caught, surface


@pytest.mark.parametrize("value", [None, "0", "true"])
@pytest.mark.parametrize("surface", SURFACES)
def test_each_surface_is_silent_without_it(surface, value, calls, monkeypatch):
    if value is not None:
        monkeypatch.setenv(UNSTABLE_VAR, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error", rp.UnstableWarning)
        calls[surface]()


def test_a_promised_surface_never_warns(monkeypatch, tmp_path):
    """The bank, its file without a formula target, the window operators
    without a state, and the eval and gram modules are promised."""
    monkeypatch.setenv(UNSTABLE_VAR, "1")
    df = frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", rp.UnstableWarning)
        bank = rp.ModelBank([ridge()])
        out = bank.fit_predict(df)
        bank.save(tmp_path / "plain.state")
        rp.ModelBank.load(tmp_path / "plain.state")
        bank.write_json()
        rp.stream.with_windows(
            df, level=rp.ewm_mean("mid", half_life=5.0, window_size=10.0), clock="t", gap_cap=50.0
        )
        rp.eval.metrics(out, "m")
        # A window run's state and a formula target inside a bank's state
        # (PLAN 542), saved, loaded and checkpointed.
        windows = tmp_path / "windows.state"
        rp.stream.with_windows(
            df,
            level=rp.ewm_mean("mid", half_life=5.0, window_size=10.0),
            clock="t",
            gap_cap=50.0,
            save_state=windows,
        )
        rp.stream.with_windows(
            df.with_columns(pl.col("t") + 1000.0),
            level=rp.ewm_mean("mid", half_life=5.0, window_size=10.0),
            clock="t",
            gap_cap=50.0,
            load_state=windows,
        )
        held = rp.ModelBank([formula_spec()])
        held.fit_predict(df)
        held.save(tmp_path / "held.state")
        rp.ModelBank.load(tmp_path / "held.state")
        held.write_json()
        rp.ModelBank.load_bytes(held.save_bytes())
        rp.stream.save_checkpoint(tmp_path / "run.checkpoint", {"windows": windows, "bank": held})
        rp.stream.load_checkpoint(tmp_path / "run.checkpoint")


def test_a_corr_call_that_runs_others_warns_once(monkeypatch):
    monkeypatch.setenv(UNSTABLE_VAR, "1")
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rp.corr.shrink(corr, alpha=0.3)
    got = [w for w in caught if issubclass(w.category, rp.UnstableWarning)]
    assert len(got) == 1, [str(w.message) for w in got]
    assert "rheon.corr.shrink" in str(got[0].message)


def test_every_corr_function_is_labelled():
    for name in rp.corr.__all__:
        fn = getattr(rp.corr, name)
        if callable(fn):
            assert hasattr(fn, "__wrapped__"), name


@pytest.mark.parametrize(
    "obj",
    [
        rp.sim,
        rp.corr,
        _gbdt,
        rp.ArrowStruct,
        rp.ModelBank.fit_predict_arrow,
        rp.ModelBank.predict_arrow,
    ],
    ids=lambda o: getattr(o, "__qualname__", getattr(o, "__name__", str(o))),
)
def test_each_surface_says_so_in_its_docstring(obj):
    doc = inspect.getdoc(obj) or ""
    assert ".. warning::" in doc and "**unstable**" in doc, doc[-600:]


@pytest.mark.parametrize(
    "obj",
    [rp.ModelBank.save, rp.stream.with_windows],
    ids=lambda o: o.__qualname__,
)
def test_a_state_surface_no_longer_says_it_is_unstable(obj):
    """A window run's state and a formula target inside a bank's state
    are promised since PLAN 542: neither docstring labels them."""
    doc = inspect.getdoc(obj) or ""
    assert "**unstable**" not in doc
    assert "every later" in doc
