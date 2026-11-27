"""``rheon.polars.ops`` is the one public home of everything that makes a
feature (PLAN 579; the user, 2026-11-24), and the API review of each wave
(PLAN 580) holds the root to it.

The seven window operators 2.0.0 exported at the root of ``rheon.polars``
stay there as forwards: each call warns with a ``RheonDeprecationWarning``
naming the ``rp.ops`` path and then is that call. The old names of
``ewm_*(reverse=True)``, released at the root too, still refuse the call.
Nothing else that makes a feature is at the root: an operator added since
2.0.0 is in ``rp.ops`` alone.
"""

from __future__ import annotations

import inspect
import warnings

import polars as pl
import pytest

import rheon
from rheon import polars as rp
from rheon._renamed import is_moved, is_renamed

TIER = "essential"

#: The operators 2.0.0 exported at the root, each kept there as a forward
#: that warns until the next major release (`git show v2.0.0:python/rheon/polars/__init__.py`).
RELEASED_AT_ROOT = (
    "ewm_mean",
    "ewm_quantile",
    "ewm_rate",
    "ewm_std",
    "ewm_sum",
    "ewm_var",
    "increment",
)

#: The old names of `ewm_*(reverse=True)`, which 2.0.0 had at the root as
#: refusals, and which still refuse the call there.
REFUSED_AT_ROOT = ("rewm_mean", "rewm_rate", "rewm_sum")

#: The operators added after 2.0.0, which never had a root name.
NEVER_AT_ROOT = (
    "cross_mean",
    "cross_quantile",
    "cross_rank",
    "cross_ref",
    "cross_std",
    "ewm_moments",
    "ewm_profile",
    "rolling_max",
    "rolling_min",
    "rolling_quantile",
    "rolling_rank",
    "tick_sign",
    "run_length",
    "clock_lag",
    "clock_drift",
    "clock_step",
    "invert",
    "unknown",
)

_CALLS: dict[str, dict[str, object]] = {
    "ewm_mean": {"half_life": 2.0},
    "ewm_quantile": {"quantile": 0.5, "half_life": 2.0},
    "ewm_rate": {"half_life": 2.0, "window_size": 3.0},
    "ewm_std": {"half_life": 2.0},
    "ewm_sum": {"half_life": 2.0, "window_size": 3.0, "reverse": True},
    "ewm_var": {"half_life": 2.0, "bias": True},
    "increment": {},
}


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [0.0, 1.0, 1.5, 3.0, 4.0, 4.0, 6.5, 7.0],
            "x": [1.0, 3.0, -2.0, 4.0, 0.5, 2.0, None, 5.0],
        }
    )


@pytest.mark.parametrize("name", RELEASED_AT_ROOT)
def test_a_released_root_name_warns_naming_its_ops_path_and_gives_the_same_result(name):
    kw = _CALLS[name]
    with pytest.warns(rp.RheonDeprecationWarning) as caught:
        old = getattr(rp, name)("x", **kw)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith(f"rp.{name}: it moved to rp.ops.{name}."), message
    assert f"Call rp.ops.{name} instead." in message
    assert caught[0].filename == __file__, "the warning points at the caller's line"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = getattr(rp.ops, name)("x", **kw)
    assert old.meta.eq(new)
    run = {"clock": "t", "gap_cap": 10.0}
    got = rp.stream.with_windows(_frame(), f=old.alias("f"), **run)
    want = rp.stream.with_windows(_frame(), f=new.alias("f"), **run)
    assert got.equals(want)
    assert got["f"].null_count() < got.height


@pytest.mark.parametrize("name", RELEASED_AT_ROOT)
def test_a_released_root_name_keeps_its_signature_and_its_errors(name):
    assert inspect.signature(getattr(rp, name)) == inspect.signature(getattr(rp.ops, name))
    refused = pytest.raises(TypeError, match=f"^rp\\.ops\\.{name}(\\(reverse=True\\))?: ")
    with pytest.warns(rp.RheonDeprecationWarning), refused:
        getattr(rp, name)(1.5, **_CALLS[name])


def test_a_released_root_name_still_imports_by_name_and_stays_out_of_all():
    from rheon.polars import ewm_mean, increment

    assert ewm_mean is rp.ewm_mean and increment is rp.increment
    for name in (*RELEASED_AT_ROOT, *REFUSED_AT_ROOT):
        assert name not in rp.__all__


@pytest.mark.parametrize("name", REFUSED_AT_ROOT)
def test_an_old_reverse_name_still_refuses_naming_the_ops_form(name):
    kind = name.removeprefix("rewm_")
    for where, who in ((rp, f"rp.{name}"), (rp.ops, f"rp.ops.{name}")):
        with pytest.raises(TypeError) as caught:
            getattr(where, name)("x", half_life=2.0, window_size=5.0)
        assert str(caught.value) == (
            f"{who}: this function was renamed rp.ops.ewm_{kind}(..., reverse=True). "
            f"Call rp.ops.ewm_{kind}(..., reverse=True) instead."
        )


@pytest.mark.parametrize("name", NEVER_AT_ROOT)
def test_an_operator_added_since_the_release_is_in_ops_alone(name):
    assert name in rp.ops.__all__
    assert not hasattr(rp, name)
    assert not hasattr(rheon, name)
    with pytest.raises(ImportError):
        exec(f"from rheon.polars import {name}", {})


def _makes_a_feature(name: str, obj: object) -> bool:
    """Whether the public ``obj`` is something that makes a feature: a name
    ``rp.ops`` exports or keeps as an old name, an object defined there, or
    a callable whose signature returns a Polars expression."""
    if name in rp.ops.__all__ or is_renamed(getattr(rp.ops, name, None)):
        return True
    target = inspect.unwrap(obj) if callable(obj) else obj
    if getattr(target, "__module__", None) == "rheon.polars.ops":
        return True
    if callable(obj) and not isinstance(obj, type):
        try:
            returns = inspect.signature(obj).return_annotation
        except (TypeError, ValueError):
            return False
        return "Expr" in str(returns)
    return False


def test_nothing_that_makes_a_feature_is_at_a_root_but_a_released_alias():
    """The rule each wave's API review holds: a new feature call goes in
    ``rp.ops``. The roots of ``rheon`` and ``rheon.polars`` may hold such a
    name only as one of the forwards or refusals 2.0.0's names left, kept
    out of ``__all__``; a new one fails here."""
    allowed = set(RELEASED_AT_ROOT) | set(REFUSED_AT_ROOT)
    found: dict[str, str] = {}
    for module, label in ((rheon, "rheon"), (rp, "rheon.polars")):
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if inspect.ismodule(obj) or not _makes_a_feature(name, obj):
                continue
            if name in getattr(module, "__all__", ()):
                found[f"{label}.{name}"] = "is in __all__"
            elif module is not rp or name not in allowed:
                found[f"{label}.{name}"] = "is not one of 2.0.0's root names"
            elif name in RELEASED_AT_ROOT and not is_moved(obj):
                found[f"{label}.{name}"] = "is not a forward that warns"
            elif name in REFUSED_AT_ROOT and not (is_renamed(obj) and not is_moved(obj)):
                found[f"{label}.{name}"] = "is not a refusal"
    assert not found, "a name that makes a feature belongs in rheon.polars.ops alone: " + "; ".join(
        f"{k} {v}" for k, v in sorted(found.items())
    )
    present = {n for n in dir(rp) if not n.startswith("_") and _makes_a_feature(n, getattr(rp, n))}
    assert present == allowed, "the test found the aliases it should"
