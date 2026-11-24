"""Names the helper modules renamed before 1.0, each refused by its old name.

Task 144's rule (docs/PLAN.md): an old name is not an alias. It is refused,
and the message names the new one, so a caller learns the new spelling from
the error rather than from a changelog. A spec's keyword goes through
``_spec._RENAMED``; a helper function's keyword goes through
:func:`renamed_keywords` on that function, and a renamed function leaves a
stub from :func:`renamed_function` under its old name (task 197). A keyword
for something removed, such as ``rp.target``'s ``relative_to`` (task 201),
is refused by :func:`removed_keywords`, saying what replaces it.

These are the refusals every module uses: the stream functions'
``_spec._renamed_keywords`` (``chunk_size``, task 196) is
:func:`renamed_keywords`. A rename is never forwarded: every old name is
refused here, naming the new one.

One exception, for a function that moved module after a release (PLAN 579):
:func:`moved_function` keeps it under its released path as a forward that
issues a ``RheonDeprecationWarning`` naming the new path, until the next
major release. It is marked as a stub is, so the reference and the API
snapshot show the new path only.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import NoReturn

from rheon._warnings import (
    MOVED_MARK,
    RENAMED_MARK,
    RheonDeprecationWarning,
    _user_stacklevel,
)

#: The qualified names a decorated function reports, as the reader calls it.
_READER_NAMES = {
    "LazyFrameRheonNamespace.fit_predict()": "lf.rheon.fit_predict",
    "LazyFrameRheonNamespace.predict()": "lf.rheon.predict",
    "DataFrameRheonNamespace.fit_predict()": "df.rheon.fit_predict",
    "DataFrameRheonNamespace.predict()": "df.rheon.predict",
    "fit_predict()": "rp.fit_predict",
    "predict()": "rp.predict",
    "refresh_time()": "rp.stream.refresh_time",
    "with_windows()": "rp.stream.with_windows",
    "embargo()": "rp.stream.embargo",
    "ModelBank.fit_predict_batches()": "ModelBank.fit_predict_batches",
    "ModelBank.fit()": "ModelBank.fit",
}


def renamed_keywords[**P, R](
    who: str, table: Mapping[str, str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator refusing each keyword of ``table`` by name, naming the
    keyword that replaced it, before ``fn`` sees the call. The signature
    ``inspect`` and the reference show is ``fn``'s own (``functools.wraps``)."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for old, new in table.items():
                if old in kwargs:
                    where = _READER_NAMES.get(who, who)
                    raise TypeError(f"{where}: {old} was renamed {new}. Use {new} instead.")
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def removed_keywords[**P, R](
    who: str, keywords: Iterable[str], why: str
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator refusing each of ``keywords``, which named something the
    library no longer has, with ``why``: what replaces it. As
    :func:`renamed_keywords`, the signature shown is ``fn``'s own, so the
    removed keywords are in neither the reference nor the API snapshot."""
    gone = tuple(keywords)

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if any(k in kwargs for k in gone):
                raise TypeError(f"{who}: {why}")
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def renamed_function(old: str, new: str, more: str = "") -> Callable[..., NoReturn]:
    """A stub for a function renamed ``new``, raising ``TypeError`` naming it
    (and ``more``, what else the rename changed) whatever it is called with.
    It is kept out of its module's ``__all__``, so the reference and the API
    snapshot show the new name only."""

    def stub(*args: object, **kwargs: object) -> NoReturn:
        who = old
        raise TypeError(f"{who}: this function was renamed {new}{more}. Call {new} instead.")

    stub.__name__ = stub.__qualname__ = old.rsplit(".", 1)[-1]
    stub.__doc__ = f"Renamed {new}: calling it raises TypeError, naming the new name."
    setattr(stub, RENAMED_MARK, new)
    return stub


def moved_function[**P, R](old: str, new: str, fn: Callable[P, R]) -> Callable[P, R]:
    """``fn`` under the path ``old`` it had in a release, now at ``new``: each
    call issues a ``RheonDeprecationWarning`` naming ``new`` and then is
    ``fn``'s call, with its arguments, result and errors unchanged. Kept out
    of its module's ``__all__``, as a stub is."""

    @functools.wraps(fn)
    def alias(*args: P.args, **kwargs: P.kwargs) -> R:
        who = old
        warnings.warn(
            f"{who}: it moved to {new}. The old path works for now, and will be removed at "
            f"the next major release. Call {new} instead.",
            RheonDeprecationWarning,
            stacklevel=_user_stacklevel(),
        )
        return fn(*args, **kwargs)

    alias.__doc__ = f"Moved to :func:`{new}`: calling it warns, then calls it."
    setattr(alias, RENAMED_MARK, new)
    setattr(alias, MOVED_MARK, new)
    return alias


def is_moved(obj: object) -> bool:
    """Whether ``obj`` is a forward from :func:`moved_function`: a released
    path kept, with a deprecation warning, for a function that moved."""
    return getattr(obj, MOVED_MARK, None) is not None


def is_renamed(obj: object) -> bool:
    """Whether ``obj`` is kept only under an old name: a refusal stub from
    :func:`renamed_function` or :func:`renamed_method`. The API snapshot and the
    reference leave such a name out, so both show the new name only."""
    target = obj.fget if isinstance(obj, property) else obj
    return getattr(target, RENAMED_MARK, None) is not None


def renamed_method(
    cls: str, old: str, new: str, *, read_as: str | None = None
) -> Callable[..., NoReturn]:
    """A method stub for a method renamed ``new``, raising ``AttributeError``
    naming it, as a missing attribute does. Marked, as
    :func:`renamed_function`'s stubs are, so the reference and the API
    snapshot show the new name only."""

    def stub(self: object, *args: object, **kwargs: object) -> NoReturn:
        who = f"{cls}.{old}"
        if read_as is not None:
            raise AttributeError(
                f"{who}: it was renamed {new}, a property. Read {read_as}, without the call."
            )
        raise AttributeError(f"{who}: this method was renamed {new}. Use {new} instead.")

    stub.__name__ = old
    stub.__qualname__ = f"{cls}.{old}"
    stub.__doc__ = f"Renamed :meth:`{new}`: calling it raises AttributeError, naming the new name."
    setattr(stub, RENAMED_MARK, new)
    return stub
