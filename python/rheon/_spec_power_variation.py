"""The power-variation builder, :func:`power_variation`: a part of
:mod:`rheon._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`rheon.spec` documents it with the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common

#: The keys a term may hold, beside its name.
_TERM_KEYS = frozenset(
    {"powers", "normalize", "kernel", "bandwidth", "flat_top", "center", "scale"}
)


def _term(name: str, options: Mapping[str, Any]) -> dict[str, Any]:
    """One term in the spec's words: its name, then each option given, the
    false flags left out as the Rust spec leaves them out."""
    if not isinstance(options, Mapping):
        raise TypeError(
            f"power_variation: term {name!r} must be a dict of its options, such as "
            f'{{"powers": [1, 1]}}, but it is {type(options).__name__}. Give each term a '
            "dict of its options."
        )
    unknown = sorted(set(options) - _TERM_KEYS)
    if unknown:
        raise ValueError(
            f"power_variation: term {name!r} has {', '.join(map(repr, unknown))}, which no "
            f"term takes. A term takes {', '.join(sorted(_TERM_KEYS))}. Leave the others out."
        )
    term: dict[str, Any] = {"name": name}
    for key in ("powers", "normalize", "kernel", "bandwidth", "flat_top", "center", "scale"):
        value = options.get(key)
        if value is None or value is False:
            continue
        term[key] = [float(p) for p in value] if key == "powers" else value
    return term


@_checked
def power_variation(
    name: str,
    *,
    targets: TargetList,
    terms: Mapping[str, Mapping[str, Any]],
    window_rows: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Power variation and kernel-weighted autocovariance sums of each
    target, one pass.

    Use it on returns, one series a target. Each term is a sum over the
    returns before the row: of products of powers of consecutive absolute
    returns, or of their autocovariances weighted by a kernel up to a
    bandwidth. Realized variance and bipower variation, the realized kernel
    and the HAC long-run variance are each one term, and a measure built
    from several, such as two-scale realized variance, is composed from the
    fields in Polars.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``session`` and ``group``,
    are in :mod:`rheon.spec`. ``half_life`` is how fast the sums forget,
    ``inf`` for not at all; set it to ``inf`` beside a window. A row's
    ``weight`` counts its terms that many times: the terms are sums, so
    they scale with it. A row of weight 0 counts in no clock: the next row
    steps as if it were not there, it is in no window and in no term. A
    null target is that target's absent return: its terms reach across the
    row to its last return. A gap past ``gap_cap`` or a session's edge is a
    step of the clock like any other; to sum each session on its own, set
    ``session_gap="reset"``.

    This model's own parameters:

    ``terms``
        A dict of the terms, each a name and a dict of its options; the
        name names its fields, ``<name>_<target>``. A term has either
        ``powers`` or ``kernel``:

        ``powers``
            A list of powers, one a return of the product, oldest first:
            ``[2]`` is the sum of squared returns, ``[1, 1]`` of products of
            neighboring absolute returns. Each finite and above 0.
        ``normalize``
            With ``powers``: multiply by ``W^(Σp/2 - 1) / Π μ_p``, ``μ_p``
            the ``p``-th absolute moment of a standard normal and ``W`` the
            returns' weight, the count of returns at unit weights without a
            decay, so the term is Barndorff-Nielsen and Shephard's
            multipower variation: ``[1, 1]`` bipower variation, ``[4/3,
            4/3, 4/3]`` tripower quarticity. The weight, not ``n``, makes
            the quarticity over the bipower variation squared 1 at a
            constant volatility under any weights, as ``realized``'s.
            Default false, the plain sum. Refused beside a ``scale`` past
            1.
        ``kernel``
            ``"bartlett"``, ``"parzen"``, ``"tukey_hanning"`` or
            ``"rectangular"``, with ``bandwidth``: the sum of ``r_j² + 2 Σ_h
            c_h r_j r_{j-h}`` over lags ``h`` from 1 to the bandwidth ``H``,
            ``c_h = k(h / (H + 1))``.
        ``bandwidth``
            With ``kernel``: ``H``, the last lag with weight, a count of
            returns.
        ``flat_top``
            With ``kernel``: the flat-top weights ``c_h = k((h - 1) / H)``,
            which give lag 1 the full weight (the flat-top realized kernel).
            Default false.
        ``center``
            With ``kernel``: take each product about the mean of the returns
            the term is of, as the HAC long-run variance does; divide by
            ``n`` for it. Default false.
        ``scale``
            ``K``: the term of the ``K``-step returns, the sums of ``K``
            consecutive returns, at every row, divided by ``K``, which is
            the average over the ``K`` offset grids of ``K``-step returns.
            A count of returns. Default 1.
    ``window_rows``
        Sum only the returns of the last this many rows of positive weight,
        in place of a decay. A term is in the window while every return in
        it is. Its bytes are checked against ``max_state_bytes`` here: a
        window past it is refused naming the largest that fits. Default
        none.
    ``window_size``
        Sum only the returns of the last this long on the clock, in clock
        units: a number of the clock's units, or of rows where the spec has
        no clock, or a duration such as ``"30m"`` on a temporal clock;
        finite and above 0. How many rows that holds is the data's, so its
        bytes are checked as the rows go in (see ``max_state_bytes``).
        Refused beside ``window_rows``. Default none.
    ``closed``
        Which edge of ``window_size`` holds the row exactly that old, in
        Polars' words: ``"right"``, the default, drops it; ``"both"`` keeps
        it, and is refused without ``window_size``.
    ``max_state_bytes``
        The bytes a group's state may hold, a window's rows included. A
        ``window_size`` window is checked as the rows go in: the first row
        to take a group past half of it raises a warning, once for the
        group, and the row that would take it past the cap stops the run
        with an error naming the group and the bytes. Default 256 KB,
        ``262144``; at most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row, each of the returns before the row and null below
    ``min_weight``:

    ``<name>_<t>``
        Per term and target: the term's sum, ``rv_<t>`` for a term named
        ``rv`` and ``lrv_<t>`` for one named ``lrv``.
    ``n_<t>``
        Per target: the count of returns ``n`` the terms are of: Kish's
        effective count ``(Σ w)² / Σ w²`` of the returns' weights, decayed
        or in the window, which is the count itself at equal weights.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay, every row's, or the weight inside the window.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take, or a term that is not a
    dict; and ``ValueError`` naming the problem for a term with neither or
    both of ``powers`` and ``kernel``, an option the other kind takes, a
    kernel without a bandwidth, a power not above 0, a scale of 0, two terms
    of one name or a term named ``n``, a window beside a decay, a window
    past ``max_state_bytes``, and the window checks every model makes.

    .. rubric:: The update

    For the returns ``r_j`` of a target and ``R_j = r_{j-K+1} + ... + r_j``
    its ``K``-step returns (``R_j = r_j`` at ``K = 1``):

    .. code-block:: text

        powers p_1..p_m:   (1/K) Σ_j Π_i |R_{j-(m-i)K}|^{p_i}
        kernel k, H:       (1/K) Σ_j [ R_j² + 2 Σ_{h=1}^{H} c_h R_j R_{j-hK} ]

    Each sum is over the ``j`` where every return of the term exists, so
    a window of ``W`` returns holds ``W - m + 1`` products of ``m`` powers.
    Online, each sum decays by ``0.5 ** (Δclock / half_life)`` a row and
    then takes the row's terms at the weight of the latest row in each. A
    centered kernel term keeps the sums of ``c R_j``, ``c R_{j-hK}`` and
    ``c`` beside its products and the mean ``m`` of the ``R_j``, and reads
    ``S - m (A + B) + m² C``: exactly the sum about the mean as it stands.
    No small-sample factor such as ``n / (n - h)`` is applied. The sums are
    compensated pairs, and a ``K``-step return is summed compensated and
    rounded once.

    .. rubric:: Example

    This code makes a day of returns with a bid-ask bounce, the noise that
    biases realized variance, and prints the realized variance beside the
    flat-top Parzen kernel at a bandwidth of 10, which takes the bounce out:
    the day's variance without the noise is ``2340 * 0.0002² = 9.4e-05``,
    and the noise adds ``2 * 2340 * 0.0004²`` to the realized variance:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        noise = 0.0004 * rng.standard_normal(2_341)
        r = 0.0002 * rng.standard_normal(2_340) + np.diff(noise)
        pl.from_dict({"r": r}).write_parquet("returns.parquet")

        model = rp.spec.power_variation(
            "pv",
            targets=["r"],
            terms={
                "rv": {"powers": [2]},
                "rk": {"kernel": "parzen", "bandwidth": 10, "flat_top": True},
            },
            half_life=float("inf"),
        )
        bank = rp.ModelBank([model])
        out = bank.fit_predict(pl.read_parquet("returns.parquet")).unnest("pv")
        print(f"rv {out['rv_r'][-1]:.2e}, kernel {out['rk_r'][-1]:.2e}")

    .. code-block:: text

        rv 8.05e-04, kernel 9.20e-05

    .. rubric:: See also

    - :func:`realized`, for realized variance, bipower and tripower
      variation and the jump tests built on them.
    - :func:`rcov`, for a block's realized covariance robust to
      microstructure noise.

    """
    if not isinstance(terms, Mapping):
        raise TypeError(
            "power_variation: terms must be a dict of each term's name and its options, "
            f'such as {{"rv": {{"powers": [2]}}}}, but it is {type(terms).__name__}. Pass '
            "terms as a dict."
        )
    model: dict[str, Any] = {
        "type": "power_variation",
        "terms": [_term(str(k), v) for k, v in terms.items()],
    }
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if window_rows is not None:
        model["window_rows"] = window_rows
    if window_size is not None:
        model["window_size"] = window_size
    if closed != "right":
        model["closed"] = closed
    return _common(name, model, targets=targets, features=[], **common)
