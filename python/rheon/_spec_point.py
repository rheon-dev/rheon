"""The point process's builder, :func:`hawkes`: a part of
:mod:`rheon._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`rheon.spec` documents it with the rest.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common
from rheon._spec_words import quoted
from rheon._warnings import RheonWarning, _user_stacklevel

#: A group's state may hold this many bytes unless ``max_state_bytes`` says
#: otherwise: 256 KB.
MAX_STATE_BYTES = 256 * 1024


def _state_bytes(d: int, u: int, window_events: int | None) -> int:
    """The bytes a group's state holds at most, as the model counts them:
    its running sums, and a full window's rows with its sum, clock and
    weight."""
    k = d * u
    sums = k + d * (1 + k) + d * (d + 1) // 2 + 2
    window = 0 if window_events is None else window_events * (2 + d + k) + k + 2
    return 8 * (sums + window)


@_checked
def hawkes(
    name: str,
    *,
    targets: TargetList,
    kernel_decays: list[float] | None = None,
    kernel_half_lives: list[float | Duration] | None = None,
    window_events: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A self-exciting point process, fitted exactly by least squares.

    Use it on a stream of events, where each event makes the next likelier
    for a while: trades, messages, failures, posts. Each target is a node,
    and a value above 0 on a row is an event of that node, of that size (its
    mark): 1 for one event, a count for that many at once.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`rheon.spec`. A row's ``weight`` only decides whether its events
    are learned: 0 skips them, and any weight above 0 learns them at their
    marks, since an event's excitation is a fact of the process rather than
    evidence to scale. A row of weight 0 counts in no clock: the next row
    steps as if it were not there, it is in no window, and its weight is in
    no ``weight_sum``; it is still read, from the last row that counted.
    ``half_life`` is how fast the fit forgets, ``inf`` for not at all.

    This model's own parameters:

    ``kernel_decays``, ``kernel_half_lives``
        The kernels, each node exciting every node through one kernel ``α β
        exp(-β t)`` per decay ``β``, which integrates to ``α``, so ``1/β`` is
        how long an event's excitation lasts. Two or three decays a decade
        apart cover a short and a long memory. Give one of the two. The
        decays ``β`` are rates per unit of the clock, each above 0 and no two
        equal; a rate has no duration form, so a temporal clock is given
        ``kernel_half_lives`` instead: each kernel's half-life ``ln 2 / β``
        as a number of the clock's units or a duration such as ``"500ms"``,
        as ``half_life`` is, and ``β = ln 2 / half-life`` in the clock's
        unit, seconds on a temporal clock. Required: one of them.
    ``window_events``
        Fit only the last this many rows that carried an event, observed
        from the clock of the row dropped last, in place of a decay: set
        ``half_life`` to ``inf`` beside it. The window keeps each of its
        rows, ``8 * (2 + D + D * U)`` bytes each for ``D`` targets and
        ``U`` decays. Default none.
    ``window_size``
        Fit only the events of the last this long on the clock, observed
        over those same ``window_size`` units (or from the first row, until
        there are that many), in place of a decay: set ``half_life`` to
        ``inf`` beside it. A number of the clock's units, or of rows where
        the spec has no clock, or a duration such as ``"5m"`` on a temporal
        clock; finite and above 0. A row leaves the window when it is
        ``window_size`` old on the clock (decayed after ``gap_cap``),
        before the row that reaches that age reads its intensity, decided
        on the rows' exact stamps (integer nanoseconds on a temporal
        clock). It is the window's span on the clock where ``window_events``
        is its count of events: how many events it holds is the data's, so
        its bytes are checked as the rows go in (see ``max_state_bytes``).
        Refused beside ``window_events``. Default none.
    ``closed``
        Which edge of ``window_size`` holds the row exactly that old, in
        Polars' words (``rolling_sum_by(closed=)``). ``"right"``, the
        default, keeps the rows less than ``window_size`` old, so the
        window is ``(T - window_size, T]``, as the window operators and
        ``rolling_sum_by`` do. ``"both"`` keeps that row too and is refused
        without ``window_size``. ``"left"`` and ``"none"`` would leave out
        the row the window ends at, which a model cannot do: both are
        refused.
    ``max_state_bytes``
        The bytes a group's state may hold, a window's rows included.
        ``window_events`` fixes the size, so it is checked here: a spec past
        it is refused, and one past half of it is warned of
        (:class:`rheon.RheonWarning`). A ``window_size`` window holds the
        events the span has, so it is checked as the rows go in: the first
        row to take a group past half of it raises a warning, once for the
        group, naming the group and the bytes, and the row that would take
        it past the cap stops the run with an error naming the group and the
        bytes, the bank having learned part of the chunk. Each row of the
        window takes ``8 * (2 + D + D * U)`` bytes and a stamp of 32 beside
        it. Default 256 KB, ``262144``; at most 64 MiB.

    .. rubric:: Output

    A value of 0 or below, or a null, is no event, so a row can be a clock
    tick alone. Every row reports each node's intensity, its expected events
    per unit of the clock, read before the row's events from the fit on every
    row before them. On a temporal clock a quantity measured on the clock
    reaches the output in seconds (as ``holt``'s trend does): the intensity
    and the base rate ``mu`` are events per second, and ``alpha``, a count
    of events, has no unit. One output column, named after the spec, holds a record
    of these fields in each row. `docs/OUTPUTS.md#hawkes
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#hawkes>`_
    gives each field's type and when it is null:

    ``intensity_<t>``
        Per target: the node's intensity just before the row's events, in
        events per unit of the clock (per second on a temporal clock), from
        the fit on the rows before. Null
        below ``min_weight``, and while the fit is not determined: before
        every node has had an event, or while two of its terms cannot be
        told apart, as when two nodes' events fall on the same rows.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay: every row's weight, an event or not. Under a
        window, the weight inside it: that of the rows with an event the
        window holds, as it stood after the last row and before this row
        takes any out.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why the row's intensities were withheld, null where they were not:
        a reason of the shared gates (:mod:`rheon.spec`), such as
        ``below_min_weight``. A fit not yet determined (``intensity_<t>``
        null with ``weight_sum`` past ``min_weight``) gives none.
    ``coef``
        The fit after the row, per target ``i``: ``mu``, the base rate (per
        unit of the clock, per second on a temporal clock), then
        ``alpha_<j>_beta<β>`` for each source target ``j`` and decay ``β``
        (``alpha_<j>_h<half-life>`` where the kernels are given by
        ``kernel_half_lives``, the half-life as written, ``h500ms``), the
        events of ``i`` one event of ``j`` causes through that kernel.
        :func:`coef_index` names them. The fit is not held to 0 or above: a
        negative ``alpha`` is an event that inhibits.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError``
    naming the problem for neither ``kernel_decays`` nor
    ``kernel_half_lives`` given, or both; ``kernel_decays`` empty, not above
    0 or with a decay twice; a ``kernel_half_lives`` entry not finite or not
    above 0; ``window_events`` below 1 or beside a finite
    ``half_life``, ``window_size`` not finite or not above 0, beside a
    finite ``half_life`` or ``window_events``, ``closed``
    other than ``"right"`` or ``"both"``, or ``"both"`` without
    ``window_size``, a ``half_life`` whose rate ``ln 2 / half_life`` is
    within 0.1% of a decay or of a sum of two, or past the largest float,
    ``kernel_decays`` two of which sum or multiply past it, and a
    ``max_state_bytes`` of 0 or past 64 MiB.

    .. rubric:: The update

    With ``λ_i(t) = μ_i + Σ_j Σ_u α_iju R_ju(t)`` and the excitation
    ``R_ju(t) = Σ β_u m_k exp(-β_u (t - t_k))`` over node ``j``'s earlier
    events of marks ``m_k``, the fit minimises the least-squares contrast
    ``Σ_i ∫ w λ_i² dt - 2 ∫ w λ_i dN_i``, each instant weighted ``w = exp(-γ
    (T - t))`` at the forgetting rate ``γ = ln 2 / half_life``. With
    ``φ = (1, R)``, it is ``θ_i = G⁻¹ b_i``. On a row ``d`` clock units
    after the last, ``f = exp(-γ d)``:

    .. code-block:: text

        R_ju ← exp(-β_u d) R_ju      b ← f b      Q ← f Q
        G₀₀  ← f G₀₀ + (1 - f) / γ                          (+ d where γ = 0)
        then, for each node j with an event of mark m_j, from R before the row's events:
        b_j  ← b_j + m_j φ        Q_jl ← Q_jl + m_j m_l        R_ju ← R_ju + m_j β_u

    and at each read, with ``j(a)`` the source node of term ``a``:

    .. code-block:: text

        G₀ₐ = (β_a b_j(a)[0] - R_a) / (β_a - γ)
        Gₐᵦ = (β_a b_j(a)[b] + β_b b_j(b)[a] + β_a β_b Q_j(a)j(b) - R_a R_b) / (β_a + β_b - γ)

    The state is ``R``, ``b``, ``Q``, ``G₀₀`` and the weight, ``8 * (k + D
    * (1 + k) + D * (D + 1) / 2 + 2)`` bytes for ``k = D * U``, whatever the
    stream's length.

    .. rubric:: Example

    This code simulates one node from a base rate of 0.5 and a kernel of
    ``α = 0.6`` at ``β = 2``, by Ogata's thinning, and prints the fit:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        t, excite, times = 0.0, 0.0, []
        while t < 5_000.0:
            bound = 0.5 + excite
            step = rng.exponential(1 / bound)
            excite *= np.exp(-2.0 * step)
            t += step
            if rng.uniform() * bound <= 0.5 + excite:
                times.append(t)
                excite += 0.6 * 2.0
        events = pl.DataFrame({"t": times, "event": 1.0})
        events.write_parquet("events.parquet")

        model = rp.spec.hawkes(
            "h", targets=["event"], kernel_decays=[2.0], clock="t", half_life=float("inf"),
            gap_cap=1e9,
        )
        bank = rp.ModelBank([model])
        bank.fit(pl.scan_parquet("events.parquet"))
        coef = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
        print(f"mu: true 0.50, found {coef['mu']:.2f}")
        print(f"alpha: true 0.60, found {coef['alpha_event_beta2']:.2f}")

    .. code-block:: text

        mu: true 0.50, found 0.51
        alpha: true 0.60, found 0.59

    .. rubric:: See also

    - :func:`seqtest`, to test whether one spec's events come faster than
      another's.
    - :func:`ew_cov`, for the correlation of counts in fixed bins rather
      than the excitation between events.

    .. rubric:: Notes

    **Why the fit is exact.** The intensity is linear in its parameters,
    so the contrast is a quadratic form in them. ``G`` is not kept: between
    events ``R_a R_b`` decays at ``β_a + β_b``, and at a row of events it
    jumps by terms whose sum over the rows is ``b`` and ``Q``, so
    integrating it by parts reads ``G`` from them. tick's
    ``ModelHawkesSumExpKernLeastSq`` computes the same contrast, divided by
    the number of events, and its minimiser agrees with this fit to about
    ``1e-12``.

    **Events on one row.** An event excites every event on a later row,
    at the same clock or not, and none on its own row: those are
    simultaneous, as in the definition ``t_k < t``.

    **The window.** Dropping the oldest row takes back exactly what it
    added, through the sums of its excitation of the later rows. Those are
    kept in two blocks of rows, so that no sum is ever rescaled by a growing
    exponential, and ``R``, ``b`` and ``Q`` are formed afresh from the rows
    each time a block turns over, so their rounding is that of one window.
    ``window_events`` drops the oldest row once the window holds more rows
    than it; ``window_size`` drops every row ``window_size`` old at the row
    it reads, and observes the interval ``(T - window_size, T]``, so ``G₀₀``
    is ``min(window_size, the clock since the first row)``. Its edge is
    decided on each row's exact stamp, never a clock summed in doubles.
    Either way, the fit is that of the window's events alone.
    """
    # `kernel_decays` is always written, null beside `kernel_half_lives`, as
    # the Rust spec writes it.
    model: dict[str, Any] = {"type": "hawkes", "kernel_decays": kernel_decays}
    if kernel_half_lives is not None:
        model["kernel_half_lives"] = kernel_half_lives
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if window_events is not None:
        model["window_events"] = window_events
    if window_size is not None:
        model["window_size"] = window_size
    if closed != "right":
        model["closed"] = closed
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    # The spec's own checks first, so a spec refused for another reason is
    # not warned of first (review 2026-11-23, PY-4).
    spec = _common(name, model, targets=targets, features=[], **common)
    cap = MAX_STATE_BYTES if max_state_bytes is None else max_state_bytes
    kernels = kernel_decays if kernel_decays is not None else kernel_half_lives
    if isinstance(kernels, list) and isinstance(window_events, int) and cap > 0:
        need = _state_bytes(len(targets), len(kernels), window_events)
        if cap // 2 < need <= cap:
            warnings.warn(
                RheonWarning(
                    f"spec {quoted(name)}: hawkes will hold {need} bytes in each group, past "
                    f"half of max_state_bytes ({cap}): its window keeps {window_events} rows. "
                    "Lower window_events, or raise max_state_bytes to keep this warning quiet."
                ),
                stacklevel=_user_stacklevel(),
            )
    return spec
