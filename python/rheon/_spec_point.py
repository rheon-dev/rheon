"""The point process's builder, :func:`hawkes`: a part of
:mod:`rheon._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`rheon.spec` documents it with the rest.
"""

from __future__ import annotations

import warnings
from typing import Any, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common
from rheon._warnings import RheonWarning, _user_stacklevel

#: A group's state may hold this many bytes unless ``max_state_bytes`` says
#: otherwise: 256 KB.
MAX_STATE_BYTES = 256 * 1024


def _state_bytes(d: int, u: int, window_events: int | None) -> int:
    """The bytes a group's state holds at most, as the model counts them:
    its running sums, and a full window's rows with its sum and clock."""
    k = d * u
    sums = k + d * (1 + k) + d * (d + 1) // 2 + 2
    window = 0 if window_events is None else window_events * (1 + d + k) + k + 1
    return 8 * (sums + window)


@_checked
def hawkes(
    name: str,
    *,
    targets: TargetList,
    kernel_decays: list[float],
    window_events: int | None = None,
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A self-exciting point process (Hawkes), fitted exactly by least squares.

    Use it on a stream of events, where each event raises the chance of the
    next for a while: trades, messages, failures, posts. Each target is a
    node, and a value above 0 on a row is an event of that node, of that
    size (its mark): 1 for one event, a count for that many at once. A
    value of 0 or below, or a null, is no event, so a row can be a clock
    tick alone. Every row reports each node's intensity, its expected
    events per unit of the clock, read before the row's events from the fit
    on every row before them.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`rheon.spec`. A row's ``weight`` only decides whether its events
    are learned: 0 skips them, and any weight above 0 learns them at their
    marks, since an event's excitation is a fact of the process rather than
    evidence to scale. ``half_life`` is how fast the fit forgets, ``inf``
    for not at all.

    This model's own parameters:

    ``kernel_decays``
        The decay ``β`` of each exponential kernel, per unit of the clock,
        each above 0 and no two equal. Each node excites every node through
        one kernel ``α β exp(-β t)`` per decay, which integrates to ``α``,
        so ``1/β`` is how long an event's excitation lasts. Two or three
        decays a decade apart cover a short and a long memory. Required.
        On a temporal clock it is refused, since a rate has no duration
        form: give the clock as a number of seconds instead.
    ``window_events``
        Fit only the last this many rows that carried an event, observed
        from the clock of the row dropped last, in place of a decay: set
        ``half_life`` to ``inf`` beside it. The window keeps each of its
        rows, ``8 * (1 + D + D * U)`` bytes each for ``D`` targets and
        ``U`` decays. Default none.
    ``max_state_bytes``
        The bytes a group's state may hold, a full window's rows included.
        The settings fix the size, so it is checked here: a spec past it is
        refused, and one past half of it is warned of
        (:class:`rheon.RheonWarning`). Default 256 KB, ``262144``.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#hawkes
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#hawkes>`_
    gives each field's type and when it is null:

    ``intensity_<t>``
        Per target: the node's intensity just before the row's events, in
        events per unit of the clock, from the fit on the rows before. Null
        below ``min_weight``, and while the fit is not determined: before
        every node has had an event, or while two of its terms cannot be
        told apart, as when two nodes' events fall on the same rows.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay: every row's weight, an event or not.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why the row's intensities were withheld, null where they were not:
        a reason of the shared gates (:mod:`rheon.spec`), such as
        ``below_min_weight``. A fit not yet determined (``intensity_<t>``
        null with ``weight_sum`` past ``min_weight``) gives none.
    ``coef``
        The fit after the row, per target ``i``: ``mu``, the base rate,
        then ``alpha_<j>_beta<β>`` for each source target ``j`` and decay
        ``β``, the events of ``i`` one event of ``j`` causes through that
        kernel. :func:`coef_index` names them. The fit is not held to 0 or
        above: a negative ``alpha`` is an event that inhibits.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError``
    naming the problem for ``kernel_decays`` empty, not above 0 or with a
    decay twice, ``window_events`` below 1 or beside a finite
    ``half_life``, and a ``half_life`` whose rate ``ln 2 / half_life`` is
    within 0.1% of a decay or of a sum of two.

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
    """
    model: dict[str, Any] = {"type": "hawkes", "kernel_decays": kernel_decays}
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if window_events is not None:
        model["window_events"] = window_events
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    cap = MAX_STATE_BYTES if max_state_bytes is None else max_state_bytes
    if isinstance(kernel_decays, list) and isinstance(window_events, int) and cap > 0:
        need = _state_bytes(len(targets), len(kernel_decays), window_events)
        if cap // 2 < need <= cap:
            warnings.warn(
                RheonWarning(
                    f"spec {name!r}: hawkes will hold {need} bytes in each group, past half of "
                    f"max_state_bytes ({cap}): its window keeps {window_events} rows. Lower "
                    "window_events, or raise max_state_bytes to keep this warning quiet."
                ),
                stacklevel=_user_stacklevel(),
            )
    return _common(name, model, targets=targets, features=[], **common)
