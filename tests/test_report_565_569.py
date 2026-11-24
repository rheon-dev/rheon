"""Five reports from a user's 27M-row transit run of 2.0.0 (2026-11-22),
each held by a test before any fix: PLAN 565 to 569.

A confirmed bug is held by a strict ``xfail`` that states the expected
behaviour and its source, so the fix flips it to a pass (and the strict mark
then fails until it is removed). A report that is the documented behaviour is
held by a plain test that pins it, naming the doc that says so.

- PLAN 565: ``reverse=True`` window edges. Which rows a window holds is the
  report's table, and Polars' ``rolling_*_by`` on the negated clock agrees
  (plain test). ``rp.ewm_mean`` is a time-weighted mean, so its far-edge row
  weighs nothing (documented, plain test). Confirmed: a window looking ahead
  under ``"right"`` or ``"both"`` whose far edge is the input's last row is
  whole at the end of the input, as the same window before a gap is (fixed).
- PLAN 566: ``ModelBank.skip_learned`` drops a row one spec ties on and
  another spec has provably not learned (confirmed); a row at the saved clock
  is dropped as learned, the predict copy of ``rp.stream.embargo`` included
  (documented, plain test).
- PLAN 567: ``with_windows(save_state=...)`` holds the newest stamp's backward
  windows, though a resumed input may not start at that stamp (confirmed).
- PLAN 568: a null target counts in ``weight_sum`` but not in its own
  ``min_weight`` gate (documented, plain test).
- PLAN 569: ``rp.spec.ftrl`` has no ``standardize`` (documented, plain test).
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

# --------------------------------------------------------------------------
# PLAN 565: rp.ewm_*(reverse=True) window edges

T0 = dt.datetime(2026, 1, 1, 8, 0)
STEP_MIN = 5
#: The report's grid: t = 0, 5, ..., 25 minutes, x = 1..6.
GRID = pl.DataFrame(
    {
        "t": [T0 + dt.timedelta(minutes=STEP_MIN * i) for i in range(6)],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
)
MINUTES = [STEP_MIN * i for i in range(6)]
#: A forward window on clock t is a backward one on the negated clock -t,
#: with its two ends swapped: (t, t+w] is [s-w, s) for s = -t.
POLARS_CLOSED = {"right": "left", "left": "right", "both": "both", "none": "none"}


def _in(closed: str, t: int, u: int, w: int) -> bool:
    """Whether a row at ``u`` is in the forward window of the row at ``t``:
    ``(t, t+w]``, ``[t, t+w]``, ``[t, t+w)`` or ``(t, t+w)``."""
    near = u >= t if closed in ("left", "both") else u > t
    far = u <= t + w if closed in ("right", "both") else u < t + w
    return near and far


def _sum_by_definition(closed: str, w: int) -> list[float]:
    return [
        sum(x for u, x in zip(MINUTES, GRID["x"], strict=True) if _in(closed, t, u, w))
        for t in MINUTES
    ]


def _sum_by_polars(closed: str, w: int) -> list[float]:
    """Polars' ``rolling_sum_by`` over the frame reversed in time: the clock
    negated (in integer minutes, so every edge is exact), ``closed``
    translated by :data:`POLARS_CLOSED`."""
    mirrored = GRID.with_columns(s=-pl.Series(MINUTES)).reverse()
    out = mirrored.select(
        pl.col("x").rolling_sum_by("s", window_size=f"{w}i", closed=POLARS_CLOSED[closed])  # type: ignore[arg-type]
    )["x"]
    return out.reverse().to_list()


def _time_weighted_mean(closed: str, t: int, w: int) -> float | None:
    """``rp.ewm_mean`` looking ahead, from WINDOWS.md's definition: a row is
    in the window by its stamp, and each value is held from its row until the
    next row, so it weighs the part of ``[u, u_next)`` inside ``[t, t+w]``.
    Even weights (``half_life=inf``)."""
    num = den = 0.0
    for i, (u, x) in enumerate(zip(MINUTES, GRID["x"], strict=True)):
        if not _in(closed, t, u, w) or i + 1 == len(MINUTES):
            continue
        held = max(0, min(MINUTES[i + 1], t + w) - max(u, t))
        num += held * x
        den += held
    return num / den if den > 0 else None


def _windows(op: str, closed: str, w: int) -> list[float | None]:
    kw: dict[str, Any] = dict(
        half_life=float("inf"), window_size=f"{w}m", closed=closed, partial="keep", reverse=True
    )
    expr = getattr(rp, op)("x", **kw)
    return rp.stream.with_windows(GRID, v=expr, clock="t", gap_cap="1h")["v"].to_list()


@pytest.mark.parametrize("w", [5, 15])
@pytest.mark.parametrize("closed", ["right", "both", "left", "none"])
def test_reverse_window_membership_is_the_reports_table_and_polars(closed: str, w: int) -> None:
    """PLAN 565, the part of the report that holds: a window looking ahead
    holds ``(t, t+w]``, ``[t, t+w]``, ``[t, t+w)`` or ``(t, t+w)`` under
    ``right``, ``both``, ``left`` and ``none`` (WINDOWS.md, "Which rows a
    window holds"), the report's table. Polars' ``rolling_sum_by`` on the
    negated clock, ``closed`` translated right<->left, agrees. Checked on
    every row whose window has passed before the input's last row."""
    got = _windows("ewm_sum", closed, w)
    want = _sum_by_definition(closed, w)
    polars = _sum_by_polars(closed, w)
    passed = [t + w < MINUTES[-1] for t in MINUTES]
    assert sum(passed) >= 2
    for i, ok in enumerate(passed):
        if ok:
            assert want[i] == polars[i], (closed, w, i)
            assert got[i] == want[i], (closed, w, i, got)


@pytest.mark.parametrize("w", [5, 15])
@pytest.mark.parametrize("closed", ["right", "both", "left", "none"])
def test_reverse_mean_is_time_weighted_so_its_far_edge_row_weighs_nothing(
    closed: str, w: int
) -> None:
    """PLAN 565, the documented part: ``rp.ewm_mean`` is a time-weighted
    mean, and looking ahead "each value is held until the next row"
    (WINDOWS.md, "A row's weight on an uneven clock"; "A row is in a window by
    its stamp, but a mean weighs it by the part of its held interval inside
    the window"; ``tests/test_window_far_edge.py``). So the far-edge row at
    ``t+w`` holds nothing inside the window under any ``closed``, and under
    ``"right"`` or ``"none"`` the step ``(t, t+5m)`` is held by the row's own
    value, which those leave out: a one-step mean is empty, so null. Polars'
    ``rolling_mean_by``, a mean by count, is another quantity: the report's
    expected means are that one, which ``ewm_sum / ewm_sum`` of ones gives."""
    got = _windows("ewm_mean", closed, w)
    for i, t in enumerate(MINUTES):
        if t + w < MINUTES[-1]:
            want = _time_weighted_mean(closed, t, w)
            value = got[i]
            if want is None:
                assert value is None, (closed, w, i, got)
            else:
                assert value is not None and math.isclose(value, want), (closed, w, i, got)
    if w == STEP_MIN and closed in ("right", "none"):
        assert all(v is None for v in got)


#: Each operator, window and ``closed`` whose far-edge window has a value
#: when a later row follows: a one-step ``"right"`` mean holds nothing
#: (the test above), so it is left out.
FAR_EDGE_CASES = [
    (closed, w, op)
    for closed in ("right", "both")
    for w in (5, 15)
    for op in ("ewm_sum", "ewm_mean", "ewm_rate", "ewm_quantile")
    if not (op == "ewm_mean" and closed == "right" and w == STEP_MIN)
]


@pytest.mark.parametrize(("closed", "w", "op"), FAR_EDGE_CASES)
def test_a_reverse_window_whose_far_edge_is_the_last_row_is_whole_at_the_end_of_the_input(
    closed: str, w: int, op: str
) -> None:
    """PLAN 565, fixed. Expected: at the end of the input, a window
    looking ahead under ``"right"`` or ``"both"`` whose far edge ``t+w`` is
    the input's last row has a value, the one it has when a row past ``t+w``
    follows. Source: WINDOWS.md, "A window that looks ahead under
    ``closed="right"`` or ``"both"`` gives its value under any ``partial``
    when its far edge falls on the last row before a gap or a reset", since
    no later row can join it; at the end of the input none can either (no
    state is saved), and Polars' ``rolling_*_by`` on the negated clock gives
    it. ``windows.rs``'s ``finish`` nulls every open forward window, where
    ``end_in`` applies the whole-window rule at a cut."""
    kw: dict[str, Any] = dict(
        half_life=float("inf"), window_size=f"{w}m", closed=closed, partial="keep", reverse=True
    )
    if op == "ewm_quantile":
        kw["quantile"] = 0.5
    expr = getattr(rp, op)("x", **kw)
    edge = MINUTES.index(MINUTES[-1] - w)
    # The same rows with one more row past the far edge: the window's value.
    longer = pl.concat(
        [GRID, pl.DataFrame({"t": [T0 + dt.timedelta(minutes=MINUTES[-1] + 60)], "x": [0.0]})]
    )
    whole = rp.stream.with_windows(longer, v=expr, clock="t", gap_cap="2h")["v"][edge]
    assert whole is not None
    if op == "ewm_sum":
        assert whole == _sum_by_polars(closed, w)[edge]
    got = rp.stream.with_windows(GRID, v=expr, clock="t", gap_cap="2h")["v"][edge]
    assert got == whole, (closed, w, op, got, whole)


def _reverse_expr(op: str, closed: str, w: int) -> pl.Expr:
    kw: dict[str, Any] = dict(
        half_life=float("inf"), window_size=f"{w}m", closed=closed, partial="keep", reverse=True
    )
    if op == "ewm_quantile":
        kw["quantile"] = 0.5
    return getattr(rp, op)("x", **kw)


#: Two groups on one clock, in clock order across them: ``a`` ends at 25
#: minutes and ``b`` at 15, so a window's far edge is its own group's last
#: stamp for ``a`` at 15 and ``b`` at 5 (w = 10).
GROUPED = pl.DataFrame(
    {
        "t": [T0 + dt.timedelta(minutes=m) for m in (0, 0, 5, 5, 10, 10, 15, 15, 20, 25)],
        "g": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "a"],
        "x": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 6.0],
    }
)


@pytest.mark.parametrize("op", ["ewm_sum", "ewm_mean", "ewm_rate", "ewm_quantile"])
@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("rows", [1, 3, 10_000])
def test_a_reverse_window_is_whole_at_the_end_of_the_input_only_up_to_the_last_row(
    closed: str, op: str, grouped: bool, rows: int
) -> None:
    """PLAN 565. Both sides of the edge, under every ``closed``, with and
    without ``group``, at any chunk size. Each row's window is read at the
    end of the input and again with a row appended per group far past every
    window (so each window is whole by the rows it holds): a window whose far
    edge is at or before its group's last stamp has the appended run's value,
    one reaching past the last stamp is null, as documented. Under ``"left"``
    and ``"none"`` the far edge holds no row, so a row on it closed the window
    already."""
    w = 10
    frame = GROUPED if grouped else GRID
    names: dict[str, Any] = {"group": "g"} if grouped else {}
    last = {
        key: max(sub["t"])
        for key, sub in (
            frame.partition_by("g", as_dict=True) if grouped else {(None,): frame}
        ).items()
    }
    far = T0 + dt.timedelta(hours=5)
    extra = pl.DataFrame(
        {"t": [far] * len(last), "x": [0.0] * len(last)}
        | ({"g": [k[0] for k in last]} if grouped else {})
    ).select(frame.columns)
    expr = _reverse_expr(op, closed, w)
    kw: dict[str, Any] = dict(clock="t", gap_cap="10h", **names)
    # Window values by the rows that follow, from the longer input.
    whole = rp.stream.with_windows(pl.concat([frame, extra]), v=expr, chunk_size=rows, **kw)["v"]
    got = rp.stream.with_windows(frame, v=expr, chunk_size=rows, **kw)["v"].to_list()
    one = rp.stream.with_windows(frame, v=expr, chunk_size=10_000, **kw)["v"].to_list()
    assert got == one, "chunk invariance"
    saw_edge = saw_past = False
    for i in range(frame.height):
        key = (frame["g"][i],) if grouped else (None,)
        edge = frame["t"][i] + dt.timedelta(minutes=w)
        if edge > last[key]:
            saw_past = True
            assert got[i] is None, (i, got[i])
        else:
            saw_edge = saw_edge or edge == last[key]
            assert got[i] == whole[i], (i, got[i], whole[i])
    assert saw_edge and saw_past


# --------------------------------------------------------------------------
# PLAN 566: ModelBank.skip_learned

T1 = T0 + dt.timedelta(minutes=5)


def _ewridge(name: str, **kw: Any) -> dict[str, Any]:
    return rp.spec.ewridge(
        name, targets=["y"], features=["x"], clock="t", gap_cap="1h", half_life="1h", **kw
    )


def test_skip_learned_keeps_a_row_one_spec_ties_on_and_another_has_not_learned() -> None:
    """PLAN 566, fixed. Expected: ``skip_learned`` keeps a row that one
    spec has provably not learned -- its clock is after that spec's group's
    last clock -- when no spec would refuse it as a step back. Source: its
    docstring's first line, "The rows of a frame or query that the bank has
    not learned"; a row the bank learned was learned by every spec, so a
    clock after any spec's last clock proves the row new; and
    :meth:`fit_predict` accepts this row, a tie on ``l1_comp``'s clock being
    no step back. ``_bank.py`` ANDs the specs' verdicts (line 680) and counts
    a tie as learned (``value > last``, line 4214), so ``l1_comp``'s tie
    vetoes ``l1_seg``'s proof; the command line's ``learned.rs`` does the
    same. A row before some spec's last clock is still dropped
    (``test_a_row_is_kept_only_where_every_spec_has_not_learned_it``)."""
    specs = [_ewridge("l1_comp", group="comp"), _ewridge("l1_seg", group="seg")]
    bank = rp.ModelBank(specs)
    bank.fit_predict(
        pl.DataFrame(
            {
                "t": [T0, T0, T1],
                "comp": ["IRT", "IRT", "IRT"],
                "seg": ["1_N", "2_N", "1_N"],
                "x": [1.0, 2.0, 3.0],
                "y": [1.0, 2.0, 3.0],
            }
        )
    )
    row = pl.DataFrame({"t": [T1], "comp": ["IRT"], "seg": ["2_N"], "x": [4.0], "y": [4.0]})
    saved = bank.save_bytes()
    # l1_seg's group 2_N last learned at T0, so T1 is new to it.
    assert rp.ModelBank.load_bytes(saved, specs).fit_predict(row).height == 1
    assert bank.skip_learned(row).height == 1


def test_skip_learned_keeps_an_embargo_predict_row_at_the_saved_clock_and_drops_its_learn_row(
    tmp_path: Path,
) -> None:
    """PLAN 566, the documented part. ``skip_learned`` reads a row at its
    group's last clock as learned (its docstring, Returns: "A row at its
    group's last clock counts as learned"; Notes: "a stream that repeats a
    clock value and was saved between two rows with that value loses the
    later ones"). ``rp.stream.embargo`` puts a learn row and then a predict
    row of weight 0 at one clock, so a save between them loses the predict
    row. With the fix of the test above a row of weight 0 at the last clock
    is kept, which a resumed bank takes harmlessly, since such a row learns
    nothing and moves no clock (CLAUDE.md, rule 9); the learn row at that
    clock, of weight 1, is still dropped."""
    spec = _ewridge("m", weight="w")
    rows = pl.DataFrame(
        {"t": [T0, T1, T1], "x": [1.0, 2.0, 2.0], "y": [1.0, 2.0, 2.0], "w": [1.0, 1.0, 0.0]}
    )
    bank = rp.ModelBank([spec])
    bank.fit_predict(rows.head(2))
    bank.save(tmp_path / "b.state")
    resumed = rp.ModelBank.load(tmp_path / "b.state")
    assert resumed.skip_learned(rows.slice(2)).height == 1
    # The learn row at the clock, of weight 1, is learned.
    assert resumed.skip_learned(rows.slice(1)).height == 1
    # The bank itself takes the predict row: a tie is no step back.
    assert resumed.fit_predict(rows.slice(2)).height == 1


def _two_banks(**kw: Any) -> rp.ModelBank:
    """``l1_comp`` by ``comp`` and ``l1_seg`` by ``seg``, each learning ``T0``
    in its groups and ``T1`` too in ``IRT`` and ``1_N``."""
    bank = rp.ModelBank(
        [_ewridge("l1_comp", group="comp", **kw), _ewridge("l1_seg", group="seg", **kw)]
    )
    bank.fit_predict(
        pl.DataFrame(
            {
                "t": [T0, T0, T1],
                "comp": ["IRT", "IRT", "IRT"],
                "seg": ["1_N", "2_N", "1_N"],
                "x": [1.0, 2.0, 3.0],
                "y": [1.0, 2.0, 3.0],
                "w": [1.0, 1.0, 1.0],
            }
        )
    )
    return bank


def test_skip_learned_reads_each_spec_before_at_and_after_its_groups_last_clock() -> None:
    """PLAN 566, both sides of every edge. ``l1_comp``'s ``IRT`` last learned
    at ``T1`` and ``l1_seg``'s ``1_N`` at ``T1``, ``2_N`` at ``T0``. A row is
    dropped when some spec has it strictly before its last clock, kept when no
    spec does and one has it after, or in an unseen group, or on a null clock,
    and dropped when every spec with a view ties."""
    T2 = T1 + dt.timedelta(minutes=5)
    rows = pl.DataFrame(
        {
            "t": [T1, T1, T1, T0, T0, T2, T1, None, T1],
            "comp": ["IRT", "IRT", "IRT", "IRT", "IRT", "IRT", "NEW", "IRT", "IRT"],
            "seg": ["1_N", "2_N", "3_N", "2_N", "1_N", "1_N", "1_N", "1_N", None],
            "x": [0.0] * 9,
            "y": [0.0] * 9,
        }
    )
    keep = [
        False,  # ties in both
        True,  # tie in comp, after in seg (2_N is at T0)
        True,  # tie in comp, seg's 3_N unseen
        False,  # before in comp; seg's 2_N at T0 ties
        False,  # before in both
        True,  # after in both
        True,  # comp's NEW unseen, tie in seg
        True,  # a null clock is the bank's to judge
        True,  # tie in comp, seg's null group unseen
    ]
    got = _two_banks().skip_learned(rows)
    assert got["t"].to_list() == [t for t, k in zip(rows["t"], keep, strict=True) if k]
    assert got.equals(rows.filter(pl.Series(keep)))
    # The lazy form agrees.
    assert _two_banks().skip_learned(rows.lazy()).collect().equals(got)


def test_skip_learned_keeps_a_weight_zero_row_at_the_last_clock_unless_a_spec_learned_it() -> None:
    """PLAN 566. Weight 0 learns nothing and counts in no clock, so replaying
    such a row at a last clock is harmless, and it is kept; a spec that learned
    the row at a positive weight there has it, and it is dropped; a row of
    weight 0 before a last clock is dropped as any such row is."""
    both = _two_banks(weight="w")
    seg_only = rp.ModelBank(
        [_ewridge("l1_comp", group="comp", weight="w"), _ewridge("l1_seg", group="seg")]
    )
    seg_only.fit_predict(
        pl.DataFrame(
            {
                "t": [T0, T0, T1],
                "comp": ["IRT", "IRT", "IRT"],
                "seg": ["1_N", "2_N", "1_N"],
                "x": [1.0, 2.0, 3.0],
                "y": [1.0, 2.0, 3.0],
                "w": [1.0, 1.0, 1.0],
            }
        )
    )

    def rows(t: list[dt.datetime], w: list[float]) -> pl.DataFrame:
        n = len(t)
        return pl.DataFrame(
            {
                "t": t,
                "comp": ["IRT"] * n,
                "seg": ["1_N"] * n,
                "x": [0.0] * n,
                "y": [0.0] * n,
                "w": w,
            }
        )

    frame = rows([T1, T1, T0, T1], [0.0, 1.0, 0.0, 2.0])
    # Both specs read the weight: the zero-weight tie is kept, a positive one
    # is learned, and a zero-weight row before the last clock is dropped.
    assert both.skip_learned(frame)["w"].to_list() == [0.0]
    # l1_seg reads no weight, so it learned the row at weight 1 as every row.
    assert seg_only.skip_learned(frame).height == 0


# --------------------------------------------------------------------------
# PLAN 567: with_windows(save_state=...) holds the newest stamp

FIVE_MIN = pl.DataFrame(
    {
        "t": [T0 + dt.timedelta(minutes=5 * i) for i in range(10)],
        "x": [float(i) for i in range(10)],
    }
)


def test_an_unsliced_save_emits_the_newest_stamps_backward_windows(tmp_path: Path) -> None:
    """PLAN 567, fixed. Expected: a run with ``save_state`` and no slice
    returns every row whose windows look back only, the newest stamp's
    included, with the value one run over the whole stream gives it. Source:
    the hold exists because "a backward window under ``closed="right"``
    waits for the next distinct stamp" (``with_windows``' Notes), in case a
    later row at that stamp joins it; but every resumed run "refuses by name
    an input whose first row is at the last stamp the state read" (the same
    Notes; ``windows_frame.rs``, the ``Peek::SameStamp`` refusal), so without
    ``group`` no row can join, and the value is final at the save.
    ``stream.py`` calls ``finish`` only when no state is saved (line 1792).
    Today 5 rows give 4, and two runs of 5 give 9 of 10."""
    state = tmp_path / "w.state"
    f: dict[str, Any] = dict(m=rp.ewm_mean("x", half_life="10m", window_size="30m"))
    one = rp.stream.with_windows(FIVE_MIN, **f, clock="t", gap_cap="1h")
    first = rp.stream.with_windows(FIVE_MIN.head(5), **f, clock="t", gap_cap="1h", save_state=state)
    # The held row's value is already final: it comes out unchanged later.
    later = rp.stream.with_windows(
        FIVE_MIN.slice(5), **f, clock="t", gap_cap="1h", load_state=state
    )
    assert pl.concat([first, later]).equals(one)
    assert first.height == 5
    assert first.equals(one.head(5))


#: Rows on a 5-minute grid with ties at some stamps, in two groups; the tied
#: stamps are at rows (3, 4) and (8, 9, 10).
_STAMPS = [0, 5, 10, 15, 15, 20, 25, 30, 35, 35, 35, 40, 50, 55]
SPLIT_FRAME = pl.DataFrame(
    {
        "t": [T0 + dt.timedelta(minutes=m) for m in _STAMPS],
        "g": [("a", "b", "b", "a", "b")[i % 5] for i in range(len(_STAMPS))],
        "x": [float((i * 7) % 11) for i in range(len(_STAMPS))],
    }
)

_SPLIT_FORMS: dict[str, Any] = {
    "back_right": dict(window_size="20m", closed="right"),
    "back_left": dict(window_size="20m", closed="left"),
    "back_both": dict(window_size="15m", closed="both"),
    "back_none": dict(window_size="15m", closed="none"),
    "ahead_right": dict(window_size="15m", closed="right", reverse=True),
    "ahead_both": dict(window_size="10m", closed="both", reverse=True),
}


def _split_exprs() -> dict[str, pl.Expr]:
    out: dict[str, pl.Expr] = {}
    for name, kw in _SPLIT_FORMS.items():
        out[f"{name}_mean"] = rp.ewm_mean("x", half_life="12m", **kw)
        out[f"{name}_sum"] = rp.ewm_sum("x", half_life="12m", partial="keep", **kw)
        out[f"{name}_q"] = rp.ewm_quantile("x", quantile=0.5, half_life="12m", **kw)
    return out


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("rows", [1, 4, 10_000])
def test_a_save_at_every_split_point_continues_to_the_bit(
    tmp_path: Path, grouped: bool, rows: int
) -> None:
    """PLAN 567. One run equals any split into two saved runs, for every
    split point, every ``closed`` looking back and ahead, with and without
    ``group``, at three chunk sizes: the first run now emits the rows whose
    backward windows waited on its newest stamp, and the second goes on from a
    state that read no row after it. A split inside a tied stamp is refused by
    name, as it always was, whichever row of the stamp is first."""
    kw: dict[str, Any] = dict(clock="t", gap_cap="2h", chunk_size=rows)
    if grouped:
        kw["group"] = "g"
    exprs = _split_exprs()
    one = rp.stream.with_windows(SPLIT_FRAME, **exprs, **kw)
    n = SPLIT_FRAME.height
    ties = {i for i in range(1, n) if _STAMPS[i] == _STAMPS[i - 1]}
    state = tmp_path / "s.state"
    for k in range(1, n):
        first = rp.stream.with_windows(SPLIT_FRAME.head(k), **exprs, **kw, save_state=state)
        if k in ties:
            with pytest.raises(Exception, match="starts at the last clock value"):
                rp.stream.with_windows(SPLIT_FRAME.slice(k), **exprs, **kw, load_state=state)
            continue
        later = rp.stream.with_windows(SPLIT_FRAME.slice(k), **exprs, **kw, load_state=state)
        got = pl.concat([first, later])
        assert got.equals(one), (k, grouped, rows)


def test_backward_windows_alone_leave_whole_at_a_save(tmp_path: Path) -> None:
    """PLAN 567. Without a window looking ahead every row of the first run
    leaves with it, the newest stamp's included, under every ``closed``."""
    state = tmp_path / "w.state"
    for name in ("back_right", "back_left", "back_both", "back_none"):
        expr = rp.ewm_mean("x", half_life="12m", **_SPLIT_FORMS[name])
        first = rp.stream.with_windows(
            SPLIT_FRAME.head(3), v=expr, clock="t", gap_cap="2h", save_state=state
        )
        assert first.height == 3, name
        one = rp.stream.with_windows(SPLIT_FRAME, v=expr, clock="t", gap_cap="2h")
        assert first.equals(one.head(3)), name


def test_a_resumed_row_at_any_groups_last_stamp_is_refused(tmp_path: Path) -> None:
    """PLAN 567. With ``group``, a resumed input that starts after the stamp
    the state read but holds a row at it later still cannot be: the stream's
    clock refuses the row as a step back, so no row of any group joins the
    windows the save read. A first row at the stamp is refused by name."""
    state = tmp_path / "g.state"
    f: dict[str, Any] = dict(m=rp.ewm_mean("x", half_life="10m", window_size="30m"))
    kw: dict[str, Any] = dict(clock="t", gap_cap="2h", group="g")
    first = SPLIT_FRAME.head(5)  # ends at stamp 15, in groups a and b
    rp.stream.with_windows(first, **f, **kw, save_state=state)
    last = _STAMPS[4]
    for key in ("a", "b"):
        row = pl.DataFrame({"t": [T0 + dt.timedelta(minutes=last)], "g": [key], "x": [1.0]})
        with pytest.raises(Exception, match="starts at the last clock value"):
            rp.stream.with_windows(row, **f, **kw, load_state=state)
        later = pl.DataFrame(
            {"t": [T0 + dt.timedelta(minutes=last + 5)] * 2, "g": ["b", "a"], "x": [1.0, 2.0]}
        )
        # A row at the stamp after one past it is a step back.
        rows = pl.concat([later.head(1), row])
        with pytest.raises(Exception, match="steps back|step back"):
            rp.stream.with_windows(rows, **f, **kw, load_state=state)


# --------------------------------------------------------------------------
# PLAN 568: a null target and weight_sum


def test_a_null_target_counts_in_weight_sum_but_not_in_its_min_weight() -> None:
    """PLAN 568, the documented behaviour. A row whose target is null is
    learned for the shared weight, so ``weight_sum`` and ``summary()``'s
    ``weight_sum`` count it, while ``min_weight`` reads the target's own
    weight: CLAUDE.md rule 8 ("Each target's ``min_weight`` is checked against
    that target's own weight ...; the emitted ``weight_sum`` is the shared
    weight either way"), MODELS.md's Warm-up gates ("The ``weight_sum`` field
    is the shared weight either way"), and ``ewridge``'s ``target_gaps``
    ("Either way ``weight_sum`` counts every row"). So a single-target spec
    shows ``weight_sum`` at or above ``min_weight`` beside
    ``below_min_weight``. A weight of 0 on the same rows withholds the same
    rows, with a ``weight_sum`` that is the target's weight."""
    y = [1.0, None, 2.0, None, 3.0, None, 4.0, None]
    df = pl.DataFrame(
        {
            "t": [float(i) for i in range(8)],
            "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "y": y,
            "w": [0.0 if v is None else 1.0 for v in y],
        }
    )
    kw: dict[str, Any] = dict(
        targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=float("inf")
    )
    nulls = rp.ModelBank([rp.spec.ewridge("m", min_weight=4.0, **kw)])
    zeros = rp.ModelBank([rp.spec.ewridge("m", min_weight=4.0, weight="w", **kw)])
    a = nulls.fit_predict(df)["m"].struct.unnest()
    b = zeros.fit_predict(df)["m"].struct.unnest()
    assert a["weight_sum"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert b["weight_sum"].to_list() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    assert a["withheld_reason"].to_list() == b["withheld_reason"].to_list()
    assert a["withheld_reason"].to_list()[4] == "below_min_weight"
    assert nulls.summary()["weight_sum"].to_list() == [8.0]
    assert zeros.summary()["weight_sum"].to_list() == [4.0]


# --------------------------------------------------------------------------
# PLAN 569: standardize on the linear specs


@pytest.mark.parametrize("kind", ["ftrl", "rls", "lasso", "expectile", "glm"])
def test_the_specs_without_standardize_refuse_it_by_name(kind: str) -> None:
    """PLAN 569, the documented behaviour for ``ftrl``: MODELS.md's table of
    shared settings says "``ftrl`` never does" standardize, and its docstring
    says why, under *Features at a level*: ``l1`` zeroes a coefficient in the
    feature's own units, so a scale inside the model would move which
    coefficients it zeroes. ``lasso`` always standardizes. ``rls``,
    ``expectile`` and ``glm`` take no ``standardize`` either, which that
    table does not say. ``ewridge``, ``huber``, ``quantile``, ``kalman``,
    ``sgd`` and ``pa`` take it."""
    with pytest.raises(TypeError, match=f'{kind}\\(\\) takes no parameter "standardize"'):
        getattr(rp.spec, kind)("m", targets=["y"], features=["x"], standardize=True)
