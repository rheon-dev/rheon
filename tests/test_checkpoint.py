"""A pipeline checkpoint (PLAN 542): the states of a run's stages as one
file, written in full or not at all, each read back to load where it was
saved.

A run of three stages -- a ``refresh_time`` grid, a window run and a bank --
is cut in two; the first half's states go into one checkpoint, and the
second half resumed from it gives what one uninterrupted run gives, to the
bit. A refused or failed save leaves the old file; a file of another kind is
named for what it is.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

pytestmark = pytest.mark.filterwarnings("ignore::rheon.ReadinessWarning")


def ticks(n: int = 300, seed: int = 4) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "series": rng.choice(["a", "b"], n).tolist(),
            "t": np.cumsum(rng.uniform(0.1, 1.0, n)),
            "v": 100.0 + np.cumsum(0.1 * rng.standard_normal(n)),
        }
    )


GRID = {"series": "series", "names": ["a", "b"], "clock": "t", "value": "v"}
WINDOWS: dict[str, Any] = {"clock": "time_refresh", "gap_cap": 100.0}


def windows(lf: pl.DataFrame, **state: Any) -> pl.DataFrame:
    return rp.stream.with_windows(
        lf,
        level=rp.ewm_mean("a_value", half_life=5.0, window_size=20.0),
        ahead=rp.ewm_mean("b_value", half_life=5.0, window_size=10.0, reverse=True),
        **WINDOWS,
        **state,
    )


def spec() -> dict:
    return rp.spec.ewridge(
        "m",
        targets=["b_value"],
        features=["a_value", "level"],
        clock="time_refresh",
        gap_cap=100.0,
        half_life=40.0,
    )


def test_a_run_resumed_from_a_checkpoint_is_one_run(tmp_path):
    whole = ticks()
    grid_all = rp.stream.refresh_time(whole, **GRID)
    win_all = windows(grid_all)
    want = rp.ModelBank([spec()]).fit_predict(win_all)

    first, second = whole.slice(0, 150), whole.slice(150)
    grid_state, win_state = tmp_path / "grid.state", tmp_path / "windows.state"
    grid1 = rp.stream.refresh_time(first, **GRID, save_state=grid_state)
    win1 = windows(grid1, save_state=win_state)
    bank = rp.ModelBank([spec()])
    out1 = bank.fit_predict(win1)
    path = tmp_path / "run.checkpoint"
    rp.stream.save_checkpoint(
        path, {"grid": grid_state, "windows": win_state.read_bytes(), "bank": bank}
    )
    assert sorted(os.listdir(tmp_path)) == ["grid.state", "run.checkpoint", "windows.state"]

    # A new process's view: the checkpoint alone.
    stages = rp.stream.load_checkpoint(path)
    assert list(stages) == ["grid", "windows", "bank"]
    assert stages["grid"] == grid_state.read_bytes()
    assert stages["windows"] == win_state.read_bytes()
    assert stages["bank"] == bank.save_bytes()
    assert rp.stream.load_checkpoint(path.read_bytes()) == stages
    grid2 = rp.stream.refresh_time(second, **GRID, load_state=stages["grid"])
    win2 = windows(grid2, load_state=stages["windows"])
    out2 = rp.ModelBank.load_bytes(stages["bank"]).fit_predict(win2)
    # A call reports the coefficients on its last row, so the first half's
    # last row carries them where the one run's does not; every other field
    # of every row is the one run's, to the bit.
    got = pl.concat([out1, out2]).unnest("m").drop("coef", "support_coef")
    assert got.equals(want.unnest("m").drop("coef", "support_coef")), "not the one run"


def test_a_refused_save_leaves_the_old_checkpoint(tmp_path):
    bank = rp.ModelBank([spec()])
    path = tmp_path / "run.checkpoint"
    rp.stream.save_checkpoint(path, {"bank": bank})
    before = path.read_bytes()
    damaged = bytearray(bank.save_bytes())
    damaged[20] ^= 1
    for stages, says in [
        ({}, "no stage is given"),
        ({"": bank}, "name is empty"),
        ({"bank": bytes(damaged)}, "damaged"),
        ({"bank": b"not a state"}, "not a rheon state"),
        ({"inner": before}, "itself a pipeline checkpoint"),
    ]:
        with pytest.raises(ValueError, match=says):
            rp.stream.save_checkpoint(path, stages)
        assert path.read_bytes() == before
    with pytest.raises(TypeError, match="not a state"):
        rp.stream.save_checkpoint(path, {"bank": 3})  # type: ignore[dict-item]
    with pytest.raises(FileNotFoundError, match="does not exist"):
        rp.stream.save_checkpoint(tmp_path / "nowhere" / "run.checkpoint", {"bank": bank})
    with pytest.raises(FileNotFoundError):
        rp.stream.save_checkpoint(path, {"bank": tmp_path / "missing.state"})
    assert path.read_bytes() == before
    assert sorted(os.listdir(tmp_path)) == ["run.checkpoint"], "a temporary was left"


def test_a_file_of_another_kind_is_named_for_what_it_is(tmp_path):
    bank = rp.ModelBank([spec()])
    with pytest.raises(ValueError, match="a model bank's state, not a pipeline checkpoint"):
        rp.stream.load_checkpoint(bank.save_bytes())
    path = tmp_path / "run.checkpoint"
    rp.stream.save_checkpoint(path, {"bank": bank})
    with pytest.raises(ValueError, match="a pipeline checkpoint, not a model bank's state"):
        rp.ModelBank.load(path)
    with pytest.raises(ValueError, match="pipeline checkpoint, not a window run's state"):
        windows(ticks().select(time_refresh="t", a_value="v", b_value="v"), load_state=path)
    cut = path.read_bytes()[:-3]
    with pytest.raises(ValueError, match="damaged"):
        rp.stream.load_checkpoint(cut)
