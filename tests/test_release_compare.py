"""`scripts/compare_release.py` and its workload, `scripts/release_probe.py`:
the release step that compares every output with the last release's, bit
for bit (docs/contributing/cut-a-release.md, step 1; docs/RELEASE-READINESS.md,
"Compare with the last release, bit for bit")."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load("compare_release")
probe = _load("release_probe")


def test_bits_are_compared_not_values():
    same = compare.same
    assert same(pl.Series([1.0, None]), pl.Series([1.0, None]))
    assert not same(pl.Series([0.0]), pl.Series([-0.0])), "-0.0 is not 0.0 here"
    assert not same(pl.Series([1.0, None]), pl.Series([1.0, 2.0])), "a null is not a value"
    assert not same(pl.Series([1.0]), pl.Series([1.0 + 2**-52])), "one unit in the last place"
    lists = pl.Series([[1.0, 2.0], None, []])
    assert same(lists, lists.clone())
    assert not same(lists, pl.Series([[1.0, 2.0], None, [0.0]]))
    assert not same(pl.Series([1.0]), pl.Series([1], dtype=pl.Int64)), "the dtype is compared"


def test_a_dtype_change_is_named_not_called_bits():
    """A field whose values are all equal but whose dtype moved -- an
    integer widened, an Enum's categories renamed -- was reported as "bits
    differ (e.g. -0.0 against 0.0)", which sent the reader after a sign of
    zero that was not there (release 1.0's comparison: `kmeans.cluster`,
    every `withheld_reason`)."""
    old = pl.Series([1, 0, None], dtype=pl.Int32)
    new = pl.Series([1, 0, None], dtype=pl.Int64)
    assert not compare.same(old, new)
    assert compare.first_difference(old, new) == "dtype Int32 -> Int64, every value equal"
    old_e = pl.Series(["a", None], dtype=pl.Enum(["a", "below_min_periods"]))
    new_e = pl.Series(["a", None], dtype=pl.Enum(["a", "below_min_weight"]))
    text = compare.first_difference(old_e, new_e)
    assert "bits differ" not in text, text
    assert "gone ['below_min_periods']" in text and "new ['below_min_weight']" in text, text
    # A row that differs is still named first, with the dtype beside it.
    assert compare.first_difference(old, pl.Series([1, 2, None])).startswith("row 1: 0 -> 2")
    # And a sign of zero, whose dtype is the same, keeps its own message.
    assert compare.first_difference(pl.Series([0.0]), pl.Series([-0.0])).startswith("bits differ")
    # An Array against a List of the same numbers is not a dtype change.
    arr = pl.Series([[1.0, 2.0]], dtype=pl.Array(pl.Float64, 2))
    assert compare.same(arr, pl.Series([[1.0, 2.0]]))


def test_this_build_runs_the_whole_workload(tmp_path):
    """A spec this build refuses would drop out of every comparison unseen:
    the workload must keep up with the API."""
    out, manifest = tmp_path / "out.parquet", tmp_path / "m.json"
    probe.main(str(out), str(manifest))
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["refused"] == {}, meta["refused"]
    assert meta["unmeasured"] == {}, meta["unmeasured"]
    specs = probe.WORKLOAD + probe.HARD_PATHS
    names = [name for name, *_ in specs]
    variants = [
        prefix + name
        for prefix, _, applies in probe.VARIANTS
        for name, _, kw, _ in specs
        if applies(kw)
    ]
    assert meta["ran"] == names + variants + [f"resume_{n}" for n in names]
    assert meta["kinds"] == {
        **{n: b for n, b, *_ in specs},
        **{v: meta["kinds"][v.split("_", 1)[1]] for v in variants},
        **{f"resume_{n}": b for n, b, *_ in specs},
    }
    frame = pl.read_parquet(out)
    assert frame.height == probe.stream().height and frame.width > 5000


def test_every_hard_path_is_reached_and_counted(tmp_path):
    """Each path `release_probe.PATHS` names is counted above 0 in this
    build's run: the comparison proves it reached each one rather than
    assuming it, and `compare` fails on a count of 0 (see
    `test_a_path_no_longer_reached_fails`). The counts, measured
    2026-11-20 on macOS arm64; a count that drops says the workload
    stopped reaching part of a path."""
    manifest = tmp_path / "m.json"
    probe.main(str(tmp_path / "out.parquet"), str(manifest))
    counts = json.loads(manifest.read_text(encoding="utf-8"))["activations"]
    assert list(counts) == probe.PATHS
    specs = len(probe.WORKLOAD + probe.HARD_PATHS)
    assert counts["a state saved mid-stream, loaded and continued"] == specs, counts
    assert counts["a reset after a step back"] == len(probe.CLOCKED) + 1, counts
    # Three groups at a decade's age in each of the three `decade_` specs.
    assert counts["a temporal clock at a decade's age"] == 9, counts
    # Each spec with a `weight_sum` and a feature: all but `ridge_grid`,
    # whose outputs are per grid point, and for null rows `holt`'s,
    # `audit`'s and `seqtest`'s, which read a row of null targets as one
    # to predict.
    assert counts["zero-weight first rows"] >= 90, counts
    assert counts["null first rows"] >= 90, counts
    for path in probe.PATHS:
        assert counts[path] > 0, (path, counts)


def _counted(df, name, builder, kw, prefix=""):
    """The counts of `release_probe.activations` for one run of a spec."""
    from rheon import polars as rp

    bank = rp.ModelBank([probe.build(builder, name, kw)])
    out = bank.fit_predict(df)
    counts = dict.fromkeys(probe.PATHS, 0)
    probe.activations(counts, prefix, name, builder, kw, bank, out, df)
    return counts


def _hard(name):
    (spec,) = [(b, dict(kw)) for n, b, kw, _ in probe.HARD_PATHS + probe.WORKLOAD if n == name]
    return spec


def test_the_bocpd_counters_count_nothing_without_a_prune_or_a_cap():
    """A count is a proof only if a run without the path counts 0: with
    `prune_below = 0` every run length from 0 to the longest is held, and
    with a cap the runs never reach, no row's `run_mode` is at it."""
    df = probe.stream()
    builder, kw = _hard("bocpd_fold_diag")
    hit = _counted(df, "b", builder, kw)
    assert hit["bocpd drops runs below prune_below"] > 0, hit
    assert hit["bocpd folds the runs at max_run"] > 0, hit
    unpruned = _counted(df, "b", builder, dict(kw, prune_below=0.0))
    assert unpruned["bocpd drops runs below prune_below"] == 0, unpruned
    assert unpruned["bocpd folds the runs at max_run"] > 0, unpruned
    uncapped = _counted(df, "b", builder, dict(kw, max_run=100_000))
    assert uncapped["bocpd folds the runs at max_run"] == 0, uncapped


def test_the_low_part_counter_counts_nothing_where_no_step_rounds():
    """A mean that never moves, every row on it, takes steps of exactly 0
    and keeps no low part: the counter reads 0 there, and above 0 on the
    probe's stream."""
    df = probe.stream()
    for name in ("kmeans", "micro"):
        builder, kw = _hard(name)
        path = f"{name} carries a compensated mean's low part"
        assert _counted(df, name, builder, kw)[path] > 0, name
        flat = df.with_columns(x1=pl.lit(1.0), x2=pl.lit(-2.0))
        assert _counted(flat, name, builder, kw)[path] == 0, name


def test_the_clock_counters_count_nothing_on_a_clock_that_does_not_take_the_path():
    """One session, a clock begun with its rows, and a clock that never
    steps back each count 0; the probe's stream counts each above 0."""
    df = probe.stream()
    builder, kw = _hard("session_kalman")
    numeric = "a numeric clock crosses a session"
    assert _counted(df, "k", builder, kw)[numeric] > 0
    assert _counted(df.with_columns(s=pl.lit("am")), "k", builder, kw)[numeric] == 0
    builder, kw = _hard("decade_ridge")
    temporal, decade = "a temporal clock crosses a session", "a temporal clock at a decade's age"
    hit = _counted(df, "d", builder, kw)
    assert hit[decade] == 3 and hit[temporal] == 3, hit
    young = _counted(df.with_columns(ts_decade=pl.col("ts")), "d", builder, kw)
    assert young[decade] == 0, young
    builder, kw = _hard("step_back_bocpd")
    reset = "a reset after a step back"
    assert _counted(df, "sb", builder, kw)[reset] == 1
    assert _counted(df.with_columns(t_back=pl.col("t")), "sb", builder, kw)[reset] == 0


def test_the_first_row_counters_count_nothing_on_a_stream_whose_first_rows_learn():
    """On the probe's own stream each group's first row has a weight (row
    0's is 0 by its draw, rows 1 and 2's are not), so the next rows'
    `weight_sum` is not 0 everywhere, and neither counter counts; on the
    variants, each does."""
    df = probe.stream()
    for name in ("ridge", "ridge_step_back", "kmeans", "rcov"):
        builder, kw = _hard(name)
        for prefix, variant, applies in probe.VARIANTS:
            if not applies(kw):
                continue
            path = "zero-weight first rows" if prefix == "w0_" else "null first rows"
            assert _counted(variant(df), name, builder, kw, prefix)[path] == 1, (name, prefix)
            assert _counted(df, name, builder, kw, prefix)[path] == 0, (name, prefix)


def test_a_resumed_run_is_the_saved_and_loaded_bank_going_on(tmp_path):
    """`_resume` saves at `RESUME_AT` and loads a bank that says it was fed
    those rows, and its output covers the whole stream: the same rows as
    the run that was never saved, and on this stream the same bits."""
    from rheon import polars as rp

    df = probe.stream()
    builder, kw = _hard("ridge")
    spec = probe.build(builder, "ridge", kw)
    out, _, fed = probe._resume([spec], "ridge", df, str(tmp_path))
    assert fed == probe.RESUME_AT
    whole = rp.ModelBank([spec]).fit_predict(df).select("ridge")
    assert out.height == whole.height
    assert compare.same(out.unnest("ridge")["pred_y"], whole.unnest("ridge")["pred_y"])


def test_a_spec_this_build_refuses_is_a_difference_not_a_silence():
    """`compare` took only the specs both builds ran and never read this
    build's refusals, so a spec the release ran and this build refuses
    dropped out, and the verdict said "identical", exit 0 (task 160, SC1).
    The gate's workload test above catches the probe refusing at HEAD; the
    script's own verdict must not depend on it."""
    meta = {"file": "a", "version": "0.0.1", "ran": ["ridge", "rls"], "refused": {}}
    new_meta = {
        "file": "b",
        "version": "0.0.2",
        "ran": ["ridge"],
        "refused": {"rls": "RuntimeError: boom"},
    }
    frame = pl.DataFrame({"ridge.pred": [1.0, 2.0]})
    old = frame.with_columns(pl.lit(3.0).alias("rls.pred"))
    out: list[str] = []
    code = compare.compare("0.0.1", old, meta, frame, new_meta, out)
    text = "\n".join(out)
    assert code == 1, text
    assert "rls" in text and "RuntimeError: boom" in text, text
    assert "identical" not in text, text


def _sgd_on(loss: str):
    return lambda b, kw: b == "sgd" and kw.get("loss") == loss


def _kalman_unsized(b: str, kw: dict) -> bool:
    """`kalman` standardized with no `obs_var`, no intercept and a lowered
    `min_weight`: rows past `min_weight` before a target's prior is sized
    (review round 9, B-1, the path 1.2.0 predicted 0.0 on). With an
    intercept the first row sizes it, diffuse, and no row waits (task 278)."""
    return (b == "kalman" and kw.get("standardize") is True and "obs_var" not in kw) and (
        "min_weight" in kw and kw.get("fit_intercept") is False
    )


#: The paths the release comparison must reach, each by the keyword that
#: takes it (review round 9, J-2: the workload ran 135 of the 346 field names
#: the fullest workload writes, and none of these). A spec of the workload
#: must satisfy each.
PATHS = {
    "a temporal clock": lambda b, kw: kw.get("clock") == "ts",
    "a duration parameter": lambda b, kw: isinstance(kw.get("half_life"), str),
    "gram_block_rows": lambda b, kw: "gram_block_rows" in kw,
    "gram_threads": lambda b, kw: "gram_threads" in kw,
    "target_gaps pairwise": lambda b, kw: kw.get("target_gaps") == "pairwise",
    "a formula target": lambda b, kw: (
        any(isinstance(t, probe._Formula) for t in kw["targets"]) if "targets" in kw else False
    ),
    "feature_sets": lambda b, kw: "feature_sets" in kw,
    "coef_prior": lambda b, kw: "coef_prior" in kw,
    "revert_half_life": lambda b, kw: "revert_half_life" in kw,
    "share_p": lambda b, kw: kw.get("share_p") is True,
    "kalman q and obs_var": lambda b, kw: "q" in kw and "obs_var" in kw,
    "kalman standardized": lambda b, kw: b == "kalman" and kw.get("standardize") is True,
    "kalman's prior sized past min_weight": _kalman_unsized,
    "drift_action reset": lambda b, kw: kw.get("drift_action") == "reset",
    "max_rows_between_solves": lambda b, kw: "max_rows_between_solves" in kw,
    "max_rows_between_coefs": lambda b, kw: "max_rows_between_coefs" in kw,
    "restart_after_step_back": lambda b, kw: "restart_after_step_back" in kw,
    "a lam accumulator": lambda b, kw: "lam" in kw,
    "closed both": lambda b, kw: kw.get("closed") == "both",
    "a lasso window": lambda b, kw: b == "lasso" and "window_size" in kw,
    "an ew_class window": lambda b, kw: b == "ew_class" and "window_size" in kw,
    "marginal window_lags": lambda b, kw: kw.get("window_lags") is True,
    "marginal bin_edges": lambda b, kw: "bin_edges" in kw,
    "group_close session": lambda b, kw: kw.get("group_close") == "session",
    "ftrl squared": lambda b, kw: b == "ftrl" and kw.get("loss") == "squared",
    "pa1": lambda b, kw: b == "pa" and kw.get("mode") == "pa1",
    "pa2": lambda b, kw: b == "pa" and kw.get("mode") == "pa2",
    "sgd adagrad": lambda b, kw: b == "sgd" and kw.get("schedule") == "adagrad",
    **{
        f"sgd {loss}": _sgd_on(loss)
        for loss in ("huber", "quantile", "epsilon_insensitive", "poisson", "logistic")
    },
}


def test_the_workload_reaches_every_field_name_and_every_named_path():
    """The release comparison is read as comparing every output: it holds
    the fullest workload's field names (the memory budgets', every output of
    every model switched on), all of them, and every path in `PATHS`."""
    import test_budgets_memory as memory
    from rheon import polars as rp

    probed: set[str] = set()
    for name, builder, kw, _ in probe.WORKLOAD:
        probed |= set(rp.spec.output_fields(probe.build(builder, name, kw)))
    fullest: set[str] = set()
    for name, builder, kw, _ in memory.WORKLOAD:
        fullest |= set(rp.spec.output_fields(getattr(rp.spec, builder)(name, **kw)))
    assert len(fullest) > 300, len(fullest)
    assert fullest <= probed, sorted(fullest - probed)
    missed = [
        p
        for p, reached in PATHS.items()
        if not any(reached(b, kw) for _, b, kw, _ in probe.WORKLOAD)
    ]
    assert not missed, missed


def test_the_kalman_unsized_prior_path_is_reached_not_only_named():
    """A spec matching `PATHS`' entry for `kalman`'s unsized prior must reach
    it on the probe's stream: at least one row withheld as
    `below_warm_rows`, the rows past `min_weight` before the prior is sized
    (review 9's R4 §6: `kalman_standardized` reported none, so the release
    comparison never compared that path's rows)."""
    from rheon import polars as rp

    specs = [(n, b, kw) for n, b, kw, _ in probe.WORKLOAD if _kalman_unsized(b, kw)]
    assert specs
    df = probe.stream()
    for name, builder, kw in specs:
        out = rp.ModelBank([probe.build(builder, name, kw)]).fit_predict(df).unnest(name)
        warm = out.filter(pl.col("withheld_reason") == "below_warm_rows")
        assert warm.height >= 1, (name, out["withheld_reason"].value_counts().to_dicts())
        # Nothing is predicted there: null, not the 0.0 of an unsized prior.
        assert warm["pred_y"].is_null().all(), warm


#: A CHANGELOG with a release above the one compared against, as while a
#: release is cut, and an older section whose clause must not be read.
CHANGELOG = """# Changelog

Header text. Moves: `header.ignored` -- above every section.

## [Unreleased]

### Fixed

- **`ewridge`'s solve adds its Gram in another order**, which moves its
  predictions by at most 2 ulps. Moves: `ewridge.pred_*`,
  `ewridge.coef` and `*.weight_sum`.

```text
- Moves: `fenced.code` -- an example, not a declaration.
```

## [0.0.3] — 2026-11-21

- `kmeans` seeds its centres from another row, so every centre and
  distance moves. Moves: `kmeans.*`.

## [0.0.2] — 2026-11-20

- Old. Moves: `rls.pred_y`, which an older release declared.
"""


def test_the_declarations_above_the_release_are_read_with_their_reasons():
    found, errors = compare.declarations(CHANGELOG, "0.0.2")
    assert errors == []
    assert [d.patterns for d in found] == [
        ("ewridge.pred_*", "ewridge.coef", "*.weight_sum"),
        ("kmeans.*",),
    ]
    first, second = found
    assert first.line == 9 and "another order" in first.reason and "Moves" not in first.reason
    assert first.covers("ewridge", "pred_y") and first.covers("rls", "weight_sum")
    assert not first.covers("ewridge", "sigma_y") and not first.covers("rls", "pred_y")
    assert second.covers("kmeans", "closed.cluster")
    # Compared against 0.0.3, its own section is released: only
    # `[Unreleased]` is read. Against a release the file does not name, as
    # 0.0.1, every section above it is.
    assert [d.patterns for d in compare.declarations(CHANGELOG, "0.0.3")[0]] == [
        ("ewridge.pred_*", "ewridge.coef", "*.weight_sum")
    ]
    assert len(compare.declarations(CHANGELOG, "0.0.1")[0]) == 3


def test_a_declaration_without_a_reason_or_a_model_field_is_an_error():
    bare = "## [Unreleased]\n\n- Moves: `ewridge.pred_y`.\n\n## [0.0.2]\n"
    found, errors = compare.declarations(bare, "0.0.2")
    assert len(found) == 1 and len(errors) == 1 and "gives no reason" in errors[0], errors
    shapeless = (
        "## [Unreleased]\n\n- The solve moved its predictions in the last bits.\n"
        "  Moves: `pred_y`.\n\n## [0.0.2]\n"
    )
    _, errors = compare.declarations(shapeless, "0.0.2")
    assert len(errors) == 1 and "is not `model.field`" in errors[0], errors
    # The release's own CHANGELOG reads, whatever it declares, without an
    # error, against the newest release it names.
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = [m.group(1) for line in text.splitlines() if (m := compare.HEADING.match(line))]
    newest = headings[1]
    assert headings[0] == "Unreleased" and compare._key(newest), headings
    assert compare.declarations(text, newest)[1] == []


def _metas(old_ran, new_ran, kinds, new_refused=None, activations=None):
    old = {"file": "a", "version": "0.0.2", "ran": old_ran, "refused": {}, "kinds": kinds}
    new = {
        "file": "b",
        "version": "0.0.3",
        "ran": new_ran,
        "refused": new_refused or {},
        "kinds": kinds,
        "activations": activations or {"a path": 3},
        "unmeasured": {},
    }
    return old, new


KINDS = {"ridge": "ewridge", "w0_ridge": "ewridge", "km": "kmeans", "new": "rls"}
OLD = pl.DataFrame(
    {
        "ridge.pred_y": [1.0, 2.0],
        "ridge.weight_sum": [0.0, 1.0],
        "w0_ridge.pred_y": [1.0, 2.0],
        "km.dist": [0.5, 0.25],
    }
)


def _compare(new, declared_text="", **meta):
    old_meta, new_meta = _metas(
        meta.pop("old_ran", ["ridge", "w0_ridge", "km"]),
        meta.pop("new_ran", ["ridge", "w0_ridge", "km"]),
        KINDS,
        **meta,
    )
    changelog = f"## [Unreleased]\n\n{declared_text}\n\n## [0.0.2]\n"
    declared, errors = compare.declarations(changelog, "0.0.2")
    out: list[str] = []
    code = compare.compare("0.0.2", OLD, old_meta, new, new_meta, out, declared, errors)
    return code, "\n".join(out)


def test_identical_outputs_pass():
    code, text = _compare(OLD.clone())
    assert code == 0 and "identical to 0.0.2" in text, text


def test_a_moved_field_no_entry_declares_fails():
    new = OLD.with_columns(pl.Series("ridge.pred_y", [1.0, 2.0 + 2**-51]))
    code, text = _compare(new)
    assert code == 1, text
    assert "not declared" in text and "ridge.pred_y: row 1" in text, text
    # A declaration of another field leaves it undeclared.
    code, text = _compare(new, "- The weights moved by a rounding. Moves: `ewridge.weight_sum`.")
    assert code == 1 and "declared, nothing moved: ewridge.weight_sum" in text, text


def test_a_moved_field_an_entry_declares_passes_and_is_listed():
    new = OLD.with_columns(
        pl.Series("ridge.pred_y", [1.0, 2.0 + 2**-51]),
        pl.Series("w0_ridge.pred_y", [1.0, 2.0 + 2**-51]),
    )
    entry = "- The solve sums in another order, by an ulp. Moves: `ewridge.pred_*`."
    code, text = _compare(new, entry)
    assert code == 0, text
    assert "moved, as the CHANGELOG declares (2 fields)" in text, text
    assert "ridge.pred_y (ewridge)" in text and "w0_ridge.pred_y (ewridge)" in text, text
    assert "sums in another order" in text, text


def test_a_field_or_a_spec_only_this_build_has_is_an_addition_that_passes():
    new = OLD.with_columns(
        pl.Series("ridge.sigma_y", [0.1, 0.2]), pl.Series("new.pred_y", [3.0, 4.0])
    )
    code, text = _compare(
        new, old_ran=["ridge", "w0_ridge", "km"], new_ran=["ridge", "w0_ridge", "km", "new"]
    )
    assert code == 0, text
    assert "added, not compared (1 fields, 1 specs)" in text, text
    assert "ridge.sigma_y: new in this build" in text and "new: refused by 0.0.2" in text, text


def test_a_field_gone_or_a_spec_refused_fails_unless_declared():
    code, text = _compare(OLD.drop("km.dist"))
    assert code == 1 and "km.dist: gone from this build" in text, text
    refused = OLD.drop("km.dist")
    kw = dict(new_ran=["ridge", "w0_ridge"], new_refused={"km": "ValueError: no"})
    code, text = _compare(refused, **kw)
    assert code == 1 and "km.dist: refused by this build: ValueError: no" in text, text
    entry = "- `kmeans` takes no `dist` any more, so its spec is refused. Moves: `kmeans.*`."
    code, text = _compare(refused, entry, **kw)
    assert code == 0, text


def test_a_malformed_declaration_or_an_unknown_model_fails_even_when_nothing_moved():
    code, text = _compare(OLD.clone(), "- The solve sums in another order. Moves: `ewrige.pred_y`.")
    assert code == 1 and "`ewrige.pred_y` names no model the workload runs" in text, text
    code, text = _compare(OLD.clone(), "- Moves: `ewridge.pred_y`.")
    assert code == 1 and "gives no reason" in text, text


def test_a_path_no_longer_reached_fails():
    code, text = _compare(OLD.clone(), activations={"a path": 3, "another path": 0})
    assert code == 1, text
    assert "the workload no longer reaches another path: counted 0" in text, text
    assert "another path: 0 / -" in text, text
