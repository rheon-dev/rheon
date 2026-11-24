"""`scripts/ci_status.py`: every job of every workflow run on a sha, read
before the next push (review 459, A5: CI on main was red for two days on
Windows while a Budgets run's expected red hid it). ``gh`` is replaced by a
table of answers, so the test runs offline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TIER = "essential"
REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("ci_status", REPO / "scripts" / "ci_status.py")
assert _spec and _spec.loader
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _run(rid, workflow, created="2026-11-17T10:00:00Z", event="push", status="completed"):
    return {"databaseId": rid, "workflowName": workflow, "event": event, "status": status,
            "conclusion": "", "createdAt": created, "url": f"u/{rid}"}  # fmt: skip


def _job(jid, name, conclusion="success", status="completed"):
    return {"databaseId": jid, "name": name, "status": status, "conclusion": conclusion,
            "url": f"j/{jid}"}  # fmt: skip


def _gh(runs, jobs, logs=None):
    def gh(args):
        if args[:2] == ["run", "list"]:
            return json.dumps(runs)
        if args[:3] == ["run", "view", "--job"]:
            return (logs or {})[int(args[3])]
        assert args[:2] == ["run", "view"]
        return json.dumps({"jobs": jobs[int(args[2])]})

    return gh


def test_every_job_passing_is_a_pass(capsys):
    gh = _gh([_run(1, "CI"), _run(2, "Budgets")],
             {1: [_job(10, "mac"), _job(11, "windows"), _job(12, "opt", "skipped")],
              2: [_job(20, "count")]})  # fmt: skip
    assert cs.check("abc", gh) == 0
    assert "4 jobs: 4 passed" in capsys.readouterr().out


def test_one_windows_leg_failing_fails_the_sha(capsys):
    gh = _gh([_run(1, "CI")], {1: [_job(10, "mac"), _job(11, "windows (3.13)", "failure")]})
    assert cs.check("abc", gh) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "windows (3.13)" in out


_STAMP = "count\tUNKNOWN STEP\t2026-11-21T03:01:35.2130520Z "
_EXIT = _STAMP + "##[error]Process completed with exit code 1."


def _log(*errors):
    """A failed Budgets job's log as ``gh run view --log-failed`` gives it:
    a setup step's script echoed, the report's lines, then each ``::error::``
    as the runner wrote it."""
    echoed = _STAMP + '\x1b[36;1m  echo "::error::linux-build-prep runs on Linux only"\x1b[0m'
    lines = [echoed] + [_STAMP + "- " + e for e in errors]
    lines += [_STAMP + "##[error]" + e for e in errors]
    return "\n".join([*lines, _EXIT])


def test_a_budget_to_tighten_rides_but_a_budget_over_fails(capsys):
    tighten = _log("x.y: 120 under its budget of 140: tighten the budget to 121")
    over = _log(
        "x.y: 120 under its budget of 140: tighten the budget to 121",
        "z.w: 160 over its budget of 140",
    )
    for log, want in ((tighten, 0), (over, 1)):
        gh = _gh([_run(1, "Budgets")], {1: [_job(10, "count", "failure")]}, {10: log})
        assert cs.check("abc", gh) == want
        assert cs.check("abc", gh, strict=True) == 1
    assert "TIGHTEN" in capsys.readouterr().out
    # The same words from another workflow are a failure.
    gh = _gh([_run(1, "CI")], {1: [_job(10, "x", "failure")]}, {10: tighten})
    assert cs.check("abc", gh) == 1


def test_cancelled_is_not_a_pass_and_pending_waits(capsys):
    gh = _gh([_run(1, "CI")], {1: [_job(10, "windows", "cancelled")]})
    assert cs.check("abc", gh) == 1
    assert "last completed run" in capsys.readouterr().out
    gh = _gh([_run(1, "CI", status="in_progress")], {1: [_job(10, "w", None, "in_progress")]})
    assert cs.check("abc", gh) == 2
    assert cs.check("abc", _gh([], {})) == 2, "no run yet is not a pass"


def test_the_latest_run_of_a_workflow_is_the_one_read():
    old, new = _run(1, "CI", "2026-11-17T10:00:00Z"), _run(2, "CI", "2026-11-17T12:00:00Z")
    gh = _gh([old, new], {1: [_job(10, "w", "failure")], 2: [_job(20, "w")]})
    assert cs.check("abc", gh) == 0
    dispatch = _run(3, "CI", "2026-11-17T09:00:00Z", event="workflow_dispatch")
    jobs = {1: [_job(10, "w", "failure")], 2: [_job(20, "w")], 3: [_job(30, "w", "failure")]}
    gh = _gh([old, new, dispatch], jobs)
    assert cs.check("abc", gh) == 1, "a dispatch on the sha is read beside the push"


@pytest.mark.parametrize(
    "other",
    [
        "scaling.ewridge.k: 1000000.000x from k=2 to k=4, past the 70.400x its growth allows",
        "instructions.ewridge_k20: budgeted, and the run did not measure it",
        "instructions.ew: the two runs disagree: 100 and 101",
        "could not read the last release's baseline (HTTP 404)",
    ],
)
def test_a_budgets_failure_beside_a_tighten_is_a_failure(other):
    """Review S1: `budgets.py check` fails four more ways than over a budget,
    none of them saying `over its budget`; each beside a budget to tighten
    is still a failure, and a log with no error line is not a tighten."""
    log = _log("x.y: 120 under its budget of 140: tighten the budget to 121", other)
    assert not cs.only_tightens(log)
    gh = _gh([_run(1, "Budgets")], {1: [_job(10, "count", "failure")]}, {10: log})
    assert cs.check("abc", gh) == 1
    assert not cs.only_tightens(_EXIT), "no error line is not a budget to tighten"
    assert not cs.only_tightens("tighten the budget to 121 (not an error line)")


@pytest.mark.parametrize("conclusion", ["timed_out", "startup_failure", "action_required"])
def test_other_conclusions_fail(conclusion):
    gh = _gh([_run(1, "CI")], {1: [_job(10, "w", conclusion)]})
    assert cs.check("abc", gh) == 1
