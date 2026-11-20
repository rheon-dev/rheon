"""Every job of every workflow run on one commit, and whether all of them passed.

    uv run python scripts/ci_status.py SHA            # exit 0 only if every job passed
    uv run python scripts/ci_status.py SHA --repo OWNER/NAME

Run it on the last pushed sha before the next push (docs/records/AGENT-WORKFLOW.md,
"Before each push"). A red workflow is easy to misread as an expected one
when another workflow on the same sha is red for a known reason; this reads
each job of each run with ``gh run list --commit`` and ``gh run view --json
jobs`` and names every one that did not pass.

A job passes when it completed with ``success`` or ``skipped`` (a job its
workflow's ``if:`` left out). Anything else fails the check:

- ``failure``, ``timed_out``, ``startup_failure``, ``action_required``: failed.
- ``cancelled``: unknown, not passed. A cancelled run says nothing about the
  commit (a later push cancelled it, or a person did); the last completed run of
  that workflow on an earlier sha is where to look, and the check says so.
- a job still queued or in progress: pending, exit 2, so a caller can wait.
- no run at all on the sha: exit 2 (gh lists none until a push is seen).

One failure is not counted: a Budgets job whose only complaint is that a
budget can be tightened (``tighten the budget to``, and no ``over its
budget``). That is a gain to keep, which rides with the next wave rather than
blocking it; it is printed as ``TIGHTEN``. ``--strict`` counts it as a failure
too. Where a workflow ran more than once on the sha (a push and a dispatch, or
a re-run), the latest run of each workflow and event is the one read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from typing import Any

PASSED = {"success", "skipped", "neutral"}
FAILED = {"failure", "timed_out", "startup_failure", "action_required", "stale"}
#: The workflow whose failures may be only budgets to tighten.
BUDGETS = "Budgets"

Gh = Callable[[list[str]], str]


def _gh(args: list[str]) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def latest_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The latest run of each workflow and event, by its creation time."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in runs:
        key = (r["workflowName"], r.get("event", ""))
        if key not in best or r["createdAt"] > best[key]["createdAt"]:
            best[key] = r
    return sorted(best.values(), key=lambda r: (r["workflowName"], r.get("event", "")))


def only_tightens(log: str) -> bool:
    """A Budgets log whose failures are all budgets to tighten, none over."""
    return "tighten the budget to" in log and "over its budget" not in log


def judge_job(workflow: str, job: dict[str, Any], log: Callable[[], str]) -> str:
    """One job's verdict: ``pass``, ``tighten``, ``fail``, ``cancelled`` or ``pending``."""
    if job.get("status") != "completed":
        return "pending"
    c = job.get("conclusion") or ""
    if c in PASSED:
        return "pass"
    if c == "cancelled":
        return "cancelled"
    if c == "failure" and workflow == BUDGETS and only_tightens(log()):
        return "tighten"
    return "fail"


def check(sha: str, gh: Gh = _gh, repo: str | None = None, strict: bool = False) -> int:
    """Print every job of every run on ``sha``; 0 if all passed, 1 if any
    failed or was cancelled, 2 if none ran yet or one is still pending."""
    where = ["--repo", repo] if repo else []
    runs = json.loads(
        gh(
            [
                "run",
                "list",
                "--commit",
                sha,
                "--limit",
                "200",
                "--json",
                "databaseId,workflowName,event,status,conclusion,createdAt,url",
                *where,
            ]
        )  # fmt: skip
    )
    if not runs:
        print(f"{sha}: no workflow run yet")
        return 2
    counts = {"pass": 0, "tighten": 0, "fail": 0, "cancelled": 0, "pending": 0}
    for run in latest_runs(runs):
        rid = str(run["databaseId"])
        jobs = json.loads(gh(["run", "view", rid, "--json", "jobs", *where]))["jobs"]
        if not jobs:
            verdict = "pending" if run.get("status") != "completed" else "fail"
            counts[verdict] += 1
            print(
                f"{verdict.upper():9s} {run['workflowName']} ({run['event']}): no jobs"
                f"  {run['url']}"
            )
            continue
        for job in jobs:

            def log(job: dict[str, Any] = job) -> str:
                return gh(["run", "view", "--job", str(job["databaseId"]), "--log-failed", *where])

            verdict = judge_job(run["workflowName"], job, log)
            if verdict == "tighten" and strict:
                verdict = "fail"
            counts[verdict] += 1
            if verdict != "pass":
                print(
                    f"{verdict.upper():9s} {run['workflowName']} / {job['name']}"
                    f" ({job.get('conclusion') or job.get('status')})  {job.get('url', run['url'])}"
                )
    total = sum(counts.values())
    print(
        f"{sha}: {total} jobs: {counts['pass']} passed, {counts['tighten']} budgets to tighten, "
        f"{counts['fail']} failed, {counts['cancelled']} cancelled, {counts['pending']} pending"
    )
    if counts["cancelled"]:
        print(
            "A cancelled job says nothing about this sha: read that workflow's last "
            "completed run before pushing on it."
        )
    if counts["fail"] or counts["cancelled"]:
        return 1
    if counts["pending"]:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sha")
    ap.add_argument("--repo", help="OWNER/NAME, where the checkout's default is not the one")
    ap.add_argument("--strict", action="store_true", help="count budgets to tighten as failures")
    args = ap.parse_args(argv)
    return check(args.sha, repo=args.repo, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
