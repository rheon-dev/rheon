<!-- kind: record -->
# How the coordinator and its workers run

Moved verbatim from CLAUDE.md on 2026-11-14 (the user: cut the size of the
docs read most), so every session loads a short summary and this file holds
the detail and the reason for each rule. CLAUDE.md's "Workflow" section is
the summary.

Iterating (task 245; docs/TESTING.md, "Iterating" and "Caches and when to
clear them"): workers run `scripts/quick.sh` while iterating and finish
with `scripts/quick.sh --extended` plus the lints, not a full gate, and
never merge `main` unless asked. The coordinator merges each wave of
finished branches onto an `integration` branch cut from `main`, regenerates
any generated files that conflict (fixtures, goldens, the API snapshot)
once, runs ONE essentials gate, and fast-forwards `main` on PASS; a failure
is mapped to a branch by the files it touched (`scripts/slot.sh overlap`)
or bisected over the wave's merges (the user, 2026-10-26: parallel gates
and the re-gates of conflicting merges were the agents' two bottlenecks).
A wave of one branch skips `integration` and the essentials gate: it merges
onto `main` (a fast-forward where it can be) and the full tier runs once, on
that sha, before the push (the user, 2026-11-03: "why bother with an
integration branch").
`scripts/gate.sh` takes one of two machine-wide tokens and waits for one
(`POLARS_ONLINE_GATES`, now `RHEON_GATES`). The full tier runs, always, before every push on
the exact sha.
Run at most two building agents at once on this machine -- three while the
1-minute load stays under about 20 (the user, 2026-11-07) -- each with
`CARGO_BUILD_JOBS=6`, pytest at `-n 4` and the gate at `GATE_WORKERS=4`;
a local `cargo mutants` runs alone, under `.claude/coord/mutants.lock`;
check `uptime` first and wait above a load of about 20 (docs/TESTING.md,
"At most two agents": four parallel mutants runs at a load of 88 panicked
the Mac twice, 2026-10-27/28). Coordination files -- briefs, the queue,
the worker rules, survivor lists -- live in `.claude/coord/` (locally
excluded), never under `/tmp`, which a restart empties. A worker gets a
worktree slot, not a fresh worktree (task 248; docs/TESTING.md, "Worktree
slots"): the coordinator runs `scripts/slot.sh take <branch> main`, starts
the worker without `isolation`, told to work only in that path, and runs
`scripts/slot.sh give <n>` after merging; a warm slot rebuilds the tests in
23 s and the dev wheel in 6 s where a fresh worktree takes 79 s and 145 s. Nobody runs a full
`cargo mutants` pass locally (the user, 2026-10-11: "Every task seems to
spend a lot of time waiting for mutants", then "we need a thorough mutants
check to run the smallest number of times"): the one thorough check is CI's
changed-lines pass on the push, sharded across runners and free; its
survivors are a follow-up batch, and a fix is checked locally only with an
`--in-diff` of the fix itself (`--copy-target=true --test-tool nextest`).
So push each merged, gated wave as it lands (the user, 2026-10-12: "Yes"
to it): a push needs the full gate on the exact sha and no open bug, and
then no further word; small pushes keep each CI mutants pass short and its
survivors fresh. Dispatching a release still needs the user's explicit
word, and its PyPI approval is always the user's click. Workers lean on activation tests instead. After each push
wave and its branch and worktree cleanup, run `scripts/clear_caches.sh
--worktrees` (or whenever free disk is under 100 GB). sccache stays off:
from a new worktree path a third of the crates miss its cache, so it is
slower there; `POLARS_ONLINE_SCCACHE=1` (now `RHEON_SCCACHE=1`) turns it on for one checkout.

Amended 2026-11-14 (the user: "a too", on the coordinator's count of the day's
gates): a wave pushed at once gets the full tier only, on the exact sha; the
essentials gate on `integration` runs only when the wave waits before its push,
since the full tier re-runs everything it does. Budget tightenings from CI's
counts ride with the next wave rather than taking a gate of their own. Workers
run fmt and clippy `--all-targets` before `scripts/quick.sh --extended`, and
that once, at the end.

## Before each push (task 459, review A5)

1. The full tier on the exact sha (above).
2. `uv run python scripts/ci_status.py <sha of the last push>`: every job of
   every workflow on the commit before this one, read from `gh run list
   --commit` and `gh run view --json jobs`. It exits 0 only when every job
   completed with success (or was skipped by its workflow's `if:`). A
   failure, a timeout or a cancelled job is exit 1; a run still queued or in
   progress, or none yet, is exit 2: wait for it, or say in the push's
   report that it was pending. A cancelled job says nothing about that sha
   (a later push cancelled it): read the workflow's last completed run
   before pushing past it. The one red it does not count is a Budgets job
   whose only complaint is a budget to tighten, which rides with the next
   wave (`--strict` counts it too).
3. Push only when step 2 is 0, or its failures are understood and the push
   fixes them (say which, and check the next run's jobs the same way). Why:
   from 2026-11-16 CI on main was red for about two days. All four Windows
   legs failed on eight runs and nine more were cancelled before a Windows
   leg reported (it reports about 90 minutes after a push), while Budgets
   was red on the same shas for tightenings that ride, so a red CI read as
   the expected one. The two failures (`os.getloadavg` in
   `compare_outputs.py`, a 0/0 from Windows' 15.6 ms CPU clock in
   `test_scaling.py`) now each have a test that fails on the Mac:
   `tests/test_platform_calls.py` walks `scripts/` and `tests/` for
   POSIX-only calls outside a platform guard, and `tests/test_scaling.py`
   runs its checks on a stepped fake clock.

## Speed claims (task 459, review A7)

A speed figure that goes into the CHANGELOG, PLAN or a decision is held to
three things before the merge that carries it, not after the push:

1. **x86 before the merge.** Every change claimed as faster, and every change
   to memory layout or call structure on a per-row path, is timed on x86-64
   (the VM) before it merges: the VM's time, not only CI's instruction
   count. The counts are a proxy that has failed both ways here (task 413:
   +5% instructions, 14-17% fewer cycles; 452b: +0.46% instructions, time
   0.87-0.89; 448's `hmm`: -6.4% instructions, 7% slower), and the two
   platform-worse changes that reached main (421: x86 grouped 0.80-0.92;
   450: +19% instructions and k=200 1.17 slower on x86, reverted) were both
   found after the push. A change that is slower on either platform does
   not merge as a speed change (the "no platform worse" rule).
2. **Both layouts.** Each platform's figure is read under the default
   layout and under the aligned one (every function aligned to 64 bytes); a
   slowdown counts only if it persists across both, a gain is quoted as the
   range the two give.
3. **The control's spread with every ABBA.** Each ABBA runs an
   identical-binary control beside the two builds (the shared harness does,
   `vm-tools/abba/abba.py`) and its result is quoted with the control's
   spread: "1.06 (identical binary 0.97-1.04, load 6)". A figure inside the
   control's spread is "no measurable change", not a gain or a loss. At the
   loads this machine runs under, the control has read +-3% to +-6.5%, so
   the many 1.02-1.08 figures the review found were inside it.

A CHANGELOG line that has only arm64 figures says "on arm64"; one with both
platforms gives both. The review's list of lines that named no platform
(`rls`, `kalman`, `ew_cov`, `ewridge` in 1.6.0) is corrected in the
CHANGELOG.
