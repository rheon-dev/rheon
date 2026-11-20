Task 479 (lasso split sweep), x86-64 measurement of the final branch
Hand to the runner (or Rheon A). Model: Opus or Sonnet at high effort, never Haiku.

The user's decision (2026-11-20): "Adopt after x86 check". The first x86
check (run 44765892752, the prototype at 3dbcb453) found the split sweep
slower at 20 features (where it did not even run: 1.025 to 1.040), at 32 and
48 with one penalty (1.06, 1.01) and at 128 with one (1.02 to 1.03). Each
was the prototype's check of the correlation matrix for a NaN or an
infinity: a second, early-exit pass over all k*k entries every solve, at
every width (twice from 64). The final branch reads it only when a sweep
first meets a zero coefficient, in one pass that runs as vectors. This run
checks the final branch against main.

What it checks
  1. Bits: below 32 features the branch runs main's sweep, so its output
     digest must equal main's (k = 8, 20, 24, 31; 1 and 3 penalties).
     bits_<layout>.txt ends in PASS or FAIL. A FAIL is a bug: report it
     first. From 32 features "moved" is expected.
  2. Speed, branch / main, both layouts, with an identical-binary control
     (ab.py's "ctl" columns), lasso at k = 8, 20, 24, 31 (below the floor:
     must be level), 32, 40, 48 (the floor and just above it), 64, 96, 128,
     200, 500, 1 and 3 penalties. On arm64 (Apple M4 Pro, both layouts)
     the branch takes 0.975-1.010 of main's time below 32 (each inside its
     control), 0.78-0.89 at 32 and 40, 0.61-0.81 from 48 to 128, 0.48 / 0.68-0.70
     at 200 and 0.41 / 0.61-0.63 at 500 (3 / 1 penalties).

Files here
  run_x86.sh   the whole run: clone, four builds, bits check, timing
  ab.py        ABBA timer (standard library only)

Commands (Linux x86-64, 4 cores; bash, git, python3, cargo/rustc stable >= 1.95
on PATH; nothing else running):
  uptime
  cp -r <this folder> ~/w479f-harness
  bash ~/w479f-harness/run_x86.sh ~/w479f 6
  # clones https://github.com/rheon-dev/rheon.git, checks out main at
  # c34869bc and branch task-479-lasso-split at the final commit (the
  # script's FINAL_SHA; it stops if the pushed branch does not hold it).
  # Builds with CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2: ~30-40 min for the
  # first tree's two layouts' dependencies, minutes for the rest.
  # Timing: ~25-30 min a layout at 6 rounds (24 cases).
  # If the timing is interrupted, rerun it without rebuilding:
  #   W479_SKIP_BUILD=1 bash ~/w479f-harness/run_x86.sh ~/w479f 6

Report back (paste the files as they are), from ~/w479f/out/:
  machine.txt
  load_before.txt, load_after_builds.txt, load_after.txt
  bits_default.txt, bits_aligned.txt
  ab_default.txt, ab_aligned.txt

Reading it: A is main, B the branch; B/A under 1 is faster. A difference
counts only if it is outside the control's range in both layouts.
  - Below 32 (k = 8, 20, 24, 31): B/A must be level with the control. A
    slowdown there in both layouts is a bug in the switch: report it.
  - From 32: every case should be at or under 1. If k=32 (or 40) is over
    the control in both layouts, with 1 penalty or 3, report which: the
    floor (`SPLIT_MIN_K` in crates/rheon-core/src/lasso/split.rs) moves up
    to the first width that is not, for every platform, and the CHANGELOG
    entry's x86 figures are replaced with this run's.

Clean up after:
  rm -rf ~/w479f ~/w479f-harness
  (the clone, both target folders, about 3-4 GB, and the binaries)
