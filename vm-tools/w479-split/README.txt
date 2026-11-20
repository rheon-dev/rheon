Task 479 (lasso split sweep), x86-64 measurement
Hand to the VM (Rheon A). Model: Opus or Sonnet at high effort, never Haiku.

The user's decision (2026-11-20): "Adopt after x86 check". The split sweep
becomes lasso's descent from 32 features once x86 confirms the gain in both
function layouts. This run is that check.

What it checks
  1. Bits: below 32 features the split build runs the current sweep, so its
     output digest must equal the current build's (k = 8 and 20, 1 and 3
     penalties). bits_<layout>.txt ends in PASS or FAIL. A FAIL is a bug:
     report it first. From 32 features "moved" is expected.
  2. Speed, split / current, both layouts, with an identical-binary control
     (ab.py's "ctl" columns), lasso at k = 20, 32, 48, 64, 128, 200, 500, 1 and
     3 penalties. On arm64 (Apple M4 Pro) the split build took 0.43 of the
     time at k=500 with 3 penalties, 0.48 at 200, 0.67-0.72 from 48 to 128,
     0.85 at 32, and 1.00-1.03 at 20 (where it runs the current sweep).

Files here
  run_x86.sh   the whole run: clone, four builds, bits check, timing
  ab.py        ABBA timer (standard library only)

Commands (Linux x86-64, 4 cores; bash, git, python3, cargo/rustc stable >= 1.95
on PATH; nothing else running):
  uptime
  cp -r <this folder> ~/w479s-harness
  bash ~/w479s-harness/run_x86.sh ~/w479s 6
  # clones https://github.com/rheon-dev/rheon.git, branch task-479-lasso-split,
  # checks out 3dbcb453 (stops if the branch does not have it).
  # Builds with CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2: ~30-40 min a layout.
  # Timing: ~15-20 min a layout at 6 rounds; each timed run (one binary,
  # one case) is under ~30 s, the longest k=500 with 3 penalties.
  # If the timing is interrupted, rerun it without rebuilding:
  #   W479_SKIP_BUILD=1 bash ~/w479s-harness/run_x86.sh ~/w479s 6

Report back (paste the files as they are), from ~/w479s/out/:
  machine.txt
  load_before.txt, load_after_builds.txt, load_after.txt
  bits_default.txt, bits_aligned.txt
  ab_default.txt, ab_aligned.txt

Reading it: B is the split build, A the current one; B/A under 1 is faster.
A difference counts only if it is outside the control's range in both
layouts. The adoption goes ahead if k >= 32 is faster in both layouts; if
some width from 32 up is slower in both, report which (the floor of 32,
`SPLIT_MIN_K` in crates/rheon-core/src/lasso/split.rs, would move up).

Clean up after:
  rm -rf ~/w479s ~/w479s-harness
  (the clone, both target folders, about 3-4 GB, and the binaries)
