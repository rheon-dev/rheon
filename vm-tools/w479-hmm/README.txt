Task 479, hmm factor-as-state prototype: x86-64 measurement
(hand to the VM; model: Opus or Sonnet at high effort, never Haiku)

What it is
  The prototype keeps each hmm state's Cholesky factor L as the state, in
  place of the co-moments, and moves L by a rank-one rotation on each learned
  row instead of updating the co-moments and refactorizing. It is a Cargo
  feature of rheon-core, `hmm-factor-state`, off by default; a second
  feature, `hmm-factor-scaled`, keeps M = s*L*L^T so the decay's factor goes
  into the scalar s rather than into every entry of L. On arm64 both were
  timed against the 479 branch (0cd9e104); this run is the x86-64 side.

What it checks
  1. The default build is unchanged on x86: digests.txt says "SAME" for the
     default build at every k. A "MOVED" there is a bug: report it first.
     (The prototypes' digests move; that is expected.)
  2. Speed, each prototype / base, hmm with 2 states and full covariance at
     k = 4, 8, 13, 20, 32, in both function layouts, with ab.py's identical-
     binary control ("ctl" columns). A difference counts only beyond the
     control's range and in both layouts.

Commands (a cloud Linux x86-64 VM, 4 cores; bash, git, python3, cargo stable)
  uptime
  cp -r <this folder> ~/w479-hmm-x86
  bash ~/w479-hmm-x86/run_x86.sh 6
  # clones https://github.com/rheon-dev/rheon.git into ~/w479-hmm/rheon
  # (WORK=... to put it elsewhere), builds seven binaries with
  # CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 (about 50-70 min), then times
  # (about 10 min; each timed run a few seconds). Run nothing else meanwhile.
  # If the VM restarts during the builds, run it again: the worktrees and
  # their target folders are reused.

Report back (paste the files as they are)
  ~/w479-hmm/out/machine.txt, branch_sha.txt, load_before.txt, load_after.txt
  ~/w479-hmm/out/digests.txt
  ~/w479-hmm/out/ab_proto_default.txt, ab_proto_aligned.txt,
  ab_scaled_default.txt, ab_scaled_aligned.txt

Clean up after: rm -rf ~/w479-hmm ~/w479-hmm-x86
