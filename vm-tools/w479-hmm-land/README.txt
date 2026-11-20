Task 479, landing: hmm's factor state from 13 features, x86-64 measurement
(model: Opus or Sonnet at high effort, never Haiku)

What it is
  Branch task-479-hmm-land: hmm with full covariance keeps each state's
  Cholesky factor as its state, M = sigma*L*L^T with sigma in [1/4, 1], and
  moves it by plane rotations on each learned row, from 13 features; below
  13 the states keep co-moments and nothing changes. The prototype's x86
  run (prototype/) found 0.91x at k=13, 0.79x at k=20, 0.535x at k=32 and
  1.05-1.07x slower at k=4 and 8. This run times the landed form, default
  build, against main.

What it checks
  1. digests.txt: k=8 must be SAME as main (a MOVED there is a bug: report
     it first); k=13, 20, 32 move by design ("moved (expected)").
  2. Speed, land / base, hmm 2 states full covariance at k = 8, 13, 20, 32,
     both function layouts, with ab.py's identical-binary control ("ctl"
     columns). A difference counts only beyond the control's range and in
     both layouts.

Commands (a Linux x86-64 runner, 4 cores; bash, git, python3, cargo stable)
  uptime
  bash run_x86.sh 6
  # BRANCH=... BASE=... REPO=... WORK=... JOBS=... override the defaults
  # (task-479-hmm-land, main, github.com/rheon-dev/rheon, ~/w479-land, 2).

Report back (paste the files as they are)
  ~/w479-land/out/machine.txt, shas.txt, load_before.txt, load_after.txt
  ~/w479-land/out/digests.txt
  ~/w479-land/out/ab_default.txt, ab_aligned.txt

Clean up after: rm -rf ~/w479-land
