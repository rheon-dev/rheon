Task 479, x86-64 measurement (hand to the VM; model: Opus or Sonnet at high effort, never Haiku)

What it checks
  1. Every change on branch task-479-per-row-factorizations is bit-identical
     to main (5122b435): ab.py prints "same" in the bits column for every case.
     A "moved" anywhere is a bug: report it first.
  2. Speed, branch / main, both function layouts, with an identical-binary
     control (ab.py's "ctl" columns). The branch:
       - ew_cov mahal: factor and solve in a thread's kept working space (all k);
       - Cholesky of order 13..64 by an own column loop (faer's bits). On an
         AVX2 CPU orders 9..12 still take faer's call, so k = 13, 16, 20, 32
         are the cases to read: hmm, ew_class full/tied, ew_cov mahal, bocpd k=16;
       - lasso: the correlation matrix written over a kept buffer (k=500).
  3. The crossover order on x86 of a rank-one factor move (SpdFactor::updated)
     against a fresh factorization: factor_crossover's last column
     (move / (fresh - alloc)) per k, main and branch builds. Under 1 means the
     move is cheaper. (The rank-one move is NOT on the branch; this is the
     study's number.)

Files here
  run_x86.sh       the whole run (builds four binaries, times, writes w479-x86-out/)
  ab.py            ABBA timer (standard library only)
  task479.bundle   the branch's commits on top of 5122b435

Commands (from a clean clone of the repo that has commit 5122b435; bash, python3, cargo):
  uptime
  cp -r <this folder> /tmp/w479-x86      # or anywhere outside the clone
  cd <clone>
  bash /tmp/w479-x86/run_x86.sh /tmp/w479-x86/task479.bundle 6
  # ~40 min of builds (4 cores), ~15 min of timing; nothing else running.

Report back (paste the files as they are)
  w479-x86-out/machine.txt, load_before.txt, load_after.txt
  w479-x86-out/ab_default.txt, ab_aligned.txt
  w479-x86-out/xover_main_default.txt, xover_branch_default.txt,
  xover_main_aligned.txt, xover_branch_aligned.txt

Clean up after: git worktree remove ../w479-main ../w479-branch (with --force),
git branch -D w479-branch, rm -rf w479-x86-out.

Reading it: a slowdown counts only if it shows in both layouts beyond the
control's range. If k=13..32 is slower on x86 in both layouts, the column
loop's lower bound on x86 should move up (solve.rs, cholesky_seq: the
`n > SMALL` test); that is the decision this run feeds.
