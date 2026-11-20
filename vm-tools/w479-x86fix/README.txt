Task 479 (x86 regressions), x86-64 measurement of the variants on
task-479-x86-regress against main (5122b435).
Hand to the runner (GitHub Actions, Linux x86-64) or Rheon A. Model: Opus or
Sonnet at high effort, never Haiku.

Why: run 44765892752 (AMD EPYC 7763, both layouts) found the branch (tip
0cd9e104) slower than main at hmm k=13/16/32 (1.02-1.08) while k=20 was
0.90-0.93, and ew_class tied k=4 at 1.07-1.09.

Found on the Mac (asm of the x86-64 build, rheon-core alone, -C debuginfo=0):
  - the column loop (llt_cols, orders 13..64) compiled on x86 to SCALAR fused
    multiply-adds (vfnmadd231sd): a batch of 8 rows = 8 dependent chains,
    4 cycles (one FMA latency) per earlier column per batch, and every row
    past the last full batch a chain of its own, run one after another
    (~2,150 dependent steps a factorization at 32 features). An Apple core
    hides the leftover rows' chains in its out-of-order window; an EPYC's
    smaller window only partly, so the loop's speed depended on what ran
    around it (factor alone: faster than faer from 16; inside hmm: slower).
  - ew_class tied k=4 never reaches the column loop: its only changed code
    is cholesky_seq's block layout (the same instructions on its path), and
    in the first run every case ran in ONE process, so the heap a case got
    depended on every case before it (main allocates faer's scratch at
    k >= 13, the branch does not). This run times each case in a process
    of its own, and also as the first run did (seq_*), to tell the two apart.

Variants (commits on task-479-x86-regress):
  tip    0cd9e104  the branch as the first run measured it
  lanes  e0d95b09  the column loop in vector lanes: up to 8 ymm sums held
                   across every earlier column, loads folded into
                   vfnmadd231pd, the last vector moved up to the column's end
                   (no row-at-a-time tail); same products, same order, same
                   bits. (The commits after it change arm64 and docs only.)

What it checks
  1. Bits: every variant's digest equals main's on every case
     (bits_<layout>.txt ends in PASS). A FAIL is a bug: report it first.
  2. Speed, variant / main, both layouts, identical-binary control:
     ab_<variant>_<layout>.txt, one process per case: hmm k=13/16/20/32,
     ew_class tied k=4/8;
     seq_<variant>_<layout>.txt, one process over every hmm and ew_class
     case, as the first run timed them.

Commands (Linux x86-64, 4 cores; bash >= 4, git, python3, cargo/rustc stable
on PATH; nothing else running):
  uptime
  cp -r <this folder> ~/w479x-harness
  bash ~/w479x-harness/run_x86.sh ~/w479x 6
  # clones https://github.com/rheon-dev/rheon.git (RHEON_REPO to change),
  # checks the variants' shas are on origin/task-479-x86-regress.
  # Builds: one target folder a layout shared by main and both variants, so
  # Polars builds twice (once a layout), ~30-40 min at 4 jobs; each further
  # build recompiles only the rheon crates.
  # Timing at 6 rounds: per-case ~1 min a case and variant and layout
  # (~25 min), seq ~4 min a variant and layout (~16 min).
  # Options: W479_SEQ=0 skips the seq timing; W479_XOVER=1 also times
  # factor_crossover (the factorization alone) for main and each variant;
  # W479_CASES="ew_cov k=32 mahal;bocpd k=16 full" adds cases;
  # W479_VARIANTS="lanes=e0d95b09" runs a subset;
  # W479_SKIP_BUILD=1 reruns the timing on the binaries already built.

Report back (paste the files as they are), from ~/w479x/out/:
  machine.txt, load_before.txt, load_after_builds.txt, load_after.txt
  bits_default.txt, bits_aligned.txt
  ab_tip_default.txt, ab_tip_aligned.txt, ab_lanes_default.txt, ab_lanes_aligned.txt
  seq_tip_default.txt, seq_tip_aligned.txt, seq_lanes_default.txt, seq_lanes_aligned.txt
  (and xover_*.txt if W479_XOVER=1)

Reading it: A is main, B the variant; B/A under 1 is faster. A difference
counts only if it is outside the control's range in both layouts.
  - lanes, hmm k=13..32: expected clearly under 1 at every k (the factor is
    most of a row, and the lanes loop is latency-bound at 4 cycles a
    column for up to 32 rows at once, where the tip took 4 a column for
    each 8-row batch plus 4 for each leftover row).
  - ew_class tied k=4: tip level in ab_tip_* but slow in seq_tip_* means the
    first run's loss was the heap history of one process, not the code;
    slow in both means code placement (cholesky_seq's layout), and a
    relayout variant is next.
  - lanes against tip at k <= 12 and ew_class tied: no code there differs
    (on x86), so any gap is placement or heap.

Clean up after:
  rm -rf ~/w479x ~/w479x-harness
