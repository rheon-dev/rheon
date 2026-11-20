Task 479 (x86), second fix: x86-64 measurement of the branch's new tip
against main (5122b435). Hand to the GitHub runner (Linux x86-64, 4 cores).
Model, if an agent runs it: Opus or Sonnet at high effort, never Haiku.

Why: run 44794745317 (AMD EPYC 9V74) found the lanes fix (e0d95b09) slower
than main for ew_class tied below 13 features, in both layouts: k=4
1.014/1.019 and k=8 1.017/1.042 a process each, k=12 1.030/1.028 in one
process. Below 13 the only changed code is cholesky_seq.

Found on the Mac (x86-64 asm of factor_bench itself, linked for
x86_64-apple-darwin with the release profile's thin LTO, against main's):
  - the lanes tip's cholesky_seq was NOT main's code with a test added: the
    column-loop test and its call sat between the order's test and the small
    factor's jump table, which LLVM moved up behind them; faer's path (orders
    9-12 on AVX2) moved to the function's end; and the small factor's last
    store was copied into five orders' branches. Same calls, other layout.
  - the new tip calls the column loop through a #[cold] #[inline(never)]
    trampoline (cols_cold): cholesky_seq is then main's code exactly, plus
    the 8-instruction (32-byte) test on faer's path and the cold call at its
    end; every later block is where main has it, 32 bytes on. Nothing else
    on ew_class's or hmm's path differs from main.
  - moving the test out of cholesky_seq (into a caller) made the rlib's
    cholesky_seq byte-identical to main's, but NOT the linked binary's: under
    LTO its error payload stayed live and the small orders' registers were
    allocated differently (985 lines differ). So the cold trampoline it is.

Variant
  new   56480f5e       main's cholesky_seq code plus the test, the column
                       loop's call laid out at the end (cold); bits main's.

What it checks
  1. Bits: every factor_bench case's digest equal to main's, both layouts
     (bits_<layout>.txt: PASS). A FAIL is a bug: report it first.
  2. Speed, new / main, both layouts, identical-binary control:
     ab_<layout>.txt, one process per case: ew_class tied and full at
     k=4/8/12, hmm k=13/20; seq_<layout>.txt, every hmm and ew_class case in
     one process, as the first run timed them.

Commands (Linux x86-64; bash >= 4, git, python3, nm, cargo/rustc stable on
PATH; nothing else running):
  uptime
  cp -r <this folder> ~/w479x2-harness
  bash ~/w479x2-harness/run_x86.sh ~/w479x2 8
  # Builds: ONE target folder; Polars compiles once (to bitcode, the release
  # profile is thin LTO), and each of the four links (main and new, two
  # layouts) generates all code; the aligned layout passes
  # -C llvm-args=-align-all-functions=6 to the final crate only (cargo rustc),
  # which aligns every function since all code is generated at that link:
  # the script checks every Rust function is 64-byte aligned
  # (alignment.txt). ~12-15 min to build Polars at 4 jobs, ~3-4 min a link.
  # Timing at 8 rounds: per-case ~4 min a layout, seq ~4 min a layout.
  # Expected total: 35-40 min.
  # Options: W479_SEQ=0 skips the seq timing; W479_SKIP_BUILD=1 reruns the
  # timing on the binaries built; W479_NEW=<sha> another tip.

Report back (paste as they are), from ~/w479x2/out/:
  machine.txt, load_before.txt, load_after_builds.txt, load_after.txt,
  alignment.txt, bits_default.txt, bits_aligned.txt,
  ab_default.txt, ab_aligned.txt, seq_default.txt, seq_aligned.txt

Reading it: A is main, B the new tip; B/A under 1 is faster. A difference
counts only if it is outside the control's range in both layouts.
  - ew_class tied and full k=4/8/12: level with main (within the control)
    means the earlier loss was cholesky_seq's layout (the moved jump table,
    the copied stores, faer's path at the end), now gone. Still above the
    control in both layouts means the 32-byte shift of the small factor's
    blocks, or something outside cholesky_seq; the next test would be main
    with 32 bytes of padding in that place.
  - hmm k=13/20: about 0.98 and 0.84-0.86 as run 44794745317 found; the
    cold call costs one jump a factorization.

Clean up after:
  rm -rf ~/w479x2 ~/w479x2-harness
