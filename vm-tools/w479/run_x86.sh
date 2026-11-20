#!/usr/bin/env bash
# Task 479's x86-64 measurement: main against the branch, in both function
# layouts, through two harnesses that use only the crates' public API:
#
#   crates/rheon-core/examples/factor_crossover.rs  a fresh factorization of
#       a k x k matrix (SpdFactor::of) against one rank-one move of a kept
#       factor (SpdFactor::updated), k = 2..64: the order from which the
#       move is the cheaper, and the factorization's own time at each order
#       (the branch factorizes orders 13..64 by its own column loop, faer's
#       bits; orders <= 12 are unchanged).
#   crates/rheon-bench/examples/factor_bench.rs  hmm, ew_class (full, tied),
#       ew_cov mahal, bocpd (full) and lasso through AnyModel::step_into, with
#       an FNV digest of every output bit (main's and the branch's must agree:
#       every change on the branch is bit-identical).
#
# Usage, from a clone of the repository (any branch checked out, clean):
#
#   bash run_x86.sh /path/to/task479.bundle  [rounds]
#
# It makes two worktrees beside the clone (main at 5122b435, the branch from
# the bundle), copies both harnesses into main's, builds each in both layouts
# (default, and RUSTFLAGS="-C llvm-args=-align-all-functions=6") into its own
# target folder, and runs ab.py (ABBA with an identical-binary control) for
# each layout, writing everything under ./w479-x86-out/. Takes about 40
# minutes on 4 cores (four release builds of rheon-bench, which builds
# Polars), then about 15 of timing. Run nothing else meanwhile; note the
# load (uptime) before and after.
set -euo pipefail
bundle=${1:?the bundle path}
rounds=${2:-6}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git rev-parse --show-toplevel)
out=$repo/w479-x86-out
mkdir -p "$out"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
uname -a > "$out/machine.txt"
(lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true) >> "$out/machine.txt"
uptime | tee "$out/load_before.txt"

git -C "$repo" fetch "$bundle" task-479-per-row-factorizations:w479-branch
git -C "$repo" worktree add -f "$repo/../w479-main" 5122b435
git -C "$repo" worktree add -f "$repo/../w479-branch" w479-branch
main=$(cd "$repo/../w479-main" && pwd)
branch=$(cd "$repo/../w479-branch" && pwd)
cp "$branch/crates/rheon-bench/examples/factor_bench.rs" "$main/crates/rheon-bench/examples/"
cp "$branch/crates/rheon-core/examples/factor_crossover.rs" "$main/crates/rheon-core/examples/"

build() { # tree label layout
  local tree=$1 label=$2 layout=$3 flags=""
  [ "$layout" = aligned ] && flags="-C llvm-args=-align-all-functions=6"
  (cd "$tree" && RUSTFLAGS="$flags" CARGO_TARGET_DIR="$tree/target-$layout" \
    cargo build --release -q -p rheon-bench --example factor_bench &&
    RUSTFLAGS="$flags" CARGO_TARGET_DIR="$tree/target-$layout" \
    cargo build --release -q -p rheon-core --example factor_crossover)
  # `.exe` on Windows (Git Bash): ab.py runs the copies by path.
  local exe=""
  [ -f "$tree/target-$layout/release/examples/factor_bench.exe" ] && exe=.exe
  cp "$tree/target-$layout/release/examples/factor_bench$exe" "$out/$label.$layout$exe"
  cp "$tree/target-$layout/release/examples/factor_crossover$exe" "$out/$label.xover.$layout$exe"
  echo "$exe" > "$out/exe"
}
for layout in default aligned; do
  build "$main" main $layout
  build "$branch" branch $layout
done

exe=$(cat "$out/exe")
for layout in default aligned; do
  for who in main branch; do
    "$out/$who.xover.$layout$exe" 7 > "$out/xover_${who}_$layout.txt"
  done
  python3 "$here/ab.py" --a "$out/main.$layout$exe" --b "$out/branch.$layout$exe" \
    --rounds "$rounds" --reps 5 > "$out/ab_$layout.txt"
done
uptime | tee "$out/load_after.txt"
echo "done: $out"
