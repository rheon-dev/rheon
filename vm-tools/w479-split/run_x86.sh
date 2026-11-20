#!/usr/bin/env bash
# Task 479 (lasso split sweep), x86-64 measurement: lasso's current
# coordinate descent against the split sweep (build feature
# `rheon-core/lasso-split`), in both function layouts, on one checkout of
# branch task-479-lasso-split.
#
# Usage (bash, git, python3, cargo on PATH; nothing else running):
#
#   bash run_x86.sh [WORKDIR] [ROUNDS] [SHA]
#
#   WORKDIR  where the clone, the builds and the results go (default ~/w479s)
#   ROUNDS   ABBA rounds per case (default 6)
#   SHA      the commit to measure (default 3dbcb453, the branch with the
#            feature; the run stops if the branch does not hold it)
#
# What it does:
#   1. clones https://github.com/rheon-dev/rheon.git (branch
#      task-479-lasso-split) into WORKDIR/rheon, checks out SHA;
#   2. builds crates/rheon-bench/examples/factor_bench four times, with
#      CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2: current and split, in the
#      default layout and with RUSTFLAGS="-C llvm-args=-align-all-functions=6",
#      one target folder per layout (the split build reuses its dependencies);
#   3. bits: below 32 features the split build must run the current sweep,
#      so its digest equals the current one (k = 8 and 20, 1 and 3 penalties);
#      from 32 it is expected to differ. Written to bits_<layout>.txt with a
#      PASS or FAIL line;
#   4. speed: ab.py (ABBA with an identical-binary control, all three from
#      paths of the same length) per case, one case per invocation, so each
#      timed run is one case's reps: k = 20, 32, 48, 64, 128, 200, 500 at 1 and
#      3 penalties. Written to ab_<layout>.txt.
#
# Each timed run (one binary, one case, reps + 1 passes) is a few seconds; the
# longest, k=500 with 3 penalties at 3 reps, about 4 x 2,000 rows at ~2 ms a
# row on a cloud x86 core: under 30 s. Builds: about 30-40 min a layout at 2
# jobs (Polars); timing: about 15-20 min a layout at 6 rounds.
set -euo pipefail
work=${1:-$HOME/w479s}
rounds=${2:-6}
sha=${3:-3dbcb453}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
work=$(cd "$work" && pwd)
out=$work/out
mkdir -p "$out"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2

repo_url=${RHEON_REPO:-https://github.com/rheon-dev/rheon.git}
if [ ! -d "$work/rheon/.git" ]; then
  git clone --branch task-479-lasso-split --single-branch "$repo_url" "$work/rheon"
else
  git -C "$work/rheon" fetch origin task-479-lasso-split
fi
git -C "$work/rheon" checkout -q --detach "$sha"
tree=$work/rheon

{
  echo "commit $(git -C "$tree" rev-parse HEAD)"
  uname -a
  (lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
  rustc -V; cargo -V; python3 -V
  nproc 2>/dev/null || true
} > "$out/machine.txt" 2>&1
uptime | tee "$out/load_before.txt"

exe=""
build() { # layout
  local layout=$1 flags=""
  [ "$layout" = aligned ] && flags="-C llvm-args=-align-all-functions=6"
  local target=$work/target-$layout
  (cd "$tree" && RUSTFLAGS="$flags" CARGO_TARGET_DIR="$target" \
    cargo build --release -q -p rheon-bench --example factor_bench)
  [ -f "$target/release/examples/factor_bench.exe" ] && exe=.exe
  cp "$target/release/examples/factor_bench$exe" "$out/current.$layout$exe"
  (cd "$tree" && RUSTFLAGS="$flags" CARGO_TARGET_DIR="$target" \
    cargo build --release -q -p rheon-core -p rheon-bench --example factor_bench \
      --features rheon-core/lasso-split)
  cp "$target/release/examples/factor_bench$exe" "$out/split.$layout$exe"
  if cmp -s "$out/current.$layout$exe" "$out/split.$layout$exe"; then
    echo "the split build is the current binary ($layout): the feature did not apply" >&2
    exit 1
  fi
}
# W479_SKIP_BUILD=1 reuses the four binaries already in WORKDIR/out (a rerun
# of the timing after an interrupted one).
if [ "${W479_SKIP_BUILD:-0}" = 1 ]; then
  [ -f "$out/current.default.exe" ] && exe=.exe
else
  for layout in default aligned; do
    echo "building $layout" >&2
    build "$layout"
  done
fi
uptime | tee "$out/load_after_builds.txt"

# Bits: one pass of each case, its digest from each build.
digest() { # binary case
  "$1" 1 "$2" | grep -F "$2" | sed -E 's/.*digest ([0-9a-f]+)$/\1/'
}
for layout in default aligned; do
  f=$out/bits_$layout.txt
  : > "$f"
  fail=0
  for c in "lasso x k=8 1 penalty" "lasso x k=8 3 penalties" \
           "lasso k=20 1 penalty" "lasso k=20 3 penalties" \
           "lasso x k=32 1 penalty" "lasso x k=32 3 penalties" \
           "lasso k=500 3 penalties"; do
    a=$(digest "$out/current.$layout$exe" "$c")
    b=$(digest "$out/split.$layout$exe" "$c")
    k=$(echo "$c" | sed -E 's/.*k=([0-9]+).*/\1/')
    if [ "$a" = "$b" ]; then verdict=same; else verdict=moved; fi
    want=same
    [ "$k" -ge 32 ] && want=moved
    ok=ok
    if [ "$verdict" != "$want" ]; then ok=UNEXPECTED; [ "$want" = same ] && fail=1; fi
    printf '%-26s current %s split %s  %s (expected %s) %s\n' "$c" "$a" "$b" "$verdict" "$want" "$ok" >> "$f"
  done
  if [ $fail = 0 ]; then echo "PASS: below 32 features the split build is the current one to the bit" >> "$f"
  else echo "FAIL: a case below 32 features moved" >> "$f"; fi
  cat "$f"
done

# Speed: one ab.py invocation per case.
cases=(
  "lasso k=20 1 penalty:5" "lasso k=20 3 penalties:5"
  "lasso x k=32 1 penalty:5" "lasso x k=32 3 penalties:5"
  "lasso x k=48 1 penalty:5" "lasso x k=48 3 penalties:5"
  "lasso x k=64 1 penalty:5" "lasso x k=64 3 penalties:5"
  "lasso x k=128 1 penalty:5" "lasso x k=128 3 penalties:5"
  "lasso k=200 1 penalty:3" "lasso k=200 3 penalties:3"
  "lasso k=500 1 penalty:3" "lasso k=500 3 penalties:3"
)
for layout in default aligned; do
  f=$out/ab_$layout.txt
  : > "$f"
  first=1
  for entry in "${cases[@]}"; do
    c=${entry%:*}; reps=${entry##*:}
    echo "timing $layout: $c" >&2
    res=$(python3 "$here/ab.py" --a "$out/current.$layout$exe" --b "$out/split.$layout$exe" \
      --rounds "$rounds" --reps "$reps" "$c" 2>/dev/null)
    if [ $first = 1 ]; then echo "$res" | head -1 >> "$f"; first=0; fi
    echo "$res" | tail -n +2 >> "$f"
  done
  uptime >> "$f"
  cat "$f"
done
uptime | tee "$out/load_after.txt"
echo "done: $out"
