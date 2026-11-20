#!/usr/bin/env bash
# Task 479 (lasso split sweep), x86-64 measurement of the final branch:
# lasso on task-479-lasso-split (the split sweep is the default from 32
# features, no build feature) against lasso on main, in both function
# layouts, at the switch's floor and around it.
#
# Usage (bash, git, python3, cargo on PATH; nothing else running):
#
#   bash run_x86.sh [WORKDIR] [ROUNDS] [SHA] [MAIN_SHA]
#
#   WORKDIR   where the clone, the builds and the results go (default ~/w479f)
#   ROUNDS    ABBA rounds per case (default 6)
#   SHA       the branch commit to measure (default FINAL_SHA below; the run
#             stops if branch task-479-lasso-split does not hold it)
#   MAIN_SHA  main's commit to measure against (default c34869bc, main when
#             the branch was finished; the run stops if main does not hold it)
#
# What it does:
#   1. clones https://github.com/rheon-dev/rheon.git (main and
#      task-479-lasso-split) into WORKDIR/rheon;
#   2. builds crates/rheon-bench/examples/factor_bench from main (the
#      branch's factor_bench.rs copied in: it uses only rheon_polars' public
#      API) and from the branch, each in the default layout and with
#      RUSTFLAGS="-C llvm-args=-align-all-functions=6", one target folder per
#      layout (the second tree reuses the first's dependencies), with
#      CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2;
#   3. bits: below 32 features the branch runs main's sweep, so its digest
#      equals main's (k = 8, 20, 24, 31 at 1 and 3 penalties); from 32 it is
#      expected to differ. Written to bits_<layout>.txt with a PASS or FAIL;
#   4. speed: ab.py (ABBA with an identical-binary control, all three from
#      paths of the same length), one case per invocation: k = 8, 20, 24, 31
#      (below the floor), 32, 40, 48 (at it and just above), 64, 96, 128, 200,
#      500, each at 1 and 3 penalties. Written to ab_<layout>.txt.
#
# Builds: about 30-40 min a layout at 2 jobs for the first tree (Polars),
# minutes for the second; timing: about 25-30 min a layout at 6 rounds.
set -euo pipefail
FINAL_SHA=959590920c3f9ddbf33165be15f961fb6e6eeab3
work=${1:-$HOME/w479f}
rounds=${2:-6}
sha=${3:-$FINAL_SHA}
main_sha=${4:-c34869bc}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
work=$(cd "$work" && pwd)
out=$work/out
mkdir -p "$out"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2

repo_url=${RHEON_REPO:-https://github.com/rheon-dev/rheon.git}
if [ ! -d "$work/rheon/.git" ]; then
  git clone --no-single-branch "$repo_url" "$work/rheon"
fi
git -C "$work/rheon" fetch -q origin main task-479-lasso-split
tree=$work/rheon
holds() { # branch sha
  git -C "$tree" merge-base --is-ancestor "$2" "origin/$1" || {
    echo "origin/$1 does not hold $2" >&2; exit 1; }
}
holds task-479-lasso-split "$sha"
holds main "$main_sha"
sha=$(git -C "$tree" rev-parse "$sha")
main_sha=$(git -C "$tree" rev-parse "$main_sha")
bench=crates/rheon-bench/examples/factor_bench.rs
git -C "$tree" show "$sha:$bench" > "$work/factor_bench.rs"

{
  echo "branch commit $sha"
  echo "main commit $main_sha"
  uname -a
  (lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
  rustc -V; cargo -V; python3 -V
  nproc 2>/dev/null || true
} > "$out/machine.txt" 2>&1
uptime | tee "$out/load_before.txt"

exe=""
build() { # name layout
  local name=$1 layout=$2 flags=""
  [ "$layout" = aligned ] && flags="-C llvm-args=-align-all-functions=6"
  local target=$work/target-$layout
  (cd "$tree" && RUSTFLAGS="$flags" CARGO_TARGET_DIR="$target" \
    cargo build --release -q -p rheon-bench --example factor_bench)
  [ -f "$target/release/examples/factor_bench.exe" ] && exe=.exe
  cp "$target/release/examples/factor_bench$exe" "$out/$name.$layout$exe"
}
# W479_SKIP_BUILD=1 reuses the four binaries already in WORKDIR/out (a rerun
# of the timing after an interrupted one).
if [ "${W479_SKIP_BUILD:-0}" = 1 ]; then
  [ -f "$out/main.default.exe" ] && exe=.exe
else
  # main, with the branch's harness copied in (main has none).
  git -C "$tree" checkout -q --detach "$main_sha"
  cp "$work/factor_bench.rs" "$tree/$bench"
  for layout in default aligned; do
    echo "building main, $layout" >&2
    build main "$layout"
  done
  rm "$tree/$bench"
  git -C "$tree" checkout -q --detach "$sha"
  for layout in default aligned; do
    echo "building the branch, $layout" >&2
    build branch "$layout"
    if cmp -s "$out/main.$layout$exe" "$out/branch.$layout$exe"; then
      echo "the branch built main's binary ($layout)" >&2
      exit 1
    fi
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
           "lasso x k=24 1 penalty" "lasso x k=24 3 penalties" \
           "lasso x k=31 1 penalty" "lasso x k=31 3 penalties" \
           "lasso x k=32 1 penalty" "lasso x k=32 3 penalties" \
           "lasso k=500 3 penalties"; do
    a=$(digest "$out/main.$layout$exe" "$c")
    b=$(digest "$out/branch.$layout$exe" "$c")
    k=$(echo "$c" | sed -E 's/.*k=([0-9]+).*/\1/')
    if [ "$a" = "$b" ]; then verdict=same; else verdict=moved; fi
    want=same
    [ "$k" -ge 32 ] && want=moved
    ok=ok
    if [ "$verdict" != "$want" ]; then ok=UNEXPECTED; [ "$want" = same ] && fail=1; fi
    printf '%-26s main %s branch %s  %s (expected %s) %s\n' "$c" "$a" "$b" "$verdict" "$want" "$ok" >> "$f"
  done
  if [ $fail = 0 ]; then echo "PASS: below 32 features the branch is main to the bit" >> "$f"
  else echo "FAIL: a case below 32 features moved" >> "$f"; fi
  cat "$f"
done

# Speed: one ab.py invocation per case (A main, B the branch).
cases=(
  "lasso x k=8 1 penalty:5" "lasso x k=8 3 penalties:5"
  "lasso k=20 1 penalty:5" "lasso k=20 3 penalties:5"
  "lasso x k=24 1 penalty:5" "lasso x k=24 3 penalties:5"
  "lasso x k=31 1 penalty:5" "lasso x k=31 3 penalties:5"
  "lasso x k=32 1 penalty:5" "lasso x k=32 3 penalties:5"
  "lasso x k=40 1 penalty:5" "lasso x k=40 3 penalties:5"
  "lasso x k=48 1 penalty:5" "lasso x k=48 3 penalties:5"
  "lasso x k=64 1 penalty:5" "lasso x k=64 3 penalties:5"
  "lasso x k=96 1 penalty:5" "lasso x k=96 3 penalties:5"
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
    res=$(python3 "$here/ab.py" --a "$out/main.$layout$exe" --b "$out/branch.$layout$exe" \
      --rounds "$rounds" --reps "$reps" "$c" 2>/dev/null)
    if [ $first = 1 ]; then echo "$res" | head -1 >> "$f"; first=0; fi
    echo "$res" | tail -n +2 >> "$f"
  done
  uptime >> "$f"
  cat "$f"
done
uptime | tee "$out/load_after.txt"
echo "done: $out"
