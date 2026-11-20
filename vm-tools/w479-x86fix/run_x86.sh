#!/usr/bin/env bash
# Task 479 (x86 regressions): main against each variant on branch
# task-479-x86-regress, in both function layouts, each case in a process of
# its own, with ab.py's identical-binary control.
#
# Usage (bash, git, python3, cargo on PATH; nothing else running):
#
#   bash run_x86.sh [WORKDIR] [ROUNDS]
#
#   WORKDIR  where the clone, the builds and the results go (default ~/w479x)
#   ROUNDS   ABBA rounds per case (default 6)
#
# Environment (all optional):
#   W479_VARIANTS  "name=sha ..." (default: the three below)
#   W479_CASES     extra case names, ';'-separated, appended to the list
#   W479_SEQ=0     skip the second timing: every variant in one process
#                  over every hmm and ew_class case, as the first x86 run
#                  timed them (each case after the cases before it),
#                  seq_<variant>_<layout>.txt
#   W479_XOVER=1   also build and run factor_crossover for main and each
#                  variant (the factorization alone, k = 2..64)
#   W479_SKIP_BUILD=1  reuse the binaries already in WORKDIR/out
#   RHEON_REPO     the repository to clone
#
# What it does:
#   1. clones the repository (main and task-479-x86-regress) into
#      WORKDIR/rheon, checks every variant's sha is on the branch;
#   2. builds crates/rheon-bench/examples/factor_bench from main (the
#      branch's factor_bench.rs copied in: public API only) and from each
#      variant, in the default layout and with
#      RUSTFLAGS="-C llvm-args=-align-all-functions=6", one target folder a
#      layout shared by every build (Polars builds once a layout; each
#      later build recompiles only the rheon crates);
#   3. bits: each case's output digest, main against each variant
#      (bits_<layout>.txt, PASS or FAIL): every variant must be main's;
#   4. speed: ab.py, one invocation (so one process a binary) per case,
#      A main, B the variant: ab_<variant>_<layout>.txt.
set -euo pipefail
work=${1:-$HOME/w479x}
rounds=${2:-6}
main_sha=5122b435
branch=task-479-x86-regress
# The variants, oldest first (each a commit on the branch):
#   tip       0cd9e104, the branch as run 44765892752 measured it, timed
#             here one case a process as well as in one (does a loss
#             survive without the other cases' allocations before it?)
#   lanes     e0d95b09, the column loop in vector lanes: up to 8 vectors
#             of 4 rows held at once, the last moved up to the column's
#             end (no row-at-a-time tail)
variants=${W479_VARIANTS:-"tip=0cd9e104 lanes=e0d95b09"}
cases=(
  "hmm k=13 K=2 full"
  "hmm k=16 K=2 full"
  "hmm k=20 K=2 full"
  "hmm k=32 K=2 full"
  "ew_class k=4 tied"
  "ew_class k=8 tied"
)
if [ -n "${W479_CASES:-}" ]; then
  IFS=';' read -r -a extra <<< "$W479_CASES"
  cases+=("${extra[@]}")
fi
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
work=$(cd "$work" && pwd)
out=$work/out
mkdir -p "$out"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}

repo_url=${RHEON_REPO:-https://github.com/rheon-dev/rheon.git}
if [ ! -d "$work/rheon/.git" ]; then
  git clone --no-single-branch "$repo_url" "$work/rheon"
fi
tree=$work/rheon
git -C "$tree" fetch -q origin main "$branch"
git -C "$tree" merge-base --is-ancestor "$main_sha" origin/main ||
  { echo "origin/main does not hold $main_sha" >&2; exit 1; }
names=()
declare -A sha_of
for v in $variants; do
  name=${v%%=*}; sha=${v#*=}
  git -C "$tree" merge-base --is-ancestor "$sha" "origin/$branch" ||
    { echo "origin/$branch does not hold $name=$sha" >&2; exit 1; }
  sha_of[$name]=$(git -C "$tree" rev-parse "$sha")
  names+=("$name")
done
main_sha=$(git -C "$tree" rev-parse "$main_sha")
tip=$(git -C "$tree" rev-parse "origin/$branch")
bench=crates/rheon-bench/examples/factor_bench.rs
xover=crates/rheon-core/examples/factor_crossover.rs
git -C "$tree" show "$tip:$bench" > "$work/factor_bench.rs"
git -C "$tree" show "$tip:$xover" > "$work/factor_crossover.rs"

{
  echo "main $main_sha"
  for n in "${names[@]}"; do echo "$n ${sha_of[$n]}"; done
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
  if [ "${W479_XOVER:-0}" = 1 ]; then
    (cd "$tree" && RUSTFLAGS="$flags" CARGO_TARGET_DIR="$target" \
      cargo build --release -q -p rheon-core --example factor_crossover)
    cp "$target/release/examples/factor_crossover$exe" "$out/$name.xover.$layout$exe"
  fi
}
checkout() { # sha; main gets the branch's harnesses copied in
  rm -f "$tree/$bench" "$tree/$xover"
  git -C "$tree" checkout -q -f --detach "$1"
  [ -f "$tree/$bench" ] || cp "$work/factor_bench.rs" "$tree/$bench"
  [ -f "$tree/$xover" ] || cp "$work/factor_crossover.rs" "$tree/$xover"
}
if [ "${W479_SKIP_BUILD:-0}" = 1 ]; then
  [ -f "$out/main.default.exe" ] && exe=.exe
else
  for layout in default aligned; do
    checkout "$main_sha"
    echo "building main, $layout" >&2
    build main "$layout"
    for n in "${names[@]}"; do
      checkout "${sha_of[$n]}"
      echo "building $n, $layout" >&2
      build "$n" "$layout"
      if cmp -s "$out/main.$layout$exe" "$out/$n.$layout$exe"; then
        echo "$n built main's binary ($layout)" >&2
        exit 1
      fi
    done
  done
  checkout "$tip"
fi
uptime | tee "$out/load_after_builds.txt"

# Bits: one pass of each case, each variant's digest against main's.
digest() { # binary case
  "$1" 1 "$2" | grep -F "$2" | sed -E 's/.*digest ([0-9a-f]+)$/\1/'
}
for layout in default aligned; do
  f=$out/bits_$layout.txt
  : > "$f"
  fail=0
  for c in "${cases[@]}" "ew_cov k=16 mahal" "ew_cov k=32 mahal" "ew_class k=20 full" \
           "bocpd k=16 full max_run=200"; do
    a=$(digest "$out/main.$layout$exe" "$c")
    for n in "${names[@]}"; do
      b=$(digest "$out/$n.$layout$exe" "$c")
      v=same; [ "$a" = "$b" ] || { v=MOVED; fail=1; }
      printf '%-30s %-9s main %s variant %s  %s\n' "$c" "$n" "$a" "$b" "$v" >> "$f"
    done
  done
  if [ $fail = 0 ]; then echo "PASS: every variant is main to the bit" >> "$f"
  else echo "FAIL: a variant moved a bit" >> "$f"; fi
  cat "$f"
done

# Speed: one ab.py invocation per case and variant (A main, B the variant).
for layout in default aligned; do
  for n in "${names[@]}"; do
    f=$out/ab_${n}_$layout.txt
    : > "$f"
    first=1
    for c in "${cases[@]}"; do
      echo "timing $layout, $n: $c" >&2
      res=$(python3 "$here/ab.py" --a "$out/main.$layout$exe" --b "$out/$n.$layout$exe" \
        --rounds "$rounds" --reps 5 "$c" 2>/dev/null)
      if [ $first = 1 ]; then echo "$res" | head -1 >> "$f"; first=0; fi
      echo "$res" | tail -n +2 >> "$f"
    done
    uptime >> "$f"
    cat "$f"
  done
done
if [ "${W479_SEQ:-1}" = 1 ]; then
  for layout in default aligned; do
    for n in "${names[@]}"; do
      f=$out/seq_${n}_$layout.txt
      echo "timing $layout, $n, one process over hmm and ew_class" >&2
      python3 "$here/ab.py" --a "$out/main.$layout$exe" --b "$out/$n.$layout$exe" \
        --rounds "$rounds" --reps 5 hmm ew_class > "$f" 2>/dev/null
      uptime >> "$f"
      cat "$f"
    done
  done
fi
if [ "${W479_XOVER:-0}" = 1 ]; then
  for layout in default aligned; do
    for n in main "${names[@]}"; do
      "$out/$n.xover.$layout$exe" 7 > "$out/xover_${n}_$layout.txt"
    done
  done
fi
uptime | tee "$out/load_after.txt"
echo "done: $out"
