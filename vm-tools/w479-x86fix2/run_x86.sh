#!/usr/bin/env bash
# Task 479 (x86), second fix: main against the branch's new tip on x86-64, in
# both function layouts, with ab.py's identical-binary control: ew_class
# tied and full below 13 features (where only cholesky_seq's layout had
# changed), and hmm at 13 and 20 as a check that the lanes' gain holds.
#
# Usage (bash >= 4, git, python3, cargo on PATH; nothing else running):
#
#   bash run_x86.sh [WORKDIR] [ROUNDS]
#
#   WORKDIR  where the clone, the builds and the results go (default ~/w479x2)
#   ROUNDS   ABBA rounds per case (default 8)
#
# Environment (all optional):
#   W479_NEW         the new tip's sha (default below)
#   W479_SEQ=0       skip the one-process timing (seq_<layout>.txt)
#   W479_SKIP_BUILD=1  reuse the binaries already in WORKDIR/out
#   RHEON_REPO       the repository to clone
#
# What it does:
#   1. clones the repository into WORKDIR/rheon and checks the new tip is on
#      origin/task-479-x86-regress;
#   2. builds crates/rheon-bench/examples/factor_bench from main (the
#      branch's factor_bench.rs copied in: public API only) and from the new
#      tip, ONE target folder for both and both layouts, so Polars builds
#      once: every dependency is compiled without extra flags, and the
#      example is linked by `cargo rustc`, which passes
#      `-C llvm-args=-align-all-functions=6` to the final crate only for the
#      aligned layout. The release profile is thin LTO, so every crate's code
#      is generated in that last step under the flag; the script checks that
#      the aligned binaries' Rust functions all start on a 64-byte boundary
#      (needs nm; it keeps symbols for that with CARGO_PROFILE_RELEASE_STRIP=none);
#   3. bits: every factor_bench case's digest, main against the new tip, in
#      each layout (bits_<layout>.txt, PASS or FAIL);
#   4. speed, one process per case (ab_<layout>.txt), then every hmm and
#      ew_class case in one process, as the first x86 run timed them
#      (seq_<layout>.txt).
set -euo pipefail
work=${1:-$HOME/w479x2}
rounds=${2:-8}
main_sha=5122b435
branch=task-479-x86-regress
new_sha=${W479_NEW:-56480f5e}
cases=(
  "ew_class k=4 tied"
  "ew_class k=8 tied"
  "ew_class k=12 tied"
  "ew_class k=4 full"
  "ew_class k=8 full"
  "ew_class k=12 full"
  "hmm k=13 K=2 full"
  "hmm k=20 K=2 full"
)
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
work=$(cd "$work" && pwd)
out=$work/out
mkdir -p "$out"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4} CARGO_PROFILE_RELEASE_STRIP=none
unset RUSTFLAGS

repo_url=${RHEON_REPO:-https://github.com/rheon-dev/rheon.git}
if [ ! -d "$work/rheon/.git" ]; then
  git clone --no-single-branch "$repo_url" "$work/rheon"
fi
tree=$work/rheon
git -C "$tree" fetch -q origin main "$branch"
git -C "$tree" cat-file -e "$main_sha^{commit}" ||
  { echo "the clone does not hold main's $main_sha" >&2; exit 1; }
git -C "$tree" merge-base --is-ancestor "$new_sha" "origin/$branch" ||
  { echo "origin/$branch does not hold $new_sha" >&2; exit 1; }
main_sha=$(git -C "$tree" rev-parse "$main_sha")
new_sha=$(git -C "$tree" rev-parse "$new_sha")
bench=crates/rheon-bench/examples/factor_bench.rs
git -C "$tree" show "$new_sha:$bench" > "$work/factor_bench.rs"

{
  echo "main $main_sha"
  echo "new  $new_sha"
  uname -a
  (lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
  rustc -V; cargo -V; python3 -V
  nproc 2>/dev/null || true
} > "$out/machine.txt" 2>&1
uptime | tee "$out/load_before.txt"

exe=""
target=$work/target
build() { # name layout
  local name=$1 layout=$2 align=""
  [ "$layout" = aligned ] && align=-align-all-functions=6
  if [ -n "$align" ]; then
    (cd "$tree" && CARGO_TARGET_DIR="$target" \
      cargo rustc --release -q -p rheon-bench --example factor_bench -- -C llvm-args="$align")
  else
    (cd "$tree" && CARGO_TARGET_DIR="$target" \
      cargo rustc --release -q -p rheon-bench --example factor_bench)
  fi
  if [ -f "$target/release/examples/factor_bench.exe" ]; then exe=.exe; fi
  cp "$target/release/examples/factor_bench$exe" "$out/$name.$layout$exe"
}
checkout() { # sha; main gets the branch's harness copied in
  rm -f "$tree/$bench"
  git -C "$tree" checkout -q -f --detach "$1"
  [ -f "$tree/$bench" ] || cp "$work/factor_bench.rs" "$tree/$bench"
}
aligned_check() { # binary: every Rust function (text symbol) on a 64-byte boundary
  # (compiler_builtins is linked as object code, outside LTO, so it is not counted)
  command -v nm > /dev/null || { echo "no nm: alignment not checked" >&2; return 0; }
  local bad total
  total=$(nm "$1" | awk '$2 ~ /^[tT]$/ && $3 ~ /^_?_R/ && $3 !~ /compiler_builtins/' | wc -l)
  bad=$(nm "$1" | awk '$2 ~ /^[tT]$/ && $3 ~ /^_?_R/ && $3 !~ /compiler_builtins/ {print $1}' |
    python3 -c 'import sys; print(sum(1 for l in sys.stdin if int(l, 16) % 64))')
  echo "$1: $total Rust functions, $bad not on a 64-byte boundary" | tee -a "$out/alignment.txt"
  [ "$total" -gt 1000 ] && [ "$bad" -eq 0 ] ||
    { echo "the aligned layout did not take" >&2; exit 1; }
}
if [ "${W479_SKIP_BUILD:-0}" = 1 ]; then
  [ -f "$out/main.default.exe" ] && exe=.exe
else
  : > "$out/alignment.txt"
  for sha_name in "$main_sha main" "$new_sha new"; do
    set -- $sha_name
    checkout "$1"
    for layout in default aligned; do
      echo "building $2, $layout" >&2
      build "$2" "$layout"
    done
    aligned_check "$out/$2.aligned$exe"
  done
  for layout in default aligned; do
    if cmp -s "$out/main.$layout$exe" "$out/new.$layout$exe"; then
      echo "new built main's binary ($layout)" >&2
      exit 1
    fi
  done
  cmp -s "$out/new.default$exe" "$out/new.aligned$exe" &&
    { echo "the two layouts built the same binary" >&2; exit 1; }
  checkout "$new_sha"
fi
uptime | tee "$out/load_after_builds.txt"

# Bits: every case once, new against main, in each layout.
digests() { # binary -> "case<TAB>digest" lines
  "$1" 1 | sed -nE 's/^(.*[^ ]) +best .*digest ([0-9a-f]+)$/\1\t\2/p'
}
for layout in default aligned; do
  f=$out/bits_$layout.txt
  digests "$out/main.$layout$exe" > "$out/digests_main_$layout.tsv"
  digests "$out/new.$layout$exe" > "$out/digests_new_$layout.tsv"
  n=$(wc -l < "$out/digests_main_$layout.tsv" | tr -d " ")
  if [ "$n" -ge 40 ] && cmp -s "$out/digests_main_$layout.tsv" "$out/digests_new_$layout.tsv"; then
    echo "PASS: $n cases, every digest main's" > "$f"
  else
    { echo "FAIL ($n cases):"; diff "$out/digests_main_$layout.tsv" "$out/digests_new_$layout.tsv" || true; } > "$f"
  fi
  cat "$f"
done

# Speed, one ab.py invocation (one process a binary) per case: A main, B new.
for layout in default aligned; do
  f=$out/ab_$layout.txt
  : > "$f"
  first=1
  for c in "${cases[@]}"; do
    echo "timing $layout: $c" >&2
    res=$(python3 "$here/ab.py" --a "$out/main.$layout$exe" --b "$out/new.$layout$exe" \
      --rounds "$rounds" --reps 5 "$c" 2>/dev/null)
    if [ $first = 1 ]; then echo "$res" | head -1 >> "$f"; first=0; fi
    echo "$res" | tail -n +2 >> "$f"
  done
  uptime >> "$f"
  cat "$f"
done
# Speed, every hmm and ew_class case in one process, as the first run did.
if [ "${W479_SEQ:-1}" = 1 ]; then
  for layout in default aligned; do
    f=$out/seq_$layout.txt
    echo "timing $layout, one process over hmm and ew_class" >&2
    python3 "$here/ab.py" --a "$out/main.$layout$exe" --b "$out/new.$layout$exe" \
      --rounds "$rounds" --reps 5 hmm ew_class > "$f" 2>/dev/null
    uptime >> "$f"
    cat "$f"
  done
fi
uptime | tee "$out/load_after.txt"
echo "done: $out"
