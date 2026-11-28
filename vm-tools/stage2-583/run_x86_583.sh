#!/usr/bin/env bash
# Stage 2 (PLAN 583): x86-64 ABBA timing of HEAD against v2.0.0 in BOTH code
# layouts (default and -align-all-functions=6) with identical-binary controls,
# plus all post-2.0.0 native cases and Python release-wheel ABBA on
# benchmark.py & wide_benchmark.py.
set -euo pipefail

work=${1:-$RUNNER_TEMP/x86-583}
rounds=${2:-6}
v2_ref=${V2_REF:-v2.0.0}
new_ref=${NEW_REF:-origin/main}

here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work/out"
work=$(cd "$work" && pwd)
out=$work/out

export CARGO_INCREMENTAL=0
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
export CARGO_PROFILE_RELEASE_STRIP=none
unset RUSTFLAGS

repo_url=${RHEON_REPO:-https://github.com/rheon-dev/rheon.git}
tree=$work/rheon
if [ ! -d "$tree/.git" ]; then
  git clone --no-single-branch "$repo_url" "$tree"
fi
git -C "$tree" fetch -q origin main --tags
v2_sha=$(git -C "$tree" rev-parse "$v2_ref")
new_sha=$(git -C "$tree" rev-parse "$new_ref")

{
  echo "v2.0.0 $v2_sha"
  echo "new    $new_sha"
  uname -a
  lscpu || true
  rustc -V; cargo -V; python3 -V; nproc
} > "$out/machine.txt" 2>&1
uptime | tee "$out/load_before.txt"

bench=crates/rheon-bench/examples/abba_583.rs
target=$work/target

build_bin() {
  local name=$1 layout=$2
  if [ "$layout" = aligned ]; then
    (cd "$tree" && CARGO_TARGET_DIR="$work/target-aligned" \
      RUSTFLAGS="-C llvm-args=-align-all-functions=6 -C symbol-mangling-version=v0" \
      cargo build --release -q -p rheon-bench --example abba_583)
    cp "$work/target-aligned/release/examples/abba_583" "$out/$name.$layout"
  else
    (cd "$tree" && CARGO_TARGET_DIR="$work/target-default" \
      RUSTFLAGS="" \
      cargo build --release -q -p rheon-bench --example abba_583)
    cp "$work/target-default/release/examples/abba_583" "$out/$name.$layout"
  fi
}

aligned_check() {
  local total bad
  total=$(nm "$1" | awk '$2 ~ /^[tT]$/ && $3 ~ /^_?_R.*rheon/ && $3 !~ /compiler_builtins/' | wc -l)
  bad=$(nm "$1" | awk '$2 ~ /^[tT]$/ && $3 ~ /^_?_R.*rheon/ && $3 !~ /compiler_builtins/ {print $1}' |
    python3 -c 'import sys; print(sum(1 for l in sys.stdin if int(l, 16) % 64))')
  echo "$1: $total rheon Rust functions, $bad not on 64B boundary" | tee -a "$out/alignment.txt"
  [ "$total" -gt 500 ] && [ "$bad" -eq 0 ]
}

: > "$out/alignment.txt"
for pair in "$v2_sha v2" "$new_sha new"; do
  set -- $pair
  git -C "$tree" checkout -q -f --detach "$1"
  cp "$here/abba_583.rs" "$tree/$bench"
  for layout in default aligned; do
    echo "building native $2 ($layout)..." >&2
    build_bin "$2" "$layout"
  done
  aligned_check "$out/$2.aligned"
  rm -f "$tree/$bench"
done

for layout in default aligned; do
  echo "=== Native ABBA ($layout, $rounds rounds, 48 v2.0.0 cases) ===" | tee "$out/native_abba_$layout.txt"
  python3 "$here/ab_583.py" --mode bin --a "$out/v2.$layout" --b "$out/new.$layout" \
    --rounds "$rounds" --reps 5 --suite v2 | tee -a "$out/native_abba_$layout.txt"
  uptime | tee -a "$out/native_abba_$layout.txt"

  echo "=== Post-2.0.0 + all native cases on HEAD ($layout) ===" | tee "$out/native_all_head_$layout.txt"
  RHEON_MAX_THREADS=1 POLARS_MAX_THREADS=1 "$out/new.$layout" 5 all | tee -a "$out/native_all_head_$layout.txt"
done

# Python release wheels: v2.0.0 vs HEAD (separate worktrees & targets)
wt_v2=$work/wt-v2
wt_new=$work/wt-new
git -C "$tree" worktree add --detach "$wt_v2" "$v2_sha"
git -C "$tree" worktree add --detach "$wt_new" "$new_sha"
for wt in "$wt_v2" "$wt_new"; do
  (
    cd "$wt"
    export UV_PYTHON=3.12
    uv sync --locked
    uv run maturin develop --release -m crates/rheon-py/Cargo.toml
  )
done

echo "=== Python benchmark.py ABBA (v2.0.0 vs HEAD, --rows 100000) ===" | tee "$out/py_benchmark_abba.txt"
python3 "$here/ab_583.py" --mode py \
  --a "$wt_v2/.venv/bin/python" --b "$wt_new/.venv/bin/python" \
  --script "$wt_v2/scripts/benchmark.py" --rounds 4 --extra "--rows 100000 --markdown" \
  | tee -a "$out/py_benchmark_abba.txt"

echo "=== Python wide_benchmark.py ABBA (v2.0.0 vs HEAD) ===" | tee "$out/py_wide_benchmark_abba.txt"
python3 "$here/ab_583.py" --mode py \
  --a "$wt_v2/.venv/bin/python" --b "$wt_new/.venv/bin/python" \
  --script "$wt_v2/scripts/wide_benchmark.py" --rounds 3 \
  | tee -a "$out/py_wide_benchmark_abba.txt"

echo "=== HEAD benchmark.py full markdown (x86-64) ===" | tee "$out/py_benchmark_head.md"
"$wt_new/.venv/bin/python" "$wt_new/scripts/benchmark.py" --rows 100000 --markdown \
  | tee -a "$out/py_benchmark_head.md"

echo "=== Linux x86-64 release wheel size ===" | tee "$out/wheel_size_linux_x86.txt"
(
  cd "$wt_new"
  uv run maturin build --release -m crates/rheon-py/Cargo.toml --out "$work/wheels"
  python3 scripts/wheel_size.py "$work"/wheels/*.whl || true
) | tee -a "$out/wheel_size_linux_x86.txt"

uptime | tee "$out/load_after.txt"
echo "done: $out"
