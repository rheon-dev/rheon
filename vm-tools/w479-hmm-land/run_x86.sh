#!/usr/bin/env bash
# Task 479, landing (branch task-479-hmm-land): hmm with full covariance
# keeps each state's Cholesky factor as its state from 13 features. This
# times the branch's default build against main on x86-64, in both function
# layouts, with an identical-binary control, at k = 8, 13, 20, 32 features
# (hmm, 2 states, full covariance), and checks the bits: k = 8 must be the
# same as main's (below 13 nothing changes), k >= 13 moves by design.
#
#   bash run_x86.sh [rounds]        # needs git, cargo, python3
#
# BRANCH (default task-479-hmm-land) and BASE (default main) are fetched
# from REPO (default https://github.com/rheon-dev/rheon.git) into $WORK
# (default ~/w479-land). factor_bench (rheon-bench's example, public API
# only) is not on main: the branch's copy is put into the base's tree
# before it is built, so both builds time the same rows.
# Builds: 4 binaries (base and branch, each layout), CARGO_INCREMENTAL=0
# CARGO_BUILD_JOBS=${JOBS:-2}, about 30-50 min on 4 cores; timing about
# 5 min a layout, each timed run a few seconds. Run nothing else meanwhile.
set -euo pipefail
rounds=${1:-6}
here=$(cd "$(dirname "$0")" && pwd)
work=${WORK:-$HOME/w479-land}
repo_url=${REPO:-https://github.com/rheon-dev/rheon.git}
branch=${BRANCH:-task-479-hmm-land}
base=${BASE:-main}
out=$work/out
mkdir -p "$out"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=${JOBS:-2}
uname -a > "$out/machine.txt"
(lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true) >> "$out/machine.txt"
nproc >> "$out/machine.txt" 2>/dev/null || true
uptime | tee "$out/load_before.txt"

repo=$work/rheon
if [ ! -d "$repo/.git" ]; then
  git clone --no-checkout "$repo_url" "$repo"
fi
git -C "$repo" fetch origin "$branch:w479-land-branch" "$base:w479-land-base" --force
{ echo "branch $(git -C "$repo" rev-parse w479-land-branch)"
  echo "base   $(git -C "$repo" rev-parse w479-land-base)"; } | tee "$out/shas.txt"
[ -d "$work/base" ] || git -C "$repo" worktree add -f --detach "$work/base" w479-land-base
[ -d "$work/branch" ] || git -C "$repo" worktree add -f --detach "$work/branch" w479-land-branch
git -C "$work/base" checkout -q --detach w479-land-base
git -C "$work/branch" checkout -q --detach w479-land-branch
cp "$work/branch/crates/rheon-bench/examples/factor_bench.rs" \
  "$work/base/crates/rheon-bench/examples/factor_bench.rs"

build() { # tree label layout
  local tree=$1 label=$2 layout=$3 flags=""
  [ "$layout" = aligned ] && flags="-C llvm-args=-align-all-functions=6"
  (cd "$tree" && RUSTFLAGS="$flags" CARGO_TARGET_DIR="$tree/target-$layout" \
    cargo build --release -q -p rheon-bench --example factor_bench)
  local exe=""
  [ -f "$tree/target-$layout/release/examples/factor_bench.exe" ] && exe=.exe
  cp "$tree/target-$layout/release/examples/factor_bench$exe" "$out/$label.$layout$exe"
  echo "$exe" > "$out/exe"
}
for layout in default aligned; do
  build "$work/base" base $layout
  build "$work/branch" land $layout
done
exe=$(cat "$out/exe")

cases=("hmm k=8 K" "hmm k=13 K" "hmm k=20 K" "hmm k=32 K")
"$out/base.default$exe" 1 "${cases[@]}" > "$out/digest_base.txt"
"$out/land.default$exe" 1 "${cases[@]}" > "$out/digest_land.txt"
python3 - "$out" <<'PY' | tee "$out/digests.txt"
import re, sys
from pathlib import Path
out = Path(sys.argv[1])
def read(name):
    got = {}
    for line in (out / name).read_text().splitlines():
        m = re.match(r"^(.*?)\s+best.*digest ([0-9a-f]+)$", line.strip())
        if m:
            got[m.group(1).strip()] = m.group(2)
    return got
b, l = read("digest_base.txt"), read("digest_land.txt")
for c in b:
    k = int(re.search(r"k=(\d+)", c).group(1))
    same = b[c] == l.get(c)
    if k < 13:
        verdict = "SAME" if same else "MOVED (a bug: report it first)"
    else:
        verdict = "moved (expected)" if not same else "SAME (unexpected: the factor form did not act)"
    print(f"{c:<24} {verdict}")
PY

for layout in default aligned; do
  python3 "$here/ab.py" --a "$out/base.$layout$exe" --b "$out/land.$layout$exe" \
    --rounds "$rounds" --reps 5 "${cases[@]}" > "$out/ab_$layout.txt"
done
uptime | tee "$out/load_after.txt"
echo "done: $out"
