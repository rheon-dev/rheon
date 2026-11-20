#!/usr/bin/env bash
# Task 479, hmm factor-as-state prototype: its speed on x86-64 against the
# 479 branch it starts from, in both function layouts, with an
# identical-binary control; and the default build's bits on x86 against
# that base (the prototype is a Cargo feature, off by default).
#
#   bash run_x86.sh [rounds]        # from anywhere; needs git, cargo, python3
#
# It clones (or reuses) https://github.com/rheon-dev/rheon.git under
# $WORK (default ~/w479-hmm), fetches branch task-479-hmm-factor-state,
# makes two worktrees -- base at 0cd9e104 (task-479-per-row-factorizations)
# and the branch tip -- and builds `factor_bench` (rheon-bench's example,
# public API only, an FNV digest of every output bit):
#   base.<layout>      base, default features
#   proto.<layout>     branch, --features rheon-core/hmm-factor-state (L moved,
#                      each entry scaled by sqrt(a) in the rotation)
#   scaled.<layout>    branch, --features rheon-core/hmm-factor-scaled (M = s*L*L^T,
#                      the decay's a in the scalar s)
#   branchdef.default  branch, default features (must give base's digests)
# each in both layouts (default, and -C llvm-args=-align-all-functions=6),
# CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2. Then ab.py (ABBA with an
# identical-binary control, every binary copied to paths of equal length)
# over hmm 2 states full covariance at k = 4, 8, 13, 20, 32.
# Builds take about 50-70 min on 4 cores; timing about 5 min a layout, each
# timed run a few seconds. Run nothing else meanwhile.
set -euo pipefail
rounds=${1:-6}
here=$(cd "$(dirname "$0")" && pwd)
work=${WORK:-$HOME/w479-hmm}
out=$work/out
mkdir -p "$out"
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2
uname -a > "$out/machine.txt"
(lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || true) >> "$out/machine.txt"
nproc >> "$out/machine.txt" 2>/dev/null || true
uptime | tee "$out/load_before.txt"

repo=$work/rheon
if [ ! -d "$repo/.git" ]; then
  git clone --no-checkout https://github.com/rheon-dev/rheon.git "$repo"
fi
git -C "$repo" fetch origin task-479-hmm-factor-state:w479-hmm-branch --force
git -C "$repo" rev-parse w479-hmm-branch | tee "$out/branch_sha.txt"
git -C "$repo" merge-base --is-ancestor 0cd9e104 w479-hmm-branch
[ -d "$work/base" ] || git -C "$repo" worktree add -f --detach "$work/base" 0cd9e104
[ -d "$work/branch" ] || git -C "$repo" worktree add -f --detach "$work/branch" w479-hmm-branch
git -C "$work/branch" checkout -q --detach w479-hmm-branch

build() { # tree label layout [features]
  local tree=$1 label=$2 layout=$3 feat=${4:-} flags=""
  [ "$layout" = aligned ] && flags="-C llvm-args=-align-all-functions=6"
  local args=(-p rheon-bench --example factor_bench)
  [ -n "$feat" ] && args=(-p rheon-bench -p rheon-core --example factor_bench --features "$feat")
  (cd "$tree" && RUSTFLAGS="$flags" CARGO_TARGET_DIR="$tree/target-$layout" \
    cargo build --release -q "${args[@]}")
  local exe=""
  [ -f "$tree/target-$layout/release/examples/factor_bench.exe" ] && exe=.exe
  cp "$tree/target-$layout/release/examples/factor_bench$exe" "$out/$label.$layout$exe"
  echo "$exe" > "$out/exe"
}
for layout in default aligned; do
  build "$work/base" base $layout
  build "$work/branch" proto $layout rheon-core/hmm-factor-state
  build "$work/branch" scaled $layout rheon-core/hmm-factor-scaled
done
build "$work/branch" branchdef default
exe=$(cat "$out/exe")

# The default build's bits on x86: every digest of the branch's default
# build must equal base's (the prototype is compiled out).
cases=("hmm k=4 K" "hmm k=8 K" "hmm k=13 K" "hmm k=20 K" "hmm k=32 K")
"$out/base.default$exe" 1 "${cases[@]}" > "$out/digest_base.txt"
"$out/branchdef.default$exe" 1 "${cases[@]}" > "$out/digest_branchdef.txt"
"$out/proto.default$exe" 1 "${cases[@]}" > "$out/digest_proto.txt"
"$out/scaled.default$exe" 1 "${cases[@]}" > "$out/digest_scaled.txt"
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
b, d = read("digest_base.txt"), read("digest_branchdef.txt")
p, s = read("digest_proto.txt"), read("digest_scaled.txt")
for c in b:
    print(f"{c:<24} default build {'SAME' if b[c] == d.get(c) else 'MOVED (a bug: report it first)'}"
          f"   prototypes {'same' if b[c] == p.get(c) else 'moved'}, "
          f"{'same' if b[c] == s.get(c) else 'moved'} (moved is expected)")
PY

for layout in default aligned; do
  for b in proto scaled; do
    python3 "$here/ab.py" --a "$out/base.$layout$exe" --b "$out/$b.$layout$exe" \
      --rounds "$rounds" --reps 5 "${cases[@]}" > "$out/ab_${b}_$layout.txt"
  done
done
uptime | tee "$out/load_after.txt"
echo "done: $out"
