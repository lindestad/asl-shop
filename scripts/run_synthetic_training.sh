#!/usr/bin/env bash
# run-synthetic-training.sh — packs, then the documented sequence runs, one GPU process at a time.
# This is what ran on the VPS on 2026-10-19 (docs/synthetic-data-2026-10-19.md); paths are
# overridable so it can run wherever the worktree and asl-data live.
set -uo pipefail
cd "${ASL_WORKTREE:-$(git rev-parse --show-toplevel)}"
PY=${PY:-.venv/bin/python}
MODELS=${ASL_MODELS:-$HOME/asl-data/models}
HANDSHAPES=$MODELS/neural-2026-10-12/handshape.pt
BASELINE=$MODELS/neural-2026-10-12/sequence.pt

pack() {
  local name=$1 generation=$2
  shift 2
  echo "== pack $name $(date -Iseconds)"
  $PY scripts/pack_openfs_synthetic.py --generation "$generation" --merge-with data/openfs-packed \
    --output "data/$name" --report "data/$name.report.json" "$@" || { echo "pack $name FAILED"; exit 1; }
}

train() {
  local name=$1 data=$2
  echo "== train $name $(date -Iseconds)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  mkdir -p "$MODELS/synthetic-$name"
  $PY scripts/train_openfs.py --data "$data" --task sequence --output "$MODELS/synthetic-$name/sequence.pt" \
    --warmstart-handshape "$HANDSHAPES" --epochs 30 --hidden 192 --batch-size 64 --device cuda --amp \
    --report "$MODELS/synthetic-$name/sequence.report.json" || { echo "train $name FAILED"; exit 1; }
  echo "== evaluate $name $(date -Iseconds)"
  $PY scripts/evaluate_synthetic_mix.py --data "$data" --checkpoint "$MODELS/synthetic-$name/sequence.pt" \
    --split test --device cuda --amp --report "data/openfs-synthetic-$name.evaluate.report.json" \
    || { echo "evaluate $name FAILED"; exit 1; }
}

pack openfs-synthetic-packed-mix15k data/openfs-synthetic-words --limit-clips 15000
pack openfs-synthetic-packed-mixall data/openfs-synthetic-words
pack openfs-synthetic-packed-letters data/openfs-synthetic-letters

train baseline data/openfs-packed
train mix15k data/openfs-synthetic-packed-mix15k
train mixall data/openfs-synthetic-packed-mixall
train mixletters data/openfs-synthetic-packed-letters

echo "== evaluate shipped-baseline $(date -Iseconds)"
$PY scripts/evaluate_synthetic_mix.py --data data/openfs-packed --checkpoint "$BASELINE" --split test \
  --device cuda --amp --report data/openfs-synthetic-baseline-shipped.evaluate.report.json

echo "== all done $(date -Iseconds)"
