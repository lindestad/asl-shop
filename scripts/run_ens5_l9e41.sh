#!/usr/bin/env bash
# gated refresh of the ens5 5th member: l9-cos-EMA e33 -> e41 (greedy val 4.719 -> 4.542)
# stages: 32-clip beam-8 smoke, 4k greedy screen, full validation beam-8 (gate 8791 errors),
# full test beam-8 only if validation beats the gate. batch 4 (GPU contended, 3591 MiB free).
set -uo pipefail
cd /home/coder/asl-wt/auto-research
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/coder/venvs/asl/bin/python
M1=data/models/openfs-recognizer-gaps-full-cosine-ema/recognizer-epoch62.pt
M2=data/models/openfs-recognizer-gaps-full/recognizer-epoch84.pt
M3=data/models/openfs-recognizer-gaps-full-layers6/recognizer-epoch82.pt
M4=data/models/openfs-recognizer-gaps-full-layers6-cosine-ema/recognizer-soup-e54e59-a50.pt
M5=data/models/openfs-recognizer-gaps-full-layers9-cosine-ema/recognizer-epoch41.pt
CKPTS="$M1 $M2 $M3 $M4 $M5"
L9DIR=data/models/openfs-recognizer-gaps-full-layers9-cosine-ema
CDIR=data/models/openfs-recognizer-gaps-full-cosine-ema
eval_run(){ # splits... output-stem beam batch limit...
  local splits="$1" out="$2" beam="$3" batch="$4"; shift 4
  nice -n 19 $PY scripts/beam_eval_openfs_recognizer.py --checkpoint $CKPTS \
    --data data/openfs-packed --splits $splits --beam-size $beam --batch-size $batch \
    --output "$out" "$@"
}
get(){ $PY -c "import json,sys;r=json.load(open(sys.argv[1]));e=r['evaluations'][0];print(e[sys.argv[2]])" "$1" "$2"; }

echo "[stage] smoke 32 clips"
eval_run validation $L9DIR/smoke-beam8-ens5-l9e41 8 4 --limit 32 || exit 1
echo "[smoke] errors=$(get $L9DIR/smoke-beam8-ens5-l9e41.beam-eval-report.json errors) exact=$(get $L9DIR/smoke-beam8-ens5-l9e41.beam-eval-report.json exact) (e33 smoke was 29/23)"

echo "[stage] greedy screen 4k clips"
eval_run validation $L9DIR/screen-subset4k-ens5-l9e41 1 4 --limit 4000 || exit 1
scr=$(get $L9DIR/screen-subset4k-ens5-l9e41.beam-eval-report.json errors)
echo "[screen] errors=$scr exact=$(get $L9DIR/screen-subset4k-ens5-l9e41.beam-eval-report.json exact) (ens5-l9e33 3113)"
if [ "$scr" -gt 3113 ]; then echo "[abort] screen worse than ens5-l9e33 (3113)"; exit 1; fi

echo "[stage] validation beam 8"
eval_run validation $CDIR/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e41 8 4 || exit 1
verr=$(get $CDIR/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json errors)
vcer=$(get $CDIR/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json cer)
echo "[val] cer=$vcer errors=$verr exact=$(get $CDIR/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json exact) (gate 8791 / 4.146%)"
if [ "$verr" -ge 8791 ]; then echo "[stop] validation gate not earned"; exit 0; fi

echo "[stage] test beam 8 (gate earned)"
eval_run test $CDIR/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e41 8 4 || exit 1
ter=$(get $CDIR/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json errors)
tcer=$(get $CDIR/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json cer)
echo "[test] cer=$tcer errors=$ter exact=$(get $CDIR/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json exact) (best 8430 / 3.813%)"
