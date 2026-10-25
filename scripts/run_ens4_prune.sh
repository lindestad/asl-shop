#!/usr/bin/env bash
# leave-one-out re-map on the PRUNED ens4 (e62 + l6e82 + soup5459 + l9soup): redundancy
# is context-dependent, so after dropping step-84 the remaining members' marginal value
# must be re-measured (drop-m2-vs-ens4 interaction untested by the ens5 prune run).
# stages: 32-clip smokes -> 4k greedy screens (baseline 2978 err / 2765 exact, reused
# from the ens5 prune drop-m2 screen, same members/subset) -> best pruned subset
# full-val beam-8 (gate 8601 err / 4.057% / 8153 exact) -> test beam-8
# (best 8186 err / 3.702% / 8892 exact). batch 4, expandable_segments (foreign tenant).
set -uo pipefail
cd /home/coder/asl-wt/auto-research
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/coder/venvs/asl/bin/python
CDIR=data/models/openfs-recognizer-gaps-full-cosine-ema
M1=$CDIR/recognizer-epoch62.pt
M3=data/models/openfs-recognizer-gaps-full-layers6/recognizer-epoch82.pt
M4=data/models/openfs-recognizer-gaps-full-layers6-cosine-ema/recognizer-soup-e54e59-a50.pt
L9=data/models/openfs-recognizer-gaps-full-layers9-cosine-ema
M5=$L9/recognizer-soup-e41e43e55-3w.pt

get(){ $PY -c "import json,sys;r=json.load(open(sys.argv[1]));e=r['evaluations'][0];print(e[sys.argv[2]])" "$1" "$2"; }
eval_run(){ # ckpt-list-string splits out beam batch [extra...]
  local ck="$1" splits="$2" out="$3" beam="$4" batch="$5"; shift 5
  nice -n 19 $PY scripts/beam_eval_openfs_recognizer.py --checkpoint $ck \
    --data data/openfs-packed --splits $splits --beam-size $beam --batch-size $batch \
    --output "$out" "$@" || exit 1
}

echo "[stage] 32-clip beam-8 smokes (fail-fast on new member subsets)"
eval_run "$M3 $M4 $M5" validation $CDIR/smoke-beam8-ens4p-drop-m1 8 4 --limit 32
eval_run "$M1 $M4 $M5" validation $CDIR/smoke-beam8-ens4p-drop-m3 8 4 --limit 32
eval_run "$M1 $M3 $M5" validation $CDIR/smoke-beam8-ens4p-drop-m4 8 4 --limit 32
eval_run "$M1 $M3 $M4" validation $CDIR/smoke-beam8-ens4p-drop-m5 8 4 --limit 32

echo "[stage] 4k greedy screens (baseline 2978 err / 2765 exact from ens5-prune drop-m2)"
declare -A SCR TAG
screen(){ # name ck-list-string
  local n="$1"; shift
  eval_run "$*" validation $CDIR/screen-subset4k-ens4p-$n 1 4 --limit 4000
  SCR[$n]=$(get $CDIR/screen-subset4k-ens4p-$n.beam-eval-report.json errors)
  echo "[screen] $n errors=${SCR[$n]} exact=$(get $CDIR/screen-subset4k-ens4p-$n.beam-eval-report.json exact)"
}
screen drop-m1 "$M3 $M4 $M5"; TAG[drop-m1]="ens3-l6e82-soup5459-l9soup"
screen drop-m3 "$M1 $M4 $M5"; TAG[drop-m3]="ens3-e62-soup5459-l9soup"
screen drop-m4 "$M1 $M3 $M5"; TAG[drop-m4]="ens3-e62-l6e82-l9soup"
screen drop-m5 "$M1 $M3 $M4"; TAG[drop-m5]="ens3-e62-l6e82-soup5459"
base=2978
best=none; beste=$base
for n in drop-m1 drop-m3 drop-m4 drop-m5; do
  if [ "${SCR[$n]}" -lt "$beste" ]; then beste=${SCR[$n]}; best=$n; fi
done
echo "[screen] baseline=$base best-drop=$best errors=$beste"
if [ "$best" = none ]; then echo "[stop] no pruned subset beats ens4 at 4k; re-pruning screen-negative"; exit 0; fi

echo "[stage] full validation beam-8 for $best"
case $best in
  drop-m1) CKS="$M3 $M4 $M5";;
  drop-m3) CKS="$M1 $M4 $M5";;
  drop-m4) CKS="$M1 $M3 $M5";;
  drop-m5) CKS="$M1 $M3 $M4";;
esac
OUT=$CDIR/validation-beam8-${TAG[$best]}
eval_run "$CKS" validation $OUT 8 4
verr=$(get $OUT.beam-eval-report.json errors)
echo "[val] cer=$(get $OUT.beam-eval-report.json cer) errors=$verr exact=$(get $OUT.beam-eval-report.json exact) (gate 8601 / 4.057% / 8153)"
if [ "$verr" -ge 8601 ]; then echo "[stop] validation gate not earned"; exit 0; fi

echo "[stage] test beam-8 (gate earned)"
TOUT=$CDIR/test-beam8-${TAG[$best]}
eval_run "$CKS" test $TOUT 8 4
echo "[test] cer=$(get $TOUT.beam-eval-report.json cer) errors=$(get $TOUT.beam-eval-report.json errors) exact=$(get $TOUT.beam-eval-report.json exact) (best 8186 / 3.702% / 8892)"
