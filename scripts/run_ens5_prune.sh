#!/usr/bin/env bash
# ens5 member-pruning (leave-one-out) map: the ensemble has only ever grown; the two
# slot-absorption findings (m1 and m4 member upgrades ensemble-invisible) motivate the
# untested inverse axis - remove a member and test whether the fused posterior sharpens.
# stages: ens5 smoke -> 4k greedy screens (full ens5 + 5 leave-one-out subsets) ->
# best pruned subset full-val beam-8 (gate 8679 errors / 4.093%) -> test beam-8
# (best 8222 / 3.719%). batch 4, expandable_segments (foreign tenant ~24 GiB).
set -uo pipefail
cd /home/coder/asl-wt/auto-research
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/coder/venvs/asl/bin/python
L9=data/models/openfs-recognizer-gaps-full-layers9-cosine-ema
CDIR=data/models/openfs-recognizer-gaps-full-cosine-ema
M1=$CDIR/recognizer-epoch62.pt
M2=data/models/openfs-recognizer-gaps-full/recognizer-epoch84.pt
M3=data/models/openfs-recognizer-gaps-full-layers6/recognizer-epoch82.pt
M4=data/models/openfs-recognizer-gaps-full-layers6-cosine-ema/recognizer-soup-e54e59-a50.pt
M5=$L9/recognizer-soup-e41e43e55-3w.pt

get(){ $PY -c "import json,sys;r=json.load(open(sys.argv[1]));e=r['evaluations'][0];print(e[sys.argv[2]])" "$1" "$2"; }
eval_run(){ # ckpt-list-string splits out beam batch [extra...]
  local ck="$1" splits="$2" out="$3" beam="$4" batch="$5"; shift 5
  nice -n 19 $PY scripts/beam_eval_openfs_recognizer.py --checkpoint $ck \
    --data data/openfs-packed --splits $splits --beam-size $beam --batch-size $batch \
    --output "$out" "$@" || exit 1
}

echo "[stage] ens5 32-clip beam-8 smoke"
eval_run "$M1 $M2 $M3 $M4 $M5" validation $CDIR/smoke-beam8-ens5-prune 8 4 --limit 32
echo "[smoke] errors=$(get $CDIR/smoke-beam8-ens5-prune.beam-eval-report.json errors) exact=$(get $CDIR/smoke-beam8-ens5-prune.beam-eval-report.json exact)"

echo "[stage] 4k greedy screens: ens5 + leave-one-out"
declare -A SCR TAG
screen(){ # name ck-list-string
  local n="$1"; shift
  eval_run "$*" validation $CDIR/screen-subset4k-ens5-$n 1 4 --limit 4000
  SCR[$n]=$(get $CDIR/screen-subset4k-ens5-$n.beam-eval-report.json errors)
  echo "[screen] $n errors=${SCR[$n]} exact=$(get $CDIR/screen-subset4k-ens5-$n.beam-eval-report.json exact)"
}
screen all "$M1 $M2 $M3 $M4 $M5"
screen drop-m1 "$M2 $M3 $M4 $M5"; TAG[drop-m1]="ens4-s84-l6e82-soup5459-l9soup"
screen drop-m2 "$M1 $M3 $M4 $M5"; TAG[drop-m2]="ens4-e62-l6e82-soup5459-l9soup"
screen drop-m3 "$M1 $M2 $M4 $M5"; TAG[drop-m3]="ens4-e62-s84-soup5459-l9soup"
screen drop-m4 "$M1 $M2 $M3 $M5"; TAG[drop-m4]="ens4-e62-s84-l6e82-l9soup"
screen drop-m5 "$M1 $M2 $M3 $M4"; TAG[drop-m5]="ens4-e62-s84-l6e82-soup5459"
base=${SCR[all]}
best=none; beste=$base
for n in drop-m1 drop-m2 drop-m3 drop-m4 drop-m5; do
  if [ "${SCR[$n]}" -lt "$beste" ]; then beste=${SCR[$n]}; best=$n; fi
done
echo "[screen] baseline=$base best-drop=$best errors=$beste"
if [ "$best" = none ]; then echo "[stop] no pruned subset beats ens5 at 4k; pruning axis screen-negative"; exit 0; fi

echo "[stage] full validation beam-8 for $best"
CKS=$(echo "$M1 $M2 $M3 $M4 $M5")
case $best in
  drop-m1) CKS="$M2 $M3 $M4 $M5";;
  drop-m2) CKS="$M1 $M3 $M4 $M5";;
  drop-m3) CKS="$M1 $M2 $M4 $M5";;
  drop-m4) CKS="$M1 $M2 $M3 $M5";;
  drop-m5) CKS="$M1 $M2 $M3 $M4";;
esac
OUT=$CDIR/validation-beam8-${TAG[$best]}
eval_run "$CKS" validation $OUT 8 4
verr=$(get $OUT.beam-eval-report.json errors)
echo "[val] cer=$(get $OUT.beam-eval-report.json cer) errors=$verr exact=$(get $OUT.beam-eval-report.json exact) (gate 8679 / 4.093%)"
if [ "$verr" -ge 8679 ]; then echo "[stop] validation gate not earned"; exit 0; fi

echo "[stage] test beam-8 (gate earned)"
TOUT=$CDIR/test-beam8-${TAG[$best]}
eval_run "$CKS" test $TOUT 8 4
echo "[test] cer=$(get $TOUT.beam-eval-report.json cer) errors=$(get $TOUT.beam-eval-report.json errors) exact=$(get $TOUT.beam-eval-report.json exact) (best 8222 / 3.719%)"
