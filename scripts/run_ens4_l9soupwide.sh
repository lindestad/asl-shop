#!/usr/bin/env bash
# m5 soup horizon extension BACKWARDS: the l9-slot 3-way soup (e41+e43+e55) found wide
# horizons win at member level in the m5 slot, but round-6 supply was blocked forward
# (trainer saves only on improvement; val-best stuck at e55 4.417% through epoch 73).
# Frozen e27/e33 snapshots allow untested backward extension: 4-way e33+e41+e43+e55
# (22-epoch span) and 5-way +e27 (28-epoch span), equal weights, as m5 swaps into the
# pruned ens4 (e62 + l6e82 + soup5459 unchanged).
# stages: soups -> 4k greedy member screens (baseline s3w 2908 err / 2750 exact) ->
# member greedy full-val confirm (baseline s3w 9004 err / 4.2466% / 7992 exact, need
# >=60 better) -> ens4 validation beam-8 (gate 8601 err / 4.057% / 8153 exact) ->
# test beam-8 (best 8186 err / 3.702% / 8892 exact). batch 4, expandable_segments.
set -uo pipefail
cd /home/coder/asl-wt/auto-research
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/coder/venvs/asl/bin/python
L9=data/models/openfs-recognizer-gaps-full-layers9-cosine-ema
CDIR=data/models/openfs-recognizer-gaps-full-cosine-ema
M1=$CDIR/recognizer-epoch62.pt
M3=data/models/openfs-recognizer-gaps-full-layers6/recognizer-epoch82.pt
M4=data/models/openfs-recognizer-gaps-full-layers6-cosine-ema/recognizer-soup-e54e59-a50.pt
E27=$L9/recognizer-epoch27.pt
E33=$L9/recognizer-epoch33.pt
E41=$L9/recognizer-epoch41.pt
E43=$L9/recognizer-epoch43.pt
E55=$L9/recognizer-epoch55.pt
S4W=$L9/recognizer-soup-e33e41e43e55-4w.pt
S5W=$L9/recognizer-soup-e27e33e41e43e55-5w.pt

get(){ $PY -c "import json,sys;r=json.load(open(sys.argv[1]));e=r['evaluations'][0];print(e[sys.argv[2]])" "$1" "$2"; }
eval_run(){
  local ck="$1" splits="$2" out="$3" beam="$4" batch="$5"; shift 5
  nice -n 19 $PY scripts/beam_eval_openfs_recognizer.py --checkpoint $ck \
    --data data/openfs-packed --splits $splits --beam-size $beam --batch-size $batch \
    --output "$out" "$@" || exit 1
}

echo "[stage] soups (equal weights via chained interpolation)"
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $E33 --checkpoint-b $E41 --alpha 0.5 --output $L9/tmp-soup-t2.pt || exit 1
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $L9/tmp-soup-t2.pt --checkpoint-b $E43 --alpha 0.6666666667 --output $L9/tmp-soup-t3.pt || exit 1
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $L9/tmp-soup-t3.pt --checkpoint-b $E55 --alpha 0.75 --output $S4W || exit 1
rm -f $L9/tmp-soup-t2.pt $L9/tmp-soup-t3.pt
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $E27 --checkpoint-b $S4W --alpha 0.2 --output $S5W || exit 1

echo "[stage] 4k greedy member screens (baseline s3w 2908 / 2750 exact)"
declare -A SCRN
for m in s4w:$S4W s5w:$S5W; do
  n=${m%%:*}; ck=${m#*:}
  eval_run "$ck" validation $L9/screen-subset4k-l9m-$n 1 4 --limit 4000
  SCRN[$n]=$(get $L9/screen-subset4k-l9m-$n.beam-eval-report.json errors)
  echo "[screen] $n errors=${SCRN[$n]} exact=$(get $L9/screen-subset4k-l9m-$n.beam-eval-report.json exact)"
done
base=2908; bestn=none; beste=$base
for n in s4w s5w; do if [ "${SCRN[$n]}" -lt "$beste" ]; then beste=${SCRN[$n]}; bestn=$n; fi; done
echo "[screen] baseline s3w=$base best=$bestn errors=$beste (margin $((base-beste)))"
if [ "$bestn" = none ] || [ $((base-beste)) -lt 40 ]; then echo "[stop] no extended soup beats s3w by >=40 at 4k"; exit 0; fi
case $bestn in s4w) SOUP=$S4W;; s5w) SOUP=$S5W;; esac

echo "[stage] member greedy full validation for $bestn"
eval_run "$SOUP" validation $L9/validation-greedy-l9m-$bestn 1 4
merr=$(get $L9/validation-greedy-l9m-$bestn.beam-eval-report.json errors)
echo "[member] $bestn cer=$(get $L9/validation-greedy-l9m-$bestn.beam-eval-report.json cer) errors=$merr exact=$(get $L9/validation-greedy-l9m-$bestn.beam-eval-report.json exact) (s3w 9004 / 4.2466% / 7992)"
if [ "$merr" -ge 8944 ]; then echo "[stop] member gain not >=60 errors at full val"; exit 0; fi

echo "[stage] smoke beam-8 (ens4 with $bestn as m5)"
eval_run "$M1 $M3 $M4 $SOUP" validation $CDIR/smoke-beam8-ens4-l9$bestn 8 4 --limit 32

echo "[stage] ens4 validation beam 8 (gate 8601 / 4.057% / 8153)"
eval_run "$M1 $M3 $M4 $SOUP" validation $CDIR/validation-beam8-ens4-e62-l6e82-soup5459-l9$bestn 8 4
OUT=$CDIR/validation-beam8-ens4-e62-l6e82-soup5459-l9$bestn
verr=$(get $OUT.beam-eval-report.json errors)
echo "[val] cer=$(get $OUT.beam-eval-report.json cer) errors=$verr exact=$(get $OUT.beam-eval-report.json exact)"
if [ "$verr" -ge 8601 ]; then echo "[stop] validation gate not earned"; exit 0; fi

echo "[stage] test beam 8 (gate earned; best 8186 / 3.702% / 8892)"
TOUT=$CDIR/test-beam8-ens4-e62-l6e82-soup5459-l9$bestn
eval_run "$M1 $M3 $M4 $SOUP" test $TOUT 8 4
echo "[test] cer=$(get $TOUT.beam-eval-report.json cer) errors=$(get $TOUT.beam-eval-report.json errors) exact=$(get $TOUT.beam-eval-report.json exact)"
