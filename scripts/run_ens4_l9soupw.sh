#!/usr/bin/env bash
# m5 soup WEIGHTING: the l9-slot member is the equal-weight 3-way soup (e41+e43+e55);
# equal thirds were built by construction, never screened against weighted variants.
# Retired axes are m4-slot soup weights and m5 BACKWARD extension - forward weight
# tilt in the one slot with a monotone member-led transfer history is untested.
# Candidates: w41h (e41 0.50, e43 0.25, e55 0.25) and w55h (e55 0.50, e43 0.25,
# e41 0.25) as m5 swaps into the pruned ens4 (e62 + l6e82 + soup5459 unchanged).
# stages: soups -> 4k greedy member screens (baseline s3w 2908 err / 2750 exact,
# margin gate >=40) -> member greedy full-val confirm (s3w 9004 err / 4.2466% /
# 7992 exact, need >=60 better) -> ens4 validation beam-8 (gate 8601 / 4.057% /
# 8153) -> test beam-8 (best 8186 / 3.702% / 8892). batch 4, expandable_segments.
set -uo pipefail
cd /home/coder/asl-wt/auto-research
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/coder/venvs/asl/bin/python
L9=data/models/openfs-recognizer-gaps-full-layers9-cosine-ema
CDIR=data/models/openfs-recognizer-gaps-full-cosine-ema
M1=$CDIR/recognizer-epoch62.pt
M3=data/models/openfs-recognizer-gaps-full-layers6/recognizer-epoch82.pt
M4=data/models/openfs-recognizer-gaps-full-layers6-cosine-ema/recognizer-soup-e54e59-a50.pt
E41=$L9/recognizer-epoch41.pt
E43=$L9/recognizer-epoch43.pt
E55=$L9/recognizer-epoch55.pt
WA=$L9/recognizer-soup-e41e43e55-w41h.pt
WB=$L9/recognizer-soup-e41e43e55-w55h.pt

get(){ $PY -c "import json,sys;r=json.load(open(sys.argv[1]));e=r['evaluations'][0];print(e[sys.argv[2]])" "$1" "$2"; }
eval_run(){
  local ck="$1" splits="$2" out="$3" beam="$4" batch="$5"; shift 5
  nice -n 19 $PY scripts/beam_eval_openfs_recognizer.py --checkpoint $ck \
    --data data/openfs-packed --splits $splits --beam-size $beam --batch-size $batch \
    --output "$out" "$@" || exit 1
}

echo "[stage] soups (alpha = weight of checkpoint-a per soup script)"
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $E43 --checkpoint-b $E55 --alpha 0.5 --output $L9/tmp-soupw-t1.pt || exit 1
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $L9/tmp-soupw-t1.pt --checkpoint-b $E41 --alpha 0.5 --output $WA || exit 1
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $E41 --checkpoint-b $E43 --alpha 0.5 --output $L9/tmp-soupw-t2.pt || exit 1
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $L9/tmp-soupw-t2.pt --checkpoint-b $E55 --alpha 0.5 --output $WB || exit 1
rm -f $L9/tmp-soupw-t1.pt $L9/tmp-soupw-t2.pt

echo "[stage] 4k greedy member screens (baseline s3w 2908 / 2750 exact)"
declare -A SCRN
for m in w41h:$WA w55h:$WB; do
  n=${m%%:*}; ck=${m#*:}
  eval_run "$ck" validation $L9/screen-subset4k-l9m-$n 1 4 --limit 4000
  SCRN[$n]=$(get $L9/screen-subset4k-l9m-$n.beam-eval-report.json errors)
  echo "[screen] $n errors=${SCRN[$n]} exact=$(get $L9/screen-subset4k-l9m-$n.beam-eval-report.json exact)"
done
base=2908; bestn=none; beste=$base
for n in w41h w55h; do if [ "${SCRN[$n]}" -lt "$beste" ]; then beste=${SCRN[$n]}; bestn=$n; fi; done
echo "[screen] baseline s3w=$base best=$bestn errors=$beste (margin $((base-beste)))"
if [ "$bestn" = none ] || [ $((base-beste)) -lt 40 ]; then echo "[stop] no weighted soup beats s3w by >=40 at 4k"; exit 0; fi
case $bestn in w41h) SOUP=$WA;; w55h) SOUP=$WB;; esac

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
