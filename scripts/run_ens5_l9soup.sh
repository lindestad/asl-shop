#!/usr/bin/env bash
# l9-slot soup member upgrade: within-trajectory soup of l9-cos-EMA snapshots as m5 swap
# into ens5 (e62 + step-84 + l6-e82 + soup5459 unchanged). The soup lever produced the
# current best m4 member but has never been applied to the m5 (l9) slot, the only slot
# with a monotone member-led transfer history.
# stages: soups -> 4k greedy member screens -> member greedy full-val confirm ->
# ens5 validation beam-8 (gate 8694 errors / 4.100%) -> test beam-8 (best 8246 / 3.730%).
# batch 4 (GPU contended, ~5.4 GiB free, two training jobs active).
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
E41=$L9/recognizer-epoch41.pt
E43=$L9/recognizer-epoch43.pt
E55=$L9/recognizer-epoch55.pt
S4155=$L9/recognizer-soup-e41e55-a50.pt
S4355=$L9/recognizer-soup-e43e55-a50.pt
S3W=$L9/recognizer-soup-e41e43e55-3w.pt
S2=$L9/recognizer-soup-e41e43-a50-tmp.pt

get(){ $PY -c "import json,sys;r=json.load(open(sys.argv[1]));e=r['evaluations'][0];print(e[sys.argv[2]])" "$1" "$2"; }
eval_run(){ # ckpt splits out beam batch limit...
  local ck="$1" splits="$2" out="$3" beam="$4" batch="$5"; shift 5
  nice -n 19 $PY scripts/beam_eval_openfs_recognizer.py --checkpoint $ck \
    --data data/openfs-packed --splits $splits --beam-size $beam --batch-size $batch \
    --output "$out" "$@"
}

echo "[stage] soups"
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $E41 --checkpoint-b $E55 --alpha 0.5 --output $S4155 || exit 1
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $E43 --checkpoint-b $E55 --alpha 0.5 --output $S4355 || exit 1
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $E41 --checkpoint-b $E43 --alpha 0.5 --output $S2 || exit 1
$PY scripts/soup_openfs_recognizer.py --checkpoint-a $S2 --checkpoint-b $E55 --alpha 0.6666666667 --output $S3W || exit 1
rm -f $S2

echo "[stage] 4k greedy member screens (batch 4)"
declare -A SCRN
for m in e43:$E43 e55:$E55 s4155:$S4155 s4355:$S4355 s3w:$S3W; do
  n=${m%%:*}; ck=${m#*:}
  eval_run "$ck" validation $L9/screen-subset4k-l9m-$n 1 4 --limit 4000 || exit 1
  SCRN[$n]=$(get $L9/screen-subset4k-l9m-$n.beam-eval-report.json errors)
  echo "[screen] $n errors=${SCRN[$n]} exact=$(get $L9/screen-subset4k-l9m-$n.beam-eval-report.json exact)"
done

p43=${SCRN[e43]}; p55=${SCRN[e55]}
bestn=s4155; beste=${SCRN[s4155]}
for n in s4355 s3w; do if [ "${SCRN[$n]}" -lt "$beste" ]; then beste=${SCRN[$n]}; bestn=$n; fi; done
parents=$p43; [ "$p55" -lt "$parents" ] && parents=$p55
echo "[screen] best soup $bestn errors=$beste vs best-parent $parents (margin $((parents-beste)))"
if [ $((parents-beste)) -lt 60 ]; then echo "[abort] soup does not beat both parents by >=60 errors at 4k"; exit 0; fi

case $bestn in s4155) SOUP=$S4155;; s4355) SOUP=$S4355;; s3w) SOUP=$S3W;; esac
echo "[stage] member greedy full validation for $bestn"
eval_run "$SOUP" validation $L9/validation-greedy-l9m-$bestn 1 4 || exit 1
mcer=$(get $L9/validation-greedy-l9m-$bestn.beam-eval-report.json cer)
merr=$(get $L9/validation-greedy-l9m-$bestn.beam-eval-report.json errors)
meexact=$(get $L9/validation-greedy-l9m-$bestn.beam-eval-report.json exact)
echo "[member] $bestn cer=$mcer errors=$merr exact=$meexact (parents e43 4.422 / e55 4.417)"
if [ "$merr" -ge 9366 ]; then echo "[abort] member not better than e55 (4.4173% = 9366 errors) at full val"; exit 0; fi

echo "[stage] ens5 validation beam 8 (m5 -> $bestn soup)"
eval_run "$M1 $M2 $M3 $M4 $SOUP" validation $CDIR/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9soup 8 4 || exit 1
verr=$(get $CDIR/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json errors)
vcer=$(get $CDIR/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json cer)
echo "[val] cer=$vcer errors=$verr exact=$(get $CDIR/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json exact) (gate 8694 / 4.100%)"
if [ "$verr" -ge 8694 ]; then echo "[stop] validation gate not earned"; exit 0; fi

echo "[stage] test beam 8 (gate earned)"
eval_run "$M1 $M2 $M3 $M4 $SOUP" test $CDIR/test-beam8-ens4-e62-s84-l6e82-soup5459-l9soup 8 4 || exit 1
ter=$(get $CDIR/test-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json errors)
tcer=$(get $CDIR/test-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json cer)
echo "[test] cer=$tcer errors=$ter exact=$(get $CDIR/test-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json exact) (best 8246 / 3.730%)"
