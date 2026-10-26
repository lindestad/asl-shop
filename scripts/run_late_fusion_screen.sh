#!/usr/bin/env bash
# Late (hypothesis-space) fusion screen: one axis, four current ens4 members decoded
# independently at member beam 1, combined by median-hypothesis medoid.
# Stages: (1) sanity, m5 member only, 4k validation subset; must reproduce the recorded
#         single-member greedy screen exactly (2908 err / 2750 exact).
#         (2) 4-member 4k validation screen; gate vs the ens4 frame-level screen baseline
#         2978 err / 2765 exact (screen-subset4k-ens5-drop-m2), margin >= 40 errors.
# Reports: *.late-fusion-eval-report.json.  No training, no checkpoints.
set -uo pipefail
PY=/home/coder/venvs/asl/bin/python
CDIR=data/models/openfs-recognizer-gaps-full-cosine-ema
M1=$CDIR/recognizer-epoch62.pt
M3=data/models/openfs-recognizer-gaps-full-layers6/recognizer-epoch82.pt
M4=data/models/openfs-recognizer-gaps-full-layers6-cosine-ema/recognizer-soup-e54e59-a50.pt
M5=data/models/openfs-recognizer-gaps-full-layers9-cosine-ema/recognizer-soup-e41e43e55-3w.pt
get(){ $PY -c "import json;d=json.load(open('$1'));e=d['evaluations'][0];print(e['$2'])"; }

echo "[stage] sanity single-member m5 beam1 4k (expect 2908 err / 2750 exact)"
nice -n 19 $PY scripts/late_fusion_eval_openfs_recognizer.py --checkpoint "$M5" \
  --data data/openfs-packed --splits validation --member-beam 1 --limit 4000 --batch-size 4 \
  --output data/models/openfs-recognizer-gaps-full-layers9-cosine-ema/lf-sanity-m5-4k
ERR=$(get data/models/openfs-recognizer-gaps-full-layers9-cosine-ema/lf-sanity-m5-4k.late-fusion-eval-report.json errors)
EXA=$(get data/models/openfs-recognizer-gaps-full-layers9-cosine-ema/lf-sanity-m5-4k.late-fusion-eval-report.json exact)
echo "[sanity] errors=$ERR exact=$EXA"
if [ "$ERR" != 2908 ] || [ "$EXA" != 2750 ]; then echo "[abort] sanity mismatch; medoid/decode path suspect"; exit 1; fi

echo "[stage] 4-member late-fusion 4k screen (baseline 2978 err / 2765 exact; gate <= 2938)"
nice -n 19 $PY scripts/late_fusion_eval_openfs_recognizer.py --checkpoint "$M1" "$M3" "$M4" "$M5" \
  --data data/openfs-packed --splits validation --member-beam 1 --limit 4000 --batch-size 4 \
  --output "$CDIR/lf-screen4k-ens4"
MED=$(get "$CDIR/lf-screen4k-ens4.late-fusion-eval-report.json" errors)
echo "[screen] medoid errors=$MED exact=$(get "$CDIR/lf-screen4k-ens4.late-fusion-eval-report.json" exact)"
echo "[screen] cross-member-mbr errors=$($PY -c "import json;d=json.load(open('$CDIR/lf-screen4k-ens4.late-fusion-eval-report.json'));e=d['evaluations'][0];print(e['cross_member_mbr']['errors'],e['cross_member_mbr']['exact'])")"
echo "[done] baseline 2978; gate 2938; medoid=$MED"
