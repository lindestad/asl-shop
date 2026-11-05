# Offline recogniser reproduction — 19 October 2026

The published OpenFS recogniser (93.7 letter accuracy, 59.4 exact top-1 on the FSboard test
split) has been trained from scratch three times inside this repo's own harness, on our packed
contract, on a V100-SXM3-32GB. Alongside those runs the released checkpoint was re-scored on the
same 12,521 clips, which is what tells us whether the pack and the metric path agree with the
authors. Machine-readable copy: `docs/offline-recognition-metrics-2026-10-19.json`.

Every row is the FSboard packed test split: 12,521 clips, 221,100 reference characters, greedy
autoregressive decoding, no dictionary and no language correction.

## The pack and the metric path agree with the published row

| Row | CER | Letter accuracy | Exact top-1 | Del / Sub / Ins |
|---|---:|---:|---:|---:|
| Published by the authors | 6.29% | 93.71% | 59.32% | n/a |
| Released checkpoint, scored 12 Oct (`scripts/evaluate_openfs_reference.py`) | 6.29% | 93.71% | 59.32% | n/a |
| Released checkpoint, scored 19 Oct (`scripts/finetune_openfs_transfer.py --epochs 0`) | 6.29% | 93.71% | 59.36% | 4,252 / 5,286 / 4,374 |

Two independent harnesses, four months and one codebase apart, agree to 0.005 CER points and
0.03 exact points across 221,100 characters. Per-clip spot checks: 23 of 30 sampled transcripts
are byte-identical between the two paths, and the seven that differ are greedy AR divergence
under different floating point ordering, the reference run having been done on CPU and this one
on the V100. The pack preserves the upstream signal and the error-type breakdown matches the
shape the authors report.

## Three input representations, trained from scratch

All three runs use the same architecture (transformer encoder plus autoregressive
cross-attention decoder, dual-level positional encodings) and the same three paper losses:
character cross entropy, signing-hand attention entropy and soft monotonicity. Only the input
representation and the positional index differ. Trainer: `scripts/train_openfs_recognizer.py`.

| Representation | Input channels | Parameters | Test CER | Letter accuracy | Exact top-1 | Character errors | Best validation CER | Wall clock | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `compressed` (upstream: detection order, arange positions) | 42 | 8,815,804 | 7.01% | 92.99% | 56.62% | 15,491 | 6.57% | 12.1 h | 5,856 |
| `gaps` (the app's timeline: every packed frame, positions from real frame deltas) | 91 | 8,828,348 | **6.50%** | 93.50% | 58.09% | 14,376 | 7.08% | 11.0 h | 7,599 |
| `wrist2d` (the `gaps` timeline, only the 42 wrist-relative coordinates of the dominant hand) | 42 | 8,815,804 | 6.59% | 93.41% | 58.44% | 14,570 | 6.98% | 10.9 h | 7,569 |

At 42 input channels, keeping the real gaps between detections rather than concatenating
detection tracks is worth 0.42 CER points, 7.01 down to 6.59. Widening from the 42 wrist-relative
coordinates to the full 91-channel packed timeline adds another 0.09 on top of that, 6.59 down to
6.50. One seed per representation, so differences under about half a CER point are not separated
by this experiment. Best validation CER ranks `compressed` first and the test split ranks `gaps`
first; validation and test are different signer-disjoint splits, and the gap between them, 6.57
against 7.08, is larger than the differences we are trying to read.

Against the released checkpoint on the same clips, our best from-scratch run is 0.21 CER points
and 1.3 exact top-1 points behind. That difference is in the training recipe rather than the data
plumbing: the released weights score 6.29 CER when fed our own pack.

Against the app's current model, which is 13.54% CER greedy CTC and 10.47% CER with beam search
plus character-LM fusion (`data/lm-rescoring-vps.report.json`, `data/models/neural-2026-10-12/sequence.pt`),
the offline recogniser is about 4 CER points better and is not causal.

## Loading released weights onto our timeline is neutral; training on it is not

Re-scored with released weights on the full test split, moving from detection-order positions to
accumulated frame-delta positions (`gaps42` in `data/transfer-teacher.report.json`) moves CER by
0.04 points the wrong way and exact top-1 by 0.05 points the right way, with 85 fewer deletions,
90 fewer substitutions and 266 more insertions. The published positional encoding does not use
real timing information, so giving it real timing changes nothing. Trained on that same timeline
from initialisation, the same change is worth 0.42 CER points. Our contract change pays off only
when the training side is built for it.

## Chunked decoding costs more than the accuracy advantage

Whole-clip autoregressive decoding does not survive being cut into windows. On a 2,000-clip test
subset with 300-frame windows at 50% overlap, CER rose 1.07 points (6.46% to 7.53%) and exact
top-1 fell 4.3 points (59.0% to 54.7%), with 366 extra insertions. Split by the number of windows
a clip needs, for the released control adapter:

| Windows per clip | Clips | CER | Exact top-1 |
|---|---:|---:|---:|
| 1 | 1,791 | 6.50% | 60.13% |
| 2 | 195 | 12.27% | 8.72% |
| 3 | 13 | 27.67% | 0% |
| 4 | 1 | 112.5% | 0% |

Single-window clips decode identically to the full-clip path, 1,791 of 1,791, so the chunker is
lossless and the decoder is what breaks. The offline model stays an offline or Finish-path tool
until something causal replaces it, and `docs/streaming-latency-2026-10-19.md` covers what a
causal student did and did not buy at 61, 201 and 601 buffered frames.

## Fine-tuning the released weights did not help

Three adapters, 30,000 clips and 4 epochs each, all inside `data/transfer-teacher.report.json`.
Fine-tuning the control adapter on our timeline left test CER unchanged (6.46% before, 6.45%
after on the 2,000-clip subset). Fine-tuning the `gaps42` adapter improved validation and made
test worse, 6.34% to 7.84%. The `gaps91` adapter, which re-initialises the pose embedding to take
our 91 channels, starts from nonsense (110% CER, the released weights do not fit that input width)
and reaches 10.4% CER and 47% exact top-1 after fine-tuning, which is the best offline number
this repo has produced on the app's own channel set. Those three comparisons are not data-paired:
the runs were launched before `scripts/finetune_openfs_transfer.py` pinned the per-epoch clip
sample, and `gaps91` used a learning rate of 3e-5 against 1e-5 for the others. One rerun of the
script as it now stands would fix both.

## Reproduce

On the training box, from a checkout with `data/openfs-packed` linked in:

```
scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation compressed --output data/models/openfs-recognizer-compressed/recognizer.pt
scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation gaps      --output data/models/openfs-recognizer-gaps/recognizer.pt
scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation wrist2d   --output data/models/openfs-recognizer-wrist2d/recognizer.pt

scripts/train_openfs_recognizer.py --data data/openfs-packed --input-representation gaps \
  --output data/models/openfs-recognizer-gaps/test-eval.pt \
  --resume data/models/openfs-recognizer-gaps/recognizer.pt \
  --evaluate-only --evaluate-splits test
```

Defaults are the paper's schedule: 100 epochs, batch 64, Adam at 1e-4, StepLR step 80 gamma 0.1,
auxiliary losses from epoch 6, early stopping off. `--evaluate-only` scores a checkpoint on any
split without training and writes `<stem>.eval-report.json` without touching the checkpoint. The
runs above completed on a Tesla V100-SXM3-32GB under torch 2.9.1+cu126 with fp16 autocast and a
GradScaler: `--amp auto` picks bf16 on Ampere and newer, and sm_70 has no bf16 tensor cores. Peak
VRAM at batch 64 was 7.6 GB for the widest variant. Single-process augmentation is the limiter,
around 530 to 900 clips/s.

## Caveats

Licence. The released weights and everything fit on FSboard data are research-only: "Commercial
use of the source code, model checkpoints, or derivative works is strictly prohibited"
(`data/external/openfs/source/LICENSE`, restated in `data/openfs-checkpoints/*/PROVENANCE.json`).
Daniel ruled on 19 October that this project is research work rather than a commercial one, so that
term covers the runs on this page instead of blocking them. It does rule out a build with
commercial intent carrying any of these weights or anything fit on FSboard data, and if that
direction ever changes the models here have to be retrained from licensed data. Our own checkpoints
carry the contract keys the app loader expects, so nothing currently prevents them being loaded
into a packaged build; a research-only guard in the loader would make the separation enforced
rather than intended, which is worth doing before the question comes up again.

These are not human-domain numbers. Every figure above is scored against FSboard transcripts of
other people's recordings, at clip level. The app-domain measurement the repo has been asking for
is `scripts/eval_held_letters.py`, which replays a recorded session through the real streaming
gate and needs recordings from real signers; `docs/training-results-2026-10-12.md` ends with the
same request.

Checkpoint payloads are at `data/models/openfs-recognizer-{compressed,gaps,wrist2d}/recognizer.pt`
with their training history, training report and test-split evaluation report beside them.
Provenance for the released weights is in `data/openfs-checkpoints/recognizer_fsboard/`.
