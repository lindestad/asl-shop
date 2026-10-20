# Autoresearch journal

Newest entry at the bottom. One block per iteration: idea, config, params, peak
VRAM, wall clock, validation CER/exact, test CER/exact if the run earned a test
pass. Update "Current best" in place when a run beats it.

## Current best

- offline transformer, gap-preserving timeline representation, 8,828,348 params
  (train-slice 12,000, batch 16, 100 epochs): **11.50% CER**, 35.07% exact
  (12,521 test strings, 20 Oct 2026,
  `data/models/openfs-recognizer-gaps-12k/test-eval.eval-report.json`)
- previous: causal TCN, 835,453 params: 13.54% CER, 18.11% exact (12,521 test
  strings, 12 Oct 2026, `docs/training-results-2026-10-12.md`)

## Iterations

(seed baseline comes from the 12 October measurement)

### 2026-10-20 — streaming student on the full train set (rf61, refined labels, no distill)

- Idea: earlier 12k-slice streaming runs plateaued at 23–26% CER; test whether the
  same recipe scales with 10x data (all 126,131 train clips, matching the baseline's
  data volume) plus frame labels refined by the released recognizer, and no teacher
  posteriors (distillation hurt in prior docs).
- Config: 61-frame causal TCN, dilations [1,2,4,8], hidden 192, CTC + auxiliary
  frame CE, no distillation (KL=0), batch 64, fp16 AMP, augmentation on, 30-epoch
  cap, patience 8, seed 20261020; test gated behind validation CER < 0.1407.
- Params: 835,453 (same count as baseline; baseline has a 40-frame RF, so 61 frames
  costs 50% more algorithmic latency).
- Peak VRAM: 1,655 MiB. Wall clock: 8,522 s training (2.4 h), 30 epochs.
- Validation (12,015 clips): **18.71% CER**, 7.07% exact; best epoch 28, plateau
  ~18.7–19.5% after epoch 20.
- Test: skipped (validation gate not earned).
- Verdict: negative. Data volume + refined labels move streaming 23% → 18.7% but do
  not approach the baseline's 14.07% validation CER; the bottleneck is not data
  volume. Retire this streaming recipe.
- In progress at harvest time: offline transformer gaps-12k run
  (`scripts/train_openfs_recognizer.py --train-slice 12000 --batch-size 16
  --epochs 100`, PID 321447, epoch 23/100, validation CER 18.5% and still
  dropping); harvest next iteration.

### 2026-10-20 — harvest: offline transformer, gaps representation, 12k train slice

- Idea: queued harvest of the in-progress run from the previous iteration; the
  pre-loop 2026-10-19 offline docs showed gap-preserving timelines beat compressed
  inputs on full data, but those runs exceeded the 6 GiB budget so they don't count
  against this loop's best.
- Config: offline Transformer recogniser (`scripts/train_openfs_recognizer.py`),
  `--input-representation gaps`, `--train-slice 12000`, batch 16, validation batch
  8, 100 epochs, fp16 AMP, inline augmentation, no early stopping, lr step at
  epoch 80.
- Params: 8,828,348.
- Peak VRAM: 1,517 MiB training, 416 MiB test eval (well inside the 6 GiB budget).
  Wall clock: 18,429 s (5.1 h) training; test eval 30 s.
- Validation (12,015 clips): best epoch 99, **12.90% CER**, 29.71% exact (epoch 24
  was 17.66%, so the lr step at 80 plus long tail mattered a lot).
- Test (earned: 12.90% < 0.1407 gate): **11.50% CER**, 35.07% exact (4,391/12,521
  strings; 25,417/221,100 char errors),
  `data/models/openfs-recognizer-gaps-12k/test-eval.eval-report.json`.
- Verdict: positive, new current best — beats causal TCN 13.54% CER / 18.11%
  exact with 10.6x params but 1/10th of the 2026-10-19 full-data budget.
- Next ideas: (a) same gaps transformer with full 126k train slice at batch 16
  (2026-10-19 full-data gaps run hit 6.98% validation CER, so ~7% test CER is
  plausible, ~50 h wall clock at this batch); (b) shrink the transformer
  (narrower d-model / fewer layers) to probe the param-efficiency frontier
  against the 835k TCN.
