# Autoresearch journal

Newest entry at the bottom. One block per iteration: idea, config, params, peak
VRAM, wall clock, validation CER/exact, test CER/exact if the run earned a test
pass. Update "Current best" in place when a run beats it.

## Current best

- causal TCN, 835,453 params: **13.54% CER**, 18.11% exact (12,521 test strings,
  12 Oct 2026, `docs/training-results-2026-10-12.md`)

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
