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

### 2026-10-20 — full 126k train slice, gaps transformer, paper schedule (launched; in progress at harvest time)

- Idea: next-idea (a). The 2026-10-19 full-data `gaps` run scored 6.50 test CER
  but at batch 64 / 7.6 GB peak, over the 6 GiB budget, so it doesn't count in
  this loop. Same gaps transformer, same paper schedule, full 126,131-clip
  train slice, batch 16 to fit the budget.
- The ~50 h estimate was ~5x too high: measured 424-435 clips/s at batch 16
  (single-process augmentation ceiling), so an epoch is 358 s training + 62 s
  validation = ~420 s, and 100 epochs ~11.7 h, consistent with the pre-loop
  11 h at batch 64. Launched as a multi-iteration run (iter-003/iter-005
  precedent).
- Config: `scripts/train_openfs_recognizer.py --input-representation gaps`, no
  `--train-slice` (full train), batch 16, validation batch 16 (halves
  validation decode; greedy per-sequence output unchanged), 100 epochs, Adam
  1e-4 with lr step at 80, fp16 AMP (V100).
- VRAM: 2,268 MiB training so far, inside the 6 GiB budget (6,059 MiB free at
  launch).
- Status: launched detached (`data/logs/openfs-recognizer-gaps-full.log`,
  `data/logs/openfs-recognizer-gaps-full.pid`, PID 408476), epoch 2/100 at
  harvest time, epoch 1 validation CER 33.38% (warmup). Harvest in later
  iterations; evaluate test (`--evaluate-only`) only if validation beats the
  current best 12.90%.
- Verdict: pending.

### 2026-10-20 — shrink the gaps transformer: ~835k-param variant, 12k slice (launched; in progress at harvest time)

- Idea: previous iteration's next-idea (b). The 11.50% test CER best uses
  8.8M params vs the causal TCN's 835k; probe the param-efficiency frontier
  by shrinking the transformer to TCN scale.
- Config: offline Transformer recogniser (`scripts/train_openfs_recognizer.py`),
  `--input-representation gaps`, `--train-slice 12000`, `--d-model 96 --heads 6
  --layers 3 --feedforward 384`, 100 epochs, Adam 1e-4 with lr step at 80,
  fp16 AMP (V100), seed 42 (default).
- Params: 827,996 — essentially matched to the 835,453-param TCN baseline.
- Batch 8 (not 16): `scripts/gpu_budget.sh 6144` failed at launch (5,479 MiB
  free vs 6,144), so per rule 2 the batch was halved; caveat vs the batch-16
  best run — 2x steps/epoch at unchanged lr. Validation batch 16 (greedy
  per-sequence output unchanged).
- VRAM: ~680 MiB training (free fell 5,479 → 4,799 MiB with the full-data run
  also active). Wall clock ~67 s train + ~60 s validation = ~2.1 min/epoch,
  ETA ~3.5 h for 100 epochs.
- Status: launched detached (`data/logs/openfs-recognizer-gaps-slim-12k.log`,
  `data/logs/openfs-recognizer-gaps-slim-12k.pid`, PID 414126), epoch 3/100 at
  harvest time. Harvest in a later iteration; evaluate test (`--evaluate-only`)
  only if validation beats the current best 12.90%.
- Verdict: pending.

### 2026-10-20 — cosine LR schedule + 150 epochs for the 8.8M gaps transformer, 12k slice (launched; in progress at harvest time)

- Idea: the current best (12k gaps, val CER 12.90%) had only one LR drop
  (step at 80) and its validation curve was still improving at epoch 100
  (0.1358@81 → 0.1306@91 → 0.1314@100), so the schedule looks under-annealed
  and too short. Test a smoother recipe: 5-epoch linear warmup + cosine to
  zero over 150 epochs, same architecture and data as the best.
- Code: added `--lr-schedule {step,cosine}` and `--warmup-epochs` to
  `scripts/train_openfs_recognizer.py` (defaults keep the old step behaviour,
  so the two in-flight runs are unaffected).
- Config: `scripts/train_openfs_recognizer.py --input-representation gaps
  --train-slice 12000 --epochs 150 --lr-schedule cosine --warmup-epochs 5`,
  default architecture (8,828,348 params, same as best), Adam 1e-4,
  fp16 AMP (V100), seed 42.
- Batch 8 (not 16): `scripts/gpu_budget.sh 6144` failed again at launch
  (4,795 MiB free), so per rule 2 the batch was halved; caveat vs the
  batch-16 best — 2x steps/epoch at unchanged lr. Validation batch 16
  (greedy per-sequence output unchanged).
- VRAM: smoke run peaked at 288 MiB (batch 4); free fell 4,795 → 3,629 MiB
  after launch, i.e. ~1.2 GiB for this run, inside budget. Wall clock
  ~76 s train + ~70 s validation = ~2.4 min/epoch, ETA ~6 h for 150 epochs.
- Status: launched detached (`data/logs/openfs-recognizer-gaps-12k-cosine.log`,
  `data/logs/openfs-recognizer-gaps-12k-cosine.pid`, PID 421615), epoch 2/150
  at harvest time. Harvest in a later iteration; evaluate test
  (`--evaluate-only`) only if validation beats the current best 12.90%.
- Verdict: pending.
