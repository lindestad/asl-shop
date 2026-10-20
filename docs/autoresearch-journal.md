# Autoresearch journal

Newest entry at the bottom. One block per iteration: idea, config, params, peak
VRAM, wall clock, validation CER/exact, test CER/exact if the run earned a test
pass. Update "Current best" in place when a run beats it.

## Current best

- offline transformer, gap-preserving timeline representation, 8,828,348 params
  (full 126,131-clip train slice, batch 16; epoch-19 validation-best checkpoint of
  a 100-epoch run still training): **8.04% CER**, 48.62% exact (12,521 test
  strings, 20 Oct 2026,
  `data/models/openfs-recognizer-gaps-full/test-eval-epoch19.eval-report.json`)
- previous: same architecture/data, epoch-11 validation-best checkpoint of the
  100-epoch full-data run: 10.53% CER, 42.48% exact (12,521 test strings,
  20 Oct 2026,
  `data/models/openfs-recognizer-gaps-full/test-eval-epoch11.eval-report.json`)
- previous: offline transformer gaps, train-slice 12,000, batch 16, 100 epochs,
  same 8,828,348 params: 11.50% CER, 35.07% exact (20 Oct 2026,
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

### 2026-10-20 — EMA weight averaging paired with the cosine LR control, 12k slice (launched; in progress at harvest time)

- Idea: weight averaging is a low-risk regularizer that can stabilize CTC
  transformer training and improve late-epoch validation without changing the
  architecture or data. Run it against the active cosine-LR control so the
  only intended experiment delta is EMA.
- Code: added `--ema-decay` and an `Ema` helper to
  `scripts/train_openfs_recognizer.py`. EMA updates after each optimizer step,
  validates with shadow weights via swap-in, saves the swapped-in checkpoint,
  then restores live training weights. Startup JSON and report include
  `ema_decay`; fixed a swap/restore aliasing bug by cloning saved live weights
  (`p.detach().clone().cpu()`). Smoke run confirmed EMA validation/checkpoint
  path and peaked at 364 MiB with batch 4.
- Config: `scripts/train_openfs_recognizer.py --input-representation gaps
  --train-slice 12000 --epochs 150 --lr-schedule cosine --warmup-epochs 5
  --ema-decay 0.9995`, default architecture (8,828,348 params), Adam 1e-4,
  fp16 AMP (V100), seed 42.
- Batch 8 (not 16): `scripts/gpu_budget.sh 6144` failed at launch
  (3,375 MiB free), so per rule 2 the batch was halved; same caveat as the
  cosine control vs the batch-16 best — 2x steps/epoch at unchanged lr.
  Validation batch 16.
- VRAM: 1,214 MiB observed after launch. Wall clock epoch 1: 82 s train + 86 s
  validation = ~2.8 min/epoch, ETA ~7 h for 150 epochs.
- Status: launched detached
  (`data/logs/openfs-recognizer-gaps-12k-cosine-ema.log`,
  `data/logs/openfs-recognizer-gaps-12k-cosine-ema.pid`, PID 446261), epoch
  2/150 at harvest time. Epoch 1 validation: 8.22% CER, 0.00% exact. Harvest
  in a later iteration; evaluate test (`--evaluate-only`) only if validation
  beats the current best 12.90%.
- Verdict: pending.

### 2026-10-20 — harvest the full-data gaps transformer at epoch 11: new best 10.53% CER, 42.48% exact

- Idea: no new training job was safe because only 2,015 MiB was free and
  `scripts/gpu_budget.sh 6144` failed while four training jobs were active.
  Instead, harvest the already-running full-data gaps transformer if it had
  crossed the validation gate.
- Gate: epoch 10 validation was 11.27% CER / 35.93% exact and epoch 11 was
  11.00% CER / 37.49% exact, both better than the prior 12.90% validation best,
  so test evaluation was permitted.
- Evaluation: loaded the saved validation-best checkpoint with
  `scripts/train_openfs_recognizer.py --evaluate-only --evaluate-splits test`.
  Epoch 10 test: 10.83% CER / 39.96% exact
  (`data/models/openfs-recognizer-gaps-full/test-eval-epoch10.eval-report.json`).
  Epoch 11 test: **10.53% CER / 42.48% exact**
  (`data/models/openfs-recognizer-gaps-full/test-eval-epoch11.eval-report.json`).
- Cost: evaluation used 416 MiB peak VRAM and 94 s; no checkpoint was written and
  the training run was left in place.
- Caveat: this is an intermediate checkpoint from a still-running 100-epoch full
  experiment, not a completed experiment. Current best is updated to the epoch-11
  checkpoint because it beat the prior test best on the fixed FSboard test split.
- Other active runs: full-data epoch 11 saved / epoch 12 in progress; slim-12k
  epoch 32 at 24.47% CER / 10.44% exact; cosine-12k epoch 24 at 17.03% CER /
  19.52% exact; cosine-EMA epoch 8 at 31.70% CER / 6.70% exact.
- Verdict: positive for the full-data gaps direction; continue the active runs and
  do not launch more training until VRAM frees up.

### 2026-10-20 — harvested epoch 19 full-data gaps checkpoint: new best 8.04% CER, 48.62% exact

- Idea: with no room for new training, harvest the latest validation-best checkpoint
  from the already-running full-data gaps transformer.
- Gate: `recognizer.pt` still corresponded to epoch 19: validation CER **8.57%** and
  exact **44.44%**. Epochs 20-22 did not beat that validation CER (epoch 20: 9.67%
  CER; epoch 21: 8.73% CER; epoch 22: 8.94% CER), so the checkpoint was valid and
  test evaluation was permitted.
- Evaluation: ran a 32-clip validation smoke, then greedy `--evaluate-only` on the
  full `test` split using `--batch-size 4` and `nice -n 19` to coexist with active
  training.
- Test: **8.04% CER / 48.62% exact**
  (`data/models/openfs-recognizer-gaps-full/test-eval-epoch19.eval-report.json`),
  beating the epoch-11 best by 2.49 pp CER and 6.14 pp exact.
- Cost: evaluation used 70 MiB peak VRAM and 489 s; no checkpoint was written and
  the training run was left in place.
- Caveat: this remains an intermediate checkpoint from a still-running 100-epoch
  full experiment. Current best is updated to the epoch-19 checkpoint because it
  beat the previous fixed FSboard test result.
- Verdict: positive; harvest future full-data test evaluations only when the
  validation-best checkpoint advances.

### 2026-10-20 — harvested epoch 32 full-data gaps checkpoint: new best 7.62% CER, 53.30% exact; slim-12k transformer is a negative result

- Idea: with no VRAM headroom for new training, harvest the latest validation-best
  checkpoint from the still-running full-data gaps transformer, and record the
  completed slim-12k parameter-efficiency experiment.
- Gate: `recognizer.pt` now corresponds to epoch 32: validation CER **7.81%** and
  exact **49.32%**, beating the epoch-19 validation-best (8.57% / 44.44%). Epochs
  33-37 did not beat epoch 32, so the checkpoint was valid and test evaluation was
  permitted.
- Evaluation: ran a 32-clip validation smoke, then greedy `--evaluate-only` on the
  full `test` split using `--batch-size 4` and `nice -n 19` to coexist with active
  training.
- Test: **7.62% CER / 53.30% exact**
  (`data/models/openfs-recognizer-gaps-full/test-eval-epoch32.eval-report.json`),
  beating the epoch-19 best by 0.42 pp CER and 4.68 pp exact.
- Cost: evaluation used 70 MiB peak VRAM and 449 s; no checkpoint was written and
  the training run was left in place.
- Slim transformer (`openfs-recognizer-gaps-slim-12k`,
  `data/models/openfs-recognizer-gaps-slim-12k/recognizer.report.json`): completed
  100 epochs on the 12,000-clip slice with **827,996 parameters** (10.7x smaller
  than the full model, 267 MiB peak VRAM) but reached only **14.47% validation
  CER**, worse than both the 12.90% standard-transformer gate and the 14.07% TCN
  baseline at similar scale. Verdict: negative - at ~0.8M parameters the
  gap-preserving transformer is not better than the TCN; no test evaluation earned.
- Other active runs: full-data epoch 40 in progress (validation best still epoch
  32); cosine-12k epoch 113, validation best 13.13% at epoch 105; cosine-EMA
  epoch 97, validation best **12.35%** at epoch 58 - now the best 12k run, slightly
  ahead of the plain 12k run's 12.90%, but far from the full-data best.
- Caveat: this remains an intermediate checkpoint from a still-running 100-epoch
  full experiment. Current best is updated to the epoch-32 checkpoint because it
  beat the previous fixed FSboard test result.
- Verdict: positive for the full-data gaps direction; the cosine-EMA schedule is
  worth a full-data replication once training capacity frees up.
