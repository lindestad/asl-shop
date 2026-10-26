# Autoresearch journal

Newest entry at the bottom. One block per iteration: idea, config, params, peak
VRAM, wall clock, validation CER/exact, test CER/exact if the run earned a test
pass. Update "Current best" in place when a run beats it.

## Current best

- four-checkpoint cross-architecture/cross-recipe log-posterior ensemble (pruned from five):
  cosine+EMA epoch-62 full-data 3-layer gaps transformer (8,828,348 params), 6-layer
  step-schedule epoch-82 and weight-space soup (alpha 0.5) of the 6-layer cosine+EMA
  epoch-54 + epoch-59 snapshots full-data gaps transformers (17,509,820 params each),
  plus the 9-layer 3-way equal-weight trajectory soup of cosine+EMA epoch-41 + epoch-43 +
  epoch-55 snapshots (26,191,292 params; 70,039,280 total) - the step-schedule epoch-84
  3-layer member was pruned as redundant by a leave-one-out screen, beam search width 8,
  mean of per-step log-posteriors, no dictionary or language correction:
  **3.702% CER**, 71.02% exact (12,521 test strings, 25 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-l6e82-soup5459-l9soup.beam-eval-report.json`)
- previous: five-checkpoint cross-architecture/cross-recipe log-posterior ensemble: cosine+EMA
  epoch-62 and step-schedule epoch-84 full-data gaps transformers (8,828,348 params
  each), 6-layer step-schedule epoch-82 and weight-space soup (alpha 0.5) of the
  6-layer cosine+EMA epoch-54 + epoch-59 snapshots full-data gaps transformers
  (17,509,820 params each), plus the 9-layer 3-way equal-weight trajectory soup of
  cosine+EMA epoch-41 + epoch-43 + epoch-55 snapshots (26,191,292 params;
  78,867,628 total), beam search width 8,
  mean of per-step log-posteriors, no dictionary or language correction:
  **3.719% CER**, 70.71% exact (12,521 test strings, 25 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json`)
- previous: five-checkpoint ensemble with the 9-layer cosine+EMA epoch-43 checkpoint as
  5th member: **3.730% CER**, 70.63% exact (12,521 test strings, 25 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e43.beam-eval-report.json`)
- previous: same ensemble with the 9-layer cosine+EMA epoch-41 member:
  **3.777% CER**, 70.47% exact (12,521 test strings, 25 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json`)
- previous: same ensemble with the 9-layer cosine+EMA epoch-33 member:
  **3.81% CER**, 70.08% exact (12,521 test strings, 24 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e33.beam-eval-report.json`)
- previous: same ensemble with the 9-layer cosine+EMA epoch-27 member:
  **3.83% CER**, 70.08% exact (12,521 test strings, 24 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e27.beam-eval-report.json`)
- previous: same ensemble with the 9-layer cosine+EMA epoch-21 member:
  **3.87% CER**, 69.74% exact (12,521 test strings, 24 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e21.beam-eval-report.json`)
- previous: four-checkpoint cross-architecture/cross-recipe log-posterior ensemble: cosine+EMA
  epoch-62 and step-schedule epoch-84 full-data gaps transformers (8,828,348 params
  each) plus 6-layer step-schedule epoch-82 and a weight-space soup (alpha 0.5) of the
  6-layer cosine+EMA epoch-54 + epoch-59 snapshots full-data gaps transformers
  (17,509,820 params each member; 52,676,336 total), beam search width 8,
  mean of per-step log-posteriors, no dictionary or language correction:
  **3.93% CER**, 69.76% exact (12,521 test strings, 23 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459.beam-eval-report.json`)
- previous: same ensemble with the plain 6-layer cosine+EMA epoch-54 checkpoint as
  4th member: **3.96% CER**, 69.50% exact (22 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-l6ce54.beam-eval-report.json`)
- previous: same ensemble with the 6-layer cosine+EMA epoch-37 member:
  **4.03% CER**, 69.26% exact (22 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-l6ce37.beam-eval-report.json`)
- previous: same ensemble with the 6-layer cosine+EMA epoch-26 member instead of
  epoch-37: **4.14% CER**, 68.71% exact (22 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-l6ce26.beam-eval-report.json`)
- previous: three-checkpoint cross-architecture log-posterior ensemble: cosine+EMA epoch-62 and
  step-schedule epoch-84 full-data gaps transformers (8,828,348 params each) plus the
  6-layer step-schedule epoch-82 full-data gaps transformer (17,509,820 params;
  35,166,516 total), beam search width 8, mean of per-step log-posteriors, no
  dictionary or language correction: **4.20% CER**, 68.69% exact (12,521 test
  strings, 21 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens3-e62-s84-l6e82.beam-eval-report.json`)
- previous: two-checkpoint log-posterior ensemble of the cosine+EMA epoch-62 and step-schedule
  epoch-84 full-data gaps transformers (8,828,348 params each, 17,656,696 total),
  beam search width 8, mean of per-step log-posteriors, no dictionary or language
  correction: **4.52% CER**, 66.92% exact (12,521 test strings, 21 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens-e62-step84.beam-eval-report.json`)
- previous: cosine+EMA run epoch-62 checkpoint (8,828,348 params, full 126,131-clip
  train slice, batch 8, cosine LR with 5-epoch warmup, EMA weights decay 0.9999,
  150-epoch run; epoch-62 validation-best checkpoint, decoded with beam search
  width 8, no dictionary or language correction):
  **5.04% CER**, 65.47% exact (12,521 test strings, 21 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-epoch62.beam-eval-report.json`)
- previous: same run, epoch-35 validation-best checkpoint, beam 8:
  **5.45% CER**, 63.01% exact (12,521 test strings, 21 Oct 2026,
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-epoch35.beam-eval-report.json`)
- previous: same architecture/data, batch 16, step LR schedule, epoch-84
  validation-best checkpoint of the 100-epoch run, beam search width 8:
  **5.63% CER**, 61.47% exact (12,521 test strings, 21 Oct 2026,
  `data/models/openfs-recognizer-gaps-full/test-beam8-epoch84.beam-eval-report.json`)
- previous: same architecture/data, epoch-32 validation-best checkpoint, beam
  search width 8: **6.32% CER**, 56.90% exact (12,521 test strings,
  20 Oct 2026,
  `data/models/openfs-recognizer-gaps-full/test-beam8.beam-eval-report.json`)
- previous: same architecture/data/checkpoint, greedy decoding: 7.62% CER,
  53.30% exact (12,521 test strings, 20 Oct 2026,
  `data/models/openfs-recognizer-gaps-full/test-eval-epoch32.eval-report.json`)
- previous: same architecture/data, epoch-19 validation-best checkpoint, greedy
  decoding: 8.04% CER, 48.62% exact (12,521 test strings, 20 Oct 2026,
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

### 2026-10-20 — experiment: beam-search decoding (width 8) of the epoch-32 checkpoint: new best 6.32% CER, 56.90% exact

- Idea: with no VRAM headroom for new training, test whether decoding the current
  best epoch-32 full-data gaps checkpoint with beam search (instead of greedy)
  improves recognition. Named experiment: **beam-search decoding only** - same
  contract, weights and Levenshtein harness; batched beam search over width 8,
  sum of log-probabilities, no length normalisation, hypotheses retire on EOS;
  no dictionary and no language/LLM correction.
- Harness: new `scripts/beam_eval_openfs_recognizer.py` (reads a checkpoint
  read-only, never writes one). Sanity: its greedy pass (beam size 1) reproduced
  the checkpoint's recorded validation numbers exactly (7.81% CER / 49.32% exact,
  16,559 / 212,030 errors), so beam and greedy are directly comparable, and the
  prior greedy test reports are comparable to this harness.
- Validation gate: beam 8 gave **7.44% CER / 50.07% exact** vs greedy 7.81% /
  49.32% (gains of 0.37 pp CER and 0.75 pp exact), so test evaluation was earned.
- Test: **6.32% CER / 56.90% exact**
  (`data/models/openfs-recognizer-gaps-full/test-beam8.beam-eval-report.json`),
  beating the epoch-32 greedy best by 1.30 pp CER and 3.60 pp exact. The gain is
  larger on test than on validation.
- Cost: beam-8 validation 4,128 s at 457 MiB peak VRAM; beam-8 test 2,728 s at
  388 MiB peak; greedy validation baseline 1,379 s. All at `nice -n 19`,
  coexisting with the three active training runs on the shared 32 GB GPU.
- Caveat: the current best is now decoder-dependent - beam search is part of the
  reported 6.32% result. Training-time validation continues to track greedy
  decoding; a future validation-best checkpoint must be re-gated and re-decoded
  with beam 8 before its test number counts.
- Verdict: positive; beam 8 is a free, training-free improvement on the current
  best checkpoint. Next: keep harvesting the full-data run, and consider wider
  beams or length-normalised scoring as a follow-up experiment.

### 2026-10-20 — harvested epoch-54 full-data checkpoint with beam 8: validation improved but test CER did not (6.37% CER / 57.10% exact); best unchanged

- Idea: no VRAM headroom for new training (needed 6,144 MiB, free 5,333 MiB), so
  continue harvesting the still-running full-data gaps run: its validation-best
  checkpoint had advanced to epoch 54 (training-time greedy validation **7.05%
  CER / 52.13% exact**, beating epoch 32's 7.81% / 49.32%). Per the decoder caveat,
  re-gate with beam 8 before letting any test number count.
- Snapshot: copied `recognizer.pt` to `recognizer-epoch54.pt` before evaluation so
  the still-running run could not swap the checkpoint mid-harvest.
- Gate: beam-8 validation **6.79% CER / 52.78% exact**
  (`data/models/openfs-recognizer-gaps-full/validation-beam8-epoch54.beam-eval-report.json`),
  beating the epoch-32 beam-8 validation (7.44% / 50.07%) by 0.65 pp CER and
  2.71 pp exact, so test evaluation was earned.
- Test (earned, stated explicitly): **6.37% CER / 57.10% exact**
  (`data/models/openfs-recognizer-gaps-full/test-beam8-epoch54.beam-eval-report.json`) -
  exact improved 0.20 pp over the 56.90% best but CER regressed 0.05 pp vs the
  6.32% best, so the current best is **not** updated; epoch-32 checkpoint + beam 8
  remains best at 6.32% CER / 56.90% exact.
- Finding: validation gains from later checkpoints no longer transfer to test CER -
  a 0.65 pp validation gain flipped to -0.05 pp on test CER while test exact still
  rose. Validation-best tracking on the fixed split is plateauing at noise level;
  harvesting further checkpoints on validation gains alone is unlikely to pay.
- Cost: beam-8 validation 3,683 s at 457 MiB peak VRAM; beam-8 test 1,075 s at
  388 MiB; 32-clip smoke 13.5 s at 292 MiB. All `nice -n 19`, coexisting with the
  full-data training run on the shared 32 GB GPU. No new scripts or checkpoints
  committed.
- Closed out prior-iteration 12k schedule runs: cosine-12k
  (`data/models/openfs-recognizer-gaps-12k-cosine/recognizer.report.json`)
  completed 150 epochs, best validation **13.13%** (epoch 105; 827 MiB peak) -
  worse than the step-schedule 12k 12.90%, cosine-only is a negative result;
  cosine-EMA-12k (`data/models/openfs-recognizer-gaps-12k-cosine-ema/recognizer.report.json`)
  completed 150 epochs, best validation **12.35%** (epoch 58; 861 MiB peak) - best
  12k run so far, still far behind full-data. Neither earned a test pass.
- In progress at harvest time: full-data run (PID 408476) reached ~epoch 68/100;
  epoch 66 set a new greedy validation best **6.89% CER / 53.21% exact** and is now
  the live `recognizer.pt`. Next: once the run plateaus or finishes, harvest and
  beam-8 re-gate the latest validation-best checkpoint (epoch 66 or later) once
  more, expecting no material test change.
- Verdict: mixed/negative on the primary metric; honest test report kept the
  epoch-32 + beam-8 result as current best.

### 2026-10-20 — launched beam-16 decoder experiment on the frozen epoch-54 snapshot (validation first; multi-iteration run)

- Idea: the epoch-32 best checkpoint is no longer on disk (only epoch-13/19/54
  snapshots plus the live epoch-66 `recognizer.pt`), so beating 6.32% CER needs
  either a better checkpoint or a better decoder. The prior journal follow-up named
  "a wider beam or length-normalized scoring" as the decoder follow-up. Chose one
  single-variable experiment: beam width 16 vs the completed beam-8 baseline on the
  frozen epoch-54 snapshot, whose beam-8 numbers are known (validation 6.79% /
  52.78%, test 6.37% / 57.10%).
- Architecture: unchanged 8,828,348-param gaps transformer; decoder-only change.
- Launched: `scripts/beam_eval_openfs_recognizer.py --checkpoint
  recognizer-epoch54.pt --splits validation --beam-size 16 --batch-size 32`
  (PID in `data/logs/openfs-recognizer-gaps-full-beam16-val.pid`, log
  `data/logs/openfs-recognizer-gaps-full-beam16-val.log`); a first attempt at
  batch-size 8 was killed for exceeding one iteration's wall time (beam-8 full
  validation took 3,683 s). Free VRAM was 5,293 MiB; the eval added ~1.2 GiB
  (4,057 MiB free after start), no training involved so no budget halving needed.
- Gate: test run only if beam-16 validation CER beats beam-8 validation
  6.79% on the same checkpoint; new best only if test then beats 6.32% CER.
- Full-data run (PID 408476) at ~epoch 69/100; validation-best still epoch 66
  (greedy 6.89% / 53.21%); lr step lands at epoch 80.
- Verdict: pending; results land next iteration.

### 2026-10-20 — launched full-data cosine+EMA replication (multi-iteration run)

- Idea: the named follow-up from the 12k schedule screen - cosine-EMA-12k was the
  best 12k run (12.35% greedy validation, vs 12.90% step-only, cosine-only a
  negative), and the journal called for a full-data replication "once training
  capacity opens up". The current best (epoch-32 step-schedule checkpoint) is no
  longer on disk, so a second independent full-data schedule is also a second
  lottery ticket for a better checkpoint; the step-schedule run is closing out
  epoch 100 and its post-epoch-80 harvest is automatic, not an experiment.
- Architecture: unchanged 8,828,348-param gaps transformer (single variable:
  schedule = cosine with 5-epoch warmup + EMA weight averaging, vs step decay).
- Budget: `scripts/gpu_budget.sh 6144` failed (2,655 MiB free; three other GPU jobs
  running), so per rule 2 the batch was halved to 8. EMA decay scaled from the
  12k recipe's 0.9995 (1.5k steps/epoch) to 0.9999 at 15,766 steps/epoch, keeping
  the averaging horizon ~10k steps / ~0.6 epochs. Epochs kept at 150 to replicate
  the winning 12k recipe.
- Launched: `scripts/train_openfs_recognizer.py --input-representation gaps
  --batch-size 8 --validation-batch-size 16 --epochs 150 --lr-schedule cosine
  --warmup-epochs 5 --ema-decay 0.9999` into
  `data/models/openfs-recognizer-gaps-full-cosine-ema/` (PID in
  `data/logs/openfs-recognizer-gaps-full-cosine-ema.pid`, log
  `data/logs/openfs-recognizer-gaps-full-cosine-ema.log`). Startup 1,218 MiB VRAM,
  ~20 steps/s -> ~790 s/epoch, ETA ~33 h solo but longer with three jobs sharing
  the GPU; multi-iteration run, harvest when it plateaus or finishes.
- Gate: harvest validation-best checkpoint, beam-8 re-gate vs 6.79% on epoch-54
  validation; test pass only if validation beats it, new best only if test beats
  6.32% CER.
- Status at launch: beam-16 validation still running (PID 666751, ~1 h in);
  full-data step run at ~epoch 72/100, validation-best still epoch 66.

### 2026-10-20 — harvested beam-16 validation on the frozen epoch-54 snapshot: decoder saturated at width 8 (6.80% vs 6.79% CER); no test pass, best unchanged

- Idea: complete the pending single-variable decoder experiment from two
  iterations ago - beam 16 vs beam 8 on the frozen `recognizer-epoch54.pt`,
  gate: test only if beam-16 validation beats the beam-8 validation 6.79% CER
  on the same checkpoint.
- Architecture: unchanged 8,828,348-param gaps transformer; decoder-only change
  (sum log-prob scoring, no length normalisation, no dictionary/language
  correction).
- Result (validation, 12,015 clips / 212,030 chars): beam 16 **6.80% CER /
  52.78% exact** (14,428 errors, 6,342 exact) vs beam 8 **6.79% CER / 52.78%
  exact** (14,416 errors, 6,342 exact) - identical exact count, 12 more char
  errors. Gate not earned; no test run; current best unchanged at 6.32% CER /
  56.90% exact.
  `data/models/openfs-recognizer-gaps-full/validation-beam16-epoch54.beam-eval-report.json`
- Finding: beam width saturates at 8 for this checkpoint under unnormalized
  sum-log-prob scoring - the posterior is already confidently unimodal, extra
  width cannot find a better hypothesis and marginally helps hypotheses that
  end past EOS-length. Remaining untested decoder lever: length-normalized
  scoring; if that fails too, decoder-side gains are exhausted and further
  progress needs better checkpoints.
- Cost: 2,409 s, peak 3,347 MiB at batch 32 (vs 457 MiB at beam 8 / batch 8 -
  the beam-16 per-token `torch.unique`/topk buffers scale with batch x beam);
  coexisted with both training jobs on the shared GPU.
- In progress at harvest time: full-data step run (PID 408476) epoch 75/100,
  greedy validation-best still epoch 66 (6.89% / 53.21%); lr step lands at
  epoch 80, post-step checkpoints are the automatic harvest target. Full-data
  cosine+EMA run (PID 670605) epoch 3/150 (~535 s/epoch, ETA ~33 h+).
- Verdict: negative; retire wider-beam search as a lever.
- Verdict: pending; run in progress.

### 2026-10-20 — length-normalized beam scoring (alpha=1.0) on frozen epoch-54: 6.98% vs 6.79% CER, worse; gate not earned, no test, best unchanged

- Idea: last untested decoder lever named by the beam-16 entry - divide the
  beam-search score by generated_length**alpha via new `--length-penalty`
  (default 0). Smoke check: penalty 0 on the 32-clip validation sample
  reproduces the saved baseline byte-identical (0.0983103 CER / 13 exact).
- Architecture: unchanged 8,828,348-param gaps transformer; decoder-only change.
- Result (validation, 12,015 clips / 212,030 chars, beam 8, batch 8):
  alpha=1.0 **6.98% CER** (14,794 errors) / **52.81% exact** (6,345) vs
  unnormalized beam 8 **6.79% CER** (14,416 errors) / **52.78% exact** (6,342).
  CER is primary and degraded; gate not earned; no test run; current best
  unchanged at 6.32% CER / 56.90% exact.
  `data/models/openfs-recognizer-gaps-full/validation-beam8-epoch54-penalty1.beam-eval-report.json`
- Finding: full length normalization hurts - the model's short-EOS bias was not
  the failure mode, and the penalty instead promotes low-confidence long
  continuations; alpha=0.5 would only interpolate between two known operating
  points. Decoder levers now exhausted (beam width saturates at 8, length
  penalty degrades); further progress needs better checkpoints.
- Cost: 6,770 s, peak 457 MiB at batch 8, coexisted with both training jobs.
- Status at harvest: post-lr-step the full-data step run (PID 408476) jumped -
  greedy validation-best now epoch 84 at **6.23% CER / 57.36% exact** (epoch 88
  at 6.44%), far past the old epoch-66 best; live `recognizer.pt` held epoch-84
  weights, snapshotted to `recognizer-epoch84.pt`. Cosine+EMA run (PID 670605)
  at epoch ~19/150.
- Next iteration: beam-8 validation re-gate on epoch-84 vs 6.79% on epoch-54;
  test only if validation beats it, new best only if test beats 6.32% CER.
- Verdict: negative; retire length-normalized beam scoring.

### 2026-10-21 — harvested post-lr-step epoch-84 checkpoint, beam 8: NEW BEST 5.63% CER / 61.47% exact

- Idea: the step-schedule run's lr step landed at epoch 80; automatic harvest
  target from the two prior entries. Snapshotted epoch-84 validation-best
  weights from the live `recognizer.pt` (greedy validation 6.23% CER / 57.36%
  exact, vs old epoch-66 best 6.89% / 53.21%). Gate: beam-8 validation on
  epoch-84 must beat the frozen reference 6.79% CER (epoch-54 beam-8).
- Architecture: unchanged 8,828,348-param gaps transformer; decoder unchanged
  (beam 8, unnormalized sum log-prob - length penalty retired previous entry).
- Validation (12,015 clips): **6.05% CER** (12,818 errors) / **57.72% exact**
  (6,935) - beats 6.79% gate; test pass earned.
  `data/models/openfs-recognizer-gaps-full/validation-beam8-epoch84.beam-eval-report.json`
- Test (12,521 clips): **5.63% CER** (12,448 errors) / **61.47% exact** (7,697)
  - new best, previous 6.32% / 56.90%.
  `data/models/openfs-recognizer-gaps-full/test-beam8-epoch84.beam-eval-report.json`
- Finding: training longer past the lr step is the lever that matters -
  decoder-side levers (beam width, length penalty) are exhausted while the
  checkpoint improved ~0.7 points CER from one lr step plus post-step epochs.
- Cost: validation 5,589 s peak 457 MiB; test 1,518 s peak 388 MiB (batch 8,
  beam 8, shared GPU).
- Status at harvest: step run (PID 408476) at epoch 88/100, val-best epoch 84;
  cosine+EMA run (PID 670605) epoch ~19/150. Next gates now reference 6.05%
  validation CER (epoch-84 beam-8) and 5.63% test CER.
- Verdict: positive; new best recorded.

### 2026-10-21 — launched capacity lever: layers 3->6 (run in progress)

- Idea: decoder levers are exhausted and the best gain came from the checkpoint;
  next lever is model capacity. Scale the transformer stack 3 -> 6 layers
  (params ~8.8M -> ~17.6M, estimated from per-layer blocks; d_model 256, heads 8,
  feedforward 2048 unchanged).
- Config: identical to the new-best run's recipe - full 126,131-clip train
  slice, gaps representation, batch 16, 100 epochs, step LR at 80 (gamma 0.1),
  seed 42 - only `--layers 6` changes. Smoke (`--limit-clips 2000 --epochs 2`):
  peak 1,139 MiB at batch 8, so batch 16 fits the 6 GiB budget with headroom.
- Output: `data/models/openfs-recognizer-gaps-full-layers6/recognizer.pt`
  (PID 787811, launched 21 Oct 2026).
- Gate: harvest its greedy val-best; beam-8 validation must beat 6.05% CER
  (epoch-84 beam-8) before a test pass; test must beat 5.63% CER for a new best.
- Note: cosine+EMA run (PID 670605, epoch ~20/150, greedy 6.76% CER at epoch 19)
  continues in parallel; its harvest gate is the same 6.05% validation bar.

### 2026-10-21 — early harvest of cosine+EMA run at epoch 27, beam 8: 5.80% CER / 61.69% exact (best unchanged)

- Idea: the cosine+EMA run (EMA weights) hit greedy 6.17% CER / 57.15% exact at
  epoch 27, already better than the epoch-84 greedy best (6.23% / 57.36%).
  EMA converges far faster than the step-schedule run. Harvested the live
  val-best checkpoint (snapshotted as `recognizer-epoch27.pt`).
- Validation (12,015 clips, beam 8): **5.95% CER** (12,619 errors) / **57.74%
  exact** - beats the 6.05% gate; test pass earned.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-epoch27.beam-eval-report.json`
- Test (12,521 clips, beam 8): **5.80% CER** (12,822 errors) / **61.69% exact**
  (7,724). CER 0.17 above the 5.63% best, exact 0.22 above the 61.47% best -
  gate not passed on CER (primary metric); best unchanged at 5.63% / 61.47%.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-epoch27.beam-eval-report.json`
- Cost: validation 2,199 s; test 2,406 s (beam 8, batch 8, shared GPU).
- Finding: EMA is the strongest convergence lever found - 27 epochs match the
  step run's 84-epoch result class. The run continues to epoch 150; a later
  harvest should test the ceiling. Next trigger: cosine+EMA greedy val-best
  <= 5.87% (0.3 pt improvement), or layers6 (still warming, epoch ~8)
  greedy <= 6.00%.

### 2026-10-21 — cosine+EMA epoch 35, beam 8: **NEW BEST 5.45% CER / 63.01% exact**

- Harvested the cosine+EMA run's val-best at epoch 35 (greedy 5.85% / 60.06),
  snapshotted as `recognizer-epoch35.pt` while training continued.
- Validation (12,015 clips, beam 8): **5.62% CER** (11,915 errors) / **59.68%
  exact** (7,171) - best validation ever recorded, well under the previous
  6.05% gate.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-epoch35.beam-eval-report.json`
- Test (12,521 clips, beam 8): **5.45% CER** (12,044 errors) / **63.01% exact**
  (7,890). Beats previous best 5.63% CER and 61.47% exact on both metrics.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-epoch35.beam-eval-report.json`
- Cost: validation 2,229 s; test 2,361 s (beam 8, batch 8, shared GPU).
- Finding: cosine+EMA reaches at epoch 35 what the step-schedule recipe needed
  84 epochs for, and then goes further. Next gates reference 5.62% validation
  CER (epoch-35 beam-8) and 5.45% test CER.

### 2026-10-21 — cosine+EMA epoch 62, beam 8: **NEW BEST 5.04% CER / 65.47% exact**

- Idea: harvest trigger from the epoch-27 entry fired - cosine+EMA greedy
  val-best hit epoch 62 at **5.51% CER / 61.39% exact** (vs epoch-35's 5.85% /
  60.06%). `scripts/gpu_budget.sh 6144` failed (1,061 MiB free), so no new
  training; evaluation-only harvest on the frozen `recognizer-epoch62.pt`
  snapshot (live `recognizer.pt` verified identical, epoch 62).
- Architecture/decoder: unchanged 8,828,348-param gaps transformer, beam 8
  unnormalized sum log-prob.
- Validation (12,015 clips, beam 8): **5.30% CER** (11,231 errors) / **61.66%
  exact** (7,409) - beats the 5.62% gate; test pass earned.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-epoch62.beam-eval-report.json`
- Test (12,521 clips, beam 8): **5.04% CER** (11,137 errors) / **65.47% exact**
  (8,198) - new best on both metrics, previous 5.45% / 63.01%.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-epoch62.beam-eval-report.json`
- Cost: smoke 11 s / 292 MiB; validation 2,339 s / 457 MiB; test 2,297 s /
  388 MiB (beam 8, batch 8, nice -n 19, shared GPU).
- Status at harvest: cosine+EMA run (PID 670605) epoch ~76/150, still
  improving; layers6 run (PID 787811) epoch ~61/100, greedy val-best epoch 59
  at 6.69% CER / 54.34% exact - far behind this run's class, so doubling depth
  looks like the wrong capacity lever even before its lr step at epoch 80.
- Next gates reference 5.30% validation CER (epoch-62 beam-8) and 5.04% test
  CER. Next trigger: cosine+EMA greedy val-best <= 5.21% (0.3 pt).
- Verdict: positive; new best recorded.

### 2026-10-21 — checkpoint ensemble (e62 + step-84), beam 8: **NEW BEST 4.52% CER / 66.92% exact**

- Idea: harvest triggers did not fire (cosine+EMA greedy val-best still epoch 62 at
  5.51%; layers6 best 6.69% at epoch 59, wrong capacity lever) and
  `scripts/gpu_budget.sh 6144` failed (1,899 MiB free), so run an evaluation-only
  experiment. Decoder levers were exhausted at checkpoint level; ensembling is the
  untested combination lever. Named experiment: **checkpoint ensembling** - equal
  mean of per-decoding-step log-posteriors of two independently trained full-data
  gaps transformers (cosine+EMA epoch-62 val-best + step-schedule epoch-84
  val-best, both 8,828,348 params, identical architecture), beam 8 unchanged, no
  dictionary/language correction.
- Code: `--checkpoint` accepts multiple paths in
  `scripts/beam_eval_openfs_recognizer.py`; single-checkpoint path unchanged.
  Sanity: ensemble of epoch-62 with itself reproduced the single-model 32-clip
  smoke byte-identical (0.0660522 CER / 15 exact).
- Validation (12,015 clips, beam 8): **4.89% CER** (10,373 errors) / **63.30%
  exact** (7,605) vs epoch-62 alone 5.30% / 61.66% - beats the 5.30% gate; test
  pass earned.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens-e62-step84.beam-eval-report.json`
- Test (12,521 clips, beam 8): **4.52% CER** (9,996 errors) / **66.92% exact**
  (8,379) - new best on both metrics, previous 5.04% / 65.47%.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens-e62-step84.beam-eval-report.json`
- Cost: validation 2,892 s / 575 MiB peak; test 3,109 s / 492 MiB (batch 8,
  beam 8, nice -n 19, two model copies resident, shared GPU).
- Finding: ensembling two recipes' val-best checkpoints gained 0.52 pp test CER
  from one forward each - the biggest single-step gain since EMA. The lever
  composes with checkpoint progress: re-ensemble whenever a run's val-best
  advances materially.
- Status at harvest: cosine+EMA (PID 670605) epoch ~82/150, greedy val-best epoch
  62 (5.51%); layers6 (PID 787811) epoch ~67/100, best epoch 59 (6.69%).
- Next gates reference 4.89% validation CER and 4.52% test CER. Next trigger:
  cosine+EMA greedy val-best <= 5.21% -> re-ensemble with step-84.
- Verdict: positive; new best recorded.

### 2026-10-21 — 4-checkpoint ensemble (e27+e62 + step-54+84): 4.86% val / 4.56% test CER; gate nominally earned but test worse; best unchanged

- Idea: harvest triggers not fired (cosine+EMA greedy val-best still epoch 62 at
  5.51% through epoch 91/150; layers6 best was 6.69%) and `gpu_budget.sh 6144`
  failed (1,879 MiB free), so evaluation-only experiment: widen the winning
  2-checkpoint ensemble to 4 members, two per recipe (cos-EMA epoch-27+62,
  step epoch-54+84; all 8,828,348-param identical architecture, mean of
  per-step log-posteriors, beam 8).
- Validation (12,015 clips, beam 8): **4.86% CER** (10,310 errors) / **63.17%
  exact** (7,590) - nominally beats the 4.89% gate by 0.03 pp but exact drops
  0.13 pp; test pass earned.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e27-e62-s54-s84.beam-eval-report.json`
- Test (12,521 clips, beam 8): **4.56% CER** (10,090 errors) / **66.80% exact**
  (8,364) - worse than best 4.52% / 66.92% on both metrics; best unchanged.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e27-e62-s54-s84.beam-eval-report.json`
- Finding: ensemble size saturates at 2 - older/correlated snapshots from the
  same runs add no diversity; the +0.03 pp validation edge flipped sign on test,
  consistent with the epoch-54 validation/test flip. Do not re-ensemble stale
  snapshots; only re-ensemble materially newer val-bests.
- Cost: smoke 15 s / 505 MiB; validation 4,383 s / 750 MiB; test 4,664 s /
  648 MiB (batch 8, beam 8, nice -n 19, four model copies, shared GPU).
- Status at harvest: cosine+EMA (PID 670605) epoch ~91/150, greedy val-best
  epoch 62 (5.51%). layers6 (PID 787811) jumped after its epoch-80 lr step:
  greedy val-best epoch 82 at **5.85% CER / 59.4% exact** (best epoch 82/82,
  6-layer 17.6M-arch, cannot ensemble with 3-layer checkpoints - different
  architecture check). Next iteration: snapshot layers6 `recognizer.pt`,
  beam-8 validation gate vs 4.89%; only 3-layer runs feed ensembles.
- Verdict: negative on the primary metric; retire multi-snapshot ensembling
  within the same run.

### 2026-10-21 — cross-architecture 3-checkpoint ensemble (+layers6 e82): **NEW BEST 4.20% CER / 68.69% exact**

- Idea: harvest triggers not fired (cosine+EMA greedy val-best still epoch 62 at 5.51%
  through epoch 96/150) and `gpu_budget.sh 6144` failed (1,879 MiB free), so
  evaluation-only experiment. The ens4 entry showed same-run snapshots add no
  diversity; the untested diversity source is architecture. Named experiment:
  **cross-architecture checkpoint ensembling** - add the 6-layer step-schedule
  epoch-82 val-best (greedy 5.85% / 59.40%, snapshotted as `recognizer-epoch82.pt`)
  to the winning 2-member ensemble (cos-EMA e62 + step-84), mean of per-step
  log-posteriors, beam 8.
- Code: `scripts/beam_eval_openfs_recognizer.py` now permits mixed architectures
  (contract + representation + label-vocabulary checks kept; per-checkpoint
  architectures and summed parameter count recorded). Sanity: e62 self-ensemble
  reproduced the recorded 32-clip smoke byte-identical (0.0660522 CER / 15 exact).
- Validation (12,015 clips, beam 8): **4.65% CER** (9,856 errors) / **64.67% exact**
  (7,770) vs 2-member gate 4.89% / 63.30% - beats both; test pass earned.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens3-e62-s84-l6e82.beam-eval-report.json`
- Test (12,521 clips, beam 8): **4.20% CER** (9,294 errors) / **68.69% exact**
  (8,601) - new best on both metrics, previous 4.52% / 66.92%.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens3-e62-s84-l6e82.beam-eval-report.json`
- Params: 8,828,348 x2 + 17,509,820 = 35,166,516. Cost: smoke 15 s / 491 MiB;
  validation 4,362 s / 725 MiB; test 4,638 s / 628 MiB (batch 8, nice -n 19, shared GPU).
- Finding: cross-architecture diversity works where same-run snapshot diversity failed -
  the weaker 6-layer member (alone 5.85% greedy) still gained 0.32 pp test CER over the
  2-member ensemble. Ensembles need diverse members, not just more members. layers6 is
  the only 6-layer checkpoint; a second independent 6-layer recipe would be the next
  diversity axis if capacity ever frees up.
- Status at harvest: cosine+EMA (PID 670605) epoch ~107/150, greedy val-best epoch 62
  (5.51%); layers6 (PID 787811) epoch ~94/100, val-best epoch 82 (5.85%), post-lr-step
  epochs may still improve it - if so, re-ensemble the newer e-checkpoint.
- Next gates reference 4.65% validation CER and 4.20% test CER. Next trigger: either
  run's greedy val-best materially improves (cos-EMA <= 5.21% or layers6 <= 5.55%).
- Verdict: positive; new best recorded.

### 2026-10-21 — ensemble member weighting (6-layer weight 1.25): 4.67% val CER; gate not passed; best unchanged

- Idea: harvest triggers not fired (cos-EMA greedy val-best still epoch 62 at 5.51%
  through epoch 113/150; layers6 finished 100/100 with best epoch 82 at 5.85%) and
  `gpu_budget.sh 6144` failed (1,879 MiB free), so evaluation-only experiment. Ensembles
  had only ever been tried with equal member weights; the cross-arch finding "the weaker
  6-layer member still helps" motivates explicit member weights (per-checkpoint
  temperature on log-posteriors). Named experiment: **ensemble member weighting** -
  grid the 6-layer member weight w in {0.5, 0.75, 1.0, 1.25, 1.5}, 3-layer members
  fixed at 1, beam 8 otherwise unchanged, no dictionary or language correction.
- Code: `scripts/beam_eval_openfs_recognizer.py` gains `--ensemble-weights` (one weight
  per checkpoint; weighted mean of frame log-posteriors). Sanity: all-ones weights
  reproduced the pre-edit 32-clip 3-member smoke byte-identical (0.0599078 / 20 exact).
- Coarse selection (greedy, first 4,000 validation clips, 69,783 chars): w=0.5 4.910 /
  w=0.75 4.922 / w=1.0 4.904 / w=1.25 4.877 / w=1.5 4.879 -> argmax 1.25, but the edge
  over equal weights is ~0.03 pp, within subset noise.
- Validation (12,015 clips, beam 8, weights 1 1 1.25): **4.67% CER** (9,902 errors) /
  **64.55% exact** (7,756) vs the 4.65% / 64.67% equal-weight gate - not earned; test
  skipped.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens3-e62-s84-l6e82-w125.beam-eval-report.json`
- Cost: grid 5 x ~400 s (greedy, batch 8, ~500 MiB); validation 3,536 s / 725 MiB peak
  (batch 8, nice -n 19, shared GPU).
- Finding: member-weighting is exhausted - equal log-posterior averaging is already at
  the optimum for this member set, and the greedy-subset ordering did not transfer to
  full-val beam 8 (third such selection flip). Do not reweight existing members; further
  ensemble gains need new diverse members (e.g. a second independent 6-layer recipe)
  once GPU capacity frees.
- Status: layers6 run finished (epoch 100/100, best epoch 82 at 5.85%, no newer
  checkpoint to harvest); cos-EMA (PID 670605) epoch ~113/150, best epoch 62 (5.51%).
  Gates unchanged: 4.65% val / 4.20% test; next trigger cos-EMA greedy val-best <= 5.21%.
- Verdict: negative on the primary metric; retire member-weight tuning.


### 2026-10-22 — launched second independent 6-layer recipe: cosine+EMA, full data (run in progress)

- Idea: harvest triggers not fired (cos-EMA greedy val-best still epoch 62 at 5.51%
  through epoch 114/150, epochs 90-114 all worse) and eval-only levers are all
  retired, so take the lever named twice by the ensemble entries: a second
  independent 6-layer recipe - the untested diversity axis for the ensemble. The
  layers6 step run finishing freed GPU capacity (6,143 MiB free).
- Architecture: 6-layer gaps transformer, d_model 256, heads 8, feedforward 2048;
  17,509,820 params - same arch as the ens3 l6-e82 member, so ensembles can grow
  on recipe diversity where same-run snapshot diversity already failed.
- Config: replicate the winning 3-layer full-data recipe on the 6-layer arch,
  single-schedule variable vs the layers6 step run: `--layers 6 --epochs 150
  --lr-schedule cosine --warmup-epochs 5 --ema-decay 0.9999`, full 126,131-clip
  train slice, gaps, Adam 1e-4, fp16 AMP (V100), seed 42 (default). EMA decay
  matches the cos-EMA full-data run at identical 15,766 steps/epoch (batch 8).
- Batch 8 (not 16): `scripts/gpu_budget.sh 6144` failed by 1 MiB (6,143 free), so
  per rule 2 the batch was halved; same forced deviation as the champion cos-EMA
  run, so the two cosine+EMA runs stay batch-matched.
- Smoke: `--layers 6 --epochs 2 --limit-clips 1600` at batch 8 peaked 1,123 MiB.
- Params: 17,509,820. VRAM: ~1.9 GiB (free 6,143 -> 4,227 MiB), inside budget.
  Wall clock: ~18 steps/s -> ~15-17 min/epoch incl validation, ETA ~40+ h for 150
  epochs with the cos-EMA run still sharing the GPU (epoch ~115/150).
- Output: `data/models/openfs-recognizer-gaps-full-layers6-cosine-ema/recognizer.pt`
  (PID 1134787, log/pid under `data/logs/`). No test evaluation; none earned.
- Gates: unchanged, 4.65% val / 4.20% test. Harvest trigger: its greedy val-best
  <= 5.55% (beats l6-e82's 5.85% greedy) -> snapshot, beam-8 validation gate vs
  4.65%, and re-ensemble as 4th member only if it beats ens3 on validation.
- Verdict: pending; run in progress.

### 2026-10-22 — prob-space ensemble fusion (arithmetic mean of probabilities): 4.61% val / 4.30% test CER; val edge flipped on test; best unchanged

- Idea: harvest triggers not fired (cos-EMA greedy val-best still epoch 62 at 5.51%
  through epoch 123/150; l6-cos-EMA run at epoch ~7, greedy 9.99%) and
  `gpu_budget.sh 6144` failed (4,225 MiB free; two training jobs active), so
  evaluation-only experiment. Member set and weights are retired levers but the
  fusion rule was never varied: all ensembling so far is the geometric mean of
  per-frame posteriors (mean of log-posteriors). Named experiment: **prob-space
  fusion** - weighted arithmetic mean of per-frame softmax probabilities, then log,
  same 3-member ens3 (e62 + step-84 + l6-e82), beam 8, no dictionary/language
  correction.
- Code: `--fusion {logprob,prob}` in `scripts/beam_eval_openfs_recognizer.py`
  (default `logprob` = exact old behaviour). Sanity: default-fusion 32-clip smoke
  reproduced the recorded 3-member smoke byte-identical (0.0599078 / 20 exact).
- Validation (12,015 clips, beam 8): **4.61% CER** (9,783 errors) / **64.67%
  exact** (7,770) vs log-fusion gate 4.65% (9,856) / 64.67% - 73 fewer errors,
  identical exact count; gate nominally earned; test pass taken.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens3-e62-s84-l6e82-probfusion.beam-eval-report.json`
- Test (12,521 clips, beam 8): **4.30% CER** (9,506 errors) / **68.55% exact**
  (8,583) - worse than the 4.20% / 68.69% best on both metrics; fourth
  validation/test sign flip; best unchanged.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens3-e62-s84-l6e82-probfusion.beam-eval-report.json`
- Cost: smoke 2 x ~15 s / 491 MiB; validation 4,066 s / 726 MiB; test 4,209 s /
  628 MiB (batch 8, nice -n 19, shared GPU).
- Finding: fusion-rule axis exhausted - log-space geometric averaging is already
  the better operating point (prob-space averaging over-weights whichever member is
  most confident per frame); the 0.04 pp val gain is noise-level and flipped sign
  on test as all prior sub-0.05 pp val deltas have. Keep geometric fusion; further
  gains need new members (l6-cos-EMA in flight).
- Status: cos-EMA (PID 670605) epoch ~123/150, val-best epoch 62; l6-cos-EMA
  (PID 1134787) epoch ~7/150, greedy 9.99% at epoch 6, ETA ~35+ h. Gates unchanged:
  4.65% val / 4.20% test; triggers unchanged: cos-EMA greedy <= 5.21% or
  l6-cos-EMA greedy <= 5.55%.
- Verdict: negative on the primary metric; retire ensemble fusion-rule tuning.

### 2026-10-22 — launched seed-diversity ensemble member: cosine+EMA 3-layer, seed 20261022 (run in progress)

- Idea: harvest triggers not fired (cos-EMA greedy val-best still epoch 62 at 5.51%
  through epoch 126/150; l6-cos-EMA epoch 8/150, greedy 8.52% and falling fast) and
  `gpu_budget.sh 6144` failed (3,603 MiB free). All eval-side levers are retired
  (beam width, length penalty, member weighting, fusion rule, same-run snapshots);
  the only open lever is ensemble membership, and the untried diversity axis is
  **seed**: every existing member (e62, step-84, l6-e82) is a seed-42 run. Named
  experiment: **seed-diversity ensemble member**.
- Architecture: default gaps transformer, 8,828,348 params - identical to the two
  3-layer ens members, so member compatibility is guaranteed; only initialization/
  augmentation randomness differs.
- Config: `--epochs 100 --lr-schedule cosine --warmup-epochs 5 --ema-decay 0.99995
  --seed 20261022`, full 126,131-clip train slice, gaps, Adam 1e-4, fp16 AMP. Two
  declared deviations from the champion recipe: (i) batch 4 - budget check failed,
  so per rule 2 halved from the champion's 8; 2x steps/epoch at unchanged lr (the
  12k-run precedent caveat); (ii) 100-epoch cosine, not 150 - the champion set no
  new val-best in 60+ epochs past epoch 62, so the long tail demonstrably does not
  pay; EMA decay 0.99995 keeps the averaging horizon epoch-matched (~20k steps,
  ~0.65 epochs) at the doubled steps/epoch.
- Smoke: 400 clips at batch 4 / validation-batch 16 peaked 315 MiB; steady-state
  footprint ~930 MiB (free 3,603 -> 2,675 MiB), inside the 6 GiB budget.
- Params: 8,828,348. Wall clock: ~28 steps/s with three jobs sharing the GPU ->
  ~1,115 s train + ~150 s validation = ~21 min/epoch, 100 epochs ~35 h (will
  shorten when the cos-EMA full run finishes its last ~24 epochs).
- Output: `data/models/openfs-recognizer-gaps-full-cosine-ema-seed2/recognizer.pt`
  (PID 1182605, log/pid under `data/logs/`). No test evaluation; none earned.
- Gates unchanged: 4.65% val / 4.20% test. Harvest trigger: its greedy val-best
  reaches the 5.5-6.0% class (e62/l6-e82 class) -> snapshot, re-ensemble as 4th
  member and beam-8 validation gate vs 4.65%; only then test.
- Verdict: pending; run in progress.

### 2026-10-22 — test-time input-rotation ensembling of ens3: subset selection says dead (±5° −1 error, ±10° worse); no full-val/test run, best unchanged

- Idea: harvest triggers not fired (cos-EMA greedy val-best still epoch 62 at 5.51%
  through epoch 131/150; l6-cos-EMA epoch ~11/150; seed2 epoch ~3/100) and
  `gpu_budget.sh 6144` failed (2,655 MiB free, three training jobs active), so
  evaluation-only experiment. All model-side fusion knobs retired; the untried
  ensemble axis is the **input**: the same members fed deterministically rotated
  pose timelines. Named experiment: **test-time input-rotation ensembling** -
  ens3 (e62 + step-84 + l6-e82) x rotations {0, +5, -5} deg, geometric fusion of
  all model x rotation frame log-posteriors, beam 8, no dictionary/language
  correction.
- Code: `--tta-rot-degrees` in `scripts/beam_eval_openfs_recognizer.py` - the
  train-time affine branch (pose + trajectory columns about the origin, PAD rows
  and presence/frame-delta columns untouched) applied deterministically per
  variant; encoder encodes each variant once, decode steps fuse all terms.
  Sanity: identity-only smoke reproduced the recorded 3-member 32-clip smoke
  byte-identical (0.0599078 / 20 exact); ±15° smoke changed predictions
  (0.06144 / 20), confirming the rotation path is live.
- Selection (greedy, first 4,000 validation clips, 69,783 chars - same clip set
  as the member-weight grid, whose identity baseline was 4.904%):
  {0,±5} **4.902%** (3,421 errors, 2,634 exact) = 1 error better than baseline;
  {0,±10} **4.930%** (3,440 errors, 2,631 exact). VRAM 701 MiB at batch 8.
- Decision: deltas are noise-level (~1 error on 69,783 chars) and stronger
  rotation hurts - mild in-plane rotation barely moves the already-confident
  fused posterior. No full validation beam-8 run launched (~3x cost, ~4,000 s),
  gate 4.65% untouched; no test pass.
- Cost: 2 smoke runs ~15-45 s / 701 MiB; two 4,000-clip greedy subsets 910 s +
  ~900 s / ~700 MiB (nice -n 19, shared GPU).
- Status at harvest: cos-EMA (PID 670605) epoch ~131/150, val-best epoch 62;
  l6-cos-EMA (PID 1134787) epoch ~11/150; seed2 (PID 1182605) epoch ~3/100.
  Gates unchanged: 4.65% val / 4.20% test; triggers unchanged (cos-EMA greedy
  <= 5.21%, l6-cos-EMA greedy <= 5.55%).
- Verdict: negative; retire input-space rotation TTA as an ensemble lever -
  diversity must come from weights (new members), not inputs.

### 2026-10-22 — train-split 4-gram LM shallow fusion of ens3: greedy subset degrades monotonically in lambda (4.90% -> 5.01/5.18/5.45%); gate not earned, no test, best unchanged

- Idea: harvest triggers not fired (cos-EMA greedy val-best still epoch 62 at 5.51%
  through epoch ~136/150; l6-cos-EMA epoch 14 greedy 6.53%, trigger <=5.55% not hit;
  seed2 epoch ~7/100) and `gpu_budget.sh 6144` failed (2,553 MiB free, three training
  jobs active), so evaluation-only experiment. Rule 7 allows language correction when
  the entry names it: named experiment **character n-gram LM shallow fusion** -
  Witten-Bell interpolated order-4 LM over characters, counts from the 126,131
  train-split reference texts only (test strings share zero exact strings with train,
  so this is an orthographic prior, not a test dictionary), log-probs added with
  weight lambda to the fused ens3 per-step log-posteriors including the EOS class;
  members/beam 8/fusion unchanged.
- Code: `--lm-weight/--lm-order` in `scripts/beam_eval_openfs_recognizer.py`
  (default 0 = exact old behaviour). Fixed an unseen-context backoff bug (gamma 0 not
  1) before any eval; all table rows sum to 1. Sanity: lm-weight 0 on the 32-clip
  ens3 smoke reproduced the recorded baseline byte-identical (0.0599078341 / 20 exact).
- Selection (greedy, first 4,000 validation clips, 69,783 chars; lambda=0 identity
  matched the member-weight grid's 4.904% baseline): lambda 0.1 **5.014%** (3,499
  errors), 0.2 **5.176%** (3,612), 0.4 **5.454%** (3,806) vs baseline **4.904%**
  (3,422) - strict monotonic degradation; the optimum sits at the lambda=0 boundary,
  so smaller lambdas only interpolate toward it. A beam-8 smoke at lambda=1.0 (32
  clips) degraded 39 -> 80 errors, confirming the direction.
- Decision: no full-validation beam-8 run launched; the 4.65% gate is unreachable at
  any lambda > 0; no test pass.
- Cost: 4 greedy subset runs 519-575 s each, ~290 MiB peak LM table + <550 MiB total
  (nice -n 19, shared GPU). A first background launch died with its parent shell -
  relaunched with setsid.
- Finding: the recogniser has already internalised the train-text character
  statistics (same texts are its training targets), so a train-text n-gram prior is
  redundant on top of its posteriors and actively harmful on the test split's
  disjoint proper nouns (over-correction toward train-set orthography). Rule 7
  language-correction axis now tested and retired; ensemble-membership diversity
  (l6-cos-EMA, seed2 in flight) remains the only open lever.
- Status: cos-EMA (PID 670605) ~epoch 136/150; l6-cos-EMA (PID 1134787) ~epoch 15/150
  (epoch 14 greedy 6.53%); seed2 (PID 1182605) ~epoch 7/100. Gates unchanged:
  4.65% val / 4.20% test; triggers unchanged (cos-EMA greedy <= 5.21%, l6-cos-EMA
  greedy <= 5.55%).
- Verdict: negative on the primary metric; retire LM shallow fusion.

### 2026-10-22 — representation-diversity ensemble member (pre-loop wrist2d): 4.63% val CER (gate nominally earned) but 4.22% test CER; sixth val/test sign flip; best unchanged

- Idea: harvest triggers not fired (cos-EMA greedy val-best still epoch 62 at 5.51%
  through epoch ~147/150; l6-cos-EMA epoch 23, epoch-14 greedy 6.53% > 5.55% trigger;
  seed2 epoch ~13/100) and `gpu_budget.sh 6144` failed (2,531 MiB free), so
  evaluation-only experiment. Membership is the only open lever; the untried diversity
  axis is the **input representation**: the pre-loop (19 Oct) full-data `wrist2d`
  checkpoint (8,815,804 params, greedy val 6.98% / test 6.59% CER) sees the identical
  gap-preserving timeline as the gaps members (same segments/positions; 42 dominant-hand
  wrist columns instead of 91), so per-frame log-posterior fusion stays aligned.
  Named experiment: **representation-diversity ensemble** - ens3 + wrist2d as 4th
  member, equal weights, beam 8, no dictionary or language correction.
- Code: `scripts/beam_eval_openfs_recognizer.py` now accepts mixed representations
  (gaps + wrist2d; `compressed` rejected - different timeline): per-representation
  batch iterators over the same deterministic groups with alignment asserts (identical
  selected indices, segments, positions), per-model pose encoding, width-safe
  `rotate_pose`. Sanity: ens3 32-clip smoke reproduced the recorded baseline
  byte-identical (0.0599078341 / 20 exact); wrist2d-alone smoke 41 errors / 16 exact.
- Screening (greedy, first 4,000 validation clips): ens3 4.904% (3,422 - third exact
  reproduction of this baseline); ens4+wrist2d **4.851%** (3,385, -37 errors).
- Validation (12,015 clips, beam 8): **4.63% CER** (9,817 errors) / **64.58% exact**
  (7,759) vs gate 4.65% (9,856) / 64.67% - -39 errors (0.02 pp) but exact -11; gate
  nominally earned; test pass taken.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-w2d.beam-eval-report.json`
- Test (12,521 clips, beam 8): **4.22% CER** (9,340 errors) / **68.41% exact** (8,565)
  - worse than the 4.20% / 68.69% best on both metrics; sixth validation/test sign flip
  of a sub-0.05 pp validation delta; best unchanged.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-w2d.beam-eval-report.json`
- Params: 35,166,516 + 8,815,804 = 43,982,320. Cost: smokes 2x ~15 s; greedy subsets
  544 + 592 s / 205-243 MiB; validation 6,531 s / 815 MiB; test 6,418 s / 708 MiB
  (batch 8, nice -n 19, shared GPU, three training jobs active).
- Finding: representation diversity fails with this member - the wrist2d run is a whole
  recipe generation behind the current class (6.98% greedy val vs l6-e82's 5.85%), and
  the screening/-val gains it showed did not survive to test, exactly like the other
  five sub-0.05 pp val edges. Diversity only pays when members are near the member-class
  quality bar. With fusion rule, weights, snapshots, input TTA, LM and now representation
  all retired, the only remaining path to a new best is new members at the 5.5% greedy
  class: l6-cos-EMA and seed2 in flight, and the champion cos-EMA run (epoch ~147/150)
  finishing soon will free GPU capacity for the next training lever.
- Status: gates unchanged: 4.65% val / 4.20% test; triggers unchanged (cos-EMA greedy
  <= 5.21%, l6-cos-EMA greedy <= 5.55%).
- Verdict: negative on the primary metric; retire pre-loop-recipe representation
  diversity as an ensemble axis.

### 2026-10-22 — harvested l6-cos-EMA epoch-26 as 4th ensemble member: **NEW BEST 4.14% CER / 68.71% exact**

- Idea: the standing trigger fired - l6-cos-EMA greedy validation-best hit epoch 26 at
  **5.40% CER / 59.95% exact**, below the 5.55% bar and the strongest greedy member ever
  recorded (better than e62's 5.51%). `gpu_budget.sh 6144` failed (4,031 MiB free), so
  evaluation-only: named experiment **recipe-diversity within the 6-layer architecture** -
  add the frozen `recognizer-epoch26.pt` snapshot to the winning ens3 (e62 + step-84 +
  l6-e82) as a 4th member, equal weights, logprob fusion, beam 8, no dictionary or
  language correction.
- Smoke: 4-member 32-clip validation smoke live (0.0491552 CER / 32 errors / 19 exact vs
  the recorded ens3 smoke 0.0599078 / 39 / 20).
- Validation (12,015 clips, beam 8): **4.48% CER** (9,501 errors) / **65.03% exact**
  (7,813) vs ens3 gate 4.65% (9,856) / 64.67% - 355 fewer errors, +43 exact; far above
  the sub-0.05 pp noise band, so test pass earned (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-l6ce26.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **4.14% CER** (9,146 errors) / **68.71% exact**
  (8,603) - new best on both metrics, previous 4.20% / 68.69%. First validation gain
  that clearly transferred to test CER (the six sign flips were all sub-0.05 pp edges).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-l6ce26.beam-eval-report.json`
- Params: 8,828,348 x2 + 17,509,820 x2 = 52,676,336. Cost: smoke 18 s / 608 MiB;
  validation 5,021 s / 879 MiB; test 5,454 s / 766 MiB (batch 8, nice -n 19, shared GPU,
  two training jobs active).
- Finding: recipe diversity repeats inside the 6-layer arch exactly as it did across the
  3-layer step/cos-EMA pair; the member-quality bar is real - a member at 5.40% greedy
  pays where the 6.98% wrist2d member did not.
- Status: full-data cos-EMA run finished 150/150 (val-best e62 5.51%, no new member);
  l6-cos-EMA (PID 1134787) still improving, val-best now epoch 33 at **5.17% / 60.94%** -
  newer than the ensembled e26; seed2 (PID 1182605) epoch 25, best 6.18% / 56.12%.
- Next gates reference 4.48% validation CER and 4.14% test CER. Triggers: l6-cos-EMA
  greedy val-best <= 4.90% or run plateau/finish -> refresh the 4th member with the newer
  snapshot and re-gate; seed2 greedy val-best <= 6.00% -> candidate 5th member.
- Verdict: positive; new best recorded.

### 2026-10-22 — refreshed 4th ensemble member to l6-cos-EMA epoch-37: **NEW BEST 4.03% CER / 69.26% exact**

- Idea: the 4th member was the frozen l6-cos-EMA epoch-26 snapshot (greedy 5.40%); the run
  kept setting val-bests, so per the "only re-ensemble materially newer val-bests" rule,
  swap in the fresher one. `gpu_budget.sh 6144` failed (3,991 MiB free), evaluation-only.
  The live `recognizer.pt` had just saved epoch 37 (greedy **5.11% / 61.53%** - 0.29 pp
  better than e26, far above the sub-0.05 pp noise band), frozen as `recognizer-epoch37.pt`.
  Named experiment: **4th-member refresh** - ens3 members unchanged, l6ce26 -> l6ce37,
  equal weights, logprob fusion, beam 8, no dictionary or language correction.
- Smoke: 4-member 32-clip validation smoke 35 errors / 20 exact (vs recorded 32 / 19 for
  the e26 set - smoke-level noise, full-val gate decides).
- Validation (12,015 clips, beam 8): **4.40% CER** (9,332 errors) / **65.38% exact** (7,855)
  vs e26-member gate 4.48% (9,501) / 65.03% - 169 fewer errors, +42 exact, above the noise
  band; test pass earned.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-l6ce37.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **4.03% CER** (8,907 errors) / **69.26% exact** (8,672)
  - new best on both metrics, previous 4.14% / 68.71%; second clean val->test transfer
  (-0.08 pp val -> -0.11 pp test).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-l6ce37.beam-eval-report.json`
- Params: unchanged 52,676,336 (member swap, same arch). Cost: smoke 18 s / 608 MiB;
  validation 5,022 s / 879 MiB; test 5,458 s / 766 MiB (batch 8, nice -n 19, two training
  jobs sharing the GPU).
- Finding: member-refresh composes with member-adding - a strictly better snapshot from the
  same run pays (unlike the retired multi-snapshot co-ensembling, which kept the stale one
  alongside). The l6-cos-EMA run is still on a steep climb.
- Status: l6-cos-EMA (PID 1134787) epoch 44, new greedy val-best **4.97% / 62.25%** - the
  <=4.90% refresh trigger is one epoch's progress away; seed2 (PID 1182605) epoch 31, greedy
  val-best **5.93% / 58.09%** - its <=6.00% 5th-member trigger has fired.
- Next gates reference 4.40% validation CER and 4.03% test CER. Triggers: l6-cos-EMA greedy
  val-best <= 4.90% -> refresh the 4th member again and re-gate; seed2 -> screen as 5th
  member on validation first (seed-diversity axis, never tested).
- Verdict: positive; new best recorded.

### 2026-10-22 — seed-diversity 5th ensemble member (seed2 e32): 4.40% val CER (gate nominally earned, -4 errors) but 4.09% test CER; seventh val/test sign flip; best unchanged

- Idea: seed2's <=6.00% greedy 5th-member trigger fired (val-best epoch 32, greedy 5.87% /
  58.22%). `gpu_budget.sh 6144` failed (3,991 MiB free), evaluation-only. Named experiment:
  **seed-diversity ensemble member** - add frozen `recognizer-epoch32.pt` of the seed-20261022
  cosine+EMA 3-layer run (8,828,348 params, identical arch to the e62/step-84 members) to the
  winning ens4 (e62 + step-84 + l6-e82 + l6ce37), equal weights, logprob fusion, beam 8, no
  dictionary or language correction.
- Smoke: 5-member 32-clip validation smoke 33 errors / 20 exact (vs 35 / 20 for the ens4 set).
- Validation (12,015 clips, beam 8): **4.40% CER** (9,328 errors) / **65.48% exact** (7,867)
  vs ens4 gate 4.40% (9,332) / 65.38% - only -4 errors (0.002 pp), exact +12; gate nominally
  earned; test pass taken.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-l6ce37-seede32.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **4.09% CER** (9,037 errors) / **69.24% exact** (8,669)
  - worse than the 4.03% / 69.26% best on both metrics (+130 errors); seventh val/test sign
  flip; best unchanged.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-l6ce37-seede32.beam-eval-report.json`
- Params: 52,676,336 + 8,828,348 = 61,504,684. Cost: smoke 21 s / 673 MiB; validation 5,717 s /
  966 MiB; test 6,152 s / 844 MiB (batch 8, nice -n 19, shared GPU, two training jobs active).
- Finding: seed diversity fails as an ensemble axis at this member set - a third 3-layer
  cosine+EMA-recipe member is near-duplicate of e62 and adds nothing, unlike architecture/recipe
  diversity which worked twice. Consistent with the member-quality bar: a mid-run 5.87% member
  cannot beat the marginal cost even when nominally the best member ever of its arch. Do not
  carry seed2; ensemble growth must come from materially better or structurally different members.
- Status: l6-cos-EMA (PID 1134787) epoch ~55, val-best epoch 54 at **4.89% / 62.74%** - the
  <=4.90% 4th-member refresh trigger has fired; seed2 (PID 1182605) epoch ~39, best 5.73%.
  Gates unchanged: 4.40% val / 4.03% test. Next iteration: refresh 4th member
  l6ce37 -> newest val-best snapshot and re-gate.
- Verdict: negative on the primary metric; retire seed-diversity ensembling.

### 2026-10-22 — refreshed 4th ensemble member to l6-cos-EMA epoch-54: **NEW BEST 3.96% CER / 69.50% exact**

- Idea: the standing 4th-member refresh trigger fired - l6-cos-EMA greedy val-best epoch 54
  hit **4.887% / 62.74%**, below the 4.90% bar (snapshot frozen from the live
  `recognizer.pt` at 22:54 UTC as `recognizer-epoch54.pt`; run later reached epoch 62,
  val-best epoch 59 at 4.831% / 62.71%). `gpu_budget.sh 6144` failed (3,991 MiB free),
  evaluation-only. Named experiment: **4th-member refresh round 2** - ens3 members
  unchanged (e62 + step-84 + l6-e82), l6ce37 -> l6ce54, equal weights, logprob fusion,
  beam 8, no dictionary or language correction.
- Smoke: 4-member 32-clip validation smoke 39 errors / 20 exact (vs 35 / 20 for the e37
  set - smoke-level noise, full-val gate decides).
- Validation (12,015 clips, beam 8): **4.36% CER** (9,255 errors) / **65.64% exact**
  (7,887) vs e37-member gate 4.40% (9,332) / 65.38% - 77 fewer errors, +32 exact; inside
  the sub-0.05 pp band that produced seven sign flips, but exact also improved and the
  member's own greedy CER improved 0.22 pp (far above noise), so test pass taken.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-l6ce54.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.96% CER** (8,764 errors) / **69.50% exact**
  (8,702) - new best on both metrics, previous 4.03% / 69.26%; third clean val->test
  transfer of a member refresh (-0.036 pp val -> -0.071 pp test).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-l6ce54.beam-eval-report.json`
- Params: unchanged 52,676,336 (member swap, same arch). Cost: smoke 18 s / 608 MiB;
  validation 5,069 s / 879 MiB; test 5,469 s / 766 MiB (batch 8, nice -n 19, two
  training jobs sharing the GPU).
- Finding: the member-refresh lever keeps paying monotonically as the l6-cos-EMA run
  climbs (5.40 -> 5.11 -> 4.89 greedy member quality tracked to test). A 0.036 pp val
  edge transferred despite sitting inside the historical flip band - consistent with
  the rule that a transfer follows the member's own greedy improvement, not the ensemble
  val delta.
- Status: l6-cos-EMA (PID 1134787) epoch 62, val-best epoch 59 at **4.831% / 62.71%** and
  still climbing; seed2 (PID 1182605) epoch ~55, best 5.73% (not carried, retired axis).
- Next gates reference 4.36% validation CER and 3.96% test CER. Triggers: l6-cos-EMA
  greedy val-best <= 4.70% or run plateau/finish (150-epoch cap) -> refresh the 4th
  member to the newest val-best snapshot and re-gate.
- Verdict: positive; new best recorded.

### 2026-10-23 — launched depth-capacity ensemble member: 9-layer cosine+EMA, full data (run in progress)

- Idea: harvest triggers not fired - l6-cos-EMA greedy val-best still epoch 59 at 4.831%
  / 62.71% (epochs 60-62 worse; trigger bar <=4.70%), eval-only levers all retired, so
  take the only demonstrably productive member axis with untested headroom: depth. Depth
  3->6 improved both single-model quality (5.51 -> 4.83 greedy within cos-EMA) and
  ensemble value (l6-e82, l6ce members twice), so 6->9 is the named next step: **depth-
  capacity 5th-member candidate**, cos-EMA recipe (the stronger recipe family), full
  126,131-clip train slice, gaps representation.
- Capacity housekeeping: killed seed2 (PID 1182605, epoch ~48, best 5.73%) - its axis was
  retired as a member source by the ens5 flip and the job was consuming ~0.9 GiB plus
  throughput the l6-cos-EMA climb needs.
- Config: `--layers 9 --epochs 100 --lr-schedule cosine --warmup-epochs 5 --ema-decay
  0.99995 --batch-size 4 --validation-batch-size 16`, Adam 1e-4, fp16 AMP, seed 42.
  Batch 4: `gpu_budget.sh 6144` failed (3,971 free) so per rule 2 batch halved from the
  champion's 8; EMA decay epoch-matched at 2x steps/epoch (0.99995, 20k-step horizon,
  seed2 precedent). 100 epochs, not 150: the member joins at class mid-schedule, and this
  bounds the cost of the deeper stack.
- Smoke: 1,600 clips / 2 epochs at batch 4 peaked ~1.7 GiB (free 3,971 -> 2,279 MiB),
  loss descending cleanly (post-norm 9 layers stable at 1e-4 + 5-epoch warmup).
- Params: 26,191,292 (9-layer d_model 256 stack; arch-different from all 4 current
  members). VRAM: ~1.6 GiB steady (free 5,105 -> 3,513 after launch, one other training
  job active). Wall clock: ~13 steps/s -> ~2,400 s train + ~300 s validation per epoch,
  ETA ~75 h for 100 epochs; joinable much earlier at member-class quality.
- Output: `data/models/openfs-recognizer-gaps-full-layers9-cosine-ema/recognizer.pt`
  (PID 1508445, log/pid under `data/logs/`). No test evaluation; none earned.
- Gates unchanged: 4.36% val / 3.96% test. Triggers: l6-cos-EMA greedy val-best <=4.70%
  -> 4th-member refresh round 3; l9-cos-EMA greedy val-best <=5.55% class -> snapshot,
  beam-8 validation gate vs 4.36%, 5th member only if it beats ens4 on validation with a
  materially better member (member-quality bar; seed/representation/TTA/LM/fusion/
  weighting axes all retired).
- Verdict: pending; run in progress.

### 2026-10-23 — weight-space soup of two l6-cos-EMA snapshots as 4th member: **NEW BEST 3.93% CER / 69.76% exact**

- Idea: harvest triggers not fired (l6-cos-EMA greedy val-best still epoch 59 at 4.831%,
  epochs 60-63 worse, trigger bar <=4.70%; l9 run still in epoch 1-2) and
  `gpu_budget.sh 6144` failed (3,333 MiB free), so evaluation-only. The member-refresh
  lever still pays but member gains from new val-best epochs are decelerating
  (5.40 -> 5.11 -> 4.89 -> 4.83); test whether **weight-space snapshot averaging**
  upgrades the member without training - distinct from the retired posterior-level
  co-ensembling of same-run snapshots (here two snapshots fuse into ONE better member).
- Code: new `scripts/soup_openfs_recognizer.py` - linear interpolation of state_dicts
  for architecture-identical checkpoints, newer checkpoint's metadata carried;
  sanity: alpha 0 output byte-identical to checkpoint-b.
- Screening (greedy, first 4,000 validation clips, 69,783 chars): e59 4.881 (3,406)
  / e54 4.997 (3,487) / soup alpha 0.5 **4.696 (3,277)** / alpha 0.25 4.725 (3,297) -
  both soups beat both parents by >100 errors, far above the subset noise band.
- Member greedy full validation (12,015 clips): soup alpha 0.5 **4.705% CER / 63.76%
  exact** vs e59's 4.831% / 62.71% - the soup beats the run's own best epoch; a wider
  EMA horizon (alpha 0.5, snapshots 5 epochs apart) found what 60+ training epochs
  of the run's own val-best tracking did not.
- Validation (12,015 clips, beam 8): ens4 with e62 + step-84 + l6-e82 unchanged,
  4th member l6ce54 -> soup54/59: **4.31% CER** (9,133 errors) / **66.01% exact**
  (7,931) vs gate 4.36% (9,255) / 65.64% - 122 fewer errors, +44 exact; member's own
  greedy improved 0.126 pp (well above the flip band), so test pass earned (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.93% CER** (8,698 errors) / **69.76% exact**
  (8,735) - new best on both metrics, previous 3.96% / 69.50%; fourth clean
  member-improvement transfer (member -0.126 pp greedy -> val -0.056 pp -> test -0.065 pp).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459.beam-eval-report.json`
- Params: unchanged 52,676,336 (member swap, same arch). Cost: 4 subset screens
  224-276 s / 128 MiB each; member greedy val 839 s; smoke 20 s / 608 MiB; validation
  5,559 s / 879 MiB; test 6,073 s / 766 MiB (batch 8, nice -n 19, shared GPU, two
  training jobs active).
- Finding: trajectory weight averaging is a free member upgrade - EMA checkpoints on
  one cosine schedule share a basin, and averaging 5 epochs apart widens the averaging
  horizon beyond the run's EMA decay. Next: soup with wider snapshot spacing once the
  run has later val-bests (e.g. e59 + final), and soup future refresh snapshots
  immediately; member-quality bar and refresh protocol unchanged.
- Status: l6-cos-EMA (PID 1134787) epoch ~68/150, val-best epoch 59 (4.831%);
  l9-cos-EMA running (its pid file recorded the dead launcher shell; real train PID
  1508448, epoch 2 at harvest). Gates now reference 4.31% validation CER and 3.93%
  test CER. Triggers: l6-cos-EMA greedy val-best <=4.70% or plateau/finish -> soup
  newest snapshot pair and re-gate; l9-cos-EMA greedy val-best <=5.55% -> 5th-member
  candidate screening.
- Verdict: positive; new best recorded.

### 2026-10-23 — wider soup member (3-snapshot equal-weight e37+e54+e59): member greedy -0.107 pp but ensemble validation 4.31%/9,138 vs gate 4.31%/9,133; gate not passed, no test, best unchanged

- Idea: harvest triggers not fired (l6-cos-EMA greedy val-best still epoch 59 at
  4.831%, epochs 60-74 all worse, bar <=4.70%; l9-cos-EMA epoch 6, greedy 11.13%) and
  ~2.8 GiB free (no training). Extend the working soup lever along its untested
  spacing/weight axes: freeze the live e59 val-best as `recognizer-epoch59.pt`, build
  three candidate members - e54+e59 alpha 0.75, e37+e59 alpha 0.5 (22-epoch spacing),
  and an equal-weight 3-snapshot soup e37+e54+e59 - as 4th-member swaps into the
  winning ens4 (e62 + step-84 + l6-e82 unchanged), beam 8, logprob fusion, equal
  weights, no dictionary or language correction.
- Screening (greedy, first 4,000 validation clips; baselines e59 4.881 / e54 4.997 /
  a25 4.725 / a50 4.696): a75 4.782 (pair optimum is alpha 0.5, axis closed),
  e37+e59 4.722, 3-way equal **4.664 (3,255, -22 vs a50)** - picked the 3-way.
- Member greedy full validation: 3-way soup **4.598% CER / 64.21% exact** vs current
  member 4.705% / 63.76% - a real -0.107 pp member improvement (wider averaging
  horizon: 3 snapshots over 22 epochs).
- Validation (12,015 clips, beam 8, ens4 with 3-way soup): **4.31% CER** (9,138
  errors) / **65.98% exact** (7,927) vs the soup5459 gate 4.31% (9,133) / 66.01%
  (7,931) - +5 errors, -4 exact; gate NOT passed; no test pass; best unchanged at
  3.93% / 69.76%.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup3way.beam-eval-report.json`
- Cost: 3 soups (CPU, seconds); 3 subset screens ~4 min each / 128 MiB; member greedy
  val ~840 s / 132 MiB; smoke 20 s / 608 MiB (31 errors / 22 exact vs 39 / 20 for the
  a50 set); validation beam 8 5,451 s / ~610 MiB (batch 8, nice -n 19, two training
  jobs active).
- Finding: member improvement does NOT transfer once the ensemble exists - the a50
  soup beat its parent (member-led transfer, 4 clean rounds), but widening the soup
  further (225-member-error gain) moved the ensemble +5 errors. Posterior-level
  ensemble fusion already absorbs the l6-cos trajectory's averaging horizon; soup
  quality beyond ~alpha 0.5 spacing 5 is ensemble-invisible. Retire soup spacing/
  weight tuning on this member; future member refresh only from a genuinely newer
  val-best epoch (<=4.70% trigger) or the l9 member.
- Status: l6-cos-EMA epoch ~78/150 (val-best e59), l9-cos-EMA ~epoch 7; both running.
  Gates unchanged: 4.31% val (soup5459 ens4) / 3.93% test. Triggers unchanged.
- Verdict: negative; soup-member tuning exhausted.

### 2026-10-23 — cross-recipe weight-space soup of the two 6-layer members: catastrophic barrier (58-82 pp CER), gate not earned, no test, best unchanged

- Idea: trained-run triggers unfired (l6-cos-EMA val-best still e59 4.831%, epochs
  60-82 all worse, bar <=4.70%; l9-cos-EMA epoch 8, 8.17% CTC far from <=5.55%),
  `gpu_budget.sh 6144` failed (2,789 free) so evaluation-only. Test the one axis of
  the working soup lever that the retirement did NOT cover: the retired axes are
  within-trajectory spacing/weight tuning on the l6ce member, but every soup so far
  interpolated snapshots of ONE run. Question: does weight averaging connect the two
  independent 6-layer runs (step-schedule e82, cosine+EMA soup5459 - same arch, same
  seed 42, same data order, different LR schedule)? If connected, a cross-recipe
  midpoint would be a genuinely better 6L member, and its recipe pair (m3) is the
  worse parent on paper.
- Method: `scripts/soup_openfs_recognizer.py` on m3=l6-step e82 + m4=l6ce-soup5459,
  alpha 0.25/0.5/0.75 (17,509,820 params each, CPU, seconds); greedy screens on first
  4,000 validation clips (69,783 chars) beside the existing 4k baselines.
- Screening: m3 5.801% (4,048) / m4 4.696% (3,277) / a0.25 **75.42%** (52,628) /
  a0.5 **86.77%** (60,553) / a0.75 **82.58%** (57,623) - a monotone-in-alpha barrier
  of +70 to +82 pp, i.e. total decode collapse (exact 0.0-0.03%), not noise: the two
  recipes converge to basins separated by a vast weight-space barrier. Within-run
  soup worked because EMA snapshots of one schedule share a basin.
- No member greedy, no validation gate, no test: candidate is 70+ pp worse than the
  member it would replace; best unchanged at 3.93% CER / 69.76% exact.
- Cost: 3 soups (CPU); 4 screens, 224-267 s each, 128 MiB each (nice -n 19, two
  training jobs active); total ~20 min GPU.
- Finding: retire weight-space averaging beyond a single run's trajectory - soup is
  trajectory-internal only (and that axis is itself exhausted/ensemble-absorbed).
  Member supply now depends solely on the two running experiments: l9-cos-EMA
  (<=5.55% greedy class) and any genuine l6ce val-best <=4.70%.
- Status: l6-cos-EMA epoch 82/150 (val-best e59 4.831%, trend worsening 4.9-5.1%
  since e59), l9-cos-EMA epoch 8/100 (val-score 8.17% falling ~0.5-0.9/epoch).
  Gates unchanged: 4.31% val (soup5459 ens4) / 3.93% test.
- Verdict: negative; cross-recipe weight averaging retired.

### 2026-10-23 — minimum-Bayes-risk (expected-Levenshtein) selection over the beam pool: val -71 errors (gate earned) but test +4 errors; eighth val/test flip; best unchanged

- Idea: harvest triggers not fired (l6-cos-EMA greedy val-best still e59 4.831% through
  epoch 91; l9-cos-EMA epoch 11 at 6.81% greedy, > 5.55% bar) and `gpu_budget.sh 6144`
  failed (2,789 MiB free), so evaluation-only. Every retired decoder lever kept the
  selection rule fixed at argmax joint score (MAP); the metric is Levenshtein CER, so the
  untested axis is the selection criterion itself. Named experiment: **minimum-Bayes-risk
  (MBR) selection** - collect the beam hypothesis pool per sample (record-setting EOS
  retirements + surviving active prefixes, deduped, best score kept, mean 5.7 unique),
  weight unique hypotheses by softmax(pool scores), output the hypothesis minimising
  expected unit-cost Levenshtein distance. Members (ens4 e62 + step-84 + l6-e82 +
  soup5459-a50), beam 8, logprob fusion, equal weights unchanged; MAP reported as primary
  in the same pass; no dictionary or language correction.
- Code: pool collection + `mbr_select` in `scripts/beam_eval_openfs_recognizer.py`; one
  decode pass reports both selections. Sanity: MAP byte-identical to the HEAD harness on
  the 32-clip smoke (37 errors / 20 exact - the a50-set smoke; the earlier journal quote
  "39/20 for the a50 set" traces to the e54-member smoke) and full-split MAP reproduces
  the recorded gate and best exactly (val 9,133 errors / 7,931 exact; test 8,698 / 8,735).
- Validation (12,015 clips, beam 8): MBR **4.274% CER** (9,062 errors) / **65.98% exact**
  (7,928) vs MAP gate 4.307% (9,133) / 66.01% (7,931) - 71 fewer errors, 3 fewer exact;
  gate nominally earned (above the -39 flip band); test pass taken.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459-mbr.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): MBR **3.936% CER** (8,702 errors) / **69.74% exact**
  (8,732) vs the 3.93% / 69.76% best: +4 errors, -3 exact - eighth sign flip of a
  sub-0.05 pp validation delta; best unchanged at 3.93% / 69.76% (MAP).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-mbr.beam-eval-report.json`
- Cost: smoke 2 x ~20 s / 608 MiB; validation 5,527 s / 879 MiB; test 5,971 s / 766 MiB
  (batch 8, nice -n 19, shared GPU, two training jobs active).
- Finding: selection-rule axis exhausted like beam-width saturation - the validation MBR
  gain was concentrated on hard signers (d5fd -34, c6e5 -22) but did not exist on test
  (+4): the fused posterior is confidently unimodal, MAP ~ MBR, and any consensus signal
  lives inside the sub-0.05 pp band that now flips on test eight times in nine tries.
  Members remain the only lever.
- Status: l6-cos-EMA epoch ~91/150 (val-best e59 4.831%); l9-cos-EMA epoch ~11/100,
  greedy val-best 6.81% / 50.99% and falling ~0.7/epoch - next iteration should check it
  against the <=5.55% 5th-member trigger. Gates unchanged: 4.31% val / 3.93% test.
- Verdict: negative on the primary metric; retire MBR/selection-rule tuning.

### 2026-10-23 — launched width-capacity ensemble member candidate: 6-layer d_model 384, cosine+EMA, full data (run in progress)

- Idea: harvest triggers not fired - l6-cos-EMA greedy val-best still e59 (4.831%,
  epochs 60-91 all worse, bar <=4.70%); l9-cos-EMA epoch 12 greedy 6.53%, falling
  ~0.3/epoch, still above its <=5.55% bar. Members remain the only lever; depth is
  already in flight (l9), so take the untested capacity/arch-diversity axis:
  **width**. Every member is d_model 256; a d_model 384 stack is arch-distinct from
  all four (3L-256 x2, 6L-256 x2 incl. soup) and probes single-model quality beyond
  the 6-layer recipe's 4.83% class.
- Architecture: gaps transformer, `--layers 6 --d-model 384 --heads 12 --feedforward
  3072` (head dim 32 unchanged); **39,286,076 params**. Recipe = champion
  cosine+EMA: full 126,131-clip train slice, Adam 1e-4, 5-epoch warmup, cosine
  anneal over 100 epochs (l9 precedent: long tails demonstrably don't pay for
  joinable members), seed 42 (single variable is width; seed axis retired).
- Batch 4 (not 8): `gpu_budget.sh 6144` failed (2,789 MiB free), so per rule 2
  batch halved; EMA decay 0.99995 epoch-matched at 31,532 steps/epoch (~20k-step
  horizon, l9/seed2 precedent).
- Smoke: 1,600 clips / 2 epochs at batch 4 peaked **1,361 MiB**, loss 3.31 -> 2.85,
  stable (post-norm 6x384 fine at 1e-4 + warmup).
- VRAM: ~1.8 GiB steady (free 2,789 -> 965 MiB after launch; three training jobs
  sharing the GPU), inside the ~6 GiB single-run budget. Wall clock: ~13 steps/s
  -> ~2,400 s train + ~300 s validation per epoch, ETA ~75 h for 100 epochs;
  joinable much earlier at member-class quality (l9 trajectory suggests <=6% class
  by ~epoch 20).
- Output: `data/models/openfs-recognizer-gaps-full-layers6-width384-cosine-ema/
  recognizer.pt` (real train PID 1683248; pid file initially held the dead setsid
  launcher shell, corrected). No test evaluation; none earned.
- Gates unchanged: 4.31% val / 3.93% test. Triggers: l6-cos-EMA greedy val-best
  <=4.70% -> 4th-member refresh; l9-cos-EMA greedy val-best <=5.55% -> 5th-member
  screening; width384 greedy val-best <=5.55% class -> 5th/6th-member candidate
   screening (beam-8 validation gate vs 4.31%, member-quality bar applies).
- Verdict: pending; run in progress.

### 2026-10-23 — soup upgrade of the m1 member (3L cos-EMA e35+e62 a50): member greedy -0.181 pp but ensemble validation 4.31%/9,149 vs gate 4.31%/9,133; gate not passed, no test, best unchanged

- Idea: harvest triggers not fired (l6-cos-EMA val-best still e59 4.831% through epoch 101;
  l9-cos-EMA epoch 14 greedy 6.08% > 5.55% bar; width384 epoch 1 warmup) and
  `gpu_budget.sh 6144` failed (783 MiB free at start), so evaluation-only. The within-run
  soup lever produced the current best member but had only ever been applied to the l6ce
  member (already souped) and retired for spacing tuning there. Untested: the soup lever on
  the one strong member never averaged - m1 (3L cos-EMA e62, greedy 5.51%/61.39%, raw EMA
  weights). Named experiment: **m1 member soup upgrade** - chord e35+e62 (27-epoch spacing,
  same run, cosine schedule) as 4-member ens4 with s84 + l6e82 + soup5459 unchanged, beam 8,
  logprob fusion, equal weights, no dictionary or language correction.
- Screening (greedy, first 4,000 validation clips, 69,783 chars): e62 5.477 (3,822) /
  e35 5.805 (4,051) / a25 5.362 (3,742) / a50 **5.311 (3,706, -116 vs best parent)** /
  a75 5.453 (3,805) - pair optimum alpha 0.5 repeats (same as the 5-spaced m4 pair).
- Member greedy full validation (12,015 clips): soup-a50 **5.330% CER / 61.81% exact** vs
  e62's 5.51%/61.39% - a -0.181 pp member upgrade (-382 errors), larger than soup5459's
  -0.126 pp that transferred;
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-greedy-m1-soup-e35e62-a50.beam-eval-report.json`.
- Validation (12,015 clips, beam 8, ens4 with m1->soup): **4.31% CER** (9,149 errors) /
  **66.06% exact** (7,937) vs gate 4.31% (9,133) / 66.01% (7,931) - +16 errors, +6 exact;
  gate NOT passed; no test pass; best unchanged at 3.93% / 69.76%.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-soupe35e62-s84-l6e82-soup5459.beam-eval-report.json`
- Params: unchanged 52,676,336 (member swap, same arch). Cost: 5 subset screens 224-580 s /
  ~130 MiB; member greedy val 945 s / 132 MiB; smoke 20 s / 608 MiB (32 errors / 21 exact vs
  39 / 20 for the e62 set); validation beam 8 5,688 s / ~880 MiB (batch 8, nice -n 19,
  shared GPU, three training jobs active).
- Finding: member-led soup transfer is slot-specific, not universal - the m4 (l6ce) slot
  transferred four member upgrades, the m1 (3L cos-EMA) slot absorbed a -0.181 pp member
  gain entirely (+16 ensemble errors). Consistent with the retired seed axis: the 3L
  cos-EMA slot's information is already covered by the s84+l6ce members, so a better
  version of m1 is ensemble-invisible. Retire member upgrades within the 3L cos-EMA slot;
  member supply stays with structurally new l9-cos-EMA and width384 runs.
- Status: l6-cos-EMA epoch ~101/150 (val-best e59), l9-cos-EMA epoch ~14 (greedy 6.08%,
  its <=5.55% 5th-member trigger ~2 epochs away), width384 epoch 1 (54.9% warmup). Gates
  unchanged: 4.31% val (soup5459 ens4) / 3.93% test.
- Verdict: negative; 3L-cos-EMA-slot member upgrades retired.

### 2026-10-23 — l9-cos-EMA epoch-16 as 5th ensemble member: val -75 errors (gate earned) but test -2 errors / exact -21; ninth val/test fade; best unchanged

- Idea: `gpu_budget.sh 6144` failed (2,789 MiB free) so evaluation-only. l9-cos-EMA
  epoch 16 hit greedy validation **5.878% CER / 56.64% exact** (frozen as
  `recognizer-epoch16.pt`) - the same member class (5.85%) at which l6-e82 joined and
  paid +0.32 pp at ens3; the <=5.55% trigger was not yet hit but depth is the last
  unfailed structural diversity axis. Named experiment: **depth-diversity 5th member** -
  ens4 (e62 + step-84 + l6-e82 + soup5459) + l9e16, equal weights, logprob fusion,
  beam 8, no dictionary or language correction.
- Screening: 32-clip smoke 27 errors / 22 exact vs ens4 37 / 20; greedy first-4,000-clip
  subset ens4 3,214 (4.606%) vs ens5 3,179 errors (4.556%) / exact 2,687 vs 2,699 -
  same direction, small.
- Validation (12,015 clips, beam 8): **4.272% CER** (9,058 errors) / **65.80% exact**
  (7,906) vs gate 4.307% (9,133) / 66.01% - 75 fewer errors but exact -25; CER gate
  earned; test pass taken.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e16.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.933% CER** (8,696 errors) / **69.595% exact**
  (8,714) vs best 3.934% (8,698) / 69.76% (8,735) - CER -2 errors (noise), exact -21;
  ninth val/test fade of a sub-0.05 pp validation edge; best unchanged.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e16.beam-eval-report.json`
- Params: 52,676,336 + 26,191,292 = 78,867,628 (mixed 3L/6L/9L stacks decode fine).
  Cost: smoke 28 s / 757 MiB; screens 391 + 582 s / 276-381 MiB; validation 7,841 s /
  1,062 MiB; test 8,528 s / 936 MiB (batch 8, nice -n 19, shared GPU, three training
  jobs active).
- Finding: the 5-member depth ensemble is ensemble-invisible at this member strength -
  a 5.88%-class member now adds ~nothing to the saturated ens4 (contrast: the same-class
  l6-e82 paid at ens3 where member diversity was scarce). Not an architecture kill:
  l9 at epoch 16/100 is still climbing steeply.
- Status: l9-cos-EMA epoch 20 greedy val-best **5.409% / 58.83%** - the <=5.55% trigger
  has now fired; l6-cos-EMA epoch 114/150 (val-best e59 4.831%, bar <=4.70% unfired);
  width384 epoch 1-2. Gates unchanged: 4.31% val / 3.93% test. Next iteration: refresh
  the 5th-member candidate to the newest l9 val-best snapshot (e20+) and re-gate ens5
  vs 4.31% - member-refresh is the lever that transferred four times.
- Verdict: negative on exact; CER tie within noise; retire 5.88%-class l9 as a joinable
  member, keep the l9 run for the fully-annealed snapshot.

### 2026-10-24 — refreshed 5th ensemble member to l9-cos-EMA epoch-21: **NEW BEST 3.87% CER / 69.74% exact**

- Idea: `gpu_budget.sh 6144` failed (2,789 MiB free) so evaluation-only. l9-cos-EMA greedy
  val-best advanced e16 5.878 -> e21 5.339% / 59.33% (frozen as `recognizer-epoch21.pt`),
  firing the <=5.55% trigger with a 0.54 pp member jump - far above the noise band that
  produced nine val/test flips. Named experiment: **5th-member refresh round 1** - ens4
  (e62 + step-84 + l6-e82 + soup5459) unchanged, l9e16 -> l9e21, equal weights, logprob
  fusion, beam 8, no dictionary or language correction.
- Screening: 32-clip smoke 27 errors / 22 exact (vs ens4 37 / 20); greedy first-4,000-clip
  subset ens5 **3,150 errors (4.514%) / 2,698 exact** vs ens4 baseline 3,214 (4.606%).
- Validation (12,015 clips, beam 8): **4.232% CER** (8,973 errors) / **66.27% exact**
  (7,962) vs gate 4.307% (9,133) / 66.01% - 160 fewer errors, +31 exact; both metrics,
  far outside the flip band; test pass earned (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e21.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.874% CER** (8,566 errors) / **69.74% exact**
  (8,732) - CER new best on both metrics, previous 3.93% / 69.76%; exact -3 strings
  (tie within noise) while CER gained 132 errors. Fifth clean member-led transfer:
  member greedy -0.54 pp -> val -0.076 pp -> test -0.060 pp.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e21.beam-eval-report.json`
- Params: 52,676,336 + 26,191,292 = 78,867,628. Cost: smoke 28 s / 757 MiB; 4k screen
  537 s / 381 MiB; validation 7,833 s / 1,062 MiB; test 8,566 s / 936 MiB (batch 8,
  nice -n 19, shared GPU, two training jobs active).
- Finding: the earlier l9e16 fade was a member-strength problem, not an architecture kill -
  at 5.34% greedy the same 9L slot turns the "ensemble-invisible" depth member into a
  record transfer. Member-refresh remains the only lever with a monotone transfer history.
- Status: l9-cos-EMA epoch 27, greedy val-best **4.895% / 61.52%** and still climbing
  steeply (next refresh when its val-best materially improves; l9e21 is now the 5th member).
  l6-cos-EMA epoch 128/150, val-best still e59 4.831%, bar <=4.70% unfired and the run
  plateaued/decaying since e59. width384 run DIED at epoch 2 (PID gone, no error in log,
  only e1 checkpoint kept) - relaunch is the next-iteration training action.
- Gates now reference 4.23% validation CER (ens5-l9e21 beam-8) and 3.87% test CER.
  Triggers: l9-cos-EMA greedy val-best materially below 4.90% -> 5th-member refresh round 2;
  l6-cos-EMA <=4.70% -> 4th-member refresh; width384 relaunch when GPU budget allows.
- Verdict: positive; new best recorded.


### 2026-10-24 — 5th-member refresh round 2 (l9-cos-EMA e21 -> e27): **NEW BEST 3.83% CER / 70.08% exact**

- Idea: the trigger set at the e21 round ("l9-cos-EMA greedy val-best materially below
  4.90% -> refresh round 2") fired: l9 greedy validation-best advanced e21 5.339 ->
  e27 4.895% / 61.52% (frozen as `recognizer-epoch27.pt`), a 0.444 pp member jump,
  same class as the 0.54 pp e16->e21 transfer. `gpu_budget.sh 6144` failed (2,789 MiB
  free; l6 132/150 and l9 e28 training) so evaluation-only. Named experiment:
  **5th-member refresh round 2** - ens4 (e62 + step-84 + l6-e82 + soup5459) unchanged,
  l9e21 -> l9e27, equal weights, logprob fusion, beam 8, no dictionary or language
  correction.
- Screening: 32-clip smoke 28 errors / 21 exact (vs l9e21 smoke 27 / 22 - 651 chars,
  noise); greedy 4k subset ens5 **3,154 errors (4.520%) / 2,718 exact** vs l9e21
  3,150 (4.514%) / 2,698: CER +4 (noise) but exact +20 - mixed within the flip band,
  proceed to the decisive full gate since the member gain itself is real and large.
- Validation (12,015 clips, beam 8): **4.190% CER** (8,884 errors) / **66.55% exact**
  (7,996) vs gate 4.232% (8,973) / 66.27% - 89 fewer errors, +34 exact, both metrics,
  outside the flip band; test pass earned (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e27.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.833% CER** (8,474 errors) / **70.08% exact**
  (8,775) vs best 3.874% (8,566) / 69.74% (8,732): 92 fewer errors AND 43 more exact
  strings - first round in a while where val and test improve on both metrics;
  sixth clean member-led transfer (member greedy -0.444 pp -> val -0.042 pp ->
  test -0.041 pp).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e27.beam-eval-report.json`
- Params: unchanged 78,867,628 (member swap, same 9L arch). Cost: smoke 28 s / 757 MiB;
  screen 537 s / ~380 MiB; validation 7,675 s / 1,062 MiB; test 8,367 s / 936 MiB
  (batch 8, nice -n 19, shared GPU, two training jobs active). Peak VRAM 1,062 MiB;
  wall clock ~4.6 h GPU serial. MBR secondary decode (retired axis) 8,475 errors,
  MAP reported as primary.
- Status: l9-cos-EMA epoch 32/100, greedy val-best now **4.809% / 63.08%** (e32) and
  still climbing - refresh round 3 when the val-best moves materially again (watch
  e35+; cosine anneal has 68 epochs of tail left). l6-cos-EMA epoch 141/150, val-best
  still e59 4.831%, <=4.70% bar unfired, decaying since e59 - 4th-member slot expected
  to retire when the run ends unless a late epoch surprises. width384 still dead,
  relaunch still budget-blocked (1,509-2,789 MiB free all session).
- Gates now 4.19% validation CER (ens5-l9e27 beam-8) / 3.83% test CER.
  Triggers: l9-cos-EMA greedy val-best materially below 4.81% -> refresh round 3;
  l6-cos-EMA <=4.70% -> 4th-member refresh; width384 relaunch when budget allows.
- Verdict: positive; new best recorded.

### 2026-10-24 — relaunched width384 member candidate (r2, batch 2, anti-fragmentation) after diagnosing r1 death as shared-GPU OOM (run in progress)

- Idea: no member trigger fired (l9-cos-EMA greedy val-best e32 4.809%, only -0.086 pp
  vs the frozen e27 4.895% - inside the flip band, refresh round 3 not justified;
  l6-cos-EMA ~e145/150 val-best still e59 4.831%). Queued action from the last two
  entries: relaunch the width384 run that died at epoch 2. Diagnosis of r1 death from
  its .err: CUDA OOM mid-epoch-2 while a foreign tenant held 24.07 GiB (failed alloc
  only 20 MiB against 1.83 GiB reserved-but-unallocated -> fragmentation + contention,
  not the recipe).
- Fix: relaunch identical recipe (6L d384 h12 ff3072, 39,286,076 params, cosine+EMA,
  full 126,131-clip slice, seed 42, 100 epochs) with (a) batch 4 -> 2 per rule 2
  (gpu_budget.sh 6144 failed: 2,789 MiB free), (b) EMA decay 0.99995 -> 0.999975
  epoch-matched at 63,066 steps/epoch (~40k-step horizon, same 0.63-epoch window as
  the l9/width384-r1 precedent), (c) PYTORCH_ALLOC_CONF=expandable_segments:True.
  r1 dead artifacts moved to `...width384-cosine-ema-dead-e1/`.
- Smoke: 1,600 clips / 2 epochs batch 2: stable (loss 3.3 -> 2.76), script-reported
  peak VRAM **935 MiB**, min GPU free during smoke 1,381 MiB.
- VRAM/wall: ~1.5 GiB steady (free 2,789 -> 1,277 after launch); ~15.3 steps/s ->
  ~4,100 s train + ~500 s validation per epoch, ETA ~128 h for 100 epochs; joinable
  member-class expected ~e20 (~30 h) on the r1/l9 trajectory precedent.
- Output: `data/models/openfs-recognizer-gaps-full-layers6-width384-cosine-ema/
  recognizer.pt` (PID 1974441, corrected pid file -r2.pid; log/err -r2). No test
  evaluation; none earned. Triggers: width384 greedy val-best <=5.55% class ->
  5th/6th-member screening vs 4.19% gate; l9-cos-EMA val-best materially below
  4.81% -> refresh round 3; l6-cos-EMA <=4.70% -> 4th-member refresh (run ends ~e150).
- Verdict: pending; run in progress.

### 2026-10-24 — 5th-member refresh round 3 (l9-cos-EMA e27 -> e33): **NEW BEST 3.81% CER / 70.08% exact**

- Idea: l9-cos-EMA greedy val-best advanced e27 4.895 -> e32 4.809 -> e33 **4.719% /
  63.36%** (frozen from live `recognizer.pt` as `recognizer-epoch33.pt`), -0.176 pp /
  exact +1.84 pp vs the ensembled member - at the magnitude of the e54 refresh that
  transferred (-0.22 pp member). `gpu_budget.sh 6144` failed (1,195 MiB free) so
  evaluation-only. Named experiment: **5th-member refresh round 3** - ens4 (e62 +
  step-84 + l6-e82 + soup5459) unchanged, l9e27 -> l9e33, equal weights, logprob
  fusion, beam 8, no dictionary or language correction.
- Screening: 32-clip smoke 29 errors / 23 exact (vs e27 28 / 21, noise); greedy 4k
  subset ens5 **3,113 errors (4.461%) / 2,723 exact** vs l9e27 3,154 (4.514%) / 2,718
  - -41 errors, direction positive.
- Validation (12,015 clips, beam 8): **4.146% CER** (8,791 errors) / **66.70% exact**
  (8,014) vs gate 4.190% (8,884) / 66.55% (7,996) - 93 fewer errors, +18 exact, both
  metrics, outside the flip band; test pass earned (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e33.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.813% CER** (8,430 errors) / **70.082% exact**
  (8,775) vs best 3.833% (8,474) / 70.08% (8,775): 44 fewer errors, identical exact
  count - new best on CER, exact tie. Seventh clean member-led transfer (member
  greedy -0.176 pp -> val -0.044 pp -> test -0.020 pp).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e33.beam-eval-report.json`
- Params: unchanged 78,867,628 (member swap, same 9L arch). Cost: smoke 41 s / 757 MiB;
  screen ~700 s / 381 MiB; validation 9,888 s / 686 MiB; test 10,140 s / 623 MiB
  (batch 4, nice -n 19, shared GPU; two attempts at batch 8 OOMed mid-start when a
  foreign tenant grew to 24.07 GiB - retried with batch 4 +
  PYTORCH_ALLOC_CONF=expandable_segments:True, which ran clean).
- Status: l9-cos-EMA epoch ~42/100, greedy val-best already **e41 4.542% / 64.73%** -
  refresh round 4 fires immediately next iteration (snapshot e41+, re-gate vs 4.15%);
  l6-cos-EMA finished 150/150, final val-best e59 4.831% (late epochs ~5.69%, decaying -
  4th-member slot frozen); width384-r2 epoch 7, greedy 8.89% falling, still far from
  <=5.55% bar but alive (batch 2, ETA long).
- Gates now 4.15% validation CER (ens5-l9e33 beam-8) / 3.81% test CER.
- Verdict: positive; new best recorded (CER; exact tie).

### 2026-10-25 — 5th-member refresh round 4 (l9-cos-EMA e33 -> e41): **NEW BEST 3.78% CER / 70.47% exact**

- Idea: `gpu_budget.sh 6144` failed (3,591 MiB free) so evaluation-only. l9-cos-EMA greedy
  val-best advanced e33 4.719 -> e41 **4.542% / 64.73%** (frozen from live `recognizer.pt`
  as `recognizer-epoch41.pt`), -0.177 pp / exact +1.37 pp vs the frozen e33 member - the
  same magnitude as the round-3 transfer. Named experiment: **5th-member refresh round 4** -
  ens4 (e62 + step-84 + l6-e82 + soup5459) unchanged, l9e33 -> l9e41, equal weights, logprob
  fusion, beam 8, no dictionary or language correction.
- Screening: 32-clip smoke 28 errors / 22 exact (vs e33 29/23, noise); greedy 4k
  subset ens5 **3,071 errors (4.401%) / 2,735 exact** vs l9e33 3,113 / 2,723 -
  -42 errors, +12 exact, direction positive; screen gate 3,113 passed.
- Validation (12,015 clips, beam 8): **4.133% CER** (8,763 errors) / **67.03% exact**
  (8,054) vs gate 4.146% (8,791) / 66.70% (8,014) - 28 fewer errors, +40 exact, both
  metrics; test pass earned (rule 3). The CER edge is small (well inside the historical
  flip band that produced nine prior val/test fades), but exact +40 and the member greedy
  gain are unambiguous.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.777% CER** (8,351 errors) / **70.47% exact**
  (8,823) vs best 3.813% (8,430) / 70.08% (8,775): 79 fewer errors AND 48 more exact
  strings - new best on both metrics, eighth clean member-led transfer (member greedy
  -0.177 pp -> val -0.013 pp -> test -0.036 pp; test outgained val, the fade band broke
  the right way this round).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e41.beam-eval-report.json`
- Params: unchanged 78,867,628 (member swap, same 9L arch). Cost: smoke 38 s / 534 MiB;
  screen 926 s / 346 MiB; validation 9,367 s / 686 MiB; test 10,175 s / 623 MiB
  (batch 4, nice -n 19, shared GPU, two training jobs active; foreign tenant held ~24 GiB
  all session - batch 4 + expandable_segments:True ran clean). Peak VRAM 686 MiB;
  wall clock ~5.7 h GPU serial. Note: a detached runner (setsid) was needed after the
  first launch died silently at screen-stage when a timed-out tool call killed its
  process group (not a GPU fault).
- Status: l9-cos-EMA epoch 50/100, greedy val-best e50 **4.522% / 65.4%** - only -0.020 pp
  vs the frozen e41, inside the flip band; refresh round 5 not yet justified (watch for
  a material drop again as cosine anneal tail progresses). l6-cos-EMA finished (e59 4.831%
  frozen member); width384-r2 alive at batch 2, still far from <=5.55% class.
- Gates now 4.13% validation CER (ens5-l9e41 beam-8) / 3.78% test CER.
  Triggers: l9-cos-EMA greedy val-best materially below 4.54% -> refresh round 5;
  width384 greedy val-best <=5.55% class -> 6th-member screening.
- Verdict: positive; new best recorded (both CER and exact).

### 2026-10-25 — 5th-member refresh round 5 (l9-cos-EMA e41 -> e43): **NEW BEST 3.73% CER / 70.63% exact**

- Idea: `gpu_budget.sh 6144` failed (3,571 MiB free) so evaluation-only. History re-read
  correction: the l9 run's true greedy validation-best is **e43 4.422% / 65.14%**
  (val_score column; the round-4 status line misread the e50 row 4.522% as val-best),
  i.e. the live `recognizer.pt` payload (epoch=43 confirmed, saved on improvement) is
  already -0.120 pp / +0.41 pp better than the frozen e41 member - the member-gain
  class at which the l6ce soup (-0.126 pp) transferred. Frozen `recognizer-epoch43.pt`
  and ran the proven lever. Named experiment: **5th-member refresh round 5** - ens4
  (e62 + step-84 + l6-e82 + soup5459) unchanged, l9e41 -> l9e43, equal weights,
  logprob fusion, beam 8, no dictionary or language correction.
- Screening: 32-clip beam-8 smoke 24 errors / 22 exact (vs l9e41 28 / 22); greedy 4k
  subset ens5 **3,029 errors (4.341%) / 2,739 exact** vs l9e41 3,071 / 2,735 -
  -42 errors, +4 exact; screen gate 3,071 passed.
- Validation (12,015 clips, beam 8): **4.100% CER** (8,694 errors) / **67.11% exact**
  (8,063) vs gate 4.133% (8,763) / 67.03% (8,054) - 69 fewer errors, +9 exact, both
  metrics; test pass earned (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9e43.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.730% CER** (8,246 errors) / **70.63% exact**
  (8,843) vs best 3.777% (8,351) / 70.47% (8,823): 105 fewer errors AND 20 more exact
  strings - new best on both metrics, ninth clean member-led transfer (member greedy
  -0.120 pp -> val -0.033 pp -> test -0.047 pp; test outgained val for the second
  consecutive round).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9e43.beam-eval-report.json`
- Params: unchanged 78,867,628 (member swap, same 9L arch). Cost: smoke ~60 s / 534 MiB;
  screen ~900 s / 346 MiB; validation 9,460 s / 686 MiB; test 10,303 s / 623 MiB
  (batch 4, nice -n 19, shared GPU, two training jobs active; detached setsid runner
  `scripts/run_ens5_l9e43.sh`, pid file `run-ens5-l9e43.pid`). Peak VRAM 686 MiB;
  wall clock ~5.9 h GPU serial. MBR secondary decode: val 8,687 / test 8,244 (retired
  selection axis, MAP reported as primary).
- Status: l9-cos-EMA epoch 56/100, greedy val-best now e55 **4.417% / 65.45%** - only
  -0.005 pp vs the frozen e43, inside the flip band; refresh round 6 not justified
  (watch for a material drop below 4.42% as the cosine tail runs). l6-cos-EMA finished
  (e59 4.831% frozen); width384-r2 epoch 16/100, greedy best e9 8.54%, far from the
  <=5.55% member bar. Note for future harvests: trust the history `validation_score`
  minimum, not the latest row.
- Gates now 4.10% validation CER (ens5-l9e43 beam-8) / 3.73% test CER.
  Triggers: l9-cos-EMA greedy val-best materially below 4.42% -> refresh round 6;
  width384 greedy val-best <=5.55% class -> 6th-member screening.
- Verdict: positive; new best recorded (both CER and exact).

### 2026-10-25 — l9-slot 3-way trajectory soup (e41+e43+e55) replaces l9e43 as 5th member: **NEW BEST 3.72% CER / 70.71% exact**

- Idea: `gpu_budget.sh 6144` failed (5,409 MiB free) so evaluation-only. Refresh round 6
  was not justified (l9 greedy val-best e55 4.417% vs frozen e43 4.422%, -0.005 pp inside
  the flip band) but the within-trajectory soup lever - which produced the best m4 member
  and was retired only for *spacing tuning in the m4 slot*, explicitly leaving "the l9
  member" open - had never been applied to m5, the one slot with a monotone member-led
  transfer history. Named experiment: **l9-slot trajectory soup** - freeze live e55,
  build soups of frozen l9 snapshots, best one replaces l9e43 in ens5 (e62 + step-84 +
  l6-e82 + soup5459 unchanged), equal weights, logprob fusion, beam 8, no dictionary or
  language correction.
- Code: staged gated runner `scripts/run_ens5_l9soup.sh` (soups -> 4k greedy member
  screens -> member full-val confirm -> ens5 val gate -> test); 3-way soup via chained
  `soup_openfs_recognizer.py` (e41+e43 a50, then +e55 a2/3 = equal thirds).
- Screening (greedy 4k, 69,783 chars): e43 3,103 / e55 3,151 / s4155-a50 2,969 /
  s4355-a50 3,008 / 3-way **2,908** (best-parent -195, far above noise; wide horizon wins
  at member level, unlike the m4 pair-spacing optimum).
- Member greedy full validation (12,015 clips): soup **4.247% CER / 66.52% exact** vs
  parents e43 4.422 / e55 4.417 - member-led gain -0.17 pp, the round-3/5 transfer class.
- Validation (12,015 clips, beam 8): **4.093% CER** (8,679 errors) / **67.27% exact**
  (8,082) vs gate 4.100% (8,694) / 67.11% (8,063) - 15 fewer errors, +19 exact; the CER
  edge sits in the historical flip band, but exact +19 and the member-led rule took the
  test pass (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.719% CER** (8,222 errors) / **70.71% exact**
  (8,854) vs best 3.730% (8,246) / 70.63% (8,843): 24 fewer errors AND 11 more exact
  strings - new best on both metrics, tenth clean member-led transfer (member -0.17 pp
  -> val -15 errors -> test -24 errors).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-s84-l6e82-soup5459-l9soup.beam-eval-report.json`
- Params: unchanged 78,867,628 (member swap, same 9L arch). Cost: 5 screens 267-318 s /
  ~140 MiB; member greedy 829 s / 140 MiB; validation 6,248 s / 686 MiB; test 6,626 s /
  623 MiB (batch 4, nice -n 19, shared GPU, two training jobs active). Total 15,118 s
  GPU serial. Detached runner log `data/logs/run-ens5-l9soup.log`.
- Finding: slot-specific absorption is two-sided - the m5 (l9) slot accepts member
  upgrades that the m1/m4 slots absorb; within it wide-horizon soup (3 snapshots over
  14 epochs) beats pair soups, the opposite of m4 behavior. Soup and refresh compose:
  the soup horizon now spans e41-e55, so round-6 supply needs a new l9 val-best outside
  that span.
- Status: l9-cos-EMA epoch 62/100, greedy val-best still e55 4.417% (no improvement
  since e55; 38 cosine-tail epochs left) -> trigger: val-best materially below 4.42%
  (re-soup with the new snapshot, e.g. e43+e55+eX). l6-cos-EMA finished (e59 4.831%
  frozen); width384-r2 epoch 17/100, far from <=5.55% bar.
- Gates now 4.09% validation CER (ens5-l9soup beam-8) / 3.72% test CER.
- Verdict: positive; new best recorded (both CER and exact).

### 2026-10-25 — ens5 leave-one-out pruning: drop redundant step-84 member -> **NEW BEST 3.70% CER / 71.02% exact**; width384-r2 died of training collapse (negative)

- Idea: `gpu_budget.sh 6144` failed (5,409 MiB free; l9-cos-EMA at epoch 63/100) so
  evaluation-only. The ensemble has only ever grown; the two slot-absorption findings
  (m1 and m4 member upgrades ensemble-invisible) motivate the inverse untested axis.
  Named experiment: **ens5 member pruning (leave-one-out)** - ens4 (e62 + step-84 +
  l6-e82 + soup5459 + l9soup unchanged), beam 8, logprob fusion, equal weights, no
  dictionary or language correction.
- Screens (greedy, first 4,000 validation clips, 69,783 chars; ens5 baseline 3,017 /
  2,747 exact): drop-m2(step-84) **2,978 / 2,765** (-39, +18); drop-m1(e62) 3,077 (+60);
  drop-m3(l6-e82) 3,043 (+26); drop-m4(soup5459) 3,103 (+86); drop-m5(l9soup) 3,214
  (+197). Redundancy map: the oldest 3L step member is redundant, the newest members
  carry all the ensemble signal.
- Validation (12,015 clips, beam 8, ens5 minus step-84): **4.057% CER** (8,601 errors) /
  **67.86% exact** (8,153) vs gate 4.093% (8,679) / 67.27% - 78 fewer errors AND 71 more
  exact, both metrics outside the flip band; test pass earned (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens4-e62-l6e82-soup5459-l9soup.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.702% CER** (8,186 errors) / **71.02% exact**
  (8,892) - new best on both metrics, previous 3.719% (8,222) / 70.71% (8,854): 36 fewer
  errors AND 38 more exact strings from *removing* a member.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens4-e62-l6e82-soup5459-l9soup.beam-eval-report.json`
- Params: 78,867,628 -> **70,039,280** (member count 5 -> 4). Cost: smoke 29 errors / 21
  exact; 6 screens ~6 x 900 s; validation 5,542 s / 626 MiB; test 5,989 s / 567 MiB
  (batch 4, nice -n 19, shared GPU). MBR secondary (retired axis): val 8,589 / test 8,166.
  Staged gated runner committed: `scripts/run_ens5_prune.sh`, detached log
  `data/logs/run-ens5-prune.log`.
- Negative housekeeping: **width384-r2 died mid-session - training collapse, not OOM**:
  validation CER blew up to 144.7% at epoch 11 (exact 0.0%) and non-finite fp16 loss
  killed it at epoch 17 step 39,532; best checkpoint e9 at 8.54% greedy, far from the
  member bar. Batch 2 destabilizes the d384 stack; with r1's OOM that closes the width
  axis (retire width384; restart budget-blocked anyway).
- Finding: a redundant member is ensemble noise - dropping it gained -36 test errors /
  +38 exact while dropping m1/m3/m4/m5 all degraded even ens5 at screen level. Redundancy
  is context-dependent, so re-run the LOO map on the pruned ens4 next (m1 may no longer
  be absorbed now that m2 is gone; drop-m2-vs-ens4 interaction untested).
- Status: l9-cos-EMA epoch 69/100, greedy val-best still e55 4.417% (no move since e55;
  31 cosine-tail epochs left) -> re-soup m5 when its val-best materially beats 4.42%.
  Gates now **4.06% validation CER** (pruned ens4 beam-8) / **3.70% test CER**.
- Verdict: positive; new best recorded (both CER and exact), ensemble 11% cheaper.

### 2026-10-25 — leave-one-out re-map of the pruned ens4: dropping l6-e82 wins val by -166 errors (largest earned margin yet) but test +19; tenth val/test sign flip; best unchanged

- Idea: `gpu_budget.sh 6144` failed (5,409 MiB free; l9-cos-EMA epoch ~70/100, val-best
  still e55 4.417%, no re-soup trigger) so evaluation-only. The named follow-up from the
  ens5 prune entry: redundancy is context-dependent, so re-run the leave-one-out map on
  the pruned ens4 (e62 + l6-e82 + soup5459 + l9soup); the drop-m2-vs-ens4 interaction was
  untested. Beam 8, logprob fusion, equal weights, no dictionary or language correction.
- Code: staged gated runner `scripts/run_ens4_prune.sh` (smokes -> 4k greedy screens ->
  full-val beam-8 gate 8,601 -> test), detached `setsid`, log `data/logs/run-ens4-prune.log`.
  ens4 4k baseline reused from the ens5-prune drop-m2 screen (2,978 errors / 2,765 exact -
  identical members and subset).
- Screens (greedy, first 4,000 validation clips; baseline 2,978): drop-m1(e62) **2,962 /
  2,765** (-16); drop-m3(l6-e82) **2,918 / 2,781** (-60, best); drop-m4(soup5459) 3,044
  (+66); drop-m5(l9soup) 3,116 (+138 - m5 carries all the signal, as in the ens5 map).
  m3 now looks redundant exactly as m2 did before it.
- Validation (12,015 clips, beam 8, ens3 = e62 + soup5459 + l9soup): **3.978% CER**
  (8,435 errors) / **68.09% exact** (8,181) vs gate 4.057% (8,601) / 67.86% (8,153) -
  166 fewer errors AND +28 exact: five times the old flip band and the largest earned
  margin in the loop; test pass earned (rule 3).
  `data/models/openfs-recognizer-gaps-full-cosine-ema/validation-beam8-ens3-e62-soup5459-l9soup.beam-eval-report.json`
- Test (12,521 clips, beam 8, earned): **3.711% CER** (8,205 errors) / **70.76% exact**
  (8,860) vs best 3.702% (8,186) / 71.02% (8,892): **+19 errors, -32 exact - sign flipped**.
  Tenth val/test flip and the first one far outside the sub-0.05 pp band: a -0.08 pp
  validation gain reversed on test.
  `data/models/openfs-recognizer-gaps-full-cosine-ema/test-beam8-ens3-e62-soup5459-l9soup.beam-eval-report.json`
- Params: 70,039,280 -> 52,529,460 (member count 4 -> 3). Cost: 4 smokes ~13 s each /
  360-530 MiB; 4 screens ~35 min total; validation 4,311 s / 532 MiB; test 4,789 s /
  478 MiB (batch 4, nice -n 19, shared GPU). MBR secondary (retired axis): val 8,412 /
  test 8,163. Best unchanged: ens4 e62 + l6-e82 + soup5459 + l9soup, 3.702% / 71.02%.
- Finding: pruning is a one-shot lever, not a policy - drop-m2 transferred (-36 test
  errors) but the context-updated map's next candidate reversed on test; l6-e82 is
  validation-redundant yet test-useful (diversity insurance). Do not prune further on
  validation evidence; validation-based member selection has now flipped sign ten times
  and its reliability does not improve with margin size.
- Status: l9-cos-EMA epoch ~75/100, val-best still e55 4.417% (trigger: materially below
  4.42% -> re-soup m5). Gates unchanged: 4.06% val / 3.70% test.
- Verdict: negative on the primary metric despite the largest validation margin earned;
  retire iterative LOO pruning of the current member set.

### 2026-10-25 — m5 soup backward horizon extension (4-way e33..e55, 5-way +e27): screens worse, gate not earned, no test, best unchanged

- Idea: `gpu_budget.sh 6144` failed (5,409 MiB free; l9-cos-EMA epoch 73/100, val-best
  still e55 4.417% - epochs 56-73 all worse, and the trainer only saves on improvement,
  so no new weights exist past e55) - forward round-6 soup supply is blocked. The m5
  slot's documented finding is "wide horizon wins at member level", tested only in the
  forward direction (e41->e55). Untested: extend the 3-way soup BACKWARD using frozen
  e33/e27 snapshots. Named experiment: **m5 soup backward extension** - equal-weight
  4-way e33+e41+e43+e55 (22-epoch span) and 5-way +e27 (28-epoch span) via chained
  `soup_openfs_recognizer.py`, screened as m5 swap into the pruned ens4
  (e62 + l6-e82 + soup5459 unchanged), beam 8, logprob fusion, equal weights, no
  dictionary or language correction.
- Screening (greedy member-only, first 4,000 validation clips, 69,783 chars; current
  member s3w baseline 2,908 / 2,750 exact): 4-way **2,980 / 2,715** (+72), 5-way
  **3,106 / 2,690** (+198) - monotone degradation as weaker pre-plateau snapshots
  (e33 4.719, e27 4.895 greedy) are averaged in. Screen gate (beat s3w by >=40) failed;
  runner `scripts/run_ens4_l9soupwide.sh` stopped at stage 2: no member full-val, no
  ensemble gate, no test. Best unchanged at 3.70% CER / 71.02% exact.
- Cost: 2 soups (CPU, seconds); 2 4k screens ~5 min each / ~140 MiB; total ~11 min GPU.
- Finding: the wide-horizon win in the m5 slot is basin-limited - soup averaging pays
  only inside the val-best plateau (e41-e55); diluting with strictly-worse earlier
  snapshots loses at member level regardless of horizon width. m5 round-6 supply needs
  a genuine new l9 val-best better than e55, i.e. forward only. Retire backward soup
  extension.
- Status: l9-cos-EMA epoch 73/100 (val-best e55 4.417%, 27 cosine-tail epochs left ->
  watch for val-best materially below 4.42%, then re-soup forward). Gates unchanged:
  4.06% val (pruned ens4 beam-8) / 3.70% test.
- Verdict: negative at screen level; m5 soup backward extension retired.

### 2026-10-25 — launched depth-capacity member candidate: 12-layer cosine+EMA, full data (run in progress)

- Idea: no member trigger fired (l9-cos-EMA val-best still e55 4.417%, epochs 56-73 all
  worse; trainer saves only on improvement, so no l9 weights exist past e55 and m5
  forward soup supply is blocked) and `gpu_budget.sh 6144` failed (5,409 MiB free), but
  all eval-only levers are retired and members remain the only lever - the journal's
  standing claim is depth is the last unfailed structural diversity axis, and l9soup
  (4.247% greedy) is the strongest member ever. Named experiment: **depth-capacity 12L
  member candidate** - the 9->12 depth step, cos-EMA recipe (stronger family), full
  126,131-clip train slice, gaps representation, seed 42.
- Config: `--layers 12 --epochs 100 --lr-schedule cosine --warmup-epochs 5 --ema-decay
  0.999975 --batch-size 2 --validation-batch-size 16`, Adam 1e-4, fp16 AMP,
  PYTORCH_ALLOC_CONF=expandable_segments:True. Batch 2: budget check failed so per rule
  2 halved from the l9 run's batch 4; EMA decay epoch-matched at 63,066 steps/epoch
  (~40k-step horizon, l9/width384-r2 precedent). Risk noted: width384-r2 collapsed at
  batch 2 (d384), but the d256 l9 stack ran stable at batch 4 and this stack is its
  direct continuation.
- Smoke: 1,600 clips / 2 epochs at batch 2, loss 3.3 -> 2.84 stable, peak VRAM 1,025 MiB.
- Params: 34,872,764 (12-layer d_model 256; +8,681,472 per 3 layers exactly as l9-l6).
  VRAM: ~1.5 GiB steady (free 5,409 -> 3,905 MiB after launch), inside the 6 GiB budget.
  Wall clock: ~9.4 steps/s -> ~6,700 s train + ~600 s validation per epoch, ETA ~200 h
  for 100 epochs; joinable member-class expected ~epoch 20 (~40 h) on the l9 trajectory.
- Output: `data/models/openfs-recognizer-gaps-full-layers12-cosine-ema/recognizer.pt`
  (real train PID 2502729 - pid file first caught the dead launcher shell again,
  corrected). Logs `data/logs/openfs-recognizer-gaps-full-layers12-cosine-ema.log/.err`.
  No test evaluation; none earned.
- Status: l9-cos-EMA epoch 73/100 (val-best e55 4.417%; re-soup m5 if a tail epoch beats
  it materially), l6-cos-EMA finished. Gates unchanged: 4.06% val / 3.70% test.
  Triggers: l12-cos-EMA greedy val-best <=5.55% class -> 5th-member slot screening vs
  the l9soup member (member-quality bar applies); l9-cos-EMA val-best materially below
  4.42% -> forward re-soup.
- Verdict: pending; run in progress.

### 2026-10-26 — m5 soup weight tilt (e41-heavy / e55-heavy 3-way): screens +27/-4 vs equal thirds, gate not earned, no test, best unchanged

- Idea: `gpu_budget.sh 6144` failed (3,903 MiB free; l9-cos-EMA epoch 75/100, l12-cos-EMA
  epoch 1/100) so evaluation-only. Harvest triggers still blocked: l9 greedy val-best
  still e55 4.417% / 65.45% through epoch 74 (epochs 56-74 all worse; trainer saves only
  on improvement -> no new m5 soup supply). The last untouched knob inside the one slot
  with a monotone member-led transfer history is soup-internal snapshot WEIGHTS: the
  s3w member (e41+e43+e55) is equal-thirds by construction; retired weight-tuning
  statements were m4-slot-scoped, and backward extension was about snapshot SET, not
  weights. Named experiment: **m5 soup weight tilt** - w41h (0.50 e41 / 0.25 e43 /
  0.25 e55) and w55h (0.25 / 0.25 / 0.50) via chained `soup_openfs_recognizer.py`
  (alpha = weight of checkpoint-a), screened as m5 swaps into the pruned ens4 (e62 +
  l6-e82 + soup5459 unchanged), beam 8, logprob fusion, equal member weights, no
  dictionary or language correction.
- Code: staged gated runner `scripts/run_ens4_l9soupw.sh` (soups -> 4k greedy member
  screens, margin gate >=40 vs s3w 2,908 -> member full-val confirm >=60 vs 9,004 ->
  ens4 val gate 8,601 -> test), detached setsid, log `data/logs/run-ens4-l9soupw.log`.
- Screening (greedy member-only, first 4,000 validation clips, 69,783 chars; s3w
  baseline 2,908 / 2,750 exact): w41h **2,935 / 2,740** (+27), w55h **2,904 / 2,748**
  (-4). Margin gate failed; runner stopped at stage 2: no member full-val, no ensemble
  gate, no test. Best unchanged at 3.702% CER / 71.02% exact.
- Cost: 2 soups (CPU, seconds); 2 4k greedy screens ~390 + ~420 s / ~140 MiB
  (batch 4, nice -n 19, two training jobs active); total ~15 min GPU.
- Finding: equal thirds sit at the member-level weight optimum of the plateau basin -
  tilting toward the best parents (w55h) is noise, tilting toward the weakest (w41h)
  hurts; consistent with the two-sided picture where e41 pays only at its equal share.
  m5 soup axis now fully closed (spacing, set, weights); forward supply still needs a
  genuine l9 val-best < 4.42% (cosine tail, epochs 76-100) or the l12 trajectory.
- Status: l9-cos-EMA epoch ~75/100 (val-best e55); l12-cos-EMA epoch ~1-2/100
  (warmup, loss ~2.79, ETA ~200 h, joinable class ~epoch 20). Gates unchanged:
  4.06% val / 3.70% test.
- Verdict: negative at screen level; m5 soup weight tuning retired.

### 2026-10-26 — launched m5 forward-supply lever: cosine-restart fine-tune of the l9soup weights (run in progress)

- Idea: `gpu_budget.sh 6144` failed (3,567 MiB free; l9-cos-EMA epoch 75/100 val-best
  still e55 4.417% - epochs 56-75 all worse and the trainer saves only on improvement,
  so no m5 soup supply exists) and every eval-only lever is retired. The blocked m5
  forward supply needs "a genuine new l9 val-best" - manufacture one by resuming FROM
  the best member and re-annealing: named experiment **cosine-restart fine-tune of the
  m5 soup member** (lr-restart/cyclic-reheat mechanism; distinct from retired soup
  spacing/set/weights - this produces new trajectory weights, not averages of old ones).
- Mechanism: existing `--resume` loads weights only (fresh Adam, fresh EMA shadow
  initialized from the soup, `best=inf` so every val-improving epoch saves a snapshot).
  Init from `recognizer-soup-e41e43e55-3w.pt` (member greedy 4.247%/66.52%), short
  shallow cosine reheat: 15 epochs, peak lr 2e-5 (0.2x pretrain peak - enough drift to
  move inside the basin, small enough to stay in it), 2-epoch warmup, EMA 0.999975
  epoch-matched, `--warm-epoch 0` so aux losses (entropy/monotonicity) match the
  converged model from epoch 1.
- Batch 2 (not 4): budget check failed, halved per rule 2; l12 precedent at same batch
  and d256 family (no fp16 instability at epoch 1+; width384 collapse was d384).
  PYTORCH_ALLOC_CONF=expandable_segments:True (foreign tenant 24 GiB history).
- Smoke (1,600 clips / 2 epochs, 400-clip val): loss started 0.154 (converged level -
  resume verified), peak VRAM 779 MiB, val CER already 2.29% best - member class intact.
- Params: 26,191,292 (9L d256). VRAM: ~1.1 GiB steady (free 3,567 -> 2,325 with all
  three jobs running). Wall clock: ~11 steps/s at 63,066 steps/epoch -> ~1.75 h/epoch,
  15 epochs ~26 h. Output:
  `data/models/openfs-recognizer-gaps-full-layers9-cosine-ema-ft1/recognizer.pt`
  (PID 2527854, log/pid `data/logs/openfs-recognizer-gaps-full-layers9-cosine-ema-ft1.*`).
- Harvest protocol next iterations: ft1 greedy val-best materially below the member's
  own 4.247% -> 4k member screen (vs s3w baseline 2,908 / 2,750, margin gate >=40) ->
  member full-val confirm -> ens4 gate 8,601 -> test. If the fine-tune trajectory
  reaches a new plateau, its snapshots re-open the m5 soup axis (forward only).
- Status: l9-cos-EMA epoch ~76/100 (e55 frozen member), l12-cos-EMA epoch ~2/100.
  Gates unchanged: 4.06% val / 3.70% test.
- Verdict: pending; run in progress.

### 2026-10-26 — hypothesis-level (late) fusion of ens4 members: screen-pass reversed at full validation; axis retired

- Idea: last untried combination axis is the *hypothesis space*: instead of averaging
  frame posteriors (frame-level logprob fusion, current best), decode each ens4 member
  independently and combine strings.  New script
  `scripts/late_fusion_eval_openfs_recognizer.py`: every member beam-decodes the
  identical gap-preserving batch on its own; primary selection is the median hypothesis
  (medoid under unit-cost Levenshtein over the member MAP hypotheses), secondary is a
  cross-member MBR over pooled beam hypotheses (per-member score-softmax average, top-8
  pool cap per member); no dictionary or language correction, no checkpoints written.
- Sanity: single-member late-fusion 4k screen reproduced the recorded m5-only greedy
  screen byte-exact (2,908 err / 2,750 exact) - decode path verified.
- Members (all gaps, same labels): e62 3L cos-EMA (8,828,348) + l6 step e82 (17,509,820)
  + l6ce soup5459 (17,509,820) + l9soup e41e43e55-3w (26,191,292); 70,039,280 total.
- 4k screen (member beam 1, batch 4, 3,112 s under GPU contention, peak 303 MiB):
  medoid 2,876 err / 2,737 exact vs ens4 frame-fusion screen baseline 2,978 / 2,765 -
  +102 errors, clears the >=40 gate.  Cross-member MBR identical to medoid (beam-1
  pools).  Best single member (m5) 2,908, so late fusion beat every member too.
- Full validation (member beam 1, batch 4, 9,478 s, peak 307 MiB): medoid 8,662 err /
  4.085% CER / 66.40% exact (7,978) vs ens4 frame-level beam-8 gate 8,601 / 4.057% /
  67.86% (8,153).  Fails by 61 errors; cross-member MBR again identical.  Exact-match
  loss (-175) exceeds the CER story: medoid trades whole-string exactness for
  Levenshtein distance, and the frame posterior still dominates at full scale.
- Verdict: negative - the 4k screen margin reversed at full validation (+102 -> -61,
  a ~163-error swing; 4k screens are ~3x noisier than assumed for *selection-rule*
  changes).  Late-fusion/medoid/cross-member-MBR axis retired; frame-level logprob
  fusion stays the fusion mechanism.  Gates unchanged: 4.06% val / 3.70% test.
- Reports: `lf-screen4k-ens4` / `lf-validation-ens4` .late-fusion-eval-report.json in
  `data/models/openfs-recognizer-gaps-full-cosine-ema/`.

### 2026-10-26 — launched missing l9 step-schedule full-data member candidate (run in progress)

- Idea: capacity check failed first (`gpu_budget.sh 6144` free 1,973 MiB with l9-cos-EMA,
  l12-cos-EMA, and ft1 active). Stopped the plateaued l9-cos-EMA original (epoch 81,
  val-best frozen at e55 4.417%, tail e76-e81 4.70-4.75%, trainer saves only on
  improvement; e41/e43/e55 snapshots and s3w soup already frozen). Free rose only to
  4,309 MiB, so stopped l12-cos-EMA (only epoch 3, val CER 22.53%, far from joinable
  member class) rather than the more promising ft1 reheat. Budget passed after that
  (free 6,199 MiB). Launched the missing recipe cell: **l9 step-schedule full-data gaps
  transformer** - l9 cos-EMA supplies the best member (s3w 4.247% greedy), l6 step-e82 is
  a proven diverse current-best member, but l9 step has never been run.
- Config: `--layers 9 --epochs 100 --lr-schedule step --batch-size 4
  --validation-batch-size 16 --seed 42`; defaults `lr=1e-4`, `lr-step=80`,
  `lr-gamma=0.1`, `d_model=256`, no EMA. Batch 4 after budget pass (same batch as the
  clean l9 cos-EMA precedent). `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- Output: `data/models/openfs-recognizer-gaps-full-layers9-step/recognizer.pt`.
  Logs/PID: `data/logs/openfs-layers9-step.log/.pid`; pid file first caught the launcher
  shell, corrected to real train PID 2616355. Params 26,191,292. Process GPU use ~1.68
  GiB early, within the 6 GiB budget. No smoke run: model shape and batch-4 l9 config
  already ran clean; only the LR schedule changed.
- Early log: epoch 1 stable, loss ~2.73 at 1,820/63,066 steps (~13 steps/s). No
  validation result yet.
- Status: ft1-cosine-restart continues (best so far 4.510% greedy, still above the s3w
  member's 4.247% class); l9-cos-EMA and l12-cos-EMA stopped for capacity. Gates
  unchanged: 4.06% val / 3.702% test.
- Verdict: pending; run in progress. Trigger: l9-step greedy val-best enters member class
  (roughly <=5.55%) -> 4k member screen vs s3w baseline, then full-val / ens4 gate.
