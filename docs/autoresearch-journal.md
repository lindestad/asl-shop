# Autoresearch journal

Newest entry at the bottom. One block per iteration: idea, config, params, peak
VRAM, wall clock, validation CER/exact, test CER/exact if the run earned a test
pass. Update "Current best" in place when a run beats it.

## Current best

- offline transformer, gap-preserving timeline representation, 8,828,348 params
  (full 126,131-clip train slice, batch 16, step LR schedule; epoch-84
  validation-best checkpoint of the 100-epoch run, decoded with beam search
  width 8, no dictionary or language correction): **5.63% CER**, 61.47% exact
  (12,521 test strings, 21 Oct 2026,
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
