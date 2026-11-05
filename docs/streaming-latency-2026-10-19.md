# Streaming student measurement — 19 October 2026

Two questions were on the table: whether distilling an offline teacher moves CER on the
streaming path at fixed latency, and what a longer causal receptive field buys per millisecond
of added lag. Both were answered on a V100-SXM3-32GB with the runtime's own causal TCN
(`model.neural.FingerspellingNet`, `asl-shop-xy-wrist-v2`), so the numbers describe a model the
app can actually run.

Distillation made accuracy worse here. On an identical 12,000-clip training slice, the same
architecture at the same 61-frame receptive field scored **23.19% CER without teacher
posteriors and 26.19% with them**, 3.0 points worse, and it was already worse on validation
(22.65% against 24.83%). Both arms share teacher-derived per-frame labels, so the KL term is
what separates them. The soft labels also cost 71% more wall clock per epoch (39.3 s against
22.9 s) for a model of identical size.

Longer receptive field bought nothing measurable. CER moved 26.19%, 26.43% and 26.55% across
61, 201 and 601 buffered frames, slightly the wrong way, while arrival cost rose from 7.12 ms
to 10.15 ms to 12.57 ms per arrival and the history a decision lags live text rose from 2.0 s
to 6.7 s to 20.0 s on the app's 30 Hz model grid.

## Measured rows

Machine-readable copy: `docs/streaming-latency-metrics-2026-10-19.json`.

| Row | Parameters | Buffered model frames | Test CER | Exact strings | Del/Sub/Ins | Validation CER (best epoch) | Seconds/epoch | Peak VRAM (MiB) | Arrival p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (shipped checkpoint) | 835,453 | 61 | **13.54%** | 18.11% | 16,803 / 6,553 / 6,586 | n/a | n/a | n/a | 7.10 / 8.73 |
| rf61, distilled | 835,453 | 61 | **26.19%** | 2.35% | 21,086 / 15,962 / 20,868 | 24.83% (15) | 39.3 | 1,227 | 7.12 / 9.31 |
| rf61, no distillation | 835,453 | 61 | **23.19%** | 3.55% | 23,934 / 13,210 / 14,135 | 22.65% (19) | 22.9 | 1,224 | 7.06 / 8.71 |
| rf201, distilled | 1,205,245 | 201 | **26.43%** | 2.78% | 17,340 / 15,813 / 25,278 | 24.65% (16) | 40.4 | 1,476 | 10.15 / 12.39 |
| rf601, distilled | 1,575,037 | 601 | **26.55%** | 2.63% | 18,497 / 16,068 / 24,143 | 24.50% (15) | 40.1 | 1,731 | 12.57 / 14.99 |

Test evaluation is the FSboard packed test split for every row: 12,521 clips, 221,100
reference characters, greedy CTC on the gap-preserving timeline, no dictionary and no language
correction. Baseline is the released app checkpoint from `docs/training-results-2026-10-12.md`
(`data/models/neural-2026-10-12/sequence.pt`), re-scored through the same metric path as the
students. Its 10-point advantage over the slice rows is mostly data size: it was trained on all
126,131 training clips, the students on 12,000. Treat it as context, not as a matched
comparison.

## Latency accounting

The model is causal, so it adds no look-ahead delay. Latency comes from the history a decision
at frame t needs, which the app has to hold in its segment buffer (cap 900 model frames).

| Buffered model frames | History at 15 Hz capture | History on the app's 30 Hz model grid | Arrival p50 / p95 |
|---:|---:|---:|---:|
| 61 | 4.00 s | 2.00 s | 7.06 to 7.12 / 8.71 to 9.31 ms |
| 201 | 13.33 s | 6.67 s | 10.15 / 12.39 ms |
| 601 | 40.00 s | 20.00 s | 12.57 / 14.99 ms |

Arrival columns are measured CPU wall clock per arrival through the live runtime
(`scripts/benchmark_neural.py --sequence <student.pt>`, four PyTorch threads, machine otherwise
idle), and are the 300-frame row: the best of repeated runs per checkpoint. Every benchmark JSON
also carries a 900-frame row built by cycling the source clip (baseline 9.54 ms p50 / 12.05 ms
p95, rf61-distill 9.66 / 12.28 ms), which is close enough to the 300-frame row that the
receptive-field ordering is unchanged. The first baseline benchmark of the day, 7.86 ms p50
against 107.9 ms p95, was taken while other work was on the machine and is superseded by the
idle numbers above.

Per added millisecond of arrival p50, going from 61 to 201 frames costs 3.0 ms and 4.7 s more
buffered history for 0.24 CER points in the wrong direction; 201 to 601 frames costs another
2.4 ms and 13.3 s for 0.13 more points. Within this architecture family the receptive field is
not the lever.

## Setup

- Runs used `--limit-clips 12000 --validation-limit 4000 --epochs 20 --patience 5 --seed 20261019`,
  batch 64, AdamW, fp16 autocast, augmentation on. The 61-frame configuration reproduces the
  shipped model's parameter count exactly (835,453), so the distilled / not-distilled pair is a
  fixed-latency comparison of one architecture.
- The paired arms really are paired: `frame_labels.clip_order` is identical in both reports, so
  both saw the same 12,000 clips in the same order.
- Teacher for the distilled arms: released OpenFS FSboard recogniser
  `data/openfs-checkpoints/recognizer_fsboard`, sha256
  `0e66c64baa9c4bb1e1299aeac2a4c4668204674677931398640caa4baaffd5d1`, 8,815,933 parameters, run
  with `--teacher-input-representation compressed`, so 42 channels at arange positions. The
  released frame classifier is `frame_classifier_fsboard/best_model.pth`, sha256
  `d0e46287d95f01429e39f3ae17998e0b063a388e6c68d167e52d31c4d04e650d`, 81,981 parameters, 61
  classes. Posteriors were cached in `data/models/streaming-student-rf61-distill/teacher-posteriors.npz`;
  the rf601 run read that cache instead of re-running the teacher.
- KL weight 0.5, temperature 2.0, restricted to frames where the teacher holds a letter
  (`--distillation-blank` was off). Auxiliary per-frame cross-entropy weight 1.0 was on in every
  arm, including the one without KL.
- Wall clock: 459 to 809 s of training per run, plus roughly four minutes of validation and final
  test evaluation. A distilled run with cached posteriors took about 16 to 17 minutes end to end;
  the run that also computed the posterior cache took about 40 minutes.

### Where the raw artifacts live

`/data/*` is gitignored in this repo, so only the merged metrics file is committed. The rest is on
the VPS under the same worktree and is what the table above was assembled from:

- per-run reports: `data/streaming-student-{baseline,rf61-distill,rf61-nodelearn,rf200-distill,rf600-distill}.report.json`, each carrying the teacher paths, sha256 values, label coverage and per-epoch losses;
- CPU benchmark JSONs: `data/streaming-latency-{baseline,rf61-distill,rf61-nodelearn,rf201-distill,rf601-distill}.benchmark.json`;
- per-epoch logs: `logs/{runA,runB,runC,runD,baseline-eval,bench-*}.log`;
- checkpoints and the posterior cache: `/home/coder/asl-data/models/streaming-student-{rf61-distill,rf61-nodelearn,rf200-distill,rf600-distill}/`.

### Licence separation

Every run report carries this line, and each student checkpoint payload records the teacher
paths and sha256 values with `research_only=true`:

> OpenFS Non-Commercial Research License: the released recogniser, the released frame classifier
> and any label or posterior derived from them are research-only and never enter app code; a
> student trained with that signal is a research-only derivative work

Both 61-frame arms are research-only derivatives. The distilled arm takes teacher posteriors,
and the not-distilled arm still trains on refined per-frame labels that come from the released
frame classifier, which is why `research_only=true` appears in its payload too. Students start
from random weights, train only on our own packed FSboard timeline and never load an upstream
tensor; the teacher is used offline as a source of labels and posteriors.

## Gaps-teacher probe

The plan was to distil from our own gaps-timeline recogniser
(`data/models/openfs-recognizer-gaps/recognizer.pt`, 7.08% validation CER in its own report),
whose output already sits on the student's gap-preserving rows. Measured through this harness it
produced unusable posteriors: argmax-decoding its frame posteriors over 200 training clips gave
95.6% CER, with the posteriors overwhelmingly blank. The released compressed teacher scored
15.0% CER on the same clips through the same decode, so the harness and the index mapping are
not the problem; that recogniser's preprocessing does not match the packed rows being fed to it.

The gaps pairing stays wired and unit-tested (`--teacher-input-representation gaps
--teacher-channels 91 --teacher-positions timeline`, with batch slots padded to `PAD_SEG` so the
teacher's own mask covers them), but a distillation run through it should not be believed until
its input preprocessing is checked. The probes are on the VPS at
`data/streaming-student-probe/probe.py` and `probe-released.py`.

## Harness fixes found by running it

`scripts/train_streaming_student.py` had never been used with our own checkpoints on a real
slice. Each of these shipped with a test; the suite went from 157 to 165 tests. Four further
guards sit on the published numbers in `docs/streaming-latency-metrics-2026-10-19.json`.

- `b75de4d` the app-model guard keyed on the payload contract, which refused our own recogniser
  checkpoints as teachers; it now keys on `task == 'sequence'`. Same commit added
  `resolve_teacher_positions()`, because `--teacher-positions auto` could be carried into
  `teacher_tokens` and silently select the wrong position index, and threaded `PAD_SEG` into the
  padded teacher batch, since `model/offline_recognizer.Recognizer` masks only `segments ==
  PAD_SEG` and was being fed zero-valued segments.
- `56f85ac` with a teacher given as a file path plus `--openfs-source`, the frame-classifier
  constructor was never imported, so the run died on an unbound local.
- `171ec03` teacher batches stayed on CPU while the teacher network was on CUDA.
- `24fc10a` the posterior cache wrote ragged records with `np.array`, which raised on
  construction; per-cell assignment fixed it.
- `01ca948` the per-step training log never accumulated the KL term, so the logged KL averaged a
  partial sum.
- `379c2bb` evaluate-only mode left the checkpoint on CPU while clips were moved to `--device`.
- `5ee8583` the worst one: the cache writer sorted clip names but wrote records in insertion
  order, so every cache hit glued one clip's posteriors onto another clip's rows. It showed up
  first as an `IndexError` on the rf200 run; before that it was silently mis-pairing supervision
  on any run that hit the cache.
- `47a9168` the merged latency table only had CER; it now carries letter accuracy, deletion,
  substitution and insertion rates, and the buffered-frame cost.

## Full sweep

The measured slice is cheap enough that the full grid is a few hours, not a day. Measured cost
at 12,000 clips: about 17 minutes per run, roughly 2 hours for six configurations, with the
first distilled run paying an extra 20 minutes for the posterior cache. The commands below are
the sweep as scripted in `scripts/train_streaming_student.py`, expanded to all six arms.

```sh
# fixed latency: one receptive field, distilled against not distilled
.venv/bin/python scripts/train_streaming_student.py --data data/openfs-packed --refined-labels data/openfs-refined --teacher data/openfs-checkpoints/recognizer_fsboard --trusted-teacher --teacher-input-representation compressed --frame-classifier data/openfs-checkpoints/frame_classifier_fsboard --openfs-source data/external/openfs/source --teacher-cache data/models/streaming-student-rf61-distill/teacher-posteriors.npz --receptive-frames 61 --limit-clips 12000 --validation-limit 4000 --epochs 20 --patience 5 --seed 20261019 --output data/models/streaming-student-rf61-distill/student-rf61.pt --report data/streaming-student-rf61-distill.report.json
.venv/bin/python scripts/train_streaming_student.py --data data/openfs-packed --refined-labels data/openfs-refined --no-distill --receptive-frames 61 --limit-clips 12000 --validation-limit 4000 --epochs 20 --patience 5 --seed 20261019 --output data/models/streaming-student-rf61-nodelearn/student-rf61.pt --report data/streaming-student-rf61-nodelearn.report.json

# receptive field, distilled
.venv/bin/python scripts/train_streaming_student.py --data data/openfs-packed --refined-labels data/openfs-refined --teacher data/openfs-checkpoints/recognizer_fsboard --trusted-teacher --teacher-input-representation compressed --frame-classifier data/openfs-checkpoints/frame_classifier_fsboard --openfs-source data/external/openfs/source --teacher-cache data/models/streaming-student-rf61-distill/teacher-posteriors.npz --receptive-frames 200 600 --limit-clips 12000 --validation-limit 4000 --epochs 20 --patience 5 --seed 20261019 --output data/models/streaming-student-sweep-distill/student.pt --report data/streaming-student-sweep-distill.report.json

# receptive field, not distilled
.venv/bin/python scripts/train_streaming_student.py --data data/openfs-packed --refined-labels data/openfs-refined --no-distill --receptive-frames 200 600 --limit-clips 12000 --validation-limit 4000 --epochs 20 --patience 5 --seed 20261019 --output data/models/streaming-student-sweep-nodelearn/student.pt --report data/streaming-student-sweep-nodelearn.report.json
```

`--receptive-frames 200 600` trains both values in one process, so the second and third commands
each cover two configurations. `--teacher-cache` is only reused when the dataset, teacher
weights and clip subset match what the cache recorded, so keep the same `--limit-clips` across
the distilled arms if you want the posterior pass done once.

Then price CPU cost per checkpoint and merge one table. Benchmarking is CPU-only and separate
from training:

```sh
.venv/bin/python scripts/benchmark_neural.py --packed data/openfs-packed --sequence data/models/streaming-student-rf61-distill/student-rf61.pt --threads 4 --frames 300 900 --report data/streaming-latency-rf61-distill.benchmark.json
.venv/bin/python scripts/train_streaming_student.py --summarize baseline:data/streaming-student-baseline.report.json:data/streaming-latency-baseline.benchmark.json --summarize rf61-distill:data/streaming-student-rf61-distill.report.json:data/streaming-latency-rf61-distill.benchmark.json --report data/streaming-latency.report.json
```

Widening the slice is the other knob. At 20,000 clips and 30 epochs expect about 40 minutes per
distilled run and 2.5 to 3 hours for the six-arm grid.

## Still open

1. Why distillation hurt. The KL term was used as configured (weight 0.5, temperature 2.0,
   letter-bearing frames only). Untested knobs: `--distillation-weight`,
   `--distillation-blank`, and `--no-frame-labels` to separate the effect of teacher frame labels
   from the effect of soft labels. One plausible mechanism worth checking first: the compressed
   teacher emits one token per signing-hand detection, so a held letter becomes a run of tokens
   that maps onto packed rows including re-hit rows, and the causal student may be fitting a
   target it cannot represent.
2. The missing control is a baseline trained on the same 12,000-clip slice with the shipped
   recipe. That separates data size from the distillation and frame-label recipe, which the
   current table cannot do.
3. Refined per-frame labels cover 24,065 of the 126,131 training clips; the other 102,066 fall
   back to the pack's own per-frame targets, so the per-frame loss is mixed supervision.
4. Nothing in `model/` or the app enforces `research_only`; the flag is recorded in the payload
   and ignored by loaders. A guard belongs in the loader before any of these checkpoints could
   ship.
5. Whether chunked autoregressive decoding of the offline OpenFS path can be made cheap enough
   to be the streaming answer at all. This study only priced the causal student.
