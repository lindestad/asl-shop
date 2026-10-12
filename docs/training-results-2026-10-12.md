# FSboard experiment — 12 October 2026

The RTX 5080 was sufficient to train the app's neural models and evaluate the
released OpenFS checkpoint. The pretrained OpenFS recognizer is substantially
stronger at complete strings than the small causal model trained here. The app
therefore supports a current-letter head, a provisional causal transcription,
and optional OpenFS analysis when a spelling segment is finished.

## Measured results

| Model | Evaluation | Result |
|---|---|---|
| Legacy-data KNN, matching XY features | 156,179 static test frames | 40.75% agreement |
| Trained handshape MLP, 50,521 parameters | Same static test frames | 83.43% agreement |
| Handshape MLP, score at least 0.8 | Accepted static test frames | 94.56% agreement at 72.13% coverage |
| Trained causal TCN, 835,453 parameters | 12,521 complete test strings | **13.54% character error; 18.11% exact strings** |
| Released OpenFS FSboard checkpoint, about 8.81M parameters | Same 12,521 complete test strings | **6.29% character error; 59.32% exact strings** |

The static frame labels are **model-generated annotations**, not human-labelled
held-letter examples. Both static models exclude J/Z and share the same test
frames and XY normalization. Their training data differs: KNN uses 55,634 rows
from the legacy alphabet CSV; the MLP uses 818,163 public sequence frames.
This comparison measures the combined data/model change, rather than isolating
neural architecture. It does not measure idle-hand rejection; there are no idle
negative frames in this public-data handshape experiment.

Sequence character error rate is Levenshtein insertions, substitutions and
deletions divided by the number of reference characters. Exact accuracy requires
the whole string to match. Test references contain 221,100 characters. OpenFS
made 13,900 character errors and matched 7,428 strings exactly. No dictionary,
LLM correction or beam search was added. References and predictions were
uppercased; internal spaces and punctuation remain part of the score.

The full OpenFS evaluation took 586.97 seconds on CPU with four PyTorch threads,
excluding archive loading/model initialization. Its greedy autoregressive decoder
sees the complete clip. This is a pretrained-checkpoint evaluation, not a model
we trained or a live streaming accuracy result. The app's optional finish adapter
uses that model's own preprocessing separately from the app model's v2 contract.

## Data and training

The official OpenFS landmark archives were joined to original FSboard metadata
from `googleai/fsboard`, cached version 13. Signer identities come from that
metadata, rather than the person IDs inside the landmark dictionaries.

| Split | Signers | Sequence clips | Static samples |
|---|---:|---:|---:|
| Train | 117 | 126,131 | 818,163 |
| Validation | 15 | 12,015 | 99,824 |
| Test | 15 | 12,521 | 156,179 |

There is **zero signer overlap**. The archive has 126,134 training clips; three
over 3,000 frames were excluded. All available validation/test archive clips were
retained. The source video's metadata contains a few more validation/test clips
than the OpenFS archives, so these results apply to the available landmark subset.

The packed app data preserves original detection frame indices and missing-hand
slots. Its timestamps are nominal 30 FPS, not original sensor timestamps. Clean
labels were aligned using both upstream filters; directly indexing them by the
full clip array would assign some labels to the wrong examples. Static supervision
excludes alignment blanks, motion letters and non-letter characters. Sequence CTC
uses human transcripts plus an auxiliary loss on available model-derived frames.

Training used PyTorch 2.11.0+cu128 on the RTX 5080 (16 GB), seed 42, AdamW,
gradient clipping, mixed precision and a cosine learning-rate schedule. Static
augmentation adds small XY rotations/noise. Sequence augmentation changes speed
without reordering frames and adds landmark noise and global wrist-origin jitter.
The sequence model warm-starts its shared encoder from the handshape checkpoint.
The best validation checkpoint is selected before final test evaluation.

The static run completed 50 epochs in 107.76 seconds. The final static evaluation
includes runtime XY clipping to [-4, 4]; its agreement is 83.4267%, versus 83.4459%
before that numerical hardening. The temporal model clamps global trajectories
to [-32, 32] and uses log-domain native CUDA CTC to avoid cuDNN underflow.

The temporal run completed 30 epochs in 2,246.74 seconds (37.45 minutes), including
validation and final test evaluation. Epoch 28 had the best validation character
error, 14.07%. Its held-out test result is 29,941 character errors and 2,267 exact
strings. The small causal model is a useful baseline, but its current accuracy
does not match the released OpenFS recognizer.

## Reproduce

Run these commands from the experimental checkout after following
[the environment and metadata setup](neural-recognition.md). Dataset and model
files are intentionally excluded from Git.

```powershell
.\.build\training-env\Scripts\python.exe scripts/prepare_openfs_training.py --input 'C:\Users\danie\dev\asl-shop\data\external\openfs\extracted\fingerspelling_fsboard_proc_clean' --metadata 'C:\Users\danie\dev\asl-shop\data\external\openfs\metadata' --output data/openfs-packed --trusted-pickle
.\.build\training-env\Scripts\python.exe scripts/train_openfs.py --data data/openfs-packed --task handshape --output data/handshape.pt --epochs 50 --batch-size 1024 --hidden 192 --device cuda --amp
.\.build\training-env\Scripts\python.exe scripts/train_openfs.py --data data/openfs-packed --task sequence --output data/sequence.pt --epochs 30 --batch-size 64 --hidden 192 --device cuda --amp --warmstart-handshape data/handshape.pt
.\.build\training-env\Scripts\python.exe scripts/benchmark_knn_transfer.py --packed data/openfs-packed --report data/knn-transfer-final.json
.\.build\training-env\Scripts\python.exe scripts/evaluate_openfs_reference.py --source 'C:\Users\danie\dev\asl-shop\data\external\openfs\source' --model-dir data/openfs-checkpoints/recognizer_fsboard --input 'C:\Users\danie\dev\asl-shop\data\external\openfs\extracted\fingerspelling_fsboard_raw\fsboard_test.npz' --metadata 'C:\Users\danie\dev\asl-shop\data\external\openfs\metadata' --report data/openfs-reference.report.json --device cpu --batch-size 16 --trusted-source
.\.venv\Scripts\python.exe scripts/benchmark_neural.py --packed data/openfs-packed --frames 100 300 900 --allow-cycle --report data/runtime-latency.json
```

The OpenFS source is the [official repository](https://github.com/AIRC-KETI/OpenFS)
at commit `cd29c853dd880816de1166bab2d9c56ff960d5e1`. The released FSboard
checkpoint archive has Google Drive ID `1ucc7mcoagNygnwhT9PJNxbfev7GY10qe`.
Its source and weights are external, subject to upstream research-use terms.

## Deployment evidence and remaining evaluation

The current-letter and text outputs are independent. Still images do not alter
the live sequence. Finishing or restarting a sequence is explicit. The app
resamples irregular captures onto the model's nominal 30 Hz timeline, preserves
missing-hand masks, and resets unfinished sequences after a long detection gap.

A real worker probe on the existing A1000 alphabet image produced **A**, model
score **0.9725**. This is one smoke example, not a held-letter accuracy estimate.
Repeating that stationary image caused an unguarded OpenFS model to hallucinate
a whole phrase. The app now requires context and movement before finish analysis,
suppresses multi-character text for stationary segments, and rejects implausibly
long decoded text. These heuristic gates do not establish idle false-positive
performance or generalize the public benchmark to webcam conditions.

Runtime latency and final checkpoint hashes are recorded alongside the local
JSON reports. Neural-only replay excludes the MediaPipe detector, camera capture,
JPEG serialization and Electron IPC. A 900-frame stress replay cycles the longest
711-frame public test clip; it is a timing test, not an accuracy example.

Final CPU replay used PyTorch 2.14.1 and four threads:

| Buffered model frames | Per-arrival p50 | Per-arrival p95 | OpenFS Finish, warmed median |
|---|---:|---:|---:|
| 100 | 2.95 ms | 3.68 ms | 17.56 ms |
| 300 | 3.86 ms | 5.42 ms | 78.67 ms |
| 900, cycled stress segment | 6.21 ms | 8.73 ms | 189.19 ms |

At the last 100 arrivals of the 900-frame stress segment, p95 was 9.31 ms. The
first OpenFS Finish took 388.06 ms including lazy source/checkpoint loading. These
costs are measurements on one replay, rather than a hardware-independent bound.
Both neural heads and the optional finish model run on CPU in this experiment.
The real Electron canvas-stream test exceeded 10 processed frames/second using
the actual detector and worker; it is not an accuracy test or a real webcam test.

The final Python suite ran 49 tests: 46 passed, with two legacy image-fixture
checks and the CPU environment's CUDA check skipped. All six packed-data tests
passed separately in the CUDA environment, including real CUDA CTC backward.
Three web tests, two Electron tests, Node protocol checks, build, lint, changed-file
formatting and whitespace checks passed. Packaged neural distribution was not
built or validated.

Machine-readable final metrics and checkpoint SHA256s are in
[experiment-metrics-2026-10-12.json](experiment-metrics-2026-10-12.json). Checkpoints,
histories and complete reports are preserved in this worktree's `data/` and backed
up in `C:\Users\danie\dev\asl-shop\data\models\neural-2026-10-12`. The app detector
is installed locally, so `scripts/start-neural.ps1` can launch the native models.
See [OpenFS refinement setup](neural-recognition.md#optional-openfs-final-text-refinement)
to enable the stronger complete-segment model.

The next accuracy evaluation should use unseen people recording deliberate held
letters, J/Z motion, natural spelling, transitions and unrelated hand activity in
the actual app. Human held-letter labels and idle negatives are essential for
calibrating rejection. For complete strings, OpenFS is the strong available
baseline; improving the causal model requires further temporal architecture/data
work before it should be described as state of the art.
