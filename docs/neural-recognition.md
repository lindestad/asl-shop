# Neural letters and fingerspelling

This branch implements two independent supervised models and the app plumbing to
use them. Checkpoints are local artifacts under `data/`, kept out of Git.
Synthetic tests exercise optimization, evaluation and checkpoint loading; they do
not measure recognition quality. Until trained checkpoints are supplied, the app
continues to use its existing KNN for static letters and explicitly reports that
the text model is unavailable. Neither static classifier emits J/Z.

## Models and input contract

`HandshapeNet` is an MLP with LayerNorm, GELU and dropout, using cross-entropy
supervision. The collection pipeline supports held letters and idle/transition
negatives. The public-data experiment uses model-derived static frame labels
and has no idle negatives; its accuracy measures teacher agreement. Its vocabulary is
24 static letters plus `nothing`. The runtime abstains below a model score of 0.8
and when more than one hand is visible; the score is not calibrated probability.
The separate sequence model has a learned hand-selection gate.

`FingerspellingNet` is a causal temporal CNN with residual, dilated convolutions
and CTC sequence supervision. A shared handshape encoder processes both hands,
and a learned gate combines their embeddings with missing-hand masks. The
temporal receptive field covers 61 resampled model frames at nominal 30 Hz.
CTC blanks and repeated-frame collapse preserve genuine double letters separated
by blanks. Text comes from this model's complete segment output, never from
concatenating the current handshape output. J/Z additionally require local motion
in the preceding second. This gate is a necessary movement check; the learned
sequence model must recognize the actual J/Z paths. It has not been validated on
real J/Z clips yet.

The current v2 contract in `model.neural.CONTRACT` uses:

- Pixel XY hand coordinates relative to the wrist, divided by the wrist-to-middle
  MCP distance, with left-hand X mirrored. Static input has 42 values. This is a
  deliberately different contract from the existing 63-value XYZ CSV.
- Sequence input has fixed Right and Left slots. Each has 42 handshape values,
  the two global pixel wrist coordinates divided by the wrist-to-MCP hand-size
  scale, and a presence mask. One elapsed
  capture-time value completes the 91-value frame. Missing hands have zero slots.
- Global wrist coordinates are measured in hand-size units, rather than fractions
  of image width/height. The same pixel-coordinate contract works for extracted
  FSboard landmarks without guessing their missing original image dimensions.
- Landmark ordering is MediaPipe's 21-joint ordering. Images are unmirrored at
  extraction; mirroring the UI preview does not alter the input.

Checkpoint loading validates task, contract, vocabulary and architecture. An
OpenFS checkpoint **cannot** be loaded into these models: upstream uses its own
normalization, hand/frame identities and Transformer architecture. An optional
external OpenFS adapter refines completed segments. Upstream research source and
weights are separate local dependencies, excluded from the packaged app.

## Environment

From the worktree root, install the project and optional neural dependencies:

```powershell
uv sync --extra neural --python 3.12
uv run --extra neural python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

On Windows, `.\setup.ps1 -Neural` additionally installs desktop dependencies and
the MediaPipe hand detector. `-OpenFS` downloads/reuses ZIPs; `-ExtractOpenFS` also
extracts them. Both accept `-OpenFSDir` pointing to your existing dataset directory.
`-Plan` prints the planned commands without changing the environment. The native
setup requires uv and Node.js 22.12+ with npm already installed.

Use a PyTorch CUDA wheel compatible with the installed NVIDIA driver and GPU.
Check that a real CUDA operation succeeds before training. CPU is supported for
tests and inference. This workstation has separate environments: CPU PyTorch 2.14
in `.venv` for app/runtime tests, and PyTorch 2.11 with CUDA 12.8 in the isolated
`.build/training-env` for RTX 5080 training. Keep setup/sync commands confined to
the runtime environment; training and app interpreters are selected separately.

The isolated Windows CUDA environment used for the reported run can be created
without changing the app environment:

```powershell
uv venv .build/training-env --python 3.12
uv pip install --python .build/training-env/Scripts/python.exe 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .build/training-env/Scripts/python.exe 'numpy==2.5.4' 'pandas==3.0.6' 'scikit-learn==1.9.1' 'kagglehub==1.0.2' 'gdown==6.4.2' 'PyYAML==6.0.3'
.\.build\training-env\Scripts\python.exe -c "import torch; x=torch.ones(4,device='cuda'); print(torch.__version__,torch.cuda.get_device_name(0),float(x.sum()))"
```

Do not run `uv sync` against this isolated environment: the project lockfile
describes the app dependencies and can replace its CUDA wheel.

MediaPipe's model can be supplied from a different checkout without copying it:

```powershell
$env:ASL_MEDIAPIPE_MODEL = 'C:\Users\danie\dev\asl-shop\data\hand_landmarker.task'
```

## Collect app-specific data

Recording is explicit and separate from the app's camera view. It saves landmarks
and monotonic capture timestamps, not video. SPACE begins a clip; SPACE saves it;
Q cancels. Supply persistent pseudonymous signer IDs and distinct session IDs.
Keep all sessions from a given signer in one split.

```powershell
uv run --extra neural python scripts/collect_signs.py --signer-id person01 --session-id day01 --kind held --hand Right --transcript A --split train
uv run --extra neural python scripts/collect_signs.py --signer-id person01 --session-id day01 --kind spelling --transcript "JAZZ" --split train
uv run --extra neural python scripts/collect_signs.py --signer-id person01 --session-id day01 --kind idle --split train
uv run --extra neural python scripts/collect_signs.py --signer-id person01 --session-id day01 --kind transition --split train
```

Held clips should contain only deliberate held letters. Record entering/leaving
the pose as separate transition clips. Idle clips should include visible relaxed
hands and unrelated hand activity, as well as absence. Collect additional signers
for validation and test with those respective `--split` values. J/Z are collected
as spelling clips, even when their transcript is one letter.

Each sample has a UUID, source, label provenance, signer/session IDs, transcript,
clip kind, split, feature-contract version, and original capture timestamps.
Training rejects signers appearing across splits instead of making a random
frame split. Separate held, fingerspelled, idle and transition conditions in the
recording plan; use unseen signers and unseen sessions for the final evaluation.

## Import the OpenFS download

The `*_raw.zip` download is re-extracted **landmark NPZs**, not raw video. Extract
the official ZIPs into a separate local directory, preserving the original files.
The downloaded ZIPs total approximately 12.7 GB compressed and 14.3 GB extracted.
Schema inspection confirmed that the raw validation partition has 12,015 clips
and includes original detection-frame indices. Actual public-data preparation
verified every original identity/transcript join and signer-disjoint partitions.
Synthetic regression fixtures additionally cover the generic importer.

`scripts/import_openfs.py` understands upstream arrays `vidnames`,
`right_hand_poses`, `left_hand_poses`, and `words`, with raw `right_frame_indices`
and `left_frame_indices` arrays. The hand entries are dictionaries
keyed by tracked person ID. Tracked person IDs are **not signer identities**.
Original FSboard metadata was fetched from `googleai/fsboard`:

- `daun_v3/metadata/daun_v3-{train,validation,test}.json`
- `dmk_v3/metadata/dmk_v3-train.json`

Join clip names to verified `signerId` and `fullVideoFilename` fields rather than
guessing signer IDs from filenames or tracked hand IDs. Archive timestamps are
reconstructed from original frame indices with **nominal 30 FPS**, explicitly
marked as reconstructed; they are not measured capture timestamps.

For a fresh installation, fetch only the four metadata JSONs (about 109 MB)
through [KaggleHub's single-file download API](https://github.com/Kaggle/kagglehub#download-dataset):

```powershell
@'
from pathlib import Path
import shutil
import kagglehub
destination = Path("data/external/openfs/metadata")
destination.mkdir(parents=True, exist_ok=True)
files = [f"daun_v3/metadata/daun_v3-{split}.json" for split in ("train", "validation", "test")]
files.append("dmk_v3/metadata/dmk_v3-train.json")
for filename in files:
    target = destination / Path(filename).name
    if not target.exists():
        downloaded = Path(kagglehub.dataset_download("googleai/fsboard", path=filename))
        shutil.copyfile(downloaded, target)
    print(target)
'@ | uv run --with kagglehub python -
```

This command downloads metadata only. It does not request FSboard videos.
Change `destination` to keep metadata alongside your existing external dataset.

The generic per-clip importer accepts a verified sidecar CSV:

```csv
video_id,signer_id,session_id,split,person_id,fps
exact-upstream-vidname,actual-participant-id,actual-recording-session,train,0,30
```

Join the archive's names with original FSboard metadata to obtain real participant
IDs and official splits. Both importers use pixel poses and wrist coordinates in
hand-size units, without guessing image dimensions.
The raw archive's `right_frame_indices` and `left_frame_indices` map each detection to its original
video frame; preserve gaps rather than treating separately compressed streams as
synchronized. Timestamps reconstructed from frame indices and FPS are recorded as
FPS-derived, not actual sensor capture timestamps. The generic CLI shown below
uses embedded indices when present, with optional `right_frame_indices` and
`left_frame_indices` CSV columns as a fallback. Omitted FPS is explicitly recorded
as nominal 30 FPS.

```powershell
uv run --extra neural python scripts/import_openfs.py --input 'C:\Users\danie\dev\asl-shop\data\external\openfs\extracted\fingerspelling_fsboard_raw\TRAIN_FILE.npz' --metadata data/openfs-metadata.csv --output data/openfs-clips --trusted-pickle
```

Object NPZ files require trusted pickle loading; use verified official files.
For an archive that discarded frame indices, an explicit `--assume-contiguous` records
that assumption in provenance. Unequal hand lengths still require original
indices. This assumption cannot reconstruct real detection gaps and is unsuitable
for claiming an exact upstream reproduction or deployment timing accuracy. Video
re-extraction is the stronger path when original timing is unavailable.

For clean archives, additionally supply `--clean-label-map data/openfs-labels.json`,
an explicit mapping from **the upstream checkpoint's** numeric frame IDs to letter
strings. The upstream vocabulary is defined in `conf/decode/fsboard.yaml`.
**Clean labels do not align directly with the full `vidnames` array.** Apply both
upstream selection steps first: retain non-`None` `signing_hand` entries, then
retain examples whose ordered unique `frame_label` IDs match the ordered unique
transcript IDs. Only then index `frame_label_clean`. This was verified on the
validation archive: 3,213 selected examples match the cleaned array length, and
every cleaned sequence matches the signing-hand detection count. Reject any
schema/count mismatch instead of assigning labels to different clips.

Do not guess label IDs. Correct alignment imports contiguous static-letter spans for
handshape supervision, excluding J/Z. Clean frame annotations are model-derived,
not human ground truth, and that provenance is stored. Use human-labelled clips
for handshape evaluation. Raw/clean imports of the same examples should not both
be added to a sequence training manifest, which would duplicate training samples.
The handshape trainer rejects model-derived evaluation labels by default;
`--allow-weak-evaluation` explicitly permits a diagnostic comparison against them.

Large NPZ object members load into RAM. Import into a configurable experimental
output directory only after verifying the mapping; do not overwrite the versioned
CSV or the original archives. `scripts/download_openfs.py` can reuse the downloaded
archives and extract them safely; it does not delete them.

## Train and evaluate

The packed public-data pipeline validates original metadata joins and keeps
whole signers in separate partitions. The verified prepared data contains:

| Partition | Signers | Sequence clips | Static frame samples |
|---|---:|---:|---:|
| Train | 117 | 126,131 | 818,163 |
| Validation | 15 | 12,015 | 99,824 |
| Test | 15 | 12,521 | 156,179 |

There is zero signer overlap. Three training clips exceeding 3,000 frames are
excluded. Static samples use model-derived clean annotations and exclude blank
labels; these partitions have no idle negatives. Human-labelled held letters and
idle activity still need their own webcam evaluation.

Prepare packed memmaps from the clean archive and verified metadata:

```powershell
.\.build\training-env\Scripts\python.exe scripts/prepare_openfs_training.py --input 'C:\Users\danie\dev\asl-shop\data\external\openfs\extracted\fingerspelling_fsboard_proc_clean' --metadata 'C:\Users\danie\dev\asl-shop\data\external\openfs\metadata' --output data/openfs-packed --trusted-pickle
```

Use the isolated CUDA interpreter for actual public-data training:

```powershell
.\.build\training-env\Scripts\python.exe scripts/train_openfs.py --data data/openfs-packed --task handshape --output data/handshape.pt --epochs 50 --hidden 192 --batch-size 1024 --device cuda --amp
.\.build\training-env\Scripts\python.exe scripts/train_openfs.py --data data/openfs-packed --task sequence --output data/sequence.pt --warmstart-handshape data/handshape.pt --epochs 30 --hidden 192 --batch-size 64 --device cuda --amp
.\.build\training-env\Scripts\python.exe scripts/train_openfs.py --data data/openfs-packed --task sequence --output data/sequence.pt --evaluate data/sequence.pt --report data/sequence-test.json --device cuda
```

Checkpoints are chosen on validation and final scores are measured on test.
Training histories and JSON reports are saved alongside checkpoints. Sequence
training can warm-start its shared encoder from the static model and use aligned
weak frame labels as an auxiliary loss; final character error uses the human
transcripts. Static agreement with weak labels must not be presented as accuracy
on human-labelled held-letter signs or idle rejection.

For app-specific collected clips, use the separate manifest trainer below.

Both tasks use the same manifest. The static model ignores spelling clips because
their full transcripts do not identify the current frame's handshape. Training
samples one frame per clip per epoch; validation/test score every eligible frame.
The sequence model uses full clip transcripts, including empty negative clips.
Speed and landmark-noise augmentation preserve temporal order; CTC alignment
capacity is checked, including the extra blanks required for repeated characters.

```powershell
uv run --extra neural python scripts/train_neural.py --manifest data/sessions/manifest.jsonl --task handshape --output data/handshape.pt --epochs 30 --device cuda --amp
uv run --extra neural python scripts/train_neural.py --manifest data/sessions/manifest.jsonl --task sequence --output data/sequence.pt --epochs 50 --device cuda --amp
uv run --extra neural python scripts/train_neural.py --manifest data/sessions/manifest.jsonl --task sequence --output data/sequence.pt --evaluate data/sequence.pt --device cuda
```

The best validation checkpoint is saved. `--evaluate` reads the test split and
does not train. Report handshape accuracy and idle false positives; for sequences,
report character error rate, exact-sequence accuracy and idle false positives.
Character error rate can exceed 100% when insertions dominate. Evaluate app replay
latency and signer/session/domain breakdowns separately. No language correction
is applied, so names and arbitrary strings are measured as recognized.

## Run the app

The worker looks for `data/handshape.pt` and `data/sequence.pt`. Override them with
`ASL_HANDSHAPE_MODEL` and `ASL_SEQUENCE_MODEL`. Restart the app after changing
checkpoints. `ASL_KNN_MODEL` can point to a trusted existing fallback model.

For native Windows launch, choose the app/runtime interpreter separately from
the CUDA training interpreter:

```powershell
.\scripts\start-neural.ps1 -Python '.venv\Scripts\python.exe' -HandshapeModel 'data\handshape.pt' -SequenceModel 'data\sequence.pt' -LandmarkerModel 'C:\Users\danie\dev\asl-shop\data\hand_landmarker.task'
```

Checkpoint paths may point to another experiment directory without copying the
files. Add `-Plan` to inspect the launch environment without starting the app.
The script validates paths, sets `ASL_SHOP_PYTHON` and the model paths for this
launch, and restores prior environment values when the app exits. Install
MediaPipe, OpenCV and PyTorch in the selected runtime environment. These launch
options do not change or synchronize `.build/training-env`.

Every live camera start gets a new sequence ID and capture clock. Still images
do not affect sequence state. Text remains provisional until **Fullfør tekst**;
**Ny sekvens** clears it. Acquisition gaps over one second reset unfinished text.
Segments are limited to 900 capture/model frames or 30 seconds. Finishing a full
segment commits its existing buffer; new frames are not appended after the limit.

`model/timeline.py` resamples capture timestamps onto the training model's 30 Hz
grid. Linear interpolation uses only hands present in both adjacent observations.
Presence transitions and acquisition intervals over 250 ms yield explicitly
missing slots, rather than invented trajectories. Captures are kept separately;
the protocol reports `frames` and `modelFrames`. The model stays causal, while
interpolation waits for the next arrived observation: at 10 Hz this adds up to
one 100 ms acquisition interval. Tests compare 10/15 Hz and irregular acquisition
with regular linear motion, and cover dropout, gaps, fast input and resets.

The CPU runtime recomputes the bounded model sequence on arrival. Default PyTorch
thread count is four; set `ASL_TORCH_THREADS=1` to compare. Reproduce public-landmark
latency measurements with:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_neural.py --packed data/openfs-packed --threads 1 4 --frames 100 300 900 --allow-cycle --report data/runtime-latency.json
```

The longest available public test clip has 711 frames, so 900-frame measurements
explicitly cycle its prefix for a capacity stress test. Timing includes feature
conversion, both neural heads, resampling and decoding, but excludes MediaPipe,
camera acquisition and IPC. These are runtime costs, not end-to-end camera rates.

## Optional OpenFS final-text refinement

The official checkpoint's full offline benchmark is a separate experiment from
our causal model and app gates. Enable the research backend only with an explicit
trusted [OpenFS checkout](https://github.com/AIRC-KETI/OpenFS) and checkpoint/config:

For a fresh source directory, clone the official release and pin the tested
revision. Keep this research dependency in the shared external-data location:

```powershell
git clone https://github.com/AIRC-KETI/OpenFS.git 'C:\Users\danie\dev\asl-shop\data\external\openfs\source'
git -C 'C:\Users\danie\dev\asl-shop\data\external\openfs\source' checkout cd29c853dd880816de1166bab2d9c56ff960d5e1
@'
from pathlib import Path
import gdown
from scripts.download_openfs import archive_ready, extract_archive
archive = Path("data/openfs-recognizer-fsboard.zip")
if not archive_ready(archive):
    gdown.download(id="1ucc7mcoagNygnwhT9PJNxbfev7GY10qe", output=str(archive), resume=True)
extract_archive(archive, Path("data/openfs-checkpoints"))
'@ | uv run --extra neural --with gdown python -
```

The checkpoint ZIP is approximately 33 MB. This workstation already has these
files; existing valid ZIPs and extracted files are reused by the Python snippet.
Launch with the explicit source and checkpoint paths:

```powershell
.\scripts\start-neural.ps1 -Python '.venv\Scripts\python.exe' -LandmarkerModel 'C:\Users\danie\dev\asl-shop\data\hand_landmarker.task' -OpenFSSource 'C:\Users\danie\dev\asl-shop\data\external\openfs\source' -OpenFSModelDir 'data\openfs-checkpoints\recognizer_fsboard'
```

Alternatively set `ASL_OPENFS_SOURCE` and `ASL_OPENFS_MODEL_DIR`. The tested source
commit is `cd29c853dd880816de1166bab2d9c56ff960d5e1`. The model directory must contain
`best_model.pth` and `.hydra/config.yaml`. Upstream code executes from that trusted
checkout; its research-use license and checkpoint terms apply. No upstream source
or weights are copied into the distributable. `PyYAML` is included in the neural
extra for reading the saved configuration.

The worker retains actual captured pixel XY independently of the causal feature
vectors. OpenFS uses its released per-axis box normalization, hand identities,
compressed per-hand detection ordering and greedy autoregressive decoding.
It loads and runs **only on Finish**, after at least six usable captures spanning
500 ms and observed hand motion. The UI distinguishes configured/ready/error
states and the final `OpenFS` method from live causal text. It does not run at
camera frame rate.

Short or stationary held-letter segments retain the independent current-letter
output and get an insufficient-context/held-letter message. Stationary segments
cannot display multi-character text from either decoder. Refined length is bounded
by 20 characters per observed second plus two, with a maximum of 128. Invalid or
overlong output and adapter failures keep the gated causal text with an explicit
refinement error. Empty input does not invoke the autoregressive decoder. J/Z need
captured movement evidence; these gates need calibration on human-labelled clips.

A real three-frame A image probe initially caused the unconstrained upstream
decoder to invent a long phrase. The eligibility gate now prevents that call and
preserves the correct current A prediction. Offline FSboard accuracy does not
establish rejection of idle hands or unrelated activity. App motion/context gates
change outputs and are evaluated separately from the raw offline benchmark.

Packaging includes neural checkpoints when present and installs the neural extra.
Packaging a CUDA runtime can make the bundle substantially larger; CPU-only
distribution and actual packaged neural inference still need verification.

```powershell
uv run --extra neural python -m unittest discover -s tests -p 'test_neural.py'
node --test tests/recognition-service.test.mjs
npm run build
```

The public-data training and pretrained comparison have been implemented and
measured. Remaining accuracy work is human-labelled webcam held/idle/transition
evaluation, camera-domain transfer, false-positive measurement, actual acquisition
latency, and calibration of confidence, movement and context thresholds.
