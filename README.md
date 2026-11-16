# ASL Shop

A desktop app that recognises American Sign Language fingerspelling from a webcam.
Everything runs locally: Electron shows the camera, and a Python process finds the
hands with MediaPipe and classifies them.

## Features

- **Current letter** from the live camera, with a choice of three models:
  - **Neural** (default): a small network trained on FSboard.
  - **KNN** and **CNN**: trained on the Mendeley ASL Alphabet images.
- **Spelled text**: a sequence model writes out words as you sign, including the
  motion letters J and Z, which the single-frame models cannot recognise.
- **Finish correction**: when a word ends, a four-model ensemble re-reads the whole
  segment and corrects the text.
- **Scratchpad** that collects finished words for editing and copying.
- **Demo replay**: plays FSboard clips through the same pipeline, without a camera.

The trained models ship with the repository, so no dataset is needed to run the app.

## Try it

Requirements: macOS, Linux or Windows, and a webcam.

**macOS / Linux** — installs uv, Node.js and all dependencies if they are missing:

```sh
./setup.sh
source data/setup-env.sh
npm run dev
```

**Windows** — install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and [Node.js 22.12+](https://nodejs.org/en/download) first:

```powershell
.\setup.ps1
npm run dev
```

Press **Start camera**, hold up a letter, and pick a model from the dropdown.

MediaPipe uses the GPU when it can and falls back to the CPU; set
`ASL_MEDIAPIPE_DELEGATE=cpu` to skip the GPU attempt.

## Optional data

None of this is needed to run the app. All data is stored in the Git-ignored `data/` folder.

| Purpose | Command | Size |
|---|---|---|
| Retrain KNN/SVM/CNN, run the desktop tests | `sh setup/data.sh` (Windows: `.\setup\data.ps1`) | 900 MB |
| Demo replay videos (needs a Kaggle API key) | `uv run python scripts/download_demo_videos.py` | ten clips |
| Retrain the neural models | `./setup.sh --extract-openfs` (Windows: `.\setup.ps1 -ExtractOpenFS`) | 12.7 GB |

## Demo replay

Demo replay plays FSboard test clips through the live pipeline, so the app can be
tried without a camera. It needs the packed FSboard landmarks:

```sh
./setup.sh --extract-openfs           # Windows: .\setup.ps1 -ExtractOpenFS
# fetch the four FSboard metadata JSONs into data/external/openfs/metadata,
# see "Import the OpenFS download" in docs/neural-recognition.md
uv run python scripts/prepare_openfs_training.py \
  --input data/external/openfs/extracted/fingerspelling_fsboard_proc_clean \
  --metadata data/external/openfs/metadata \
  --output data/openfs-packed --trusted-pickle
uv run python scripts/demo_replay.py --list    # check that the clips are found
```

The video footage is optional. Without it, replay sends the reconstructed landmarks
straight to the recogniser, skipping MediaPipe. To show the clips as video, set
`KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`)
and run `uv run python scripts/download_demo_videos.py`.

Start the app with `npm run dev`. Once the landmarks are found, a **Dataset replay**
card appears below the camera: pick a clip and press **Play clip**. Clips with video are
marked "footage". `ASL_PACKED_DIR` and `ASL_DEMO_VIDEO_DIR` point to the landmarks
and the clips when they are stored somewhere else.

## Retrain the letter models

```sh
sh setup/data.sh                      # images to data/Train and data/Test
uv run python load_data.py            # landmarks to data/extractedLandmarks.csv
uv run python model/knn.py            # or model/svm.py, model/cnn-landmarks.py
```

`load_data.py` and `model/knn.py` accept `--input` and `--output`, so an experiment
can leave the shipped model untouched. The accuracy they print is measured on images
from the same dataset and does not show how well a model handles new people or lighting.

Training and evaluation of the neural models is covered in the
[neural guide](docs/neural-recognition.md) and the
[measured results](docs/training-results-2026-10-12.md).

## Project layout

| Path | Contents |
|---|---|
| `frontend/` | React interface: camera view, model picker, scratchpad |
| `electron/` | Window, camera permission, and the bridge to the Python process |
| `scripts/recognition_worker.py` | Hand detection and classification for each frame |
| `handTracker/` | MediaPipe helpers and landmark normalisation |
| `model/` | Model definitions, the live runtime, and KNN/SVM/CNN training scripts |
| `model/trained_models/`, `data/*.pt`, `data/models/` | Shipped model weights |
| `scripts/` | Dataset preparation, training and evaluation tools |
| `docs/` | [Pipeline walkthrough](docs/pipeline.md), training guide and results |

## Develop

```sh
npm run lint
npm run format:check
npm run build
npm test                                                    # Playwright, web and desktop
uv run python -m unittest discover -s tests -p "test_*.py"
npm run package                                             # unpacked app in release/
```

`npm run dist` builds an installer for the current platform, with Python and all
models bundled.

## Licences and safety

- The neural weights are trained on FSboard through [OpenFS](https://github.com/AIRC-KETI/OpenFS)
  and are for research and non-commercial use only.
- The Mendeley [ASL Alphabet dataset](https://data.mendeley.com/datasets/jdyksv2jhh/1) is CC BY 4.0.
- The KNN and CNN models are pickle files; only load ones you trained or trust.
