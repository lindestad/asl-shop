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
| Demo replay without a camera (needs a Kaggle API key) | see [Demo replay](#demo-replay-no-camera) | 6.5 GB download, 24 GB on disk |
| Retrain the neural models | `./setup.sh --extract-openfs` (Windows: `.\setup.ps1 -ExtractOpenFS`) | 12.7 GB |

## Demo replay (no camera)

Demo replay plays ten FSboard clips through the live pipeline. It needs the clip
videos **and** the packed FSboard landmarks; the "Dataset replay" card stays hidden
until both are in place. One script fetches and prepares everything.

**1. Get a Kaggle API key** (free, needed for the videos and the signer metadata):

1. Create an account or sign in at [kaggle.com](https://www.kaggle.com).
2. Open [kaggle.com/settings/api](https://www.kaggle.com/settings/api) and generate an API token.
3. Set it in the shell you will run the script from:

   ```sh
   export KAGGLE_API_TOKEN='your-token'          # macOS / Linux
   ```

   ```powershell
   $env:KAGGLE_API_TOKEN = 'your-token'          # Windows PowerShell
   ```

   A legacy `kaggle.json` saved to `~/.kaggle/kaggle.json` (Windows:
   `%USERPROFILE%\.kaggle\kaggle.json`) also works; on macOS/Linux run
   `chmod 600 ~/.kaggle/kaggle.json`. Keep the key outside the repository.

**2. Run the setup script, then start the app:**

```sh
uv run --with gdown python scripts/setup_demo_replay.py
npm run dev
```

**This takes a while.** A first run downloads about 6.5 GB and uses about 24 GB of
disk; expect 15–30 minutes or more depending on your connection. If it stops
halfway (the large download can drop), run the same command again: finished steps
are skipped and the download resumes.

The script checks the Kaggle key first, then:

1. downloads the ten demo videos from Kaggle to `data/external/openfs/demo-clips/`;
2. downloads the FSboard signer metadata from Kaggle to `data/external/openfs/metadata/` (about 100 MB);
3. downloads and extracts the OpenFS landmark archive from Google Drive to `data/external/openfs/` (6.4 GB, the slow step);
4. packs the landmarks into `data/openfs-packed/` and checks that all ten clips are listed.

Afterwards only `data/openfs-packed/` and `demo-clips/` are needed. Deleting
`data/external/openfs/fingerspelling_fsboard_proc_clean.zip` and
`data/external/openfs/extracted/` frees about 13 GB.

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

These commands are only tested and working on Linux. On macOS and Windows, run the
app with `npm run dev` instead (see [Try it](#try-it)); it works just as well there.

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
