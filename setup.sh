#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

download() {
    local url="$1" destination="$2"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "$destination" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --output-document="$destination" "$url"
    else
        echo "Install curl or wget with your system package manager, then rerun ./setup.sh." >&2
        return 1
    fi
}

installer=""
trap 'if [[ -n "$installer" ]]; then rm -f -- "$installer"; fi' EXIT

if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    echo "Installing uv..."
    installer="$(mktemp)"
    download "https://astral.sh/uv/install.sh" "$installer"
    UV_INSTALL_DIR="$HOME/.local/bin" sh "$installer"
    uv_bin="$HOME/.local/bin/uv"
fi
export PATH="$(dirname -- "$uv_bin"):$PATH"

echo "Setting up Python 3.12 and project dependencies..."
"$uv_bin" sync --locked --python 3.12

mkdir -p data
model="data/hand_landmarker.task"
if [[ ! -s "$model" ]]; then
    echo "Downloading the MediaPipe Hand Landmarker model..."
    download \
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
        "$model.part"
    "$uv_bin" run --no-sync python - "$model.part" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile

model = Path(sys.argv[1])
with ZipFile(model) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("Downloaded model is corrupt; rerun ./setup.sh.")
model.replace(model.with_suffix(""))
PY
else
    echo "MediaPipe model is already installed."
fi

"$uv_bin" run --no-sync python scripts/download_dataset.py
echo "Setup complete. Start the webcam demo with: uv run python handTracker/hand_features.py"
