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

setup_tmp="$(mktemp -d)"
trap 'rm -rf -- "$setup_tmp"' EXIT

if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    echo "Installing uv..."
    installer="$setup_tmp/uv-install.sh"
    download "https://astral.sh/uv/install.sh" "$installer"
    UV_INSTALL_DIR="$HOME/.local/bin" sh "$installer"
    uv_bin="$HOME/.local/bin/uv"
fi
uv_dir="$(dirname -- "$uv_bin")"
export PATH="$uv_dir:$PATH"

echo "Setting up Python 3.12 and project dependencies..."
"$uv_bin" sync --locked --python 3.12

has_node() {
    command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 &&
        node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 12) ? 0 : 1)'
}

if ! has_node; then
    if command -v fnm >/dev/null 2>&1; then
        fnm_bin="$(command -v fnm)"
    elif [[ -x "$HOME/.local/bin/fnm" ]]; then
        fnm_bin="$HOME/.local/bin/fnm"
    else
        case "$(uname -s):$(uname -m)" in
            Linux:x86_64) fnm_archive="fnm-linux" ;;
            Linux:aarch64|Linux:arm64) fnm_archive="fnm-arm64" ;;
            Darwin:x86_64|Darwin:arm64) fnm_archive="fnm-macos" ;;
            *) echo "Automatic Node installation supports macOS and Linux on x64 or arm64." >&2; exit 1 ;;
        esac
        echo "Installing fnm..."
        download "https://github.com/Schniz/fnm/releases/latest/download/$fnm_archive.zip" "$setup_tmp/fnm.zip"
        fnm_bin="$HOME/.local/bin/fnm"
        "$uv_bin" run --no-sync python - "$setup_tmp/fnm.zip" "$fnm_bin" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile

destination = Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(sys.argv[1]) as archive:
    binary = archive.read("fnm")
partial = destination.with_suffix(".part")
partial.write_bytes(binary)
partial.chmod(0o755)
partial.replace(destination)
PY
    fi
    echo "Installing Node.js 24 and npm..."
    eval "$("$fnm_bin" env --shell bash)"
    "$fnm_bin" install 24
    "$fnm_bin" use 24
    if ! has_node; then
        echo "Node.js 22.12+ and npm are required; Node installation failed." >&2
        exit 1
    fi
else
    echo "Using Node.js $(node --version) and npm $(npm --version)."
fi

echo "Installing frontend and desktop dependencies..."
npm ci --include=dev
npx --no-install install-electron
npx --no-install playwright install chromium

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

echo "Preparing the packaging backend..."
npm run build:backend

# An executable script cannot update its parent shell's PATH. Save the actual
# runtime paths, including fnm's persistent Node directory, for this terminal.
node_dir="$(node -p 'require("node:path").dirname(process.execPath)')"
printf "export PATH=%q:%q:\$PATH\n" "$node_dir" "$uv_dir" > data/setup-env.sh
echo "Setup complete. Start the app with:"
echo "source data/setup-env.sh"
echo "npm run dev"
