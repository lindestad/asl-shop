"""Download the demo selection from googleai/fsboard, without the full video dataset.

    uv run python scripts/download_demo_videos.py
"""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import is_zipfile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "data/external/openfs/demo-clips"
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/googleai/fsboard/"
# Same selection as demo_replay.pick_indices, in display order.
CLIPS = (
    "67fc_535_20221021_200720_909133_911200",
    "66d7_497_20220924_080738_502433_508478",
    "08db_193_20220922_170755_510056_530267",
    "9fc2_1_20220920_112246_230533_253956",
    "66d7_391_20220923_130506_637456_641822",
    "08db_766_20220929_223057_417078_423633",
    "a1ba_524_20220929_004033_526083_530544",
    "52ff_716_20221020_213842_205378_208833",
    "515f_80_20220921_132016_111078_115633",
    "9fc2_101_20220920_160722_165950_171378",
)


def credentials():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return "Bearer " + token.strip()
    username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (username and key):
        directory = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
        token_file = directory / "access_token"
        if token_file.is_file() and token_file.read_text().strip():
            return "Bearer " + token_file.read_text().strip()
        config = directory / "kaggle.json"
        if config.is_file():
            try:
                values = json.loads(config.read_text())
                username, key = values.get("username"), values.get("key")
            except (ValueError, OSError):
                raise ValueError("Cannot read Kaggle credentials. " + credential_help()) from None
    if username and key:
        return "Basic " + base64.b64encode(f"{username}:{key}".encode()).decode()
    raise ValueError(credential_help())


def credential_help():
    return (
        "Set up your Kaggle API key, then rerun this command:\n"
        "  1. Open https://www.kaggle.com/settings/api and generate an API token.\n"
        "  2. Set KAGGLE_API_TOKEN in your shell (PowerShell: $env:KAGGLE_API_TOKEN = 'TOKEN'; "
        "bash: export KAGGLE_API_TOKEN='TOKEN').\n"
        "  Or create a Legacy API Key and save kaggle.json to ~/.kaggle/kaggle.json "
        "(Windows: %USERPROFILE%\\.kaggle\\kaggle.json).\n"
        "  On macOS/Linux, run: chmod 600 ~/.kaggle/kaggle.json\n"
        "  KAGGLE_USERNAME + KAGGLE_KEY and KAGGLE_CONFIG_DIR are also supported.\n"
        "  Keep credentials outside the repository."
    )


def valid_video(path):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        header = stream.read(12)
    return len(header) == 12 and header[4:8] == b"ftyp"


def download_clip(name, directory, authorization):
    target = directory / f"{name}.mp4"
    partial = target.with_suffix(".mp4.part")
    unpacked = target.with_suffix(".mp4.unpacking")
    source = f"daun_v3/video_clips/daun_v3-test/{target.name}"
    request = Request(DATASET_URL + quote(source, safe=""),
                      headers={"Authorization": authorization, "User-Agent": "asl-shop"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        # Kaggle may return a single-file ZIP instead of the MP4 itself.
        if is_zipfile(partial):
            with ZipFile(partial) as archive:
                matches = [item for item in archive.infolist() if Path(item.filename).name == target.name]
                if len(matches) != 1:
                    raise ValueError(f"Kaggle archive does not contain {target.name}.")
                with archive.open(matches[0]) as source_file, unpacked.open("wb") as output:
                    shutil.copyfileobj(source_file, output)
            unpacked.replace(partial)
        if not valid_video(partial):
            raise ValueError("Kaggle returned no MP4 video. " + credential_help())
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
        unpacked.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(os.environ.get("ASL_DEMO_VIDEO_DIR", DEFAULT_DIRECTORY)))
    args = parser.parse_args()
    missing = [name for name in CLIPS if not valid_video(args.directory / f"{name}.mp4")]
    if not missing:
        print(f"All {len(CLIPS)} demo videos are already available in {args.directory}.")
        return
    try:
        authorization = credentials()
        args.directory.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(missing, 1):
            print(f"Downloading {index}/{len(missing)}: {name}.mp4", flush=True)
            download_clip(name, args.directory, authorization)
    except HTTPError as error:
        if error.code in (401, 403):
            parser.exit(1, "Kaggle rejected the credentials. " + credential_help() + "\n")
        parser.exit(1, f"Kaggle download failed (HTTP {error.code}). Rerun to retry missing clips.\n")
    except (ValueError, OSError, URLError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Demo videos ready in {args.directory}. Start the app with npm run dev.")


if __name__ == "__main__":
    main()
