"""Set up demo replay in one go: demo videos, FSboard metadata, OpenFS landmarks, packing.

    uv run --with gdown python scripts/setup_demo_replay.py

Needs a Kaggle API key (https://www.kaggle.com/settings/api). A first run downloads
about 6.5 GB and uses about 24 GB of disk, so it can take 15-30 minutes or more.
Finished steps are skipped, so rerun the same command if it stops halfway.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import is_zipfile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import demo_replay, download_demo_videos as videos, download_openfs

OPENFS = ROOT / "data/external/openfs"
ARCHIVE = "fingerspelling_fsboard_proc_clean.zip"
METADATA = ("daun_v3/metadata/daun_v3-train.json", "daun_v3/metadata/daun_v3-validation.json",
            "daun_v3/metadata/daun_v3-test.json", "dmk_v3/metadata/dmk_v3-train.json")
SPLITS = ("train", "validation", "test")
ATTEMPTS = 5
NEEDED_GB = 24


def packed_ready():
    return (demo_replay.packed_split_dir(demo_replay.PACKED) / "test_clips.json").is_file()


def missing_videos():
    return [name for name in videos.CLIPS if not videos.valid_video(demo_replay.CLIP_VIDEOS / f"{name}.mp4")]


def missing_metadata(openfs):
    return [source for source in METADATA if not (openfs / "metadata" / Path(source).name).is_file()]


def fetch_kaggle_file(source, target, authorization):
    """Download one JSON file of googleai/fsboard; Kaggle may wrap it in a single-file ZIP."""
    partial = target.with_name(target.name + ".part")
    unpacked = target.with_name(target.name + ".unpacking")
    request = Request(videos.DATASET_URL + quote(source, safe=""),
                      headers={"Authorization": authorization, "User-Agent": "asl-shop"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if is_zipfile(partial):
            with ZipFile(partial) as archive:
                matches = [item for item in archive.infolist() if Path(item.filename).name == target.name]
                if len(matches) != 1:
                    raise ValueError(f"Kaggle archive does not contain {target.name}.")
                with archive.open(matches[0]) as source_file, unpacked.open("wb") as output:
                    shutil.copyfileobj(source_file, output)
            unpacked.replace(partial)
        with partial.open("rb") as stream:
            if stream.read(64).lstrip()[:1] not in (b"[", b"{"):
                raise ValueError("Kaggle returned no JSON metadata. " + videos.credential_help())
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
        unpacked.unlink(missing_ok=True)


def download_videos(authorization):
    missing = missing_videos()
    if not missing:
        print(f"[1/4] Demo videos: all {len(videos.CLIPS)} already downloaded.", flush=True)
        return
    demo_replay.CLIP_VIDEOS.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(missing, 1):
        print(f"[1/4] Demo videos: downloading {index}/{len(missing)}: {name}.mp4", flush=True)
        videos.download_clip(name, demo_replay.CLIP_VIDEOS, authorization)


def download_metadata(authorization, openfs):
    missing = missing_metadata(openfs)
    if not missing:
        print("[2/4] FSboard metadata: already downloaded.", flush=True)
        return
    (openfs / "metadata").mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(missing, 1):
        name = Path(source).name
        print(f"[2/4] FSboard metadata: downloading {index}/{len(missing)}: {name}", flush=True)
        fetch_kaggle_file(source, openfs / "metadata" / name, authorization)


def download_landmarks(openfs):
    extracted = openfs / "extracted" / "fingerspelling_fsboard_proc_clean"
    if all((extracted / f"fsboard_{split}_proc_clean.npz").is_file() for split in SPLITS):
        print("[3/4] OpenFS landmarks: already downloaded and extracted.", flush=True)
        return extracted
    print("[3/4] OpenFS landmarks: downloading 6.4 GB from Google Drive, then extracting. "
          "This is the slow step.", flush=True)
    openfs.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            download_openfs.fetch_archive(ARCHIVE, openfs, extract=True)
            return extracted
        except Exception as error:  # gdown raises its own types on a dropped connection
            if attempt == ATTEMPTS or isinstance(error, RuntimeError):
                raise
            print(f"Download interrupted ({error}); resuming, attempt {attempt + 1}/{ATTEMPTS}.", flush=True)


def pack(extracted, openfs):
    print("[4/4] Packing landmarks into " + str(demo_replay.PACKED) + " (a few minutes).", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_openfs_training.py"),
                    "--input", str(extracted), "--metadata", str(openfs / "metadata"),
                    "--output", str(demo_replay.PACKED), "--trusted-pickle"], check=True)


def verify():
    listing = subprocess.run([sys.executable, str(ROOT / "scripts/demo_replay.py"), "--list"],
                             check=True, capture_output=True, text=True).stdout
    clips = json.loads(listing)
    with_video = sum(1 for clip in clips if clip["video"])
    print(f"Demo replay is ready: {len(clips)} clips, {with_video} with video. Start the app with npm run dev.")


def main(openfs=OPENFS):
    try:
        packed = packed_ready()
        authorization = None
        if missing_videos() or (not packed and missing_metadata(openfs)):
            authorization = videos.credentials()  # fail here, before the long download
        if not packed:
            free = shutil.disk_usage(ROOT).free / 1e9
            print(f"A first run downloads about 6.5 GB and uses about {NEEDED_GB} GB of disk "
                  f"({free:.0f} GB free). Expect 15-30 minutes or more; rerun to resume if it stops.",
                  flush=True)
        download_videos(authorization)
        if packed:
            print("[2-4/4] Packed landmarks: already present in " + str(demo_replay.PACKED) + ".", flush=True)
        else:
            download_metadata(authorization, openfs)
            pack(download_landmarks(openfs), openfs)
        verify()
    except HTTPError as error:
        if error.code in (401, 403):
            sys.exit("Kaggle rejected the credentials. " + videos.credential_help())
        sys.exit(f"Download failed (HTTP {error.code}). Rerun the same command to resume.")
    except subprocess.CalledProcessError as error:
        sys.exit(f"Step failed: {Path(error.cmd[1]).name} exited with code {error.returncode}.")
    except (ValueError, OSError, URLError, RuntimeError) as error:
        sys.exit(f"{error}\nRerun the same command to resume.")


if __name__ == "__main__":
    main()
