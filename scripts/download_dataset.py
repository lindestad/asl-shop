"""Download and extract the public Kaggle ASL Alphabet dataset into data/."""

from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet"
ARCHIVE_NAME = "asl-alphabet.zip"
MARKER_NAME = ".asl-alphabet.complete"
DATASET_DIRS = ("asl_alphabet_train", "asl_alphabet_test")


def download(destination: Path) -> None:
    partial = destination.with_suffix(".zip.part")
    request = Request(DATASET_URL, headers={"User-Agent": "asl-shop"})
    print("Downloading ASL Alphabet from Kaggle (about 1 GB)...", flush=True)
    with urlopen(request, timeout=60) as response, partial.open("wb") as output:
        expected_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 32 * 1024 * 1024
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                progress = f" ({downloaded / expected_size:.0%})" if expected_size else ""
                print(f"Downloaded {downloaded / 1024**2:.0f} MiB{progress}", flush=True)
                next_report = downloaded + 32 * 1024 * 1024
        if expected_size and downloaded != expected_size:
            raise OSError("Download was incomplete; rerun the script to retry.")

    # Kaggle can return an HTML error page with an HTTP success status.
    with ZipFile(partial):
        pass
    partial.replace(destination)


def install(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / MARKER_NAME
    if marker.exists() and all((data_dir / name).is_dir() for name in DATASET_DIRS):
        print(f"ASL Alphabet is already installed in {data_dir}")
        return

    archive_path = data_dir / ARCHIVE_NAME
    if not archive_path.exists():
        download(archive_path)

    print(f"Extracting {archive_path}...", flush=True)
    with ZipFile(archive_path) as archive:
        members = archive.infolist()
        roots = {Path(member.filename).parts[0] for member in members if member.filename}
        if not set(DATASET_DIRS).issubset(roots):
            raise BadZipFile("Archive is missing the ASL training or test directory.")
        for member in members:
            target = (data_dir / member.filename).resolve()
            if not target.is_relative_to(data_dir.resolve()):
                raise BadZipFile(f"Unsafe archive path: {member.filename}")
        for index, member in enumerate(members, start=1):
            archive.extract(member, data_dir)
            if index % 10000 == 0:
                print(f"Extracted {index:,}/{len(members):,} entries", flush=True)

    marker.write_text("grassknoted/asl-alphabet\n", encoding="utf-8")
    print(f"ASL Alphabet installed in {data_dir}")


if __name__ == "__main__":
    try:
        install()
    except (OSError, URLError, BadZipFile) as error:
        print(f"Dataset installation failed: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInstallation interrupted; rerun the script to retry.", file=sys.stderr)
        sys.exit(130)
