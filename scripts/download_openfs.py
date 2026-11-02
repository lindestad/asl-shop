"""Download/resume the official OpenFS FSboard landmark and cleaned-annotation ZIPs.

Existing valid archives are reused. Extraction is explicit and leaves ZIPs intact.
"""
import argparse
from pathlib import Path
import shutil
from zipfile import ZipFile, BadZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "fingerspelling_fsboard_raw.zip": "1M6jOtBfsoyGdhP4--8qimGA2BVqhwtTa",
    "fingerspelling_fsboard_proc_clean.zip": "1dhcrbZeb3-5IshIkzatVmNO3MOTVFjJW",
}


def archive_ready(path):
    try:
        with ZipFile(path) as archive:
            return bool(archive.infolist())
    except (FileNotFoundError, BadZipFile):
        return False


def extract_archive(source, destination):
    """Write each member atomically, validate CRC while reading, refuse path escapes."""
    destination = Path(destination).resolve()
    with ZipFile(source) as archive:
        entries = [(entry, (destination / entry.filename).resolve()) for entry in archive.infolist()]
        for entry, target in entries:
            if not target.is_relative_to(destination):
                raise ValueError(f"ZIP member leaves the extraction directory: {entry.filename}")
        for entry, target in entries:
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if target.stat().st_size != entry.file_size:
                    raise ValueError(f"Existing extracted file has a different size: {target}; choose another output directory.")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extract-part")
            with archive.open(entry) as input_file, temporary.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=4 * 1024 * 1024)
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/external/openfs")
    parser.add_argument("--extract", action="store_true", help="Extract into DIRECTORY/extracted, preserving ZIPs")
    parser.add_argument("--verify", action="store_true", help="Read every ZIP member to validate CRC; can take minutes")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for filename, file_id in ARCHIVES.items():
        archive_path = args.directory / filename
        if archive_ready(archive_path):
            print(f"Using existing archive: {archive_path}", flush=True)
        else:
            try:
                import gdown
            except ImportError as error:
                raise RuntimeError("Run: uv run --with gdown python scripts/download_openfs.py") from error
            gdown.download(id=file_id, output=str(archive_path), resume=True, quiet=False)
            if not archive_ready(archive_path):
                raise ValueError(f"Download is incomplete or not a ZIP: {archive_path}; rerun to resume.")
        if args.verify:
            with ZipFile(archive_path) as archive:
                failed = archive.testzip()
                if failed:
                    raise ValueError(f"Corrupt ZIP member: {filename}/{failed}")
        if args.extract:
            extract_archive(archive_path, args.directory / "extracted")
    print("OpenFS archives are ready. Signer metadata and preprocessing are described in docs/neural-recognition.md.")


if __name__ == "__main__":
    main()
