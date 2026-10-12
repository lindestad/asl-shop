"""Validate the downloaded MediaPipe ZIP container before promoting its .part file."""
from pathlib import Path
import sys
from zipfile import ZipFile


def promote_model(partial_path):
    partial_path = Path(partial_path)
    if partial_path.suffix != ".part":
        raise ValueError("Expected a .part download; existing models are never replaced directly.")
    with ZipFile(partial_path) as archive:
        if not archive.infolist() or archive.testzip() is not None:
            raise ValueError("Downloaded MediaPipe model is empty or corrupt; rerun setup.")
    partial_path.replace(partial_path.with_suffix(""))


if __name__ == "__main__":
    promote_model(sys.argv[1])
