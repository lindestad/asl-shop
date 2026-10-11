"""
Bygg et landmark datasett fra Kaggle ASL Alphabet bildene.

Kjører MediaPipe på hvert treningsbilde, tar ut 63-verdiers featurevektor og lagrer alt i data/landmarks.npz:

Forutsetninger:
- uv run python scripts/download_dataset.py
- data/hand_landmarker.task

Kjør:
- uv run python -m handTracker.build_dataset
- uv run python -m handTracker.build_dataset --limit 200   # rask test

"""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

try:
    from handTracker.hand_features import create_detector, extractHands
except ModuleNotFoundError:  # kjørt som `python handTracker/build_dataset.py`
    from hand_features import create_detector, extractHands

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
OUT_PATH = ROOT / "data" / "landmarks.npz"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_train_dir():
    """Kaggle-zippen har ofte en ekstra mappe (asl_alphabet_train/asl_alphabet_train)."""
    outer = ROOT / "data" / "asl_alphabet_train"
    for candidate in (TRAIN_DIR, outer):
        if candidate.is_dir() and any(p.is_dir() and len(p.name) == 1 for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        f"Fant ikke bokstavmappene under {outer}. Kjør scripts/download_dataset.py først."
    )


def image_number(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def list_images(letter_dir, limit):
    files = sorted(
        (p for p in letter_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=image_number,
    )
    if limit and len(files) > limit:
        # jevnt fordelt utvalg, slik at hele nummerområdet er med
        keep = np.linspace(0, len(files) - 1, limit).astype(int)
        files = [files[i] for i in keep]
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="maks bilder per bokstav (0 = alle)")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="BOKSTAV",
                        help="bokstaver som skal utelates, f.eks. --exclude J Z")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    train_dir = find_train_dir()
    excluded = {letter.upper() for letter in args.exclude}
    letter_dirs = sorted(
        p for p in train_dir.iterdir()
        if p.is_dir() and len(p.name) == 1 and p.name.isalpha() and p.name.upper() not in excluded
    )
    if not letter_dirs:
        sys.exit(f"Ingen bokstavmapper funnet i {train_dir}")

    detector = create_detector(num_hands=1)
    features, labels, numbers = [], [], []
    handedness = Counter()
    start = time.time()

    print(f"Leser bilder fra {train_dir}")
    for letter_dir in letter_dirs:
        letter = letter_dir.name.upper()
        files = list_images(letter_dir, args.limit)
        found = 0
        for path in files:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            hands = extractHands(frame, detector)
            if not hands:
                continue  # hopper over hvis den ikke finner hånd
            features.append(hands[0]["features"])
            labels.append(letter)
            numbers.append(image_number(path))
            handedness[hands[0]["hand"]] += 1
            found += 1
        print(f"{letter}: {found}/{len(files)} bilder med hånd ({found / max(len(files), 1):.0%})"
              f"  [{time.time() - start:.0f}s]", flush=True)

    if not features:
        sys.exit("Ingen hender funnet. Sjekk at data/hand_landmarker.task finnes.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    idx = np.asarray(numbers, dtype=np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, idx=idx)

    print(f"\nLagret {len(X)} eksempler ({X.shape[1]} features) til {args.out}")
    print(f"Sidighet funnet av MediaPipe: {dict(handedness)}")


if __name__ == "__main__":
    main()