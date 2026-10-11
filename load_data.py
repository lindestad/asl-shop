"""Extract normalized landmarks from the downloaded ASL Alphabet training images."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return result


def extract_image(item):
    # Each process imports one detector and reuses it for all its images.
    import cv2
    from handTracker.hand_features import detector, extractHands

    label, image_path = item
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read {image_path}")
    return label, [hand["features"].tolist() for hand in extractHands(image, detector)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "extracted_landmarks.csv")
    parser.add_argument("--samples-per-class", type=positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=positive_int, default=min(4, os.cpu_count() or 1),
        help="Number of independent extraction processes (default: up to 4 CPU workers).",
    )
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Missing dataset: {args.input}. Run uv run python scripts/download_dataset.py first.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(args.seed)
    images = []
    for class_dir in sorted(args.input.iterdir()):
        if not class_dir.is_dir():
            continue
        selected = sorted(class_dir.glob("*.jpg"))
        if args.samples_per_class is not None:
            selected = randomizer.sample(selected, min(args.samples_per_class, len(selected)))
        images.extend((class_dir.name, image_path) for image_path in selected)

    started = time.monotonic()
    print(f"Extracting {len(images)} images with {args.workers} worker(s).", flush=True)
    pool = (
        ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
        if args.workers > 1 else nullcontext()
    )
    rows = 0
    temporary = None
    try:
        with pool as executor, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=args.output.parent,
            prefix=args.output.name + ".", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            writer = csv.writer(output)
            writer.writerow(["label", *[f"{axis}{i}" for i in range(21) for axis in "xyz"]])
            results = (
                executor.map(extract_image, images, chunksize=16)
                if executor else map(extract_image, images)
            )
            try:
                # map preserves input order, so the seed and training split stay reproducible.
                for index, (label, features) in enumerate(results, 1):
                    for hand in features:
                        writer.writerow([label, *hand])
                    rows += len(features)
                    if index % 1000 == 0 or index == len(images):
                        elapsed = time.monotonic() - started
                        rate = index / elapsed
                        remaining = (len(images) - index) / rate
                        print(
                            f"{index}/{len(images)} images; {rate:.1f}/s; "
                            f"~{remaining / 60:.1f} min remaining", flush=True,
                        )
            except BaseException:
                if executor:
                    executor.shutdown(wait=True, cancel_futures=True)
                raise
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved {rows} landmark rows to {args.output}")


if __name__ == "__main__":
    main()
