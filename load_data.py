"""Extract normalized landmarks from the downloaded ASL Alphabet training images."""

import argparse
import csv
from pathlib import Path
import random

import cv2
from handTracker.hand_features import detector, extractHands

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Use a positive number of samples.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "extracted_landmarks.csv")
    parser.add_argument("--samples-per-class", type=positive_int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Missing dataset: {args.input}. Run uv run python scripts/download_dataset.py first.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(args.seed)
    rows = 0
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["label", *[f"{axis}{i}" for i in range(21) for axis in "xyz"]])
        for class_dir in sorted(args.input.iterdir()):
            if not class_dir.is_dir():
                continue
            images = sorted(class_dir.glob("*.jpg"))
            if args.samples_per_class is not None:
                images = randomizer.sample(images, min(args.samples_per_class, len(images)))
            class_rows = 0
            for image_path in images:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not read {image_path}")
                for hand in extractHands(image, detector):
                    writer.writerow([class_dir.name, *hand["features"].tolist()])
                    class_rows += 1
            rows += class_rows
            print(f"{class_dir.name}: {class_rows} hands from {len(images)} images", flush=True)
    print(f"Saved {rows} landmark rows to {args.output}")


if __name__ == "__main__":
    main()
