"""Train a five-neighbor ASL letter classifier from the ASL Alphabet images.

The saved NPZ contains only normalized landmark features and labels. The desktop
worker uses the same feature extractor and performs nearest-neighbor inference
without loading a Python pickle.
"""

import argparse
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from handTracker.hand_features import detector, extractHands  # noqa: E402

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NEIGHBORS = 5
FEATURE_SCHEMA = "wrist_relative_mirrored_scaled_63_v1"


def features_from_image(filename: Path):
    frame = cv2.imread(str(filename))
    if frame is None:
        return None
    hands = extractHands(frame, detector)
    if not hands:
        return None
    return max(hands, key=lambda hand: hand["score"])["features"].astype(np.float32)


def collect_training_samples(directory: Path, samples_per_class: int, seed: int):
    randomizer = random.Random(seed)
    features = []
    labels = []
    for letter in LETTERS:
        files = sorted((directory / letter).glob("*.jpg"))
        if not files:
            raise FileNotFoundError(f"No training images found for {letter} in {directory}.")
        chosen = randomizer.sample(files, min(samples_per_class, len(files)))
        detected = 0
        for filename in chosen:
            values = features_from_image(filename)
            if values is None:
                continue
            features.append(values)
            labels.append(letter)
            detected += 1
        print(f"{letter}: {detected}/{len(chosen)} hands found", flush=True)
        if detected < NEIGHBORS:
            raise RuntimeError(f"Too few hands detected for {letter}.")
    return np.stack(features), np.asarray(labels, dtype="U1")


def evaluate(classifier: KNeighborsClassifier, directory: Path):
    expected = []
    predicted = []
    missing = []
    for letter in LETTERS:
        filename = directory / f"{letter}_test.jpg"
        if not filename.exists():
            continue
        values = features_from_image(filename)
        if values is None:
            missing.append(letter)
            continue
        expected.append(letter)
        predicted.append(classifier.predict(values.reshape(1, -1))[0])
    if expected:
        print(
            f"Separate test images: {accuracy_score(expected, predicted):.1%} "
            f"({sum(a == b for a, b in zip(expected, predicted))}/{len(expected)} detected)."
        )
        mistakes = [f"{a}→{b}" for a, b in zip(expected, predicted) if a != b]
        if mistakes:
            print("Misclassified:", ", ".join(mistakes))
    if missing:
        print("No hand detected in test images:", ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "data/asl_alphabet_train/asl_alphabet_train",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=ROOT / "data/asl_alphabet_test/asl_alphabet_test",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "model/asl_knn.npz")
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_class < NEIGHBORS:
        parser.error(f"--samples-per-class must be at least {NEIGHBORS}.")

    features, labels = collect_training_samples(args.train_dir, args.samples_per_class, args.seed)
    classifier = KNeighborsClassifier(n_neighbors=NEIGHBORS)
    classifier.fit(features, labels)
    evaluate(classifier, args.test_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        labels=labels,
        schema=np.asarray(FEATURE_SCHEMA),
        neighbors=np.asarray(NEIGHBORS),
    )
    print(f"Saved {len(labels)} examples to {args.output}.")


if __name__ == "__main__":
    main()
