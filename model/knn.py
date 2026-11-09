"""Train the runtime classifier from locally extracted hand landmarks.

Reads a landmarks CSV with a label column followed by 63 coordinates in
x0,y0,z0,... order. A header row is optional: files from the extraction
explorations are headerless and assumed to be in that canonical order.
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "extractedLandmarks.csv"
OUTPUT_PATH = ROOT / "model" / "trained_models" / "knn_model.pkl"
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]


def main():
    with INPUT_PATH.open() as file:
        has_header = file.readline().lower().startswith("label")
    df = (
        pd.read_csv(INPUT_PATH)
        if has_header
        else pd.read_csv(INPUT_PATH, header=None, names=["label", *FEATURE_COLUMNS])
    )
    features, labels = df[FEATURE_COLUMNS].to_numpy(), df["label"].astype(str)

    # 60% train, 20% validation, 20% test.
    x_train, x_rest, y_train, y_rest = train_test_split(
        features, labels, test_size=0.4, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.5, random_state=42
    )

    classifier = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
    print(f"Train: {len(x_train)}; validation: {len(x_val)}; test: {len(x_test)}")
    print(f"Validation accuracy: {accuracy_score(y_val, classifier.predict(x_val)):.2%}")
    print(f"Test accuracy: {accuracy_score(y_test, classifier.predict(x_test)):.2%}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as output:
        pickle.dump(classifier, output)
    print(f"Saved classifier to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
