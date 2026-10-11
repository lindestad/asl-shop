"""Train the runtime classifier from locally extracted hand landmarks."""

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "extracted_landmarks.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "knn.pkl")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Missing landmarks: {args.input}. Run uv run python load_data.py first.")

    df = pd.read_csv(args.input)
    if list(df.columns) != ["label", *FEATURE_COLUMNS]:
        parser.error("Expected a label column followed by 63 coordinates in x0,y0,z0,... order.")
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if df["label"].isna().any() or not np.isfinite(features).all():
        parser.error("Labels and coordinates must not contain missing or non-finite values.")
    if len(df) < 8:
        parser.error("At least eight landmark rows are required for training and evaluation.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, df["label"].astype(str), test_size=0.3, random_state=42
    )
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print(f"Training rows: {len(x_train)}; held-out rows: {len(x_test)}")
    print(f"Held-out accuracy (k=5): {accuracy_score(y_test, prediction):.2%}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        pickle.dump(classifier, output)
    print(f"Saved classifier to {args.output}")


if __name__ == "__main__":
    main()
