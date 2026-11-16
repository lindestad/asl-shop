"""Train the runtime KNN classifier from locally extracted hand landmarks."""

import argparse
from pathlib import Path
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from landmarks import CSV_PATH, FEATURE_COLUMNS, ROOT, load_landmarks

OUTPUT_PATH = ROOT / "model" / "trained_models" / "knn_model.pkl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    # k=28 chosen from cross-validated log loss (see knn-cross-validation.py).
    parser.add_argument("--neighbors", type=int, default=28)
    args = parser.parse_args()

    df = load_landmarks(args.input)
    features, labels = df[FEATURE_COLUMNS].to_numpy(), df["label"]

    # 60% train, 20% validation, 20% test.
    x_train, x_rest, y_train, y_rest = train_test_split(
        features, labels, test_size=0.4, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.5, random_state=42
    )

    classifier = KNeighborsClassifier(n_neighbors=args.neighbors).fit(x_train, y_train)
    print(f"Train: {len(x_train)}; validation: {len(x_val)}; test: {len(x_test)}")
    print(f"Validation accuracy: {accuracy_score(y_val, classifier.predict(x_val)):.2%}")
    print(f"Test accuracy: {accuracy_score(y_test, classifier.predict(x_test)):.2%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        pickle.dump(classifier, output)
    print(f"Saved classifier to {args.output}")


if __name__ == "__main__":
    main()
