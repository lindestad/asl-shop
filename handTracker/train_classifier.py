"""
Tren og sammenlign classifiers på landmark datasettet.

Leser data/landmarks.npz, evaluerer flere modeller,
skriver ut nøyaktighet, klasserapport og forvekslingsmatrise for den beste,
og lagrer den beste modellen (trent på alle data) til data/asl_classifier.joblib
som en ordbok {"model": ..., "labels": [...]}.

Kjør fra repo root:
    uv run python -m handTracker.train_classifier
    uv run python -m handTracker.train_classifier --models knn rf     # utvalg
    uv run python -m handTracker.train_classifier --split random

Splitting:
    block   (standard) siste 20 % av bildenumrene per bokstav er test. Bildene i
            Kaggle-settet er fortløpende bilder av samme hånd, så en tilfeldig
            splitt gir nesten-duplikater i testsettet og for høy nøyaktighet.
    random  vanlig stratifisert tilfeldig splitt, for sammenligning.
"""

import argparse
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "landmarks.npz"
MODEL_PATH = ROOT / "data" / "asl_classifier.joblib"
MATRIX_PATH = ROOT / "data" / "confusion_matrix.csv"


def make_models(seed):
    # Funksjonene er allerede normalisert, men SVM og MLP
    # trenger likevel StandardScaler for å fungere godt.
    return {
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        #"svm": make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=seed)),
        "svm": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=10), ensemble=False)),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=seed),
        ),
    }


def test_mask(y, idx, mode, test_size, seed):
    """Boolean mask som er True for testeksempler."""
    mask = np.zeros(len(y), dtype=bool)
    if mode == "random":
        _, test_rows = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
        mask[test_rows] = True
        return mask
    for letter in np.unique(y):
        rows = np.flatnonzero(y == letter)
        rows = rows[np.argsort(idx[rows], kind="stable")]
        n_test = max(1, int(round(len(rows) * test_size)))
        mask[rows[-n_test:]] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    parser.add_argument("--models", nargs="+", choices=list(make_models(0)), default=list(make_models(0)))
    parser.add_argument("--split", choices=["block", "random"], default="block")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Fant ikke {args.data}. Kjør build_dataset.py først.")
    data = np.load(args.data)
    X, letters, idx = data["X"], data["y"], data["idx"]
    # Heltallsetiketter (MLP med early stopping takler ikke tekstetiketter)
    labels, y = np.unique(letters, return_inverse=True)
    print(f"{len(X)} eksempler, {X.shape[1]} features, {len(labels)} bokstaver")

    test = test_mask(y, idx, args.split, args.test_size, args.seed)
    X_train, y_train, X_test, y_test = X[~test], y[~test], X[test], y[test]
    print(f"Splitt: {args.split} -> {len(X_train)} trening / {len(X_test)} test\n")

    models = make_models(args.seed)
    results = {}
    for name in args.models:
        model = models[name]
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start
        start = time.time()
        pred = model.predict(X_test)
        predict_time = (time.time() - start) / len(X_test) * 1000
        results[name] = (accuracy_score(y_test, pred), pred)
        print(f"{name:4s} nøyaktighet {results[name][0]:.4f}   trening {fit_time:6.1f}s   "
              f"prediksjon {predict_time:.3f} ms/eksempel", flush=True)

    best = max(results, key=lambda name: results[name][0])
    print(f"\nBeste modell: {best} ({results[best][0]:.4f})\n")
    best_pred = results[best][1]
    print(classification_report(y_test, best_pred, target_names=labels, zero_division=0))

    matrix = pd.DataFrame(
        confusion_matrix(y_test, best_pred, labels=range(len(labels))), index=labels, columns=labels
    )
    matrix.to_csv(MATRIX_PATH)
    off_diagonal = [
        (int(matrix.loc[a, b]), a, b) for a in labels for b in labels if a != b and matrix.loc[a, b] > 0
    ]
    print("Vanligste forvekslinger (sann -> predikert):")
    for count, true, predicted in sorted(off_diagonal, reverse=True)[:10]:
        print(f"  {true} -> {predicted}: {count}")
    print(f"Forvekslingsmatrise lagret til {MATRIX_PATH}")

    # Til slutt: tren den beste modellen på alle data før lagring
    final = make_models(args.seed)[best]
    final.fit(X, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # labels[i] er bokstaven for klasse i, brukes ved prediksjon:
    #   bundle = joblib.load(path); proba = bundle["model"].predict_proba([features])[0]
    #   letter = bundle["labels"][proba.argmax()]; confidence = proba.max()
    joblib.dump({"model": final, "labels": labels}, args.out)
    print(f"Modell ({best}, trent på alle data) lagret til {args.out}")


if __name__ == "__main__":
    main()