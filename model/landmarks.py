"""Shared loader for the landmark CSV behind the KNN, SVM and CNN letter models.

The CSV is a label column followed by 63 coordinates in x0,y0,z0,... order.
A header row is optional: load_data.py writes one, older extracts have none.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "extractedLandmarks.csv"
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
COLUMNS = ["label", *FEATURE_COLUMNS]
FORMAT_ERROR = "Expected a label column followed by 63 coordinates in x0,y0,z0,... order"


def load_landmarks(path=CSV_PATH):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Missing landmarks: {path}. Run load_data.py to extract them.")
    with path.open(encoding="utf-8") as file:
        has_header = file.readline().lower().startswith("label")
    df = pd.read_csv(path, header=0 if has_header else None)
    if has_header and list(df.columns) != COLUMNS or df.shape[1] != len(COLUMNS):
        raise SystemExit(f"{FORMAT_ERROR}: {path}")
    df.columns = COLUMNS
    df["label"] = df["label"].astype(str)
    return df
