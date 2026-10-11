"""Regression checks for the CSV-to-runtime-model contract."""

import csv
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "model" / "knn.py"
COLUMNS = ["label", *[f"{axis}{i}" for i in range(21) for axis in "xyz"]]


class TrainingTests(unittest.TestCase):
    def test_trains_a_runtime_model_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "landmarks.csv"
            output = Path(directory) / "models" / "knn.pkl"
            with source.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(COLUMNS)
                for label, coordinate in (("A", 0.0), ("B", 1.0)):
                    for _ in range(20):
                        writer.writerow([label, *[coordinate] * 63])
            result = subprocess.run(
                [sys.executable, str(TRAINER), "--input", str(source), "--output", str(output)],
                cwd=directory, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Training rows: 28; held-out rows: 12", result.stdout)
            with output.open("rb") as file:
                classifier = pickle.load(file)
            self.assertEqual(classifier.n_features_in_, 63)
            self.assertEqual(classifier.predict([[0.0] * 63, [1.0] * 63]).tolist(), ["A", "B"])

    def test_rejects_reordered_coordinates_without_writing_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "landmarks.csv"
            output = Path(directory) / "knn.pkl"
            columns = COLUMNS.copy()
            columns[1], columns[2] = columns[2], columns[1]
            with source.open("w", newline="") as file:
                csv.writer(file).writerow(columns)
            result = subprocess.run(
                [sys.executable, str(TRAINER), "--input", str(source), "--output", str(output)],
                cwd=directory, text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("63 coordinates", result.stderr)
            self.assertFalse(output.exists())

    def test_default_input_is_anchored_to_the_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            # A same-named file in cwd must never override the repo's training input.
            source = Path(directory) / "data" / "extracted_landmarks.csv"
            source.parent.mkdir()
            source.write_text("not the dataset", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(TRAINER), "--output", str(Path(directory) / "knn.pkl")],
                cwd=directory, text=True, capture_output=True,
            )
            if (ROOT / "data" / "extracted_landmarks.csv").exists():
                self.assertEqual(result.returncode, 0, result.stderr)
            else:
                self.assertIn(str(ROOT / "data" / "extracted_landmarks.csv"), result.stderr)


if __name__ == "__main__":
    unittest.main()
