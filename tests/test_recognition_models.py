"""The worker loads both trained models and gets real letters out of them."""

import csv
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recognition_worker as worker

LANDMARKS = ROOT / "data" / "extractedLandmarks.csv"


@unittest.skipUnless(LANDMARKS.exists(), "Needs data/extractedLandmarks.csv")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with LANDMARKS.open(newline="") as file:
            rows = [row for _, row in zip(range(500), csv.reader(file))]
        cls.samples = [(row[0], np.array(row[1:], dtype=float)) for row in rows]

    def assert_recognizes_samples(self, name, minimum_accuracy):
        classifier = worker.get_classifier(name)
        correct = 0
        for label, features in self.samples:
            prediction = worker.predict_letter([{"score": 1.0, "features": features}], classifier)
            correct += prediction is not None and prediction["letter"] == label
        self.assertGreaterEqual(correct / len(self.samples), minimum_accuracy)

    def test_knn_recognizes_samples(self):
        self.assert_recognizes_samples("knn", 0.9)

    def test_cnn_recognizes_samples(self):
        self.assert_recognizes_samples("cnn", 0.8)

    def test_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            worker.get_classifier("svm")

    def test_no_hands_means_no_letter(self):
        self.assertIsNone(worker.predict_letter([], worker.get_classifier("knn")))


if __name__ == "__main__":
    unittest.main()
