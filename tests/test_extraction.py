"""Real-detector checks for serial and multiprocessing extraction."""

import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "load_data.py"
SAMPLE = ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train" / "A" / "A1000.jpg"


@unittest.skipUnless(
    SAMPLE.exists() and (ROOT / "data" / "hand_landmarker.task").exists(),
    "Download the ASL dataset and MediaPipe model to test real extraction.",
)
class ExtractionTests(unittest.TestCase):
    def run_extractor(self, directory, source, output, workers):
        return subprocess.run(
            [sys.executable, str(EXTRACTOR), "--input", str(source), "--output", str(output),
             "--workers", str(workers), "--samples-per-class", "16", "--seed", "42"],
            cwd=directory, text=True, capture_output=True, timeout=60,
        )

    def test_parallel_output_matches_serial_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "images"
            for label in ("A", "B"):
                (source / label).mkdir(parents=True)
                for index in range(17):
                    shutil.copyfile(SAMPLE, source / label / f"{index}.jpg")
            outputs = [Path(directory) / f"landmarks-{workers}.csv" for workers in (1, 2)]
            for workers, output in zip((1, 2), outputs):
                result = self.run_extractor(directory, source, output, workers)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            with outputs[1].open(newline="") as file:
                rows = list(csv.reader(file))
            self.assertEqual(len(rows[0]), 64)
            self.assertEqual([row[0] for row in rows[1:]], ["A"] * 16 + ["B"] * 16)

    def test_failed_worker_preserves_existing_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "images" / "A"
            source.mkdir(parents=True)
            (source / "broken.jpg").write_bytes(b"not an image")
            output = Path(directory) / "landmarks.csv"
            output.write_bytes(b"previous extraction")
            result = self.run_extractor(directory, source.parent, output, 2)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Could not read", result.stderr)
            self.assertEqual(output.read_bytes(), b"previous extraction")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
