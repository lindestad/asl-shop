"""Dataset replay fidelity: exported landmarks must reproduce the packed model frames,
and the worker landmark path must run the full live pipeline to a corrected final text."""
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKED = Path(os.environ.get("ASL_PACKED_DIR", ROOT / "data" / "openfs-packed"))
ENSEMBLE = ROOT / "data" / "models" / "ensemble-2026-10-25"


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (PACKED / "test_clips.json").is_file():
            raise unittest.SkipTest("Packed OpenFS dataset is required for replay tests")
        import numpy as np

        cls.clips = json.loads((PACKED / "test_clips.json").read_text(encoding="utf-8"))
        cls.x = np.load(PACKED / "test_frames.npy", mmap_mode="r")

    def test_reconstructed_landmarks_reproduce_packed_features(self):
        from types import SimpleNamespace

        import numpy as np

        from model.neural import frame_features
        from scripts.demo_replay import clip_frames

        for index in (0, self.clips.index(min(self.clips, key=lambda c: c["end"] - c["start"]))):
            frames = clip_frames(self.clips[index], self.x)
            rows = np.asarray(self.x[self.clips[index]["start"]:self.clips[index]["end"]],
                              dtype=np.float32)
            self.assertEqual(len(frames), len(rows))
            for i, frame in enumerate(frames):
                hands = [{"hand": hand["hand"],
                          "landmarks": [SimpleNamespace(x=x, y=y) for x, y in hand["points"]]}
                         for hand in frame["hands"]]
                rebuilt = frame_features(hands, 1000, 1000, 0 if i == 0 else 1000 / 30)
                self.assertLess(float(np.max(np.abs(rebuilt - rows[i]))), 1e-4)

    def test_worker_landmark_path_finishes_with_corrected_text(self):
        if not any(ENSEMBLE.glob("*.pt")):
            raise unittest.SkipTest("Ensemble checkpoints are required for the corrected-text test")
        previous = os.environ.get("ASL_ENSEMBLE_MODEL_DIR")
        os.environ["ASL_ENSEMBLE_MODEL_DIR"] = str(ENSEMBLE)
        self.addCleanup(
            lambda: os.environ.pop("ASL_ENSEMBLE_MODEL_DIR", None)
            if previous is None
            else os.environ.__setitem__("ASL_ENSEMBLE_MODEL_DIR", previous)
        )
        from scripts.demo_replay import clip_frames
        import scripts.recognition_worker as worker

        row = next(c for c in self.clips if "-" in c["text"] and "+61" in c["text"])
        frames = clip_frames(row, self.x)
        session = "unittest-replay"
        last = None
        for i, frame in enumerate(frames):
            last = worker.analyze_landmarks(frame["hands"], session, frame["t"],
                                            finish=i == len(frames) - 1)
        sequence = last["fingerspelling"]
        self.assertEqual(sequence["status"], "finished")
        self.assertEqual(sequence["method"], "ensemble")
        self.assertEqual(sequence["text"], row["text"])
        self.assertEqual(sequence["ensemble"]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
