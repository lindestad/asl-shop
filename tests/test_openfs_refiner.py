"""Finish-only refinement, honest fallback, and external-checkpoint smoke coverage."""
import os
import base64
import json
import subprocess
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
except ImportError:
    torch = None


class FakeRefiner:
    enabled = True
    def __init__(self, text="BEST", fail=False):
        self.text, self.fail, self.calls = text, fail, 0
    def readiness(self):
        return {"enabled": True, "available": True, "status": "ready"}
    def refine(self, right, left):
        self.calls += 1
        if self.fail:
            raise ValueError("fixture refinement failed")
        return self.text


def hand(offset=0):
    return {"hand": "Right", "landmarks": [SimpleNamespace(x=.2 + i * .002 + offset,
                                                          y=.3 + i * .001) for i in range(21)]}


@unittest.skipIf(torch is None, "Neural extra required")
class RefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def runtime(self, directory, refiner):
        from model.live_neural import NeuralRuntime
        from model.neural import VOCAB
        class AModel:
            def __call__(self, frames):
                logits = torch.zeros((*frames.shape[:2], len(VOCAB)))
                logits[..., VOCAB.index("A")] = 10
                return logits
        runtime = NeuralRuntime(directory)
        runtime.loaded = True
        runtime.models["sequence"] = AModel()
        runtime.refiner = refiner
        return runtime

    def test_finish_refines_once_keeps_raw_pixels_and_stays_sticky(self):
        with tempfile.TemporaryDirectory() as directory:
            refiner = FakeRefiner()
            runtime = self.runtime(directory, refiner)
            for i in range(6):
                result = runtime.analyze([hand(i * .02)], 640, 480, video=True, session_id="one", timestamp_ms=i * 100)
            self.assertEqual(refiner.calls, 0)
            self.assertEqual(result["fingerspelling"]["text"], "A")
            np.testing.assert_allclose(runtime.raw_hands["Right"][0][0], [128, 144])
            result = runtime.analyze([hand(.12)], 640, 480, video=True, session_id="one", timestamp_ms=510, finish=True)
            self.assertEqual(refiner.calls, 1)
            self.assertEqual(result["fingerspelling"]["text"], "BEST")
            self.assertEqual(result["fingerspelling"]["method"], "openfs")
            self.assertFalse(result["fingerspelling"]["provisional"])
            runtime.analyze([hand()], 640, 480)  # Still images do not touch the segment.
            runtime.analyze([hand()], 640, 480, video=True, session_id="one", timestamp_ms=3000)
            self.assertEqual(refiner.calls, 1)
            result = runtime.analyze([hand()], 640, 480, video=True, session_id="two", timestamp_ms=0)
            self.assertEqual(result["fingerspelling"]["method"], "causal-tcn")
            self.assertEqual(len(runtime.raw_hands["Right"]), 1)

    def test_failed_refinement_keeps_causal_text_with_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, FakeRefiner(fail=True))
            for i in range(6):
                result = runtime.analyze([hand(i * .02)], 640, 480, video=True, session_id="one",
                                         timestamp_ms=i * 100, finish=i == 5)
            self.assertEqual(result["fingerspelling"]["text"], "A")
            self.assertEqual(result["fingerspelling"]["method"], "causal-tcn")
            self.assertIn("fixture refinement failed", result["fingerspelling"]["refinementError"])

    def test_final_jz_requires_captured_motion_and_empty_input_never_hallucinates(self):
        with tempfile.TemporaryDirectory() as directory:
            for moving in (False, True):
                refiner = FakeRefiner("JAZZ")
                runtime = self.runtime(directory, refiner)
                for i in range(6):
                    result = runtime.analyze([hand(.1 * i if moving else 0)], 640, 480, video=True,
                                             session_id="one", timestamp_ms=i * 100, finish=i == 5)
                self.assertEqual(result["fingerspelling"]["text"], "JAZZ" if moving else "A")
            refiner = FakeRefiner("PHANTOM")
            runtime = self.runtime(directory, refiner)
            result = runtime.analyze([], 640, 480, video=True, session_id="empty", timestamp_ms=0, finish=True)
            self.assertEqual(result["fingerspelling"]["text"], "")
            self.assertEqual(refiner.calls, 0)

    def test_finished_limit_refines_existing_buffer_without_append(self):
        with tempfile.TemporaryDirectory() as directory:
            refiner = FakeRefiner("NEW")
            runtime = self.runtime(directory, refiner)
            for i in range(6):
                runtime.analyze([hand(i * .02)], 640, 480, video=True, session_id="one", timestamp_ms=i * 100)
            runtime.model_frames = [runtime.model_frames[0]] * 900
            runtime.model_timestamps = np.arange(900).tolist()
            captured = len(runtime.frames)
            result = runtime.analyze([hand()], 640, 480, video=True, session_id="one", timestamp_ms=2000, finish=True)
            self.assertEqual(result["fingerspelling"]["status"], "finished")
            self.assertEqual(result["fingerspelling"]["modelFrames"], 900)
            self.assertEqual(result["fingerspelling"]["frames"], captured)
            self.assertEqual(refiner.calls, 1)

    def test_short_stationary_and_implausible_outputs_cannot_create_words(self):
        with tempfile.TemporaryDirectory() as directory:
            refiner = FakeRefiner("SANTA CLAUS GOT STUCK")
            runtime = self.runtime(directory, refiner)
            for i in range(3):
                result = runtime.analyze([hand()], 640, 480, video=True, session_id="short",
                                         timestamp_ms=i * 100, finish=i == 2)
            self.assertEqual(refiner.calls, 0)
            self.assertEqual(result["fingerspelling"]["contextStatus"], "insufficient-context")
            self.assertEqual(result["fingerspelling"]["method"], "causal-tcn")
            for i in range(8):
                result = runtime.analyze([hand()], 640, 480, video=True, session_id="held",
                                         timestamp_ms=i * 100, finish=i == 7)
            self.assertEqual(refiner.calls, 0)
            self.assertEqual(result["fingerspelling"]["contextStatus"], "held-letter")
            for i in range(6):
                result = runtime.analyze([hand(i * .02)], 640, 480, video=True, session_id="moving",
                                         timestamp_ms=i * 100, finish=i == 5)
            self.assertEqual(refiner.calls, 1)
            self.assertEqual(result["fingerspelling"]["text"], "A")
            self.assertIn("too long", result["fingerspelling"]["refinementError"])

    def test_finish_after_duration_limit_preserves_segment_despite_capture_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            refiner = FakeRefiner("NEW")
            runtime = self.runtime(directory, refiner)
            for i in range(6):
                runtime.analyze([hand(i * .02)], 640, 480, video=True, session_id="one", timestamp_ms=i * 100)
            result = runtime.analyze([hand()], 640, 480, video=True, session_id="one", timestamp_ms=31000, finish=True)
            self.assertEqual(result["fingerspelling"]["status"], "finished")
            self.assertEqual(result["fingerspelling"]["frames"], 6)
            self.assertEqual(result["fingerspelling"]["text"], "NEW")
            self.assertEqual(refiner.calls, 1)

    def test_multiple_same_side_hands_abstain_without_stopping_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.runtime(directory, FakeRefiner())
            result = runtime.analyze([hand(), hand(.2)], 640, 480, video=True, session_id="one", timestamp_ms=0)
            self.assertIn("Håndvalget", result["fingerspelling"]["error"])
            self.assertEqual(runtime.raw_hands["Right"], [])
            result = runtime.analyze([hand()], 640, 480, video=True, session_id="one", timestamp_ms=100)
            self.assertNotIn("error", result["fingerspelling"])
            self.assertEqual(len(runtime.frames), 2)

    def test_external_adapter_matches_reference_normalization_and_compressed_indices(self):
        from model.openfs_refiner import preprocess
        right = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3)
        left = right[:1].copy() + 5
        poses, segments, indices = preprocess({0: right}, {0: left})
        expected = right[..., :2].copy()
        expected -= (expected.min(axis=1, keepdims=True) + expected.max(axis=1, keepdims=True)) * .5
        expected /= np.abs(expected).max(axis=1, keepdims=True) + 1e-6
        expected *= .5
        np.testing.assert_allclose(poses[:2].numpy(), expected)
        self.assertEqual(segments.tolist(), [0, 0, 1])
        self.assertEqual(indices.tolist(), [0, 1, 0])
        self.assertIsNone(preprocess({}, {}))

    def test_real_checkpoint_smoke_if_explicit_source_is_available(self):
        from model.openfs_refiner import OpenFSRefiner
        source = Path(os.environ.get("ASL_OPENFS_SOURCE", ROOT / ".build/openfs"))
        directory = Path(os.environ.get("ASL_OPENFS_MODEL_DIR", ROOT / "data/openfs-checkpoints/recognizer_fsboard"))
        if not (source / "lib/models/recognizer.py").is_file() or not (directory / "best_model.pth").is_file():
            self.skipTest("External research source/checkpoint not available")
        refiner = OpenFSRefiner(source, directory)
        points = np.array([[point.x * 640, point.y * 480] for point in hand()["landmarks"]], dtype=np.float32)
        text = refiner.refine({0: [points] * 4}, {})
        self.assertIsInstance(text, str)
        self.assertEqual(refiner.readiness()["status"], "ready")
        self.assertIsNotNone(refiner.network)

    def test_real_a_image_short_capture_keeps_letter_without_autoregressive_word(self):
        source = Path(os.environ.get("ASL_OPENFS_SOURCE", ROOT / ".build/openfs"))
        model_directory = Path(os.environ.get("ASL_OPENFS_MODEL_DIR", ROOT / "data/openfs-checkpoints/recognizer_fsboard"))
        data = Path(os.environ.get("ASL_SHOP_TEST_DATA_ROOT", ROOT / "data"))
        image = data / "asl_alphabet_train/asl_alphabet_train/A/A1000.jpg"
        landmarker = Path(os.environ.get("ASL_MEDIAPIPE_MODEL", data / "hand_landmarker.task"))
        if not all(path.is_file() for path in (image, landmarker, ROOT / "data/handshape.pt", ROOT / "data/sequence.pt", source / "lib/models/recognizer.py", model_directory / "best_model.pth")):
            self.skipTest("Real trained models, detector and public test image not available")
        url = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
        requests = [dict(id=i, dataUrl=url, video=True, sessionId="held-A", timestampMs=i * 100, finish=i == 2)
                    for i in range(3)]
        environment = {**os.environ, "ASL_MEDIAPIPE_MODEL": str(landmarker), "ASL_OPENFS_SOURCE": str(source),
                       "ASL_OPENFS_MODEL_DIR": str(model_directory), "PYTHONPATH": ""}
        result = subprocess.run([sys.executable, str(ROOT / "scripts/recognition_worker.py")],
                                input="".join(json.dumps(request) + "\n" for request in requests),
                                env=environment, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertTrue(all("error" not in response for response in responses), str(responses))
        final = responses[-1]["result"]
        self.assertEqual(final["prediction"]["letter"], "A")
        self.assertEqual(final["prediction"]["method"], "neural")
        self.assertEqual(final["fingerspelling"]["method"], "causal-tcn")
        self.assertEqual(final["fingerspelling"]["contextStatus"], "insufficient-context")
        self.assertLessEqual(len(final["fingerspelling"]["text"]), 1)
        self.assertFalse(final["fingerspelling"]["refiner"]["available"])


if __name__ == "__main__":
    unittest.main()
