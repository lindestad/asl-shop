"""Acquisition timing and dropout behavior independent of neural weights."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.timeline import TimelineResampler, resample_frames


def feature(timestamp, *, right=True, left=False):
    frame = np.zeros(91, dtype=np.float32)
    for base, present in ((0, right), (45, left)):
        if present:
            frame[base:base + 44] = timestamp / 1000
            frame[base + 44] = 1
    return frame


class TimelineTests(unittest.TestCase):
    def test_irregular_10_and_15_hz_samples_match_a_regular_linear_motion_grid(self):
        regular_times = np.arange(31) * 1000 / 30
        expected = np.array([feature(time, right=True, left=True) for time in regular_times])
        expected[1:, -1] = 1 / 30
        for capture_times in (np.arange(11) * 100, np.arange(16) * 1000 / 15,
                              np.array([0, 45, 110, 185, 310, 395, 520, 655, 730, 850, 1000])):
            frames, ticks = resample_frames([feature(time, right=True, left=True) for time in capture_times], capture_times)
            np.testing.assert_allclose(ticks, regular_times, atol=1e-6)
            np.testing.assert_allclose(frames, expected, atol=1e-6)

    def test_dropout_never_interpolates_an_unobserved_hand_and_keeps_other_hand(self):
        capture_times = [0, 100, 200, 300]
        frames = [feature(0, right=True, left=True), feature(100, right=False, left=True),
                  feature(200, right=True, left=True), feature(300, right=True, left=True)]
        values, ticks = resample_frames(frames, capture_times)
        np.testing.assert_allclose(ticks, np.arange(10) * 1000 / 30)
        np.testing.assert_array_equal(values[:, 44], [1, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        self.assertTrue((values[1:6, :45] == 0).all())
        np.testing.assert_array_equal(values[:, 89], np.ones(10))
        np.testing.assert_allclose(values[:, 45], ticks / 1000, atol=1e-6)

    def test_long_gap_is_explicit_missing_data_until_arriving_observation(self):
        values, ticks = resample_frames([feature(0), feature(500)], [0, 500])
        self.assertEqual(len(values), 16)
        np.testing.assert_array_equal(values[1:-1, 44], np.zeros(14))
        self.assertEqual(values[-1, 44], 1)
        self.assertAlmostEqual(ticks[-1], 500)

    def test_fast_acquisition_does_not_duplicate_model_frames_and_timestamps_are_ordered(self):
        sampler = TimelineResampler()
        self.assertEqual(len(sampler.feed(feature(0), 0)), 1)
        self.assertEqual(sampler.feed(feature(10), 10), [])
        self.assertEqual(len(sampler.feed(feature(40), 40)), 1)
        with self.assertRaisesRegex(ValueError, "increase"):
            sampler.feed(feature(40), 40)
        for fps in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TimelineResampler(fps)

    def test_runtime_counts_capture_and_model_frames_resets_and_handles_fast_finish(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Install neural dependencies for runtime checks")
        from model.live_neural import NeuralRuntime
        from model.neural import VOCAB
        class BlankModel:
            def __call__(self, frames):
                logits = torch.zeros((*frames.shape[:2], len(VOCAB)))
                logits[..., 0] = 10
                return logits
        with tempfile.TemporaryDirectory() as directory:
            runtime = NeuralRuntime(directory)
            runtime.loaded = True
            runtime.models["sequence"] = BlankModel()
            for timestamp in (0, 100, 200):
                result = runtime.analyze([], 640, 480, video=True, session_id="one", timestamp_ms=timestamp)
            self.assertEqual(result["fingerspelling"]["frames"], 3)
            self.assertEqual(result["fingerspelling"]["modelFrames"], 7)
            result = runtime.analyze([], 640, 480, video=True, session_id="one", timestamp_ms=1500)
            self.assertEqual(result["fingerspelling"]["frames"], 1)
            self.assertEqual(result["fingerspelling"]["modelFrames"], 1)
            result = runtime.analyze([], 640, 480, video=True, session_id="two", timestamp_ms=0)
            self.assertEqual(result["fingerspelling"]["frames"], 1)
            result = runtime.analyze([], 640, 480, video=True, session_id="two", timestamp_ms=10, finish=True)
            self.assertEqual(result["fingerspelling"]["status"], "finished")
            self.assertEqual(result["fingerspelling"]["frames"], 2)
            self.assertEqual(result["fingerspelling"]["modelFrames"], 1)


if __name__ == "__main__":
    unittest.main()
