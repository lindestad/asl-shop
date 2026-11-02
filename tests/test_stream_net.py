"""Streaming FSboard model (model/stream_net.py) and its live-runtime wiring."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None


def hand(side, rng, shift=0.0):
    points = rng.random((21, 3)).astype(np.float32) * [0.2, 0.2, 0.05] + [0.4 + shift, 0.4, -0.02]
    return {"hand": side, "landmarks": [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in points]}


@unittest.skipIf(torch is None, "Install the neural extra to exercise neural models")
class StreamNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def small(self):
        from model.stream_net import StreamNet
        torch.manual_seed(0)
        return StreamNet(hidden=16, dilations=(1, 2, 4), lookahead=3).eval()

    def test_raw_frame_slots_and_scale(self):
        from model.stream_net import Z_REFERENCE_WIDTH, raw_frame
        rng = np.random.default_rng(0)
        left = hand("Left", rng)
        xyz, present = raw_frame([left], 640, 480)
        self.assertEqual(present.tolist(), [False, True])
        self.assertTrue(np.all(xyz[0] == 0))
        point = left["landmarks"][5]
        np.testing.assert_allclose(xyz[1, 5], [point.x * 640, point.y * 480, point.z * 640 / Z_REFERENCE_WIDTH], rtol=1e-6)
        with self.assertRaises(ValueError):
            raw_frame([left, hand("Left", rng)], 640, 480)

    def test_output_does_not_see_beyond_lookahead(self):
        model = self.small()
        xyz, present = torch.randn(1, 40, 2, 21, 3) * 100, torch.ones(1, 40, 2, dtype=torch.bool)
        changed = xyz.clone()
        changed[:, 25:] = torch.randn(1, 15, 2, 21, 3) * 100
        with torch.no_grad():
            a, b = model(xyz, present), model(changed, present)
        torch.testing.assert_close(a[:, :25 - model.lookahead_frames], b[:, :25 - model.lookahead_frames])

    def test_incremental_decoder_matches_whole_clip(self):
        from model.stream_net import StreamDecoder
        model = self.small()
        rng = np.random.default_rng(1)
        frames = [(rng.normal(size=(2, 21, 3)).astype(np.float32) * 100, rng.random(2) > 0.2) for _ in range(90)]
        decoder = StreamDecoder(model)
        for k in range(0, 90, 7):
            decoder.push(frames[k:k + 7])
        xyz = torch.from_numpy(np.stack([f[0] for f in frames]))[None]
        present = torch.from_numpy(np.stack([f[1] for f in frames]))[None]
        with torch.no_grad():
            whole = model(xyz, present).argmax(-1)[0].tolist()
        self.assertEqual(len(decoder.tokens()), 90)
        self.assertEqual(decoder.final, whole[:90 - model.lookahead_frames])
        self.assertEqual(decoder.tail, whole[90 - model.lookahead_frames:])

    def test_timeline_interpolates_short_gaps_only(self):
        from model.stream_net import RawTimeline
        timeline = RawTimeline()
        a, b = np.zeros((2, 21, 3), np.float32), np.full((2, 21, 3), 3, np.float32)
        both, right = np.array([True, True]), np.array([True, False])
        self.assertEqual(len(timeline.feed(a, both, 0)), 1)
        ticks = timeline.feed(b, right, 100)  # 3 ticks: two interpolated, one endpoint
        self.assertEqual(len(ticks), 3)
        np.testing.assert_allclose(ticks[0][0][0], 1.0, rtol=1e-5)
        self.assertEqual(ticks[0][1].tolist(), [True, False])  # left vanished: not synthesized
        self.assertTrue(np.all(ticks[0][0][1] == 0))
        late = timeline.feed(a, both, 1000)  # gap > 250 ms: no interpolation
        self.assertFalse(any(present.any() for _, present in late[:-1]))

    def test_checkpoint_round_trip_through_load_checkpoint(self):
        from model.neural import VOCAB, load_checkpoint
        from model.stream_net import CONTRACT
        model = self.small()
        checkpoint = dict(contract=CONTRACT, task="sequence", labels=VOCAB, config=model.config,
                          state_dict={k: v.half() for k, v in model.state_dict().items()})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stream.pt"
            torch.save(checkpoint, path)
            loaded = load_checkpoint(path, "sequence")
        self.assertEqual((loaded.history_frames, loaded.lookahead_frames), (model.history_frames, 3))
        xyz, present = torch.randn(1, 30, 2, 21, 3) * 100, torch.ones(1, 30, 2, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(loaded(xyz, present), model(xyz, present), atol=0.05, rtol=0.05)

    @unittest.skipUnless((ROOT / "assets/models/fsboard-stream.pt").is_file(), "exported model not present")
    def test_live_runtime_uses_streaming_model(self):
        from model.live_neural import NeuralRuntime
        runtime = NeuralRuntime(ROOT)
        runtime.paths["sequence"] = ROOT / "assets/models/fsboard-stream.pt"
        runtime.paths["handshape"] = ROOT / "missing.pt"
        rng = np.random.default_rng(2)
        result = None
        for k in range(45):
            hands = [hand("Right", rng, shift=0.1 * np.sin(k / 3))]
            result = runtime.analyze(hands, 1280, 720, video=True, session_id="s", timestamp_ms=1000 + k * 33.3,
                                     finish=k == 44)
        sequence = result["fingerspelling"]
        self.assertTrue(sequence["available"])
        self.assertEqual(sequence["method"], "fsboard-stream")
        self.assertEqual(sequence["status"], "finished")
        self.assertNotIn("error", sequence)
        self.assertGreaterEqual(sequence["modelFrames"], 44)
        self.assertEqual(len(runtime.stream.tokens()), sequence["modelFrames"])


if __name__ == "__main__":
    unittest.main()
