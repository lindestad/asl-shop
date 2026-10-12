"""Behavioral checks, including real CPU training smoke tests (not accuracy claims)."""
import csv
import json
import os
from pathlib import Path
import subprocess
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


@unittest.skipIf(torch is None, "Install the neural extra to exercise neural models")
class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ctc_keeps_double_letters_only_after_blank(self):
        from model.neural import VOCAB, decode_ctc
        a, b = VOCAB.index("A"), VOCAB.index("B")
        self.assertEqual(decode_ctc([a, a, 0, a, b, b, 0]), "AAB")

    def test_temporal_network_does_not_see_future_frames(self):
        from model.neural import FingerspellingNet
        model = FingerspellingNet(16).eval()
        frames = torch.randn(2, 20, 91)
        changed = frames.clone()
        changed[:, 10:] = torch.randn(2, 10, 91) * 50
        with torch.no_grad():
            original, later = model(frames), model(changed)
        self.assertTrue(torch.allclose(original[:, :10], later[:, :10], atol=1e-6))
        self.assertFalse(torch.allclose(original[:, 10:], later[:, 10:]))

    def test_preprocessing_retains_trajectory_and_missing_hands(self):
        from model.neural import frame_features
        points = [SimpleNamespace(x=.2 + i * .001, y=.3 + i * .002) for i in range(21)]
        moved = [SimpleNamespace(x=p.x + .1, y=p.y) for p in points]
        a = frame_features([{"hand": "Right", "landmarks": points}], 640, 480, 30)
        b = frame_features([{"hand": "Right", "landmarks": moved}], 640, 480, 30)
        np.testing.assert_allclose(a[:42], b[:42], atol=1e-5)
        pixels = np.array([[point.x * 640, point.y * 480] for point in points], dtype=np.float32)
        scale = np.linalg.norm(pixels[9] - pixels[0]) + 1e-6
        self.assertAlmostEqual(float(b[42] - a[42]), .1 * 640 / scale, places=4)
        self.assertEqual(a[44], 1)
        self.assertEqual(a[89], 0)
        self.assertTrue((a[45:90] == 0).all())

    def test_packed_openfs_and_camera_features_match_without_image_dimension_guesses(self):
        from model.neural import frame_features
        from scripts.prepare_openfs_training import encode_poses
        right = np.arange(21 * 3, dtype=np.float32).reshape(1, 21, 3) + 50
        left = right.copy() + 15
        packed = encode_poses({0: right}, {0: left}, [0], [0])
        hands = []
        for side, poses in (("Right", right), ("Left", left)):
            hands.append({"hand": side, "landmarks": [SimpleNamespace(x=float(p[0]) / 640,
                                                                        y=float(p[1]) / 480) for p in poses[0]]})
        camera = frame_features(hands, 640, 480, 0)
        np.testing.assert_allclose(camera, packed[0], atol=1e-5)
        scaled_camera = frame_features(hands, 1280, 960, 0)
        np.testing.assert_allclose(scaled_camera, camera, atol=1e-5)

    def test_manifest_rejects_signer_leakage_and_wrong_timestamps(self):
        from model.sessions import write_clip, read_manifest
        with tempfile.TemporaryDirectory() as directory:
            common = dict(signer_id="person", session_id="session", kind="idle", transcript="",
                          source="test", label_provenance="human")
            with self.assertRaisesRegex(ValueError, "strictly increase"):
                write_clip(directory, np.zeros((2, 91)), [0, 0], **common)
            write_clip(directory, np.zeros((2, 91)), [0, 30], **common)
            write_clip(directory, np.zeros((2, 91)), [0, 30], split="test", **common)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                read_manifest(Path(directory) / "manifest.jsonl")

    def test_openfs_reconstructs_missing_detection_times(self):
        from scripts.import_openfs import convert
        poses = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3)
        meta = {"width": "640", "height": "480", "fps": "25", "right_frame_indices": "[0, 3]"}
        frames, times = convert(poses, [], meta)
        self.assertEqual(frames.shape, (4, 91))
        np.testing.assert_array_equal(frames[:, 44], [1, 0, 0, 1])
        np.testing.assert_allclose(times, [0, 40, 80, 120])
        with self.assertRaisesRegex(ValueError, "indices missing"):
            convert(poses, [], {"width": 640, "height": 480, "fps": 25})

    def test_openfs_import_roundtrip_preserves_verified_identity_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "raw.npz"
            np.savez(source, vidnames=["clip"], words=["AA"],
                     right_hand_poses=np.array([{0: np.ones((4, 21, 3))}], dtype=object),
                     left_hand_poses=np.array([{}], dtype=object))
            metadata = directory / "meta.csv"
            with metadata.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["video_id", "signer_id", "session_id", "split", "width", "height", "fps", "person_id", "right_frame_indices"])
                writer.writerow(["clip", "actual-person", "actual-session", "train", 640, 480, 30, 0, "[0, 1, 3, 4]"])
            output = directory / "output"
            command = [sys.executable, str(ROOT / "scripts/import_openfs.py"), "--input", str(source),
                       "--metadata", str(metadata), "--output", str(output), "--trusted-pickle"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            entry = json.loads((output / "manifest.jsonl").read_text())
            self.assertEqual(entry["signer_id"], "actual-person")
            self.assertEqual(entry["transcript"], "AA")
            self.assertIn("fps-derived-time", entry["label_provenance"])
            self.assertIn("OpenFS:raw.npz:clip", entry["source"])
            with np.load(output / entry["file"]) as clip:
                self.assertEqual(len(clip["frames"]), 5)

    def test_generic_openfs_import_uses_v2_hand_units_and_embedded_indices(self):
        from scripts.import_openfs import convert
        from scripts.prepare_openfs_training import encode_poses
        right = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3) + 50
        left = right[:1].copy() + 15
        metadata = {"fps": 25}
        frames, _ = convert(right, left, metadata, frame_indices={"Right": [0, 3], "Left": [1]})
        expected = encode_poses({0: right}, {0: left}, [0, 3], [1], fps=25)
        np.testing.assert_allclose(frames, expected, atol=1e-5)
        with self.assertRaisesRegex(ValueError, "finite integers"):
            convert(right, [], metadata, frame_indices={"Right": [0, 3.5]})

    def test_clean_openfs_import_aligns_filtered_annotations_not_full_clip_array(self):
        from scripts.prepare_openfs_training import UPSTREAM_CHARS
        a, b = UPSTREAM_CHARS.index("a"), UPSTREAM_CHARS.index("b")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "clean.npz"
            right = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3) + 30
            right_hands = np.array([{0: right}, {0: right}, {0: right}], dtype=object)
            np.savez(source, vidnames=["no-signing-hand", "bad-coarse-labels", "correct"],
                     words=["a", "a", "b"], right_hand_poses=right_hands,
                     left_hand_poses=np.array([{}, {}, {}], dtype=object),
                     right_frame_indices=np.array([[0, 3]] * 3, dtype=np.int16),
                     left_frame_indices=np.empty((3, 0), dtype=np.int16),
                     signing_hand=np.array([None, 0, 0], dtype=object),
                     frame_label=np.array([[a, a], [b, b], [b, b]], dtype=np.int64),
                     frame_label_clean=np.array([[b, b]], dtype=np.int64))
            metadata = directory / "meta.csv"
            with metadata.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["video_id", "signer_id", "session_id", "split", "person_id"])
                for name in ("no-signing-hand", "bad-coarse-labels", "correct"):
                    writer.writerow([name, "actual-person", "actual-session", "train", 0])
            mapping = directory / "labels.json"
            mapping.write_text(json.dumps({str(a): "A", str(b): "B"}))
            output = directory / "output"
            command = [sys.executable, str(ROOT / "scripts/import_openfs.py"), "--input", str(source),
                       "--metadata", str(metadata), "--output", str(output), "--trusted-pickle",
                       "--clean-label-map", str(mapping)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            entries = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len([e for e in entries if e["kind"] == "spelling"]), 3)
            held = [e for e in entries if e["kind"] == "held"]
            self.assertEqual(len(held), 1)
            self.assertEqual(held[0]["transcript"], "B")
            self.assertTrue(held[0]["source"].endswith(":correct"))
            self.assertIn("model-derived-clean-frame-label", held[0]["label_provenance"])
            self.assertIn("nominal-30-fps", held[0]["label_provenance"])
            self.assertNotIn("assumed-contiguous", held[0]["label_provenance"])
            with np.load(output / held[0]["file"]) as clip:
                np.testing.assert_array_equal(clip["timestamps_ms"], [0, 100])
            with np.load(source, allow_pickle=True) as archive:
                broken = {key: archive[key] for key in archive.files}
            broken["frame_label_clean"] = np.array([[b, b], [b, b]], dtype=np.int64)
            wrong_source = directory / "wrong-clean-count.npz"
            np.savez(wrong_source, **broken)
            wrong_output = directory / "wrong-output"
            wrong_command = command.copy()
            wrong_command[wrong_command.index("--input") + 1] = str(wrong_source)
            wrong_command[wrong_command.index("--output") + 1] = str(wrong_output)
            result = subprocess.run(wrong_command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("filtering/count mismatch", result.stderr)
            self.assertFalse((wrong_output / "manifest.jsonl").exists())

    def test_runtime_readiness_sessions_finish_and_no_static_j(self):
        from model.live_neural import NeuralRuntime
        from model.neural import VOCAB
        class FixedSequence:
            def __call__(self, frames):
                logits = torch.zeros((*frames.shape[:2], len(VOCAB)))
                logits[..., VOCAB.index("J")] = 10
                return logits
        with tempfile.TemporaryDirectory() as directory:
            runtime = NeuralRuntime(directory)
            result = runtime.analyze([], 640, 480)
            self.assertFalse(result["handshapeAvailable"])
            self.assertFalse(result["fingerspelling"]["available"])
            runtime.models["sequence"] = FixedSequence()
            for timestamp in (0, 30, 60):
                result = runtime.analyze([], 640, 480, video=True, session_id="one", timestamp_ms=timestamp)
            self.assertEqual(result["fingerspelling"]["text"], "")
            self.assertEqual(result["fingerspelling"]["frames"], 3)
            with self.assertRaisesRegex(ValueError, "increase"):
                runtime.analyze([], 640, 480, video=True, session_id="one", timestamp_ms=60)
            result = runtime.analyze([], 640, 480, video=True, session_id="two", timestamp_ms=0, finish=True)
            self.assertEqual(result["fingerspelling"]["frames"], 1)
            self.assertFalse(result["fingerspelling"]["provisional"])
            result = runtime.analyze([], 640, 480, video=True, session_id="two", timestamp_ms=5000)
            self.assertEqual(result["fingerspelling"]["status"], "finished")
            self.assertEqual(result["fingerspelling"]["frames"], 1)

    def test_j_requires_recent_motion_with_visible_hand(self):
        from model.live_neural import NeuralRuntime
        from model.neural import VOCAB
        class EmitLastJ:
            def __call__(self, frames):
                logits = torch.zeros((*frames.shape[:2], len(VOCAB)))
                logits[..., 0] = 5
                logits[:, -1, VOCAB.index("J")] = 10
                return logits
        with tempfile.TemporaryDirectory() as directory:
            runtime = NeuralRuntime(directory)
            runtime.loaded = True
            runtime.models["sequence"] = EmitLastJ()
            for moving, session in ((False, "held"), (True, "moving")):
                for i in range(4):
                    points = [SimpleNamespace(x=.2 + n * .001 + (.1 * i if moving else 0),
                                              y=.3 + n * .002) for n in range(21)]
                    result = runtime.analyze([{"hand": "Right", "landmarks": points}], 640, 480,
                                             video=True, session_id=session, timestamp_ms=i * 1000 / 30)
                self.assertEqual(result["fingerspelling"]["text"], "J" if moving else "")

    def test_worker_protocol_with_neural_checkpoints_and_real_detector(self):
        import importlib.util
        from model.neural import CONTRACT, HANDSHAPES, VOCAB, HandshapeNet, FingerspellingNet
        model_path = Path(os.environ.get("ASL_MEDIAPIPE_MODEL", ROOT / "data/hand_landmarker.task"))
        if not model_path.is_file() or importlib.util.find_spec("cv2") is None or importlib.util.find_spec("mediapipe") is None:
            self.skipTest("MediaPipe dependencies/model not available for real worker protocol test")
        import base64
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for task, network, labels in (("handshape", HandshapeNet(16), HANDSHAPES),
                                          ("sequence", FingerspellingNet(16), VOCAB)):
                path = Path(directory) / f"{task}.pt"
                torch.save({"contract": CONTRACT, "task": task, "labels": labels,
                            "hidden": 16, "state_dict": network.state_dict()}, path)
                paths[f"ASL_{task.upper()}_MODEL"] = str(path)
            _, encoded = cv2.imencode(".png", np.zeros((128, 128, 3), dtype=np.uint8))
            (Path(directory) / "bad-knn.pkl").write_bytes(b"not a pickle")
            image = "data:image/png;base64," + base64.b64encode(encoded).decode()
            requests = [dict(id=1, dataUrl=image), dict(id=2, dataUrl=image, video=True,
                                                      sessionId="camera-one", timestampMs=0)]
            result = subprocess.run([sys.executable, str(ROOT / "scripts/recognition_worker.py")],
                                    input="".join(json.dumps(request) + "\n" for request in requests),
                                    env={**os.environ, **paths, "ASL_KNN_MODEL": str(Path(directory) / "bad-knn.pkl")},
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            responses = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([response["id"] for response in responses], [1, 2])
            for response in responses:
                self.assertNotIn("error", response, str(response))
                self.assertTrue(response["result"]["handshapeAvailable"])
                self.assertTrue(response["result"]["fingerspelling"]["available"])
                self.assertIsNone(response["result"]["prediction"])
                self.assertEqual(response["result"]["fingerspelling"]["text"], "")

    def test_real_train_save_load_and_evaluate_both_tasks(self):
        from model.neural import load_checkpoint
        from model.sessions import write_clip
        with tempfile.TemporaryDirectory() as directory:
            for split in ("train", "validation", "test"):
                for letter in ("A", "B"):
                    frames = np.zeros((6, 91), dtype=np.float32)
                    frames[:, :42] = 0.2 if letter == "A" else .8
                    frames[:, 44] = 1
                    frames[1:, -1] = .03
                    write_clip(directory, frames, np.arange(6) * 30,
                               signer_id=split, session_id=split, kind="held", transcript=letter,
                               source="synthetic-smoke", label_provenance="synthetic", split=split, hand="Right")
            for task in ("handshape", "sequence"):
                output = Path(directory) / f"{task}.pt"
                command = [sys.executable, str(ROOT / "scripts/train_neural.py"), "--manifest",
                           str(Path(directory) / "manifest.jsonl"), "--task", task, "--output", str(output),
                           "--epochs", "1", "--hidden", "16", "--device", "cpu"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(output.is_file())
                self.assertIsNotNone(load_checkpoint(output, task))
                result = subprocess.run([*command, "--evaluate", str(output)], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("idle_false_positive_rate", json.loads(result.stdout))
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    load_checkpoint(output, "sequence" if task == "handshape" else "handshape")


if __name__ == "__main__":
    unittest.main()
