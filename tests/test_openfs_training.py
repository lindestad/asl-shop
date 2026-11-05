"""Packed FSboard identity, clean alignment and real short optimization roundtrips.

Fixtures are synthetic protocol checks, never recognition-accuracy evidence.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
except ImportError:
    torch = None


def objects(values):
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result


def fixture(directory):
    from scripts.prepare_openfs_training import UPSTREAM_CHARS
    directory = Path(directory)
    source, metadata, packed = (directory / name for name in ("source", "metadata", "packed"))
    source.mkdir()
    metadata.mkdir()
    a, b = UPSTREAM_CHARS.index("a"), UPSTREAM_CHARS.index("b")
    poses = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3) + 40
    for split in ("train", "validation", "test"):
        names = [f"{split}-{kind}" for kind in ("no-hand-id", "wrong-coarse", "a", "bb")]
        words = ["a", "a", "a", "bb"]
        np.savez(source / f"fsboard_{split}_proc_clean.npz", vidnames=names, words=words,
                 signing_hand=objects([None, 0, 0, 1]),
                 frame_label=objects([np.array([a, a]), np.array([b, b]), np.array([a, a]), np.array([b, b])]),
                 frame_label_clean=objects([np.array([a, a]), np.array([b, b])]),
                 right_hand_poses=objects([{0: poses}, {0: poses}, {0: poses}, {}]),
                 left_hand_poses=objects([{}, {}, {}, {0: poses + 30}]),
                 right_frame_indices=objects([np.array([0, 3], dtype=np.int16)] * 3 + [np.array([], dtype=np.int16)]),
                 left_frame_indices=objects([np.array([], dtype=np.int16)] * 3 + [np.array([1, 4], dtype=np.int16)]))
        rows = [dict(clipFilename=name + ".mp4", signerId=f"person-{split}",
                     fullVideoFilename=f"original-{split}.mp4", phrase=word) for name, word in zip(names, words)]
        (metadata / f"daun_v3-{split}.json").write_text(json.dumps(rows), encoding="utf-8")
    return source, metadata, packed


@unittest.skipIf(torch is None, "Install the neural extra for packed model tests")
class PackedOpenFSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prepare(self, source, metadata, output):
        return subprocess.run([sys.executable, str(ROOT / "scripts/prepare_openfs_training.py"),
                               "--input", str(source), "--metadata", str(metadata), "--output", str(output),
                               "--trusted-pickle"], capture_output=True, text=True, timeout=30)

    def test_clean_filters_preserve_first_occurrence_order_and_skip_unlabelled_hands(self):
        from scripts.prepare_openfs_training import UPSTREAM_CHARS, clean_indices
        a, b = UPSTREAM_CHARS.index("a"), UPSTREAM_CHARS.index("b")
        selected = clean_indices(["ab", "ab", "ba", "abb"], [None, 0, 0, 0],
                                 [[a, b], [b, a], [b, -1, a], [a, a, b, b]])
        self.assertEqual(selected, [2, 3])

    def test_preparation_joins_real_identity_preserves_gaps_and_filtered_label_assignment(self):
        from model.neural import CONTRACT, HANDSHAPES, VOCAB
        with tempfile.TemporaryDirectory() as directory:
            source, metadata, packed = fixture(directory)
            result = self.prepare(source, metadata, packed)
            self.assertEqual(result.returncode, 0, result.stderr)
            schema = json.loads((packed / "dataset.json").read_text())
            self.assertEqual(schema["contract"], CONTRACT)
            self.assertEqual([part["clips"] for part in schema["partitions"]], [4, 4, 4])
            self.assertEqual([part["clean_clips"] for part in schema["partitions"]], [2, 2, 2])
            prepared = packed / schema.get("generation", "")
            clips = json.loads((prepared / "train_clips.json").read_text())
            self.assertEqual(clips[-1]["text"], "BB")
            self.assertTrue(all(row["signer"] == "person-train" for row in clips))
            self.assertTrue(all(row["session"] == "original-train.mp4" for row in clips))
            frames = np.load(prepared / "train_frames.npy")
            aux = np.load(prepared / "train_targets.npy")
            np.testing.assert_array_equal(frames[:4, 44], [1, 0, 0, 1])
            np.testing.assert_array_equal(aux[:8], [-100] * 8)
            np.testing.assert_array_equal(aux[8:12], [VOCAB.index("A"), 0, 0, VOCAB.index("A")])
            np.testing.assert_array_equal(aux[12:], [0, VOCAB.index("B"), 0, 0, VOCAB.index("B")])
            with np.load(prepared / "train_handshape.npz") as shape:
                np.testing.assert_array_equal(shape["clips"], [2, 2, 3, 3])
                np.testing.assert_array_equal(shape["labels"], [HANDSHAPES.index("A")] * 2 + [HANDSHAPES.index("B")] * 2)

    def test_failed_repreparation_preserves_active_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, metadata, packed = fixture(directory)
            result = self.prepare(source, metadata, packed)
            self.assertEqual(result.returncode, 0, result.stderr)
            original_schema = (packed / "dataset.json").read_bytes()
            schema = json.loads(original_schema)
            prepared = packed / schema.get("generation", "")
            original_frames = (prepared / "train_frames.npy").read_bytes()
            path = metadata / "daun_v3-train.json"
            rows = json.loads(path.read_text())
            rows[-1]["phrase"] = "WRONG"
            path.write_text(json.dumps(rows))
            result = self.prepare(source, metadata, packed)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Transcript mismatch", result.stderr)
            self.assertEqual((packed / "dataset.json").read_bytes(), original_schema)
            self.assertEqual((prepared / "train_frames.npy").read_bytes(), original_frames)

    def test_preparation_rejects_missing_identities_and_signer_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            source, metadata, packed = fixture(directory)
            path = metadata / "daun_v3-validation.json"
            rows = json.loads(path.read_text())
            path.write_text(json.dumps(rows[1:]))
            result = self.prepare(source, metadata, packed)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing original identity metadata", result.stderr)
            self.assertFalse((packed / "dataset.json").exists())
            for row in rows:
                row["signerId"] = "person-train"
            path.write_text(json.dumps(rows))
            result = self.prepare(source, metadata, packed)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Signer overlap", result.stderr)
            self.assertFalse((packed / "dataset.json").exists())

    def test_cpu_train_checkpoint_test_report_and_warmstarted_ctc_roundtrip(self):
        from model.neural import load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            source, metadata, packed = fixture(directory)
            result = self.prepare(source, metadata, packed)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs = {}
            for task in ("handshape", "sequence"):
                outputs[task] = Path(directory) / f"{task}.pt"
                command = [sys.executable, str(ROOT / "scripts/train_openfs.py"), "--data", str(packed),
                           "--task", task, "--output", str(outputs[task]), "--device", "cpu", "--hidden", "16",
                           "--epochs", "1", "--batch-size", "2"]
                if task == "sequence":
                    command.extend(["--warmstart-handshape", str(outputs["handshape"])])
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsNotNone(load_checkpoint(outputs[task], task))
                report = json.loads(outputs[task].with_suffix(".report.json").read_text())
                self.assertEqual(report["test"]["split"], "test")
                self.assertEqual(report["epochs"], 1)
                if task == "sequence":
                    self.assertEqual(report["test"]["clips"], 4)
                    self.assertEqual(report["test"]["characters"], 5)
                    self.assertEqual(list(report["test"]["per_signer"]), ["person-test"])
                    self.assertTrue(np.isfinite(report["test"]["cer"]))
                else:
                    self.assertIn("model-derived", report["test"]["label_provenance"])
            resumed = Path(directory) / "resumed.pt"
            # Deliberately omit --hidden: the checkpoint width must override the CLI default.
            resume_command = [sys.executable, str(ROOT / "scripts/train_openfs.py"), "--data", str(packed),
                              "--task", "handshape", "--output", str(resumed), "--device", "cpu",
                              "--epochs", "1", "--batch-size", "2", "--resume", str(outputs["handshape"])]
            result = subprocess.run(resume_command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(torch.load(resumed, weights_only=True)["hidden"], 16)
            self.assertIsNotNone(load_checkpoint(resumed, "handshape"))

    def test_ctc_float32_targets_and_lengths_follow_requested_device(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA runtime unavailable; CPU roundtrip above exercises CTC")
        from model.neural import FingerspellingNet, VOCAB
        model = FingerspellingNet(16).cuda()
        frames = torch.zeros((2, 6, 91), device="cuda")
        frames[..., 44] = 1
        with torch.autocast("cuda"):
            logits = model(frames)
        log_probs = logits.float().log_softmax(-1).transpose(0, 1).contiguous()
        targets = torch.tensor([VOCAB.index("A"), VOCAB.index("A"), VOCAB.index("B")], dtype=torch.long, device="cuda")
        lengths = torch.tensor([6, 6], dtype=torch.long, device="cpu")
        counts = torch.tensor([2, 1], dtype=torch.long, device="cpu")
        loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(log_probs, targets, lengths, counts)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
