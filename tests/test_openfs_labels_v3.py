import json
import numpy as np
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import HANDSHAPES, SEQUENCE_SIZE, VOCAB
from scripts.prepare_openfs_labels_v3 import (CONTRACT_V3, FEATURE_NOTE, STATIC_NOTE, audit_disjointness,
                                              refined_lookup, target_handshape, upstream_handshape, upstream_target)
from scripts.prepare_openfs_training import UPSTREAM_CHARS

A, B = UPSTREAM_CHARS.index('a'), UPSTREAM_CHARS.index('b')


class LabelMappings(unittest.TestCase):
    def test_blank_and_letters_across_spaces(self):
        self.assertEqual(upstream_handshape([0, A, B, 60, -1]).tolist(),
                         [0, HANDSHAPES.index('A'), HANDSHAPES.index('B'), -1, -1])
        self.assertEqual(upstream_target([0, A, B, 60]).tolist(), [0, VOCAB.index('A'), VOCAB.index('B'), -100])

    def test_movement_letters_stay_out_of_static_labels_but_not_ctc(self):
        for letter in ('j', 'z'):
            upstream_id = UPSTREAM_CHARS.index(letter)
            self.assertEqual(upstream_handshape([upstream_id]).tolist(), [-1])
            self.assertEqual(upstream_target([upstream_id]).tolist(), [VOCAB.index(letter.upper())])

    def test_target_handshape_reads_packed_targets(self):
        self.assertEqual(target_handshape([0, VOCAB.index('A'), VOCAB.index('J'), -100]).tolist(),
                         [0, HANDSHAPES.index('A'), -1, -1])


class RefinedJoin(unittest.TestCase):
    @staticmethod
    def _write(path, names, offsets, frame_index, labels, hands):
        np.savez(path, names=np.asarray(names), offsets=np.asarray(offsets),
                 frame_index=np.asarray(frame_index), label=np.asarray(labels),
                 clean_label=np.asarray(labels), hand=np.asarray(hands))

    def test_round_trip_by_clip_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'validation.npz'
            self._write(path, ['c1', 'c2'], [0, 2, 5], [3, 9, 0, 4, 8], [0, A, B, 0, 60], [0, 1])
            lookup = refined_lookup(path)
            self.assertEqual(sorted(lookup), ['c1', 'c2'])
            index, labels, hand = lookup['c2']
            self.assertEqual(index.tolist(), [0, 4, 8])
            self.assertEqual(labels.tolist(), [B, 0, 60])
            self.assertEqual(hand, 1)

    def test_rejects_malformed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            duplicated = Path(tmp) / 'duplicated.npz'
            self._write(duplicated, ['c1', 'c1'], [0, 1, 2], [0, 1], [0, A], [0, 0])
            with self.assertRaises(ValueError):
                refined_lookup(duplicated)
            offsets = Path(tmp) / 'offsets.npz'
            self._write(offsets, ['c1'], [0, 9], [0], [0], [0])
            with self.assertRaises(ValueError):
                refined_lookup(offsets)
            descending = Path(tmp) / 'descending.npz'
            self._write(descending, ['c1'], [0, 2], [5, 2], [0, A], [0])
            with self.assertRaises(ValueError):
                refined_lookup(descending)


class SplitHygiene(unittest.TestCase):
    @staticmethod
    def _clip(name, signer, session):
        return dict(name=name, signer=signer, session=session)

    def test_disjoint_splits_pass_and_overlap_fails_loudly(self):
        audit = audit_disjointness({'train': [self._clip('t1', 's1', 'v1')], 'test': [self._clip('e1', 's2', 'v2')]})
        self.assertEqual(audit['overlap'], 'none')
        with self.assertRaisesRegex(ValueError, 'signer overlap'):
            audit_disjointness({'train': [self._clip('t1', 's1', 'v1')], 'test': [self._clip('e1', 's1', 'v2')]})
        with self.assertRaisesRegex(ValueError, 'session overlap'):
            audit_disjointness({'train': [self._clip('t1', 's1', 'v1')], 'test': [self._clip('e1', 's2', 'v1')]})


def _pose(seed):
    rng = np.random.default_rng(seed)
    pose = rng.normal(size=(21, 3)).astype(np.float32)
    pose[0] = 0.0  # wrist origin
    pose[9] = [0.1, 0.0, 0.0]  # non-zero wrist-pinky scale
    return np.stack([pose] * 4)


class PackGenerationV3(unittest.TestCase):
    """Runs the real v3 packer end to end over a two-clip fixture: clean fallback plus refined join."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.frames = {}
        for name, words, coarse, clean, indices, labels in (
                ('clip1', 'ab', [A, A, B, -1], [A, 0, B, 0], [1, 2, 5, 7], [A, 0, B, 0]),
                ('clip2', 'ba', [B, B, A], [B, B, A, 0], [0, 1, 2, 4], [B, B, A, 0])):
            cls.frames[name] = (words, coarse, clean, indices, labels)
        arrays = dict(vidnames=np.array(list(cls.frames)),
                      words=np.array([cls.frames[n][0] for n in cls.frames]),
                      signing_hand=np.array([0] * 2),
                      frame_label=np.array([cls.frames[n][1] for n in cls.frames], dtype=object),
                      frame_label_clean=np.array([cls.frames[n][2] for n in cls.frames], dtype=object),
                      right_hand_poses=np.array([{0: _pose(i)} for i in range(2)], dtype=object),
                      left_hand_poses=np.array([{} for _ in range(2)], dtype=object),
                      right_frame_indices=np.array([cls.frames[n][3] for n in cls.frames], dtype=object),
                      left_frame_indices=np.array([[] for _ in range(2)], dtype=object))
        source = tmp / 'source'
        (source).mkdir()
        np.savez(source / 'fsboard_validation_proc_clean.npz', **arrays)
        metadata = tmp / 'metadata'
        metadata.mkdir()
        (metadata / 'batch-1.json').write_text(json.dumps([
            dict(clipFilename=f'{n}.mp4', signerId=f'signer{i}', phrase='AB' if n == 'clip1' else 'BA',
                 fullVideoFilename=f'signer{i}_session') for i, n in enumerate(cls.frames)]))
        refined = tmp / 'refined'
        refined.mkdir()
        np.savez(refined / 'validation.npz', names=np.array(['clip1']), offsets=np.array([0, 4]),
                 frame_index=np.array([1, 2, 5, 7]), label=np.array([A, 0, B, 0], dtype=np.int8),
                 clean_label=np.array([A, 0, B, 0], dtype=np.int8), hand=np.array([0], dtype=np.int8))
        output = tmp / 'packed'
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_openfs_labels_v3.py'),
                              '--trusted-pickle', '--input', str(source), '--metadata', str(metadata),
                              '--output', str(output), '--refined-labels', str(refined),
                              '--splits', 'validation', '--static-per-letter', '2', '--idle-per-letter', '1'],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        cls.output = output
        pack = next(output.glob('generations/*'))
        cls.split = pack

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_feature_contract_and_vocab_are_unchanged(self):
        frames = np.load(self.split / 'validation_frames.npy')
        self.assertEqual(frames.shape[1:], (SEQUENCE_SIZE,))
        dataset = json.loads((self.output / 'dataset.json').read_text())
        self.assertEqual(dataset['contract'], CONTRACT_V3)
        self.assertEqual(dataset['labels'], VOCAB)
        self.assertEqual(dataset['static_labels'], HANDSHAPES)
        self.assertEqual(dataset['feature_contract'], FEATURE_NOTE)
        self.assertIn('J and Z', dataset['static_labels_note'])
        self.assertNotIn('J', dataset['static_labels'])

    def test_idle_negatives_respect_cap_and_join_refined_origin(self):
        with np.load(self.split / 'validation_handshape.npz') as samples:
            labels, origin = samples['labels'], samples['origin']
            self.assertEqual(len(labels), len(origin))
            idle = labels == HANDSHAPES.index('nothing')
            self.assertTrue(idle.any(), 'blank frames must become nothing negatives')
            self.assertTrue((labels >= 0).all() and (labels < len(HANDSHAPES)).all())
            # clip1 (refined): classes A,B sampled 2 each, cap idle = 1 class-multiple * 2 classes.
            clip1 = samples['clips'] == 0
            self.assertTrue((origin[clip1] == 1).all())
            self.assertLessEqual(int((idle & clip1).sum()), 2)
            # clip2 keeps the clean fallback even though another clip was refined.
            self.assertTrue((origin[samples['clips'] == 1] == 0).all())

    def test_frame_targets_match_the_packed_label_space(self):
        targets = np.load(self.split / 'validation_targets.npy')
        clips = json.loads((self.split / 'validation_clips.json').read_text())
        start = {row['name']: row['start'] for row in clips}
        self.assertEqual(targets[start['clip1'] + np.array([1, 2, 5, 7])].tolist(),
                         [VOCAB.index('A'), 0, VOCAB.index('B'), 0])  # blanks stay 0, never -100
        partition = next(p for p in json.loads((self.output / 'dataset.json').read_text())['partitions']
                         if p['split'] == 'validation')
        self.assertEqual(partition['refined_clips'], 1)
        self.assertEqual(partition['clean_labelled_clips'] if 'clean_labelled_clips' in partition
                         else partition['labelled_clips'] - partition['refined_clips'], 1)
        self.assertGreater(partition['idle_negative_frames'], 0)
        self.assertLess(partition['idle_negative_ratio'], 0.5)

    def test_dataset_records_split_hygiene_and_license(self):
        dataset = json.loads((self.output / 'dataset.json').read_text())
        self.assertEqual(dataset['split_hygiene']['overlap'], 'none')
        self.assertIn('research', dataset['license_note'])


if __name__ == '__main__':
    unittest.main()
