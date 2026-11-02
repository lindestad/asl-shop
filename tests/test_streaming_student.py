"""Guards for the causal streaming student trainer (scripts/train_streaming_student.py).

The student is what ships in the app; the OpenFS teacher weights are research-only.  These tests
pin the properties the accuracy-versus-latency result rests on:

- the single teacher-token <-> packed-row map, its padding sentinels, and its survival of the
  packer's temporal augmentation;
- every built network really is causal and really has the receptive field the report claims
  (measured off the conv layers, never off the CLI value);
- the distillation losses are differentiable, mask pad slots, and the per-frame label chain falls
  back in the documented order with the fallback counted;
- wall-clock latency is measured only by scripts/benchmark_neural.py, which really does accept a
  student checkpoint through the live streaming runtime;
- a whole study run end to end on synthetic packed clips: checkpoint naming, report columns, the
  research_only/distilled_from provenance, evaluate-only mode and the merged summary table;
- the published study artifacts in docs/ stay arithmetically self-consistent and agree with the
  writeup they summarise.

The released teacher constructors live in the pinned upstream checkout and the weights are not in
this repo, so load_teacher/teacher_posteriors themselves are exercised on the VPS, not here.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('OMP_NUM_THREADS', '2')
torch.set_num_threads(2)

REPO = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location('streaming_student',
                                                  REPO / 'scripts' / 'train_streaming_student.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def quiet():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def packed_values(rows=6, present=(), left=()):
    """A clip slice in the packed gap-preserving layout: 45-wide hand slots (42 shape columns,
    2 wrist, 1 presence at column 44 right / 89 left) plus the column-90 delta column."""
    values = np.zeros((rows, 91), dtype=np.float32)
    rng = np.random.default_rng(7)
    for base, rows_with_hand in ((0, present), (45, left)):
        for row in rows_with_hand:
            values[row, base:base + 42] = rng.normal(0, .08, 42).astype(np.float32)
            values[row, base + 42:base + 44] = rng.normal(0, .02, 2).astype(np.float32)
            values[row, base + 44] = 1
    values[:, 90] = 1 / 30
    return values


def write_pack(directory, splits=(('train', 6, 34), ('validation', 3, 32), ('test', 3, 32))):
    """Synthetic packed directory in the layout scripts/train_openfs.read_data expects, with
    per-frame pack targets so the pack_targets frame-label route is the fallback that fires."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    texts = ('ACE', 'BDF', 'CEG', 'AB', 'CD', 'GH', 'AC', 'BD')
    for split, clip_count, rows in splits:
        total = clip_count * rows
        values = np.zeros((total, 91), dtype=np.float32)
        rng = np.random.default_rng(abs(hash(split)) % 1000)
        values[:, 0:42] = rng.normal(0, .08, (total, 42)).astype(np.float32)
        values[:, 42:44] = rng.normal(0, .02, (total, 2)).astype(np.float32)
        values[:, 44] = 1
        values[:, 87:89] = rng.normal(0, .02, (total, 2)).astype(np.float32)
        values[:, 90] = 1 / 30
        targets = np.full(total, -100, dtype=np.int16)
        clips = []
        for clip in range(clip_count):
            text = texts[clip % len(texts)]
            start, end = clip * rows, (clip + 1) * rows
            block = rows // len(text)
            for offset, character in enumerate(text):
                targets[start + offset * block:start + (offset + 1) * block] = \
                    module_vocab().index(character)
            clips.append(dict(name=f'{split}{clip}', start=start, end=end, text=text,
                              signer='s0', session='synthetic'))
        np.save(directory / f'{split}_frames.npy', values)
        np.save(directory / f'{split}_targets.npy', targets)
        (directory / f'{split}_clips.json').write_text(json.dumps(clips), encoding='utf-8')
    (directory / 'dataset.json').write_text(json.dumps({'generation': ''}), encoding='utf-8')
    return directory


_VOCAB = None


def module_vocab():
    global _VOCAB
    if _VOCAB is None:
        sys.path.insert(0, str(REPO))
        from model.neural import VOCAB
        _VOCAB = VOCAB
    return _VOCAB


class IndexMap(unittest.TestCase):
    """The one bridge between the teacher's compressed detection order and the student's rows."""

    @classmethod
    def setUpClass(cls):
        cls.s = load_script()

    def test_compressed_tokens_carry_packed_rows_and_normalized_poses(self):
        s = self.s
        values = packed_values(rows=6, present=(0, 1, 4))
        poses, segments, positions, rows = s.teacher_tokens(values, 'compressed')
        np.testing.assert_array_equal(rows, [0, 1, 4])          # gaps 2,3,5 vanish from the index
        self.assertEqual(poses.shape, (3, 42))
        expected = s.normalized_box(values[[0, 1, 4], 0:42].reshape(-1, 21, 2)).reshape(-1, 42)
        np.testing.assert_allclose(poses, expected, rtol=1e-5)  # upstream +-0.5 normalisation
        np.testing.assert_array_equal(segments, [0, 0, 0])      # the signing hand's own track
        np.testing.assert_array_equal(positions, [0., 1., 2.])  # arange: gaps invisible to teacher
        inverse = s.student_rows(rows, length=6)
        np.testing.assert_array_equal(inverse, [0, 1, -1, -1, 2, -1])  # -1: no token on that row

    def test_rebased_positions_keep_the_gap_that_arange_hides(self):
        s = self.s
        values = packed_values(rows=6, present=(0, 1, 4))
        values[3, 90] = 2 / 30  # a doubled frame delta before the third detection
        _, _, positions, rows = s.teacher_tokens(values, 'compressed', positions='rebased')
        expected = s.timeline_indices(values)[rows] - s.timeline_indices(values)[rows[0]]
        np.testing.assert_allclose(positions, [0., 1., 5.], rtol=1e-4)   # doubled delta widens it
        np.testing.assert_allclose(positions, expected)

    def test_gaps_representation_keeps_every_row_and_hand_identity(self):
        s = self.s
        values = packed_values(rows=6, present=(0, 1, 4))
        values[3, 90] = 2 / 30
        poses, segments, positions, rows = s.teacher_tokens(values, 'gaps', channels=91,
                                                            positions='rebased')
        np.testing.assert_array_equal(rows, np.arange(6))       # every packed row is a token
        np.testing.assert_allclose(poses, values[:, :91])
        np.testing.assert_array_equal(segments, [0, 0, 2, 2, 0, 2])  # 2 = no detection at all
        np.testing.assert_allclose(positions, [0, 1, 2, 4, 5, 6])    # gap preserved, rebased to 0

    def test_hand_is_whichever_has_more_detections_and_uses_its_own_slot(self):
        s = self.s
        values = packed_values(rows=5, present=(0,), left=(1, 3))
        poses, _, _, rows = s.teacher_tokens(values, 'compressed')
        np.testing.assert_array_equal(rows, [1, 3])             # left wins
        nonzero = np.flatnonzero(values[1, 45:87])
        self.assertTrue(len(nonzero))                            # left slot carries the shape
        self.assertFalse(np.any(values[1, 0:42]))

    def test_padded_slots_use_upstream_sentinels_and_never_become_rows(self):
        s = self.s
        sets = [(np.ones((3, 42), dtype=np.float32), np.zeros(3, dtype=np.int64),
                 np.arange(3, dtype=np.float64), np.array([0, 2, 5])),
                (np.ones((1, 42), dtype=np.float32), np.zeros(1, dtype=np.int64),
                 np.zeros(1, dtype=np.float64), np.array([1]))]
        poses, segments, positions, rows = s.padded_teacher(sets, 42)
        self.assertEqual(poses.shape, (2, 3, 42))
        np.testing.assert_array_equal(poses[1, 1:], -2)          # upstream pose sentinel
        np.testing.assert_array_equal(positions[1, 1:], 1024)    # upstream position sentinel
        self.assertEqual([list(row) for row in rows], [[0, 2, 5], [1]])  # rows live outside pads
        with self.assertRaises(ValueError):
            s.padded_teacher(sets, 91)                           # wrong channel width caught

    def test_padded_slots_for_a_recognizer_teacher_use_the_maskable_segment(self):
        # our model/offline_recognizer masks padding only via segments==PAD_SEG(32), unlike the
        # released model which masks by the -2 pose sentinel; the gaps teacher runs on VPS would
        # otherwise attend to garbage pose tokens from inside every real token's memory.
        s = self.s
        sets = [(np.ones((3, 42), dtype=np.float32), np.zeros(3, dtype=np.int64),
                 np.arange(3, dtype=np.float64), np.array([0, 2, 5])),
                (np.ones((1, 42), dtype=np.float32), np.zeros(1, dtype=np.int64),
                 np.zeros(1, dtype=np.float64), np.array([1]))]
        poses, segments, positions, rows = s.padded_teacher(sets, 42, pad_segment=32)
        np.testing.assert_array_equal(segments[1, 1:], 32)          # pad slots become maskable
        np.testing.assert_array_equal(segments[0], 0)               # real tokens keep their own id
        np.testing.assert_array_equal(segments[1, :1], 0)
        np.testing.assert_array_equal(s.padded_teacher(sets, 42)[1][1, 1:], 0)  # released default

    def test_timeline_positions_are_the_raw_frame_index_the_gaps_recogniser_consumes(self):
        # scripts/train_openfs_recognizer.py trains the gaps recogniser on raw timeline_indices
        # (not rebased); feeding it rebased or arange positions shifts its sinusoidal index.
        s = self.s
        values = packed_values(rows=6, present=(0, 1, 4))
        values[3, 90] = 2 / 30
        _, _, positions, rows = s.teacher_tokens(values, 'gaps', channels=91, positions='timeline')
        np.testing.assert_allclose(positions, s.timeline_indices(values))
        np.testing.assert_allclose(positions, [1., 2., 3., 5., 6., 7.])  # doubled delta widens

    def test_augmented_resampling_glues_teacher_rows_to_the_resampled_timeline(self):
        s = self.s
        rows, probabilities = s.align_rows(np.array([0, 3, 9]), np.ones((3, 2)), np.arange(10), 10)
        np.testing.assert_array_equal(rows, [0, 3, 9])           # identity map changes nothing
        rows, probabilities = s.align_rows(np.array([1, 2]), np.array([[1., 0.], [0., 1.]]),
                                           np.array([0, 1, 5]), 7)
        np.testing.assert_array_equal(rows, [1])                 # both join sampled frame index 1
        np.testing.assert_allclose(probabilities, [[.5, .5]])    # collided rows average
        rows, probabilities = s.align_rows(np.array([], dtype=np.int64), np.zeros((0, 4)),
                                           np.array([0, 4]), 10)
        self.assertEqual(len(rows), 0)                           # clip without detections passes
        rows, _ = s.align_rows(np.array([8]), np.ones((1, 2)), np.array([0, 4]), 10)
        self.assertEqual(rows.max(initial=0), 1)                 # past the last sample: nearest end

    def test_class_matrix_maps_all_61_head_classes_into_student_space(self):
        s = self.s
        matrix = s.class_matrix()
        self.assertEqual(matrix.shape, (s.TEACHER_CLASSES, len(s.VOCAB)))  # head emits 61
        self.assertEqual([row.sum() for row in matrix[:len(s.UPSTREAM_CHARS)]],
                         [1.] * len(s.UPSTREAM_CHARS))           # every released class maps once
        self.assertEqual(matrix[0].argmax(), s.VOCAB.index(''))  # blank phi -> packed blank token
        letter_row = s.UPSTREAM_CHARS.index('a')
        self.assertEqual(matrix[letter_row].argmax(), s.VOCAB.index('A'))  # upstream 'a' -> 'A'
        np.testing.assert_array_equal(matrix[len(s.UPSTREAM_CHARS):], 0)  # class 60 drops out
        posterior = np.full((2, s.TEACHER_CLASSES), 1 / s.TEACHER_CLASSES)
        mapped = posterior @ matrix
        self.assertAlmostEqual(float(mapped.sum()), 2 * len(s.UPSTREAM_CHARS) / s.TEACHER_CLASSES)


class TeacherPositions(unittest.TestCase):
    """--teacher-positions auto must read the index the teacher was actually trained on; an
    unresolved 'auto' leaking into teacher_tokens silently selects the timeline branch."""

    def test_auto_reads_the_trained_index_off_the_checkpoint(self):
        s = load_script()
        self.assertEqual(s.resolve_teacher_positions('upstream', None, 'auto'), 'arange')
        self.assertEqual(s.resolve_teacher_positions('recognizer', 'gaps', 'auto'), 'timeline')
        self.assertEqual(s.resolve_teacher_positions('recognizer', 'gaps42', 'auto'), 'rebased')
        self.assertEqual(s.resolve_teacher_positions('recognizer', 'gaps91', 'auto'), 'rebased')
        self.assertEqual(s.resolve_teacher_positions('recognizer', 'compressed', 'auto'), 'arange')
        self.assertEqual(s.resolve_teacher_positions('recognizer', 'control42', 'auto'), 'arange')
        self.assertEqual(s.resolve_teacher_positions('recognizer', None, 'auto'), 'arange')

    def test_an_explicit_choice_is_never_overridden(self):
        s = load_script()
        for requested in ('arange', 'rebased', 'timeline'):
            self.assertEqual(s.resolve_teacher_positions('recognizer', 'gaps', requested),
                             requested)
            self.assertEqual(s.resolve_teacher_positions('upstream', None, requested), requested)


class TeacherPayload(unittest.TestCase):
    """The app-model guard keys on task, not on the contract key: our own retrained OpenFS-recipe
    recognisers carry the contract (scripts/train_openfs_recognizer.py) and are valid teachers."""

    def _args(self, teacher):
        return argparse.Namespace(teacher=teacher, openfs_source=None, trusted_teacher=False,
                                  teacher_input_representation='gaps', teacher_positions='auto',
                                  teacher_channels='auto',
                                  frame_classifier=teacher.parent / 'no-frame-classifier',
                                  device='cpu')

    def test_recognizer_checkpoints_carrying_the_contract_are_teachers_not_app_models(self):
        s = load_script()
        sys.path.insert(0, str(REPO))
        from model.offline_recognizer import Recognizer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recognizer.pt'
            network = Recognizer(input_dim=91, d_model=8, heads=2, layers=1, feedforward=16)
            torch.save(dict(contract=s.CONTRACT, task='recognizer', representation='gaps',
                            architecture=dict(input_dim=91, d_model=8, heads=2, layers=1,
                                              feedforward=16, dropout=0.1),
                            state_dict=network.state_dict()), path)
            with self.assertRaises(ValueError) as caught:                    # stops at the classifier
                s.load_teacher(self._args(path))
        message = str(caught.exception)
        self.assertNotIn('app model', message)
        self.assertIn('frame classifier checkpoint is missing', message)    # the task guard passed

    def test_a_file_teacher_without_openfs_source_names_the_classifier_gap(self):
        # the released frame-classifier constructor is imported even for a file-path teacher,
        # which used to raise UnboundLocalError instead of naming the missing --openfs-source
        s = load_script()
        sys.path.insert(0, str(REPO))
        from model.offline_recognizer import Recognizer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recognizer.pt'
            network = Recognizer(input_dim=91, d_model=8, heads=2, layers=1, feedforward=16)
            torch.save(dict(contract=s.CONTRACT, task='recognizer', representation='gaps',
                            architecture=dict(input_dim=91, d_model=8, heads=2, layers=1,
                                              feedforward=16, dropout=0.1),
                            state_dict=network.state_dict()), path)
            classifier = Path(directory) / 'frame_classifier'
            classifier.mkdir()
            (classifier / 'best_model.pth').write_bytes(b'not a checkpoint')     # weights present
            args = self._args(path)
            args.frame_classifier = classifier
            with self.assertRaises(ValueError) as caught:
                s.load_teacher(args)
        self.assertIn('--openfs-source', str(caught.exception))

    def test_app_sequence_checkpoints_are_still_refused_as_teachers(self):
        s = load_script()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sequence.pt'
            torch.save(dict(contract=s.CONTRACT, task='sequence'), path)
            with self.assertRaises(ValueError) as caught:
                s.load_teacher(self._args(path))
        self.assertIn('app model', str(caught.exception))


class TeacherCache(unittest.TestCase):
    """The longer-rf runs rerun on a cached teacher pass, so the ragged records must round-trip."""

    def test_ragged_posterior_records_survive_the_cache_round_trip(self):
        s = load_script()
        with tempfile.TemporaryDirectory() as directory:
            data = {'clips': [{'name': f'c{i}'} for i in range(3)]}

            def fake(teacher, data, order):   # rows length == posterior rows: np.array() would
                return ({int(i): (np.arange(4 + 2 * int(i), dtype=np.int32),            # crash
                                  np.full((4 + 2 * int(i), 5), int(i), np.float16))      # and
                         for i in (2, 0, 1)}, {'clips': 3})  # insertion order != sorted order:
                # a writer that pairs sorted names with insertion-ordered records mis-glues clips

            args = argparse.Namespace(data=Path(directory) / 'pack',
                                      teacher_cache=Path(directory) / 'teacher.npz')
            teacher = {'provenance': {'paths': {'teacher': 'weights'}}, 'representation': 'compressed',
                       'positions': 'arange', 'channels': 42}
            original = s.teacher_posteriors
            s.teacher_posteriors = fake
            try:
                posteriors, stats = s.teacher_posteriors_cached(args, teacher, data, [0, 1, 2])
                self.assertEqual(stats['cache'], 'miss')
                cached, stats = s.teacher_posteriors_cached(args, teacher, data, [0, 1, 2])
                self.assertEqual(stats['cache'], 'hit')
                self.assertEqual(sorted(posteriors), sorted(cached))
                for index in posteriors:
                    np.testing.assert_array_equal(posteriors[index][0], cached[index][0])
                    np.testing.assert_allclose(posteriors[index][1], cached[index][1])
                _, stats = s.teacher_posteriors_cached(args, teacher, data, [0, 1])  # other subset
                self.assertEqual(stats['cache'], 'miss')
            finally:
                s.teacher_posteriors = original


class ReceptiveField(unittest.TestCase):
    """No report row may claim a latency the built network does not actually have."""

    @classmethod
    def setUpClass(cls):
        cls.s = load_script()

    def test_stacks_hit_the_request_without_wasting_frames(self):
        for kernel in (3, 5):
            for requested in (0, 1, 2, 60, 61, 62, 200, 600, 900, 2000):
                stack = self.s.dilation_stack(requested, kernel)
                field = self.s.receptive_field(stack, kernel)
                self.assertGreaterEqual(field, requested, (stack, kernel, requested))
                self.assertLess(field - requested, kernel - 1,
                                (stack, kernel, requested))     # smallest reachable field >= r
                if len(stack) > 1:                              # dropping the fill undershoots
                    self.assertLess(self.s.receptive_field(stack[:-1], kernel), requested)
                for position, dilation in enumerate(stack[:-1]):
                    self.assertEqual(dilation, 2 ** position, stack)  # doubles while it fits

    def test_known_stacks(self):
        self.assertEqual(self.s.dilation_stack(61), (1, 2, 4, 8))   # exactly the shipped model
        self.assertEqual(self.s.receptive_field((1, 2, 4, 8)), 61)
        self.assertEqual(self.s.dilation_stack(41), (1, 2, 4, 3))
        self.assertEqual(self.s.dilation_stack(200), (1, 2, 4, 8, 16, 19))
        self.assertEqual(self.s.dilation_stack(1), ())              # field 1 = no stack at all
        self.assertEqual(self.s.receptive_field(self.s.dilation_stack(0)), 1)

    def test_field_is_measured_from_the_built_network(self):
        s = self.s
        net = s.FingerspellingNet(hidden=16, dilations=s.dilation_stack(301))
        self.assertEqual(s.model_receptive_field(net), 301)
        self.assertEqual(s.model_receptive_field(s.FingerspellingNet(hidden=16)), 61)
        self.assertEqual(s.FingerspellingNet(hidden=16).dilations, (1, 2, 4, 8))  # shipped stack
        net = s.FingerspellingNet(hidden=16, dilations=s.dilation_stack(200))
        self.assertEqual([block.conv.dilation[0] for block in net.blocks], [1, 2, 4, 8, 16, 19])
        self.assertEqual(s.model_receptive_field(net), 201)

    def test_latency_columns_convert_on_the_app_grid(self):
        latency = self.s.app_latency((1, 2, 4, 8))
        self.assertEqual(latency['receptive_field_frames'], 61)
        self.assertEqual(latency['algorithmic_latency_frames'], 60)
        self.assertEqual(latency['algorithmic_latency_ms_at_10hz'], 6000.0)
        self.assertEqual(latency['algorithmic_latency_ms_at_15hz'], 4000.0)
        self.assertEqual(latency['history_ms_on_model_grid'], 2000.0)  # 60 frames on the 30 Hz grid
        self.assertEqual(latency['buffered_model_frames_required'], 61)

    def test_checkpoint_round_trip_remembers_the_stack(self):
        s = self.s
        net = s.FingerspellingNet(hidden=16, dilations=s.dilation_stack(200))
        payload = dict(contract=s.CONTRACT, task='sequence', labels=s.VOCAB, hidden=16,
                       dilations=list(net.dilations),
                       receptive_field_frames=s.model_receptive_field(net),
                       state_dict=net.state_dict(), experimental=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'student.pt'
            torch.save(payload, path)
            loaded = s.load_checkpoint(path, 'sequence')
            self.assertEqual(loaded.dilations, net.dilations)
            self.assertEqual([block.conv.dilation[0] for block in loaded.blocks],
                             [1, 2, 4, 8, 16, 19])
            legacy = dict(payload)
            legacy.pop('dilations')                               # pre-sweep checkpoints
            legacy['state_dict'] = s.FingerspellingNet(hidden=16).state_dict()  # shipped 4-stack
            torch.save(legacy, path)
            self.assertEqual(s.load_checkpoint(path, 'sequence').dilations, (1, 2, 4, 8))


class CausalChain(unittest.TestCase):
    """Every swept configuration must answer frame t from frames <= t only."""

    @classmethod
    def setUpClass(cls):
        cls.s = load_script()

    def assert_causal(self, net, length=140, cut=100):
        torch.manual_seed(3)
        net.eval()
        with torch.no_grad():
            frames = torch.randn(2, length, 91)
            full = net(frames).numpy()
            prefix = net(frames[:, :cut + 1]).numpy()
            changed = frames.clone()
            changed[:, cut + 1:] += 5.0
            later = net(changed).numpy()
        # conv blocking over different sequence lengths gives tiny float drift; causality is exact
        # structurally and equal within float tolerance (same convention as tests/test_neural.py).
        np.testing.assert_allclose(full[:, :cut + 1], prefix[:, :cut + 1], atol=1e-6)
        np.testing.assert_allclose(full[:, :cut + 1], later[:, :cut + 1], atol=1e-6)

    def test_shipped_and_every_swept_stack_is_causal(self):
        s = self.s
        for requested in (1, 61, 200, 600, 1200):
            net = s.FingerspellingNet(hidden=24, dilations=s.dilation_stack(requested))
            with self.subTest(receptive_field=s.model_receptive_field(net)):
                self.assert_causal(net)

    def test_measured_field_matches_the_formula_for_built_models(self):
        s = self.s
        for requested in (61, 200, 600):
            net = s.FingerspellingNet(hidden=24, dilations=s.dilation_stack(requested))
            self.assertEqual(s.model_receptive_field(net),
                             s.receptive_field(net.dilations))


class DistillationLosses(unittest.TestCase):
    """KL and frame CE must carry gradient only through supervised frames."""

    @classmethod
    def setUpClass(cls):
        cls.s = load_script()

    def test_soft_kl_flows_only_through_masked_frames(self):
        s = self.s
        logits = torch.randn(2, 5, len(s.VOCAB), requires_grad=True)
        targets = torch.softmax(torch.randn(2, 5, len(s.VOCAB)) * 4, dim=-1)
        mask = torch.tensor([[True, True, False, False, False],
                             [False, False, True, False, False]])
        loss = s.soft_kl(logits, targets, mask, 1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(logits.grad[0, :2].abs().sum().item(), 0)
        self.assertGreater(logits.grad[1, 2].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 2:].abs().sum().item(), 0)  # pads and blanks contribute 0

    def test_soft_kl_matches_the_temperated_formula(self):
        s = self.s
        from torch.nn import functional as F
        logits = torch.randn(3, 4, len(s.VOCAB))
        targets = torch.softmax(torch.randn(3, 4, len(s.VOCAB)) * 3, dim=-1)
        mask = torch.ones(3, 4, dtype=torch.bool)
        temperature = 2.0
        expected = temperature ** 2 * F.kl_div(
            F.log_softmax(logits.reshape(-1, len(s.VOCAB)).float() / temperature, dim=-1),
            targets.reshape(-1, len(s.VOCAB)), reduction='batchmean')   # masked frames flatten first
        self.assertAlmostEqual(s.soft_kl(logits, targets, mask, temperature).item(),
                               expected.item(), places=5)

    def test_matching_posteriors_and_empty_masks_give_zero_kl(self):
        s = self.s
        logits = torch.randn(2, 3, len(s.VOCAB))
        posterior = torch.softmax(logits, dim=-1)                    # identical distributions
        self.assertAlmostEqual(s.soft_kl(logits, posterior,
                                         torch.ones(2, 3, dtype=torch.bool), 1.0).item(), 0.0, places=5)
        empty = s.soft_kl(torch.randn(2, 3, len(s.VOCAB), requires_grad=True), posterior,
                         torch.zeros(2, 3, dtype=torch.bool), 1.0)
        self.assertEqual(float(empty), 0.0)
        empty.backward()                                        # graph stays alive for DDP safety

    def test_posterior_batch_masks_pads_and_blank_argmax_frames(self):
        s = self.s
        def one_hot(token):
            value = torch.zeros(len(s.VOCAB))
            value[token] = 1
            return value
        clip_a = torch.stack([one_hot(1), one_hot(0), one_hot(2)])   # A, blank, B
        clip_b = torch.stack([one_hot(1)])
        values, used = s.posterior_batch([clip_a, clip_b, None, None],
                                         [[0, 1, 2], [0], [], []],
                                         torch.tensor([3, 1, 0, 0]), 'cpu', False)
        self.assertEqual(values.shape, (4, 3, len(s.VOCAB)))
        np.testing.assert_allclose(values[0, 0].numpy(), one_hot(1).numpy())
        np.testing.assert_allclose(values[0, 2].numpy(), one_hot(2).numpy())
        self.assertEqual(used.numpy().tolist(),
                         [[True, False, True],        # blank-argmax frame masked by default
                          [True, False, False],       # frames past the clip length stay pads
                          [False, False, False],      # clips without posteriors contribute nothing
                          [False, False, False]])
        values, used = s.posterior_batch([clip_a, clip_b, None, None],
                                         [[0, 1, 2], [0], [], []],
                                         torch.tensor([3, 1, 0, 0]), 'cpu', True)
        self.assertTrue(bool(used[0, 1]))                        # --distillation-blank widens in
        self.assertFalse(bool(used[1, 1]))                       # ...never onto a pad slot

    def test_frame_labels_follow_the_refined_to_pack_to_handshape_chain(self):
        s = self.s
        rows_per_clip = 5
        clips = [dict(name=name, start=index * rows_per_clip,
                      end=(index + 1) * rows_per_clip, text='AB')
                 for index, name in enumerate(('refined-clip', 'packed-clip', 'hand-clip',
                                               'bare-clip'))]
        aux = np.full(len(clips) * rows_per_clip, -100, dtype=np.int16)
        aux[5:10] = [s.VOCAB.index('A')] * 3 + [-100] * 2        # the pack's own per-frame targets
        data = dict(clips=clips, aux=aux, x=np.zeros((len(aux), 91), dtype=np.float32))
        refined = {'refined-clip': (np.array([0, 1, 2]),
                                    np.array([s.UPSTREAM_CHARS.index('a'),
                                              s.UPSTREAM_CHARS.index('b'), 0]), np.array([0] * 3))}
        handshape = dict(clips=np.array([2, 2]), frames=np.array([0, 2]), labels=np.array([1, 0]))
        labels, info = s.frame_labels(data, refined, handshape)
        origin = info['frame_label_clips']
        self.assertEqual(origin, dict(refined=1, pack_targets=1, handshape_npz=1, none=1))
        self.assertTrue(info['uses_teacher_labels'])             # refined/handshape labels are model-derived
        self.assertEqual(labels[0:5].tolist(),
                         [s.VOCAB.index('A'), s.VOCAB.index('B'), 0, 0, 0])  # upstream a,b; blank
        self.assertEqual(labels[5:10].tolist(),
                         [s.VOCAB.index('A')] * 3 + [-100] * 2)  # pack fallback keeps -100 holes
        self.assertEqual(labels[10:15].tolist()[0], s.VOCAB.index('A'))  # handshape 'A' at frame 0
        self.assertEqual(labels[12], 0)                          # handshape 'nothing'
        self.assertEqual(labels[13:15].tolist(), [-100, -100])   # uncovered frames stay unsupervised
        self.assertEqual(info['frame_label_letter_rows'], 6)     # 2 refined + 3 pack + 1 handshape
        self.assertEqual(info['frame_label_unlabelled_rows'], 10)
        outside = {'refined-clip': (np.array([9]), np.array([34]), np.array([0]))}
        with self.assertRaises(ValueError):                       # an index off the end is corruption
            s.frame_labels(data, outside, None)

    def test_batches_keep_frame_labels_and_posteriors_on_the_resampled_timeline(self):
        s = self.s
        clips = [dict(name=f'clip{index}', start=index * 20, end=(index + 1) * 20, text='AB')
                 for index in range(3)]
        data = dict(clips=clips, aux=np.zeros(60, dtype=np.int16),
                    x=packed_values(rows=60, present=tuple(range(60))))
        labels, _ = s.frame_labels(data, {}, None)
        posteriors = {0: (np.array([2, 9, 17], dtype=np.int32),
                          np.tile([0., 1., 0., 0.], (3, 1)).astype(np.float16))}
        import random
        random.seed(0)
        got = list(s.batches(data, 3, labels, posteriors, augment=True, order=[0, 1, 2],
                             distillation=True, seed=0))
        self.assertTrue(got)
        for frames, _, lengths, _, frame_label_rows, posterior_rows, posterior_values, _ in got:
            self.assertEqual(frames.shape[0], len(lengths))
            for row, length in enumerate(lengths.tolist()):
                self.assertLessEqual(length, frames.shape[1])
                unsupervised = int((frame_label_rows[row, length:] == -100).all())
                self.assertEqual(unsupervised, 1)                # padded CE slots stay ignored
                if posterior_rows[row] is not None:
                    self.assertLess(int(posterior_rows[row].max()), length)  # glued to real frames
                    self.assertGreater(len(posterior_values[row]), 0)
                    self.assertEqual(posterior_values[row].sum(-1).max().item(), 1.0)  # still probs


class RuntimeMetrics(unittest.TestCase):
    """Wall-clock latency comes only from scripts/benchmark_neural.py; the trainer must not time."""

    @classmethod
    def setUpClass(cls):
        cls.s = load_script()

    def test_benchmark_block_delegates_to_the_existing_harness(self):
        source = (REPO / 'scripts' / 'train_streaming_student.py').read_text(encoding='utf-8')
        block = source.split('def benchmark_block')[1].split('\ndef ')[0]
        self.assertIn('benchmark_neural.py', block)
        self.assertIn("'--sequence'", block)
        self.assertNotIn('perf_counter', block)                  # it never times anything itself

    def test_runtime_and_metrics_stay_teacher_free(self):
        # the student path (architecture, runtime, metrics) must never reach the research teacher;
        # model/openfs_refiner.py is the separate, pre-existing app refiner and is not in scope.
        neural = (REPO / 'model' / 'neural.py').read_text(encoding='utf-8')
        self.assertNotIn('openfs', neural.lower())
        self.assertNotIn('lib', neural)
        for path in sorted((REPO / 'model').glob('*.py')):
            text = path.read_text(encoding='utf-8')
            self.assertNotIn('from scripts.', text, path)        # trainers stay out of runtime
            self.assertNotIn('import scripts.', text, path)
        source = (REPO / 'scripts' / 'train_streaming_student.py').read_text(encoding='utf-8')
        for forbidden in ('import finetune_openfs_transfer', 'from scripts.finetune_openfs_transfer',
                          'importlib.import_module(\'scripts.finetune_openfs_transfer'):
            self.assertNotIn(forbidden, source)                  # other agents' script, never ours
        self.assertIs(self.s.ErrorTally,
                      __import__('model.recognition_metrics', fromlist=['ErrorTally']).ErrorTally)

    def test_harness_measures_a_student_through_the_live_runtime(self):
        handshape = REPO / 'data' / 'handshape.pt'
        if not handshape.is_file():     # the live runtime also needs the shipped handshape head
            self.skipTest(f'{handshape} is not present in this checkout')
        s = self.s
        with tempfile.TemporaryDirectory() as directory:
            pack = write_pack(Path(directory) / 'pack', splits=(('train', 2, 30),
                                                                ('validation', 2, 30),
                                                                ('test', 2, 40)))
            net = s.FingerspellingNet(hidden=16, dilations=s.dilation_stack(41))
            checkpoint = Path(directory) / 'student-rf41.pt'
            torch.save(dict(contract=s.CONTRACT, task='sequence', labels=s.VOCAB, hidden=16,
                            dilations=list(net.dilations), receptive_field_frames=41,
                            state_dict=net.state_dict(), metrics={}, epoch=1, seed=0,
                            dataset=str(pack), experimental=True), checkpoint)
            args = argparse.Namespace(data=pack, packed_generation=None, handshape=handshape,
                                      benchmark_threads=[1], benchmark_frames=[12])
            with quiet():
                measured = s.benchmark_block(args, checkpoint)
        self.assertEqual(measured['source'], 'scripts/benchmark_neural.py')
        self.assertEqual(len(measured['measurements']), 1)
        row = measured['measurements'][0]
        for key in ('threads', 'segment_frames', 'p50_ms', 'p95_ms',
                    'tail100_p50_ms', 'tail100_p95_ms'):
            self.assertIn(key, row)
        self.assertEqual(row['segment_frames'], 12)
        self.assertGreater(row['p50_ms'], 0)
        self.assertLess(row['p50_ms'], 1000)                      # a hidden=16 student is not absurd
        self.assertEqual(measured['fastest_p50_ms_by_threads'], {1: row['p50_ms']})


class StudyPipeline(unittest.TestCase):
    """Whole-script runs on synthetic packed clips: naming, provenance, report shape."""

    @classmethod
    def setUpClass(cls):
        cls.s = load_script()

    def run_main(self, argv):
        old = sys.argv
        sys.argv = ['train_streaming_student.py'] + argv
        try:
            with quiet() as out:
                self.s.main()
            return out.getvalue()
        finally:
            sys.argv = old

    def test_trains_records_provenance_and_falls_back_without_any_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pack = write_pack(directory / 'pack')
            empty = directory / 'no-refined'
            empty.mkdir()
            self.run_main(['--data', str(pack), '--refined-labels', str(empty), '--no-distill',
                           '--device', 'cpu', '--epochs', '1', '--patience', '1',
                           '--hidden', '16', '--receptive-frames', '41', '--batch-size', '4',
                           '--output', str(directory / 'student.pt'),
                           '--report', str(directory / 'report.json')])
            report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            block, = report['configs']
            self.assertEqual(block['config'], 'rf41')
            self.assertEqual(block['dilations'], [1, 2, 4, 3])
            self.assertEqual(block['receptive_field_frames'], 41)
            self.assertEqual(block['algorithmic_latency_ms_at_15hz'], 2666.7)
            self.assertFalse(block['training']['distillation'])
            self.assertIsNone(report['teacher'])
            self.assertEqual(report['frame_labels']['frame_label_clips'], {'pack_targets': 6})
            checkpoint = directory / 'student-rf41.pt'
            self.assertTrue(checkpoint.is_file())                 # stem-rf<measured field>
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.assertEqual(payload['dilations'], [1, 2, 4, 3])
            self.assertEqual(payload['receptive_field_frames'], 41)
            self.assertFalse(payload['research_only'])            # no teacher touched these weights
            self.assertIsNone(payload['distilled_from'])
            self.assertEqual(block['test']['clips'], 3)
            self.assertGreaterEqual(block['test']['cer'], 0.0)

    def test_distillation_without_a_teacher_or_ablation_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pack = write_pack(directory / 'pack', splits=(('train', 2, 30), ('validation', 2, 30),
                                                          ('test', 2, 30)))
            with self.assertRaises(ValueError) as caught:
                self.run_main(['--data', str(pack), '--device', 'cpu', '--epochs', '1',
                               '--output', str(directory / 'student.pt'),
                               '--report', str(directory / 'report.json')])
            self.assertIn('--teacher', str(caught.exception))

    def test_sweeps_several_fields_and_enforces_the_parameter_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pack = write_pack(directory / 'pack', splits=(('train', 2, 30), ('validation', 2, 30),
                                                          ('test', 2, 30)))
            self.run_main(['--data', str(pack), '--refined-labels', str(directory / 'none'),
                           '--no-distill', '--device', 'cpu', '--epochs', '1',
                           '--hidden', '16', '--receptive-frames', '41', '101', '--batch-size', '2',
                           '--output', str(directory / 'student.pt'),
                           '--report', str(directory / 'report.json')])
            report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            fields = [block['receptive_field_frames'] for block in report['configs']]
            self.assertEqual(fields, [41, 101])
            for field in fields:
                self.assertTrue((directory / f'student-rf{field}.pt').is_file())
            with self.assertRaises(ValueError) as caught:
                self.run_main(['--data', str(pack), '--refined-labels', str(directory / 'none'),
                               '--no-distill', '--device', 'cpu', '--epochs', '1', '--hidden', '192',
                               '--max-parameters', '10000', '--receptive-frames', '600',
                               '--output', str(directory / 'capped.pt'),
                               '--report', str(directory / 'capped.json')])
            self.assertIn('max-parameters', str(caught.exception))

    def _student_checkpoint(self, directory, pack):
        s = self.s
        net = s.FingerspellingNet(hidden=16, dilations=s.dilation_stack(41))
        checkpoint = Path(directory) / 'student-rf41.pt'
        torch.save(dict(contract=s.CONTRACT, task='sequence', labels=s.VOCAB, hidden=16,
                        dilations=list(net.dilations), receptive_field_frames=41,
                        state_dict=net.state_dict(), metrics={}, epoch=1, seed=0,
                        dataset=str(pack), experimental=True), checkpoint)
        return checkpoint

    def test_evaluate_only_mode_scores_a_checkpoint_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = write_pack(Path(directory) / 'pack', splits=(('train', 2, 30),
                                                                ('validation', 2, 30),
                                                                ('test', 3, 34)))
            checkpoint = self._student_checkpoint(directory, pack)
            report = Path(directory) / 'eval.json'
            self.run_main(['--data', str(pack), '--evaluate-checkpoint', str(checkpoint),
                           '--test-limit', '2', '--device', 'cpu', '--report', str(report)])
            document = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(document['mode'], 'evaluate')
            self.assertEqual(document['dilations'], [1, 2, 4, 3])
            self.assertEqual(document['receptive_field_frames'], 41)
            self.assertEqual(document['history_ms_on_model_grid'], 1333.3)
            self.assertEqual(document['test']['clips_evaluated'], 2)
            self.assertEqual(document['test']['split'], 'test')
            self.assertNotIn('examples', document['test'])          # reports stay free of transcripts

    def test_summary_merges_rows_and_reads_the_field_from_a_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pack = write_pack(directory / 'pack', splits=(('train', 2, 30), ('validation', 2, 30),
                                                          ('test', 2, 30)))
            checkpoint = self._student_checkpoint(directory, pack)
            legacy = dict(json.loads(json.dumps(dict(checkpoint=str(checkpoint), parameters=26653,
                                                     test=dict(cer=0.5, exact_sequence_accuracy=0.0,
                                                                 characters=6, clips=2,
                                                                 decoder='greedy CTC')))))
            legacy_report = directory / 'legacy.json'
            legacy_report.write_text(json.dumps(legacy), encoding='utf-8')
            sweep_report = directory / 'sweep.json'
            sweep_report.write_text(json.dumps(dict(configs=[dict(
                config='rf61', checkpoint=str(checkpoint), parameters=835453,
                receptive_field_frames=61, algorithmic_latency_ms_at_15hz=4000.0,
                test=dict(cer=0.13, exact_sequence_accuracy=0.18, characters=99, clips=9),
                training=dict(seconds_per_epoch=10.0, peak_vram_mib=512, distillation=True))])),
                encoding='utf-8')
            benchmark = directory / 'bench.json'
            benchmark.write_text(json.dumps(dict(measurements=[
                dict(threads=4, segment_frames=300, p50_ms=6.5, p95_ms=11.0,
                     tail100_p50_ms=6.4, tail100_p95_ms=9.9)])), encoding='utf-8')
            self.run_main(['--data', str(pack), '--device', 'cpu',
                           '--summarize', f'baseline:{legacy_report}:{benchmark}',
                           '--summarize', f'sweep:{sweep_report}',
                           '--report', str(directory / 'merged.json')])
            merged = json.loads((directory / 'merged.json').read_text(encoding='utf-8'))
            self.assertEqual([row['config'] for row in merged['rows']], ['baseline', 'sweep'])
            first, second = merged['rows']
            self.assertEqual(first['receptive_field_frames'], 41)  # measured off the checkpoint
            self.assertEqual(first['arrival_p50_ms_best'], 6.5)
            self.assertEqual(first['algorithmic_latency_ms_at_15hz'], 2666.7)
            self.assertEqual(second['receptive_field_frames'], 61)
            self.assertEqual(second['cer'], 0.13)
            self.assertIsNone(second.get('arrival_p50_ms_best'))   # no benchmark file given


class PublishedMeasurements(unittest.TestCase):
    """The 2026-10-19 study artifacts in docs/ are the answer to the accuracy-versus-latency
    question, so they are checked against themselves: rows internally consistent, the
    fixed-latency pair actually fixed-latency, and the prose table agreeing with the JSON."""

    METRICS = REPO / 'docs' / 'streaming-latency-metrics-2026-10-19.json'
    NOTES = REPO / 'docs' / 'streaming-latency-2026-10-19.md'

    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(cls.METRICS.read_text(encoding='utf-8'))
        cls.rows = {row['config']: row for row in cls.document['rows']}

    def test_every_row_is_arithmetic_that_adds_up(self):
        self.assertEqual(len(self.document['rows']), 5)
        for config, row in self.rows.items():
            with self.subTest(config=config):
                errors = row['deletions'] + row['substitutions'] + row['insertions']
                self.assertAlmostEqual(row['cer'] * row['characters'], errors, delta=1.0)
                self.assertAlmostEqual(row['letter_accuracy'], 1 - row['cer'], places=9)
                self.assertAlmostEqual(row['deletion_rate'] * row['characters'],
                                       row['deletions'], delta=1.0)
                self.assertEqual(row['clips'], 12521)               # the packed FSboard test split
                self.assertEqual(row['characters'], 221100)
                self.assertEqual(row['buffered_model_frames_required'],
                                 row['receptive_field_frames'])
                self.assertIsNotNone(row['arrival_p50_ms_best'])    # every row is priced

    def test_the_fixed_latency_pair_differs_only_by_the_kl_term(self):
        distilled, plain = self.rows['rf61-distill'], self.rows['rf61-no-distill']
        for field in ('parameters', 'dilations', 'receptive_field_frames',
                      'buffered_model_frames_required', 'algorithmic_latency_ms_at_15hz'):
            self.assertEqual(distilled[field], plain[field], field)
        self.assertTrue(distilled['distillation'])
        self.assertFalse(plain['distillation'])
        # same architecture and shape, so per-arrival CPU cost cannot differ by construction
        self.assertLess(abs(distilled['arrival_p50_ms_best'] - plain['arrival_p50_ms_best']), .5)
        self.assertLess(plain['cer'], distilled['cer'])             # measured: distillation cost CER

    def test_longer_history_costs_time_and_bought_nothing_in_accuracy(self):
        arms = [self.rows[name] for name in ('rf61-distill', 'rf201-distill', 'rf601-distill')]
        buffered = [row['buffered_model_frames_required'] for row in arms]
        self.assertEqual(buffered, [61, 201, 601])
        self.assertEqual([row['arrival_p50_ms_best'] for row in arms],
                         sorted(row['arrival_p50_ms_best'] for row in arms))
        for row, frames in zip(arms, buffered):
            self.assertAlmostEqual(row['history_ms_on_model_grid'], frames * 1000 / 30, delta=50)
        widest, narrowest = arms[-1], arms[0]
        self.assertLess(abs(widest['cer'] - narrowest['cer']), .01)  # flat across a 10x field

    def test_the_writeup_table_matches_the_machine_readable_rows(self):
        notes = self.NOTES.read_text(encoding='utf-8')
        for config, row in self.rows.items():
            with self.subTest(config=config):
                self.assertIn(f"{row['cer'] * 100:.2f}", notes)
                self.assertIn(f"{row['arrival_p50_ms_best']:.2f}", notes)


if __name__ == '__main__':
    unittest.main()
