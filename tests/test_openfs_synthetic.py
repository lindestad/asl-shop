"""Synthetic-data pipeline guards: pose normalisation round trip, pack contract and provenance,
the model-derived quality filter, the trusted-source gate, and the app-runtime import firewall.

Fixtures are tiny synthetic protocol checks, never recognition-accuracy evidence. The one place
real data is read is the pose-normalisation round trip against packed FSboard rows, and that test
skips when the pack is not in the checkout.
"""
import json
import os
from pathlib import Path
import shutil
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

from model.neural import CONTRACT, HANDSHAPES, VOCAB
from scripts.prepare_openfs_training import UPSTREAM_CHARS, encode_poses

PACK = ROOT / 'data/openfs-packed'
GENERATOR = ROOT / 'scripts/generate_openfs_synthetic.py'
PACKER = ROOT / 'scripts/pack_openfs_synthetic.py'


def normalized_box(pose):
    """The repo's parity copy of the upstream +-0.5 frame normalisation (per-axis divisor)."""
    from scripts.train_openfs_recognizer import normalized_box as upstream
    return upstream(pose)


def isotropic_box(pose):
    """The wrong inverse: one max-abs divisor for both axes (the fsboard_sessions bug class)."""
    values = pose.copy()
    values -= 0.5 * (values.min(axis=1, keepdims=True) + values.max(axis=1, keepdims=True))
    return values / (np.abs(values).max(axis=(1, 2), keepdims=True) + 1e-6) * .5


def sampled_pose(seed, frames):
    """A 21x3 clip in the generator's own output convention: bbox-centred, +-0.5 per axis.

    x and y extents are deliberately unequal, because a normalisation bug that divides both axes by
    one scale is invisible on a fixture whose extents happen to match.
    """
    rng = np.random.default_rng(seed)
    layout = np.zeros((21, 3), dtype=np.float32)
    layout[:, 0] = np.linspace(-.4, .5, 21)                     # wide in x
    layout[:, 1] = np.linspace(-.25, .2, 21)                    # narrow in y
    layout[:, 2] = np.linspace(-.1, .1, 21)
    layout[0] = 0.
    layout[9] = (0., .3, 0.)                                    # wrist to middle knuckle
    pose = np.stack([layout + rng.normal(0, .01, layout.shape) + shift * np.array([.02, .015, 0])
                     for shift in range(frames)], axis=0).astype(np.float32)
    pose[..., :2] = normalized_box(pose[..., :2])
    return pose


def make_generation(directory, clips, *, generator_sha='ab' * 32, conditioner_sha='cd' * 32):
    """Write a generation directory shaped exactly like scripts/generate_openfs_synthetic.py's output."""
    directory = Path(directory)
    (directory / 'words').mkdir(parents=True, exist_ok=True)
    poses, labels, records, offset = [], [], [], 0
    for position, clip in enumerate(clips):
        pose = sampled_pose(clip.get('seed', position), clip['frames'])
        word = clip['word']
        per_letter = max(1, clip['frames'] // len(word))
        sequence = (list(word) * (len(word) + 1))[:clip['frames']]
        poses.append(pose)
        labels.append(np.array([UPSTREAM_CHARS.index(character) for character in sequence], dtype=np.int16))
        records.append(dict(name=f"trial{clip.get('trial', 0)}-{word}", word=word.upper(), set=clip['set'],
                            trial=clip.get('trial', 0), start=offset, end=offset + clip['frames'],
                            transcription=clip.get('transcription', word.upper()),
                            accepted=bool(clip.get('accepted', True)),
                            quality_filter='released recognizer transcription, exact word match required'))
        offset += clip['frames']
    np.save(directory / 'poses.npy', np.concatenate(poses).astype(np.float32))
    np.save(directory / 'frame_labels.npy', np.concatenate(labels))
    (directory / 'clips.json').write_text(json.dumps(records), encoding='utf-8')
    (directory / 'words' / 'generated-union.txt').write_text(''.join(f"{clip['word']}\n" for clip in clips),
                                                             encoding='utf-8')
    (directory / 'provenance.json').write_text(json.dumps(dict(
        kind='openfs-synthetic-generation', research_only=True,
        license=dict(quote='Commercial use of the source code, model checkpoints, or derivative works is '
                           'strictly prohibited', source='data/external/openfs/source/LICENSE'),
        generator=dict(path='data/openfs-checkpoints/generator_df_fsboard', artifact='best_model.pth',
                       artifact_sha256=generator_sha, upstream_commit='cd29c85'),
        conditioner=dict(path='data/openfs-checkpoints/recognizer_fsboard', artifact='best_model.pth',
                         artifact_sha256=conditioner_sha),
        seed=7, trials=1, batch_size=8,
        word_lists=dict(train=dict(words=len(clips), sha256='ef' * 32)),
        sampler=dict(class_name='lib.models.diffusion.Diffusion',
                     sampling_method='lib.models.diffusion.Diffusion.sampling'),
        dataset=dict(class_name='lib.datasets.fingerspelling.WordDataset', use_frame_label_clean=True,
                     alphabet=UPSTREAM_CHARS),
        clips=len(records), frames=offset,
        quality_filter=dict(kind='model-derived quality filter (released recognizer transcription), not a human '
                                  'check', pass_rate=1., accepted_clips=len(records), rejected_clips=0),
    ), indent=2), encoding='utf-8')
    return directory


def run(script, arguments, *, expect_zero=True):
    result = subprocess.run([sys.executable, str(script), *arguments], capture_output=True, text=True, timeout=180)
    if expect_zero and result.returncode:
        raise AssertionError(f'{Path(script).name} failed: {result.stdout}\n{result.stderr}')
    if not expect_zero and not result.returncode:
        raise AssertionError(f'{Path(script).name} unexpectedly succeeded: {result.stdout}')
    return result


class RealPackRoundTrip(unittest.TestCase):
    """The inverse of the packer's shape channels against real packed FSboard rows."""

    @classmethod
    def setUpClass(cls):
        if not (PACK / 'dataset.json').is_file():
            raise unittest.SkipTest(f'{PACK} is not present in this checkout')
        clips = json.loads((PACK / 'validation_clips.json').read_text(encoding='utf-8'))
        frames = np.load(PACK / 'validation_frames.npy', mmap_mode='r')
        rows, tracks, cls.anisotropic = [], [], 0.
        for clip in clips:
            values = np.array(frames[clip['start']:clip['end']], dtype=np.float32)
            detected = values[:, 44] > 0
            if detected.sum() < 8:
                continue
            shape = values[detected][:, :42].reshape(-1, 21, 2)
            wrist = values[detected][:, 42:44]
            rows.append(shape)
            tracks.append(shape + wrist[:, None, :])              # the reconstruct_landmarks inverse
            per_axis_max = np.stack([np.abs(shape - 0.5 * (shape.min(1, keepdims=True) +
                                                           shape.max(1, keepdims=True))
                                           ).max(axis=1) for axis in (0, 1)], -1)
            cls.anisotropic = max(cls.anisotropic, float(np.abs(per_axis_max[:, 0] -
                                                                per_axis_max[:, 1]).max()))
            if len(rows) >= 12:
                break
        cls.rows, cls.tracks = rows, tracks
        if not rows:
            raise unittest.SkipTest('no validation clip in this pack has a long enough right-hand track')

    def test_upstream_normalisation_of_real_rows_is_recovered_by_the_packed_inverse(self):
        """normalize() of reconstructed tracks equals normalized_box() of the stored shape channels."""
        for shape, points in zip(self.rows, self.tracks):
            # the wrist offset is constant per frame and the hand-size divisor is common to both
            # axes, so the two normalisations must land on the same +-0.5 frame (max-abs per axis)
            np.testing.assert_allclose(normalized_box(points), normalized_box(shape), atol=1e-4)
            recovered = normalized_box(shape)
            self.assertAlmostEqual(float(recovered.max()), .5, places=3)
            self.assertAlmostEqual(float(recovered.min()), -.5, places=3)
            self.assertLess(float(recovered.max()), .501)          # exactly +-0.5, not +-0.5 after rescale
        self.assertGreater(self.anisotropic, 1e-3, 'fixture is not anisotropic, so the test would be vacuous')

    def test_real_rows_are_normalised_per_axis_not_isotropically(self):
        """Re-deriving the upstream frame normalisation from packed rows must use the per-axis divisor."""
        self.assertGreater(self.anisotropic, 1e-3, 'fixture is not anisotropic, so the test would be vacuous')
        widest = 0.
        for shape in self.rows[:6]:
            recovered = normalized_box(shape)
            wrong = isotropic_box(shape)                            # one max-abs divisor for both axes
            widest = max(widest, float(np.abs(recovered - wrong).max()))
            # the isotropic inverse keeps the +-0.5 promise on the wide axis and silently shrinks the
            # narrow one, so it still looks plausible while being wrong on most frames
            self.assertGreaterEqual(float(np.abs(recovered[..., 0]).max()), .4999)
        self.assertGreater(widest, .02, 'the isotropic inverse would be undetectable on these rows')

    def test_the_released_normalisation_matches_the_repo_parity_copy(self):
        """train_openfs_recognizer.normalized_box is what upstream normalize() computes."""
        source = os.environ.get('ASL_OPENFS_SOURCE')
        module = Path(source) / 'lib/utils/dataset.py' if source else None
        if not module or not module.is_file():
            self.skipTest('ASL_OPENFS_SOURCE is not set to an OpenFS checkout')
        import importlib.util
        spec = importlib.util.spec_from_file_location('upstream_dataset_parity', module)
        released_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(released_module)
        except ImportError as error:
            self.skipTest(f'the released dataset module needs a dependency this venv lacks ({error})')
        for shape in self.rows[:4]:
            np.testing.assert_allclose(released_module.normalize(shape.astype(np.float64)),
                                       normalized_box(shape), atol=1e-5)


class GenerationGuards(unittest.TestCase):
    """The trusted-source and pinned-provenance gates run before any checkpoint is loaded."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'data/openfs-packed').mkdir(parents=True)
        (self.root / 'data/openfs-packed/train_clips.json').write_text(json.dumps(
            [dict(name='train-a', text='JIM CALLANAN', signer='p1', start=0, end=40),
             dict(name='train-b', text='AZTEC RUN', signer='p1', start=0, end=40),
             dict(name='train-c', text='ZEBRA CROSSING', signer='p1', start=0, end=40)]), encoding='utf-8')
        (self.root / 'data/openfs-packed/validation_clips.json').write_text(json.dumps(
            [dict(name='val-a', text='ZEBRA CROSSING', signer='p2', start=0, end=40)]), encoding='utf-8')

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, output, *extra):
        return ['--words-from-pack', str(self.root / 'data/openfs-packed'), '--train-words', '2',
                '--jz-words', '1', '--output', str(output), *extra]

    def test_refuses_without_the_trusted_source_opt_in(self):
        result = run(GENERATOR, self.arguments(self.root / 'openfs-synthetic-guard'), expect_zero=False)
        self.assertIn('--trusted-source', result.stderr)
        self.assertFalse((self.root / 'openfs-synthetic-guard').exists())

    def test_refuses_without_pinned_checkpoint_digests(self):
        result = run(GENERATOR, self.arguments(self.root / 'openfs-synthetic-guard', '--trusted-source',
                                               '--source', str(ROOT / 'data/external/openfs/source')),
                     expect_zero=False)
        self.assertIn('generator-sha256', result.stderr)
        self.assertFalse((self.root / 'openfs-synthetic-guard').exists())

    def test_refuses_a_digest_that_does_not_match_the_checkpoint(self):
        checkpoint = self.root / 'generator'
        checkpoint.mkdir()
        if torch is not None:
            torch.save(dict(weight=torch.zeros(1)), checkpoint / 'best_model.pth')
        else:
            (checkpoint / 'best_model.pth').write_bytes(b'not-a-checkpoint')
        result = run(GENERATOR, self.arguments(self.root / 'openfs-synthetic-guard', '--trusted-source',
                                               '--source', str(ROOT / 'data/external/openfs/source'),
                                               '--generator', str(checkpoint), '--generator-sha256', 'ab' * 32,
                                               '--conditioner-sha256', 'cd' * 32), expect_zero=False)
        self.assertIn('does not match the pinned', result.stderr + result.stdout)
        self.assertFalse((self.root / 'openfs-synthetic-guard').exists())

    def test_refuses_an_output_directory_outside_the_synthetic_namespace(self):
        result = run(GENERATOR, self.arguments(self.root / 'openfs-packed-accident', '--trusted-source',
                                               '--source', str(ROOT / 'data/external/openfs/source'),
                                               '--generator-sha256', 'ab' * 32,
                                               '--conditioner-sha256', 'cd' * 32), expect_zero=False)
        self.assertIn('openfs-synthetic*', result.stderr + result.stdout)
        self.assertFalse((self.root / 'openfs-packed-accident').exists())

    def test_module_import_never_touches_upstream_code(self):
        probe = subprocess.run([sys.executable, '-c', 'import sys;sys.path.insert(0,%r);'
                                'import scripts.generate_openfs_synthetic as m;'
                                'print(sorted(k for k in sys.modules if k.startswith("lib.")))' % str(ROOT)],
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(probe.stdout.strip(), '[]', probe.stderr)

    def test_upstream_import_refuses_a_directory_that_is_not_a_checkout(self):
        if torch is None:
            self.skipTest('the generation module needs torch')
        from scripts.generate_openfs_synthetic import import_upstream
        with self.assertRaises(ValueError):
            import_upstream(self.root / 'nowhere')


class WordChoice(unittest.TestCase):
    """Word sets come from our own train transcripts and never from held-out text."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.pack = Path(self.temporary.name) / 'openfs-packed'
        self.pack.mkdir()
        (self.pack / 'train_clips.json').write_text(json.dumps([
            dict(name='t1', text='JIM CALLANAN'), dict(name='t2', text='AZTEC RUN'),
            dict(name='t3', text='CALLANAN RUN'), dict(name='t4', text='A'), dict(name='t5', text='J'),
            dict(name='t6', text='ZZZZZZZZZZZZZZZZZZ'), dict(name='t7', text='GUITAR PLAYING'),
            dict(name='t8', text='PYTHON CODE'), dict(name='t9', text='JAVA LATTICE')]), encoding='utf-8')
        (self.pack / 'validation_clips.json').write_text(json.dumps([dict(name='v1', text='AZTEC RUN')]),
                                                         encoding='utf-8')
        (self.pack / 'test_clips.json').write_text(json.dumps([dict(name='e1', text='JIM CALLANAN')]),
                                                   encoding='utf-8')

    def tearDown(self):
        self.temporary.cleanup()

    def test_word_sets_exclude_held_out_tokens_and_cover_movement_letters(self):
        from scripts.generate_openfs_synthetic import derive_word_lists
        chosen, counts, held = derive_word_lists(self.pack, ['train', 'jz', 'letters'],
                                                 train_words=10, jz_words=10, seed=5)
        # GUITAR/PLAYING/PYTHON/CODE/LATTICE are train-only without movement letters; JAVA is the only
        # train-only token containing one; JIM/CALLANAN sit in the test split and RUN/AZTEC in
        # validation, so those four never enter a generation set
        self.assertEqual(chosen['train'], ['code', 'guitar', 'lattice', 'playing', 'python'])
        self.assertEqual(chosen['jz'], ['java'])                # J/Z-bearing train-only tokens
        self.assertEqual(chosen['letters'], list('abcdefghijklmnopqrstuvwxyz'))
        self.assertEqual(set(held), {'JIM', 'CALLANAN', 'AZTEC', 'RUN'})
        self.assertEqual(counts['letters']['chosen'], 26)
        self.assertNotIn('jim', chosen['train'] + chosen['jz'])
        self.assertNotIn('zzzzzzzzzzzzzzzzzz', chosen['train'])   # over the word-length bound

    def test_smaller_word_counts_are_subsets_of_bigger_ones(self):
        from scripts.generate_openfs_synthetic import select_words
        pool = [f'word{index:03d}' for index in range(200)]
        small = select_words(pool, 20, 11)
        large = select_words(pool, 120, 11)
        self.assertTrue(set(small) <= set(large))


class QualityRules(unittest.TestCase):
    """Both keep rules are honest about what the greedy released decoder does to generated clips."""

    def test_exact_rejects_a_word_the_decoder_rambles_past_and_prefix_accepts_it(self):
        from scripts.generate_openfs_synthetic import accepted_by
        self.assertTrue(accepted_by('exact', 'CAMIONES', 'CAMIONES'))
        self.assertFalse(accepted_by('exact', 'CAMIONES', 'CAMIONES BRING'))
        self.assertTrue(accepted_by('prefix', 'CAMIONES', 'CAMIONES BRING'))
        self.assertTrue(accepted_by('prefix', 'J', 'J------'))
        self.assertFalse(accepted_by('prefix', 'J', 'AJ'))
        self.assertFalse(accepted_by('prefix', '', ''))
        with self.assertRaises(ValueError):
            accepted_by('regex', 'J', 'J')


class PackerContract(unittest.TestCase):
    """Manifest keys, the synthetic flag, the quality filter's bookkeeping and the round trip gate."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clips = [dict(word='callanan', set='train', frames=24, accepted=True),
                      dict(word='jim', set='jz', frames=12, accepted=True),
                      dict(word='jay', set='jz', frames=9, accepted=True),
                      dict(word='zebra', set='jz', frames=15, accepted=False, transcription='LEBRA'),
                      dict(word='ross', set='train', frames=18, accepted=False, transcription='LOSS')]
        self.generation = make_generation(self.root / 'generation', self.clips)

    def tearDown(self):
        self.temporary.cleanup()

    def pack(self, output, *extra):
        run(PACKER, ['--generation', str(self.generation), '--output', str(output), *extra])
        directory = Path(output)
        return (json.loads((directory / 'dataset.json').read_text(encoding='utf-8')),
                json.loads((directory / 'train_clips.json').read_text(encoding='utf-8')),
                np.load(directory / 'train_frames.npy'), np.load(directory / 'train_targets.npy'))

    def test_manifest_carries_the_contract_provenance_and_synthetic_flag(self):
        manifest, clips, frames, targets = self.pack(self.root / 'openfs-synthetic-packed')
        self.assertEqual(manifest['contract'], CONTRACT)
        self.assertEqual(manifest['labels'], VOCAB)
        self.assertEqual(manifest['static_labels'], HANDSHAPES)
        self.assertIs(manifest['research_only'], True)
        self.assertIn('strictly prohibited', manifest['license']['quote'])
        provenance = manifest['provenance']['generators'][0]
        self.assertEqual(provenance['generator']['artifact_sha256'], 'ab' * 32)
        self.assertEqual(provenance['conditioner']['artifact_sha256'], 'cd' * 32)
        self.assertEqual(provenance['seed'], 7)
        self.assertEqual(provenance['word_lists']['train']['sha256'], 'ef' * 32)
        self.assertTrue(all(clip['synthetic'] for clip in clips))
        self.assertEqual(len(clips), 3)
        self.assertEqual(manifest['synthetic']['selection']['packed'], 3)
        self.assertIn('model-derived quality filter', manifest['synthetic']['selection']['quality_filter'])
        self.assertEqual(manifest['synthetic']['selection']['quality_rules'], ['exact'])
        self.assertIn('arange', manifest['timeline'])
        self.assertEqual(manifest['partitions'][0]['signers'], ['synthetic'])
        self.assertLess(manifest['synthetic']['roundtrip']['maximum_abs_error'], 1e-5)
        self.assertIn('per axis', manifest['synthetic']['roundtrip']['note'])

    def test_a_rejected_clip_never_reaches_the_pack(self):
        manifest, clips, frames, targets = self.pack(self.root / 'openfs-synthetic-packed')
        packed = {clip['word'] for clip in clips}
        self.assertEqual(packed, {'CALLANAN', 'JIM', 'JAY'})
        self.assertEqual(manifest['synthetic']['selection']['dropped'], {'model_derived_quality_filter': 2})
        self.assertEqual(manifest['synthetic']['selection']['generated'], 5)
        ablated, ablated_clips, _, _ = self.pack(self.root / 'openfs-synthetic-kept', '--keep-rejected')
        self.assertEqual({clip['word'] for clip in ablated_clips},
                         {'CALLANAN', 'JIM', 'JAY', 'ZEBRA', 'ROSS'})
        self.assertEqual(ablated['synthetic']['selection']['packed'], 5)

    def test_frame_targets_are_dense_and_can_name_movement_letters(self):
        manifest, clips, frames, targets = self.pack(self.root / 'openfs-synthetic-packed')
        jay = next(clip for clip in clips if clip['word'] == 'JAY')
        row_targets = targets[jay['start']:jay['end']]
        self.assertEqual(set(np.unique(row_targets)) >= {VOCAB.index('J'), VOCAB.index('A'), VOCAB.index('Y')}, True)
        self.assertNotIn(-100, row_targets)
        self.assertEqual(len(row_targets), jay['end'] - jay['start'])
        # the app's plain statement: the CTC alphabet has J and Z, the static handshape taxonomy does not
        self.assertIn('J', VOCAB)
        self.assertIn('Z', VOCAB)
        self.assertNotIn('J', HANDSHAPES)
        self.assertNotIn('Z', HANDSHAPES)
        self.assertIn('J/Z', manifest['frame_targets'])

    def test_rows_satisfy_the_pack_timeline_and_the_upstream_round_trip(self):
        from scripts.train_openfs_recognizer import timeline_indices
        manifest, clips, frames, targets = self.pack(self.root / 'openfs-synthetic-packed')
        for clip in clips:
            rows = frames[clip['start']:clip['end']]
            np.testing.assert_array_equal(rows[:, 44], np.ones(len(rows)))     # signing hand present
            np.testing.assert_array_equal(rows[:, 89], np.zeros(len(rows)))    # no second hand
            self.assertAlmostEqual(float(rows[1, -1]), 1 / 30, places=6)
            self.assertAlmostEqual(float(rows[0, -1]), 0., places=6)
            np.testing.assert_allclose(timeline_indices(rows), np.arange(len(rows)), atol=1e-6)
            self.assertEqual(clip['generated_frames'], len(rows))
            self.assertTrue(set(clip['letters']) >= set(clip['word'].lower()))
        self.assertLess(manifest['synthetic']['roundtrip']['maximum_timeline_index_error_from_arange'], 1e-5)

    def test_smaller_mixes_are_nested_subsets(self):
        _, small, _, _ = self.pack(self.root / 'openfs-synthetic-small', '--limit-clips', '2')
        _, large, _, _ = self.pack(self.root / 'openfs-synthetic-large', '--limit-clips', '3')
        self.assertTrue({clip['name'] for clip in small} <= {clip['name'] for clip in large})

    def test_a_round_trip_that_does_not_hold_fails_the_pack(self):
        result = run(PACKER, ['--generation', str(self.generation), '--output',
                              str(self.root / 'openfs-synthetic-strict'), '--roundtrip-atol', '0'],
                     expect_zero=False)
        self.assertIn('round trip failed', result.stderr + result.stdout)

    def test_pack_refuses_a_generation_without_a_conditioner_digest(self):
        broken = self.root / 'broken-generation'
        shutil.copytree(self.generation, broken)
        manifest = json.loads((broken / 'provenance.json').read_text(encoding='utf-8'))
        del manifest['conditioner']['artifact_sha256']
        (broken / 'provenance.json').write_text(json.dumps(manifest), encoding='utf-8')
        result = run(PACKER, ['--generation', str(broken), '--output', str(self.root / 'openfs-synthetic-broken')],
                     expect_zero=False)
        self.assertIn('conditioner sha256', result.stderr + result.stdout)

    def test_pack_refuses_an_output_outside_the_synthetic_namespace(self):
        result = run(PACKER, ['--generation', str(self.generation), '--output', str(self.root / 'openfs-packed-x')],
                     expect_zero=False)
        self.assertIn('openfs-synthetic*', result.stderr + result.stdout)

    def test_word_split_pack_is_word_disjoint(self):
        run(PACKER, ['--generation', str(self.generation), '--output', str(self.root / 'openfs-synthetic-split'),
                     '--split-words'])
        directory = self.root / 'openfs-synthetic-split'
        words = {}
        for split in ('train', 'validation', 'test'):
            path = directory / f'{split}_clips.json'
            if not path.is_file():
                continue
            rows = json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(all(clip['signer'] == f'synthetic-{split}' for clip in rows))
            for clip in rows:
                self.assertNotIn(clip['word'], words)
                words[clip['word']] = split
        manifest = json.loads((directory / 'dataset.json').read_text(encoding='utf-8'))
        signers = [set(part['signers']) for part in manifest['partitions']]
        self.assertEqual(len(signers) >= 2, True)
        for index, left in enumerate(signers):
            for right in signers[index + 1:]:
                self.assertFalse(left & right)


class MergedPack(unittest.TestCase):
    """Mixing synthetic rows into a real pack must leave the real splits untouched."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.real = self.root / 'openfs-packed-real'
        self.real.mkdir()
        rows, targets, clips, offset = [], [], [], 0
        for index in range(3):
            pose = sampled_pose(100 + index, 12)
            values = encode_poses({0: pose}, {}, np.arange(12), np.array([], dtype=np.int64))
            rows.append(values)
            targets.append(np.full(12, VOCAB.index('C'), dtype=np.int16))
            clips.append(dict(name=f'real-{index}', text='CC', signer=f'person-{index}', session='s',
                              start=offset, end=offset + 12))
            offset += 12
        validation = [dict(clips[0], signer='person-3', name='real-validation')]
        test = [dict(clips[0], signer='person-4', name='real-test')]
        for split in ('train', 'validation', 'test'):
            members = clips if split == 'train' else (validation if split == 'validation' else test)
            body = np.concatenate(rows) if split == 'train' else np.concatenate(rows[:1])
            marks = np.concatenate(targets) if split == 'train' else np.concatenate(targets[:1])
            np.save(self.real / f'{split}_frames.npy', body)
            np.save(self.real / f'{split}_targets.npy', marks)
            (self.real / f'{split}_clips.json').write_text(json.dumps(members), encoding='utf-8')
        partitions = []
        for split, members in (('train', clips), ('validation', validation), ('test', test)):
            partitions.append(dict(split=split, clips=len(members), frames=len(members) * 12,
                                   signers=[clip['signer'] for clip in members]))
        (self.real / 'dataset.json').write_text(json.dumps(dict(
            contract=CONTRACT, labels=VOCAB, static_labels=HANDSHAPES, partitions=partitions)),
            encoding='utf-8')
        self.generation = make_generation(self.root / 'generation',
                                          [dict(word='jim', set='jz', frames=10, accepted=True),
                                           dict(word='ana', set='train', frames=8, accepted=True)])

    def tearDown(self):
        self.temporary.cleanup()

    def test_merging_appends_synthetic_rows_and_keeps_real_rows_byte_identical(self):
        before = {split: (self.real / f'{split}_frames.npy').read_bytes() for split in ('validation', 'test')}
        output = self.root / 'openfs-synthetic-packed-mix'
        run(PACKER, ['--generation', str(self.generation), '--output', str(output), '--merge-with', str(self.real)])
        manifest = json.loads((output / 'dataset.json').read_text(encoding='utf-8'))
        for split in ('validation', 'test'):
            self.assertEqual((output / f'{split}_frames.npy').read_bytes(), before[split])
        frames = np.load(output / 'train_frames.npy')
        real_frames = np.load(self.real / 'train_frames.npy')
        np.testing.assert_array_equal(frames[:len(real_frames)], real_frames)
        clips = json.loads((output / 'train_clips.json').read_text(encoding='utf-8'))
        real_clips = json.loads((self.real / 'train_clips.json').read_text(encoding='utf-8'))
        self.assertEqual(len(clips), len(real_clips) + 2)
        appended, synthetic = clips[:len(real_clips)], clips[len(real_clips):]
        self.assertEqual([clip['start'] for clip in appended], [clip['start'] for clip in real_clips])
        self.assertTrue(all(clip['synthetic'] for clip in synthetic))
        self.assertTrue(all(not clip.get('synthetic') for clip in appended))
        for clip in synthetic:
            self.assertEqual(frames[clip['end'] - 1, 44], 1)
        train = next(part for part in manifest['partitions'] if part['split'] == 'train')
        self.assertIn('synthetic', train['signers'])
        self.assertEqual(train['real_clips'], len(real_clips))
        self.assertEqual(train['synthetic_clips'], 2)
        sets = [set(part['signers']) for part in manifest['partitions']]
        for index, left in enumerate(sets):
            for right in sets[index + 1:]:
                self.assertFalse(left & right)

    def test_the_real_trainer_reads_a_merged_pack(self):
        if torch is None:
            self.skipTest('the trainer needs torch')
        output = self.root / 'openfs-synthetic-packed-trainer'
        run(PACKER, ['--generation', str(self.generation), '--output', str(output), '--merge-with', str(self.real)])
        from scripts.train_openfs import read_data
        data = read_data(output, 'train', 'sequence')
        self.assertEqual(len(data['clips']), 5)
        self.assertEqual(data['x'].shape[1], 91)
        manifest = json.loads((output / 'dataset.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['contract'], CONTRACT)
        self.assertEqual([set(part['signers']) for part in manifest['partitions']][0] &
                         set(VOCAB), set())      # signers are ids, not vocabulary


class AppRuntimeFirewall(unittest.TestCase):
    """Nothing the app loads may reach the research generator.

    The app legitimately contains the word openfs (its shipped refiner uses released OpenFS
    recognizers), so this guard forbids reaching for the synthetic generator, upstream diffusion
    sampling and anything under scripts/, the same way tests/test_streaming_student.py keeps the
    shipped runtime teacher-free.
    """

    FORBIDDEN = ('generate_openfs_synthetic', 'pack_openfs_synthetic', 'openfs-synthetic',
                 'openfs_synthetic', 'models.diffusion', 'Diffusion', 'diffusion',
                 'from scripts.', 'import scripts.')

    def assert_firewalled(self, path, text):
        lowered = text.lower()
        for needle in self.FORBIDDEN:
            self.assertNotIn(needle.lower(), lowered, f'{needle} reaches the app runtime: {path}')

    def test_model_packages_never_import_scripts_or_upstream(self):
        for path in sorted((ROOT / 'model').rglob('*.py')):
            self.assert_firewalled(path, path.read_text(encoding='utf-8'))

    def test_the_generator_script_is_imported_only_by_tests(self):
        hits = subprocess.run(['git', '-C', str(ROOT), 'grep', '-l', 'generate_openfs_synthetic'],
                              capture_output=True, text=True).stdout.split()
        self.assertIn('scripts/generate_openfs_synthetic.py', hits)
        self.assertIn('tests/test_openfs_synthetic.py', hits)
        for directory in ('model', 'electron', 'frontend', 'handTracker'):
            self.assertFalse([hit for hit in hits if hit.startswith(directory + '/')], hits)
        for directory in ('model', 'electron', 'frontend', 'handTracker'):
            for path in (ROOT / directory).rglob('*'):
                if path.suffix in {'.py', '.ts', '.tsx', '.js', '.mjs'} and path.is_file():
                    self.assert_firewalled(path, path.read_text(encoding='utf-8', errors='ignore'))

    def test_the_committed_metrics_guard(self):
        path = ROOT / 'docs/synthetic-data-metrics-2026-10-19.json'
        if not path.is_file():
            self.skipTest('the synthetic-data metrics file is written with the result')
        document = json.loads(path.read_text(encoding='utf-8'))
        for row in document['rows']:
            for block in (row.get('test'), row.get('generation_quality')):
                if not block:
                    continue
                self.assertEqual(block['deletions'] + block['substitutions'] + block['insertions'], block['errors'])
                self.assertAlmostEqual(block['cer'], block['errors'] / block['characters'], places=12)
        self.assertTrue(any('research_only' in json.dumps(row) for row in document['rows']))


if __name__ == '__main__':
    unittest.main()
