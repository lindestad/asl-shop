"""Checks for the held-letter harness: span validation, the scorer, and the report's own honesty fields.

scripts/eval_held_letters.py replays a recorded app-domain session through model/live_neural.py and scores
held letters, idle and unrelated-activity rejection, J/Z motion and natural spelling against the spans a
human confirmed. What that number rests on is pinned here:

- the validator refuses annotations that would silently change what accuracy means: overlapping held
  letters, a held letter inside an idle span, frame references outside the recording, letters the static
  head has no class for (J and Z are movement letters, so they are either motion spans or rejected),
  motion spans that show no movement, a missing consent grant, a schema version nobody reads;
- the scorer is run against scripted landmark tracks plus a scripted emitter, so the right answer is known
  by construction: an emitter that says exactly the annotated letters must score held accuracy 1.00 with
  zero false accepts on idle spans, and a gate that stays silent must score coverage 0 rather than an
  empty table;
- the report keeps its honesty fields: required keys, human_labels, and numbers_from_suggested_labels true
  whenever the spans were only the model's own unreviewed draft;
- the labelling assist proposes spans that do not contradict each other, and --confirm refuses a draft
  that still does, while a scripted recording can never be promoted into a human label;
- landmark rows rebuilt from the packed FSboard layout come back through the live feature path unchanged,
  which is what lets real FSboard clips score as a known-good plumbing check in their own report file.

ScriptedRuntime is a stand-in for the checkpoint-backed NeuralRuntime. It keeps the per-frame confidence
gate and the J/Z local-motion requirement, because the harness mirrors those two when it reads emissions;
it does not re-implement the app's held-letter text gate (spelling_context in model/live_neural.py), which
is what the live smoke test below and the recorded runs cover instead.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.neural import HANDSHAPES, VOCAB, frame_features  # noqa: E402
from scripts import eval_held_letters, label_sign_session  # noqa: E402
from scripts.fsboard_sessions import (reconstruct_landmarks, signed_run, spans_of,  # noqa: E402
                                      visible_run, write_probe_session)
from scripts.sign_sessions import (ANNOTATION_SCHEMA, FIXTURE_PLANS, MOVEMENT_GATE, SESSION_FILE,  # noqa: E402
                                   SignSessionError, fixture_pose, make_synthetic_session, movement_score,
                                   read_annotation, read_session, validate, write_annotation, write_session)
from scripts.validate_sign_session import check as check_session  # noqa: E402

HUMAN_PROVENANCES = ('human', 'human-reviewed-model-suggestions')


def scratch(test):
    """A scratch directory per test, removed when the test ends."""
    holder = tempfile.TemporaryDirectory(prefix='asl-held-letter-')
    test.addCleanup(holder.cleanup)
    return Path(holder.name)


def call(function, *arguments):
    """Run something that prints, and hand back its result plus everything it wrote."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        return function(*arguments), out.getvalue() + err.getvalue()


def scripted_session(directory, *, session_id='scripted-1', synthetic=True, plan='easy'):
    """Write a scripted session, optionally dropping the synthetic consent marker.

    That marker is what keeps a fixture out of the human-label numbers, so validation tests that need a
    recording which could be a person ask for it explicitly.
    """
    info = make_synthetic_session(directory, session_id=session_id, plan=FIXTURE_PLANS[plan])
    if not synthetic:
        path = Path(directory) / SESSION_FILE
        entry = json.loads(path.read_text(encoding='utf-8'))
        entry['consent'].pop('synthetic', None)
        path.write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
    return info


def run_position(span, index, length):
    """The frame a spelling run's index-th character is scripted onto, evenly inside the span."""
    return span['start'] + (span['end'] - span['start']) * (2 * index + 1) // (2 * length)


def emitter_for(annotation):
    """Scripted model output that says exactly what the annotation claims: the held letter on every frame
    of a held span, the spelling run's characters spread across its frames, and the motion letter inside
    each J/Z span. Idle, activity and transition spans get nothing at all."""
    letters, sequence = {}, {}
    for span in annotation['spans']:
        if span['kind'] == 'held':
            for frame in range(span['start'], span['end']):
                letters[frame] = (span['letter'], .95)
        elif span['kind'] == 'motion':
            sequence[(span['start'] + span['end']) // 2] = span['letter']
        elif span['kind'] == 'spelling':
            for index, char in enumerate(span['text']):
                sequence[run_position(span, index, len(span['text']))] = char
    return letters, sequence


class ScriptedRuntime:
    """The app's per-frame letter gate and motion requirement, with letters chosen by the test."""

    def __init__(self):
        self.paths = {'handshape': 'scripted-handshape.pt', 'sequence': 'scripted-sequence.pt'}
        self.models = {'handshape': object(), 'sequence': object()}
        self.errors = {}
        self.logs = {task: Recorder() for task in ('handshape', 'sequence')}
        self.sessions = {}
        self.letters, self.sequence = {}, {}
        self.gate = .9  # this stand-in's own claim threshold, mirroring the fixed one in the live path
        self.block_from, self.session_id = 0, None
        self.model_timestamps, self.sequence_rows, self.text = [], [], ''
        self.finished, self.frames = False, 0

    def add(self, session):
        self.sessions[session['session_id']] = session

    def analyze(self, hands, width, height, *, video=False, session_id=None, timestamp_ms=None, finish=False):
        session = self.sessions[str(session_id).split('#')[0]]
        frame = int(np.argmin(np.abs(session['timestamps_ms'] - timestamp_ms)))
        if session_id != self.session_id:  # replay opens a segment per visible-hand run, like a fresh session
            self.session_id, self.block_from = session_id, frame
            self.model_timestamps, self.sequence_rows, self.text = [], [], ''
            self.finished, self.frames = False, 0
        prediction = None
        if len(hands) == 1:  # the live static head abstains when the signing hand is ambiguous
            letter, confidence = self.letters.get(frame, (None, .999))
            index = HANDSHAPES.index(letter) if letter else 0
            table = np.full(len(HANDSHAPES), (1 - confidence) / (len(HANDSHAPES) - 1))
            table[index] = confidence
            self.logs['handshape'].probabilities.append(table)
            self.logs['handshape'].calls += 1
            if index and confidence >= self.gate:
                prediction = {'letter': HANDSHAPES[index], 'confidence': float(confidence), 'method': 'scripted'}
        if video and hands:
            if not self.finished:
                char = self.sequence.get(frame)
                if char and not (char in 'JZ' and movement_score(session, self.block_from, frame + 1)
                                 < MOVEMENT_GATE):  # model/live_neural.py demands the same evidence
                    self.text += char
                self.frames += 1
                self.model_timestamps.append(timestamp_ms)
                row = np.full(len(VOCAB), (1 - .9) / (len(VOCAB) - 1))
                row[VOCAB.index(char) if char else 0] = .9
                self.sequence_rows.append(row)
            self.logs['sequence'].probabilities = [np.array(self.sequence_rows)]
            self.logs['sequence'].calls += 1
        if finish:
            self.finished = True
        return {'handshapeAvailable': True, 'handshapeError': None, 'prediction': prediction,
                'fingerspelling': {'available': True, 'text': self.text, 'frames': self.frames,
                                   'modelFrames': len(self.model_timestamps), 'method': 'scripted',
                                   'provisional': not finish, 'status': 'finished' if finish else 'recognizing'}}


class Recorder:
    """Stands in for eval_held_letters.ProbabilityRecorder: the frame evidence the live gate throws away."""

    def __init__(self):
        self.calls = 0
        self.probabilities = []


def scripted_fixture(directory, **kwargs):
    """A scripted session, its annotation, and the runtime and logs that replay it deterministically."""
    info = scripted_session(directory, **kwargs)
    session, annotation = read_session(directory), read_annotation(directory)
    runtime = ScriptedRuntime()
    runtime.add(session)
    runtime.letters, runtime.sequence = emitter_for(annotation)
    return dict(info=info, directory=directory, session=session, annotation=annotation, runtime=runtime,
                logs=runtime.logs)


def replay(fixture, annotation=None):
    """Run the real replay path and fill in the fields every scorer expects, as main() does."""
    annotation = annotation or fixture['annotation']
    data = eval_held_letters.replay(fixture['session'], annotation, fixture['runtime'], fixture['logs'])
    data.update(session=fixture['session'], spans=annotation['spans'], replayed_frames=data['frames'],
                measured_fps=fixture['session']['measured_fps'])
    data['kinds'] = eval_held_letters.frame_kinds(data['expected'], data['frames'])
    return data


def scored(fixture, *, thresholds=None):
    data = replay(fixture)
    return data, eval_held_letters.score_session(fixture['session'], fixture['annotation'], data,
                                                 thresholds or eval_held_letters.DEFAULT_THRESHOLDS)


def at_gate(block, threshold=eval_held_letters.APP_HANDSHAPE_GATE):
    return next(row for row in block['held_letters']['curve'] if row['threshold'] == threshold)


def held_spans(annotation):
    return [span for span in annotation['spans'] if span['kind'] == 'held']


def presence_runs(present):
    runs, start = [], None
    for index, visible in enumerate(list(present) + [False]):
        if visible and start is None:
            start = index
        if not visible and start is not None:
            runs.append((start, index))
            start = None
    return runs


class SpanValidationTests(unittest.TestCase):
    """Annotations that contradict themselves, or the recording, must never reach the scorer."""

    def setUp(self):
        self.directory = scratch(self) / 'session'
        scripted_session(self.directory, session_id='person01', synthetic=False)
        self.session = read_session(self.directory)
        self.annotation = read_annotation(self.directory)

    def containing(self, problems, phrase):
        return [problem for problem in problems if phrase in problem]

    def test_held_letters_may_not_overlap(self):
        first, second = held_spans(self.annotation)[:2]
        second['start'], second['end'] = first['start'] + 2, first['end'] + 2
        errors, _ = validate(self.session, self.annotation)
        self.assertTrue(self.containing(errors, 'contradictory spans'))
        self.assertTrue(self.containing(errors, 'held and held overlap'))

    def test_held_letter_inside_an_idle_span_is_rejected(self):
        held = held_spans(self.annotation)[0]
        idle = next(span for span in self.annotation['spans'] if span['kind'] == 'idle')
        idle['start'], idle['end'] = held['start'] - 5, held['end'] + 5
        errors, _ = validate(self.session, self.annotation)
        self.assertTrue(self.containing(errors, 'contradictory spans'))
        self.assertTrue(self.containing(errors, 'idle span has a visible hand'))

    def test_frame_references_must_stay_inside_the_recording(self):
        held = held_spans(self.annotation)[0]
        start = held['start']
        for broken in ((start, self.session['frame_count'] + 1), (start, start), (-1, start + 10),
                       (start + 20, start + 5)):
            held['start'], held['end'] = broken
            errors, _ = validate(self.session, self.annotation)
            self.assertTrue(self.containing(errors, 'outside the recorded range'), f'{broken} was accepted')
        for broken in ('2', None):
            held['start'], held['end'] = broken, start + 10
            errors, _ = validate(self.session, self.annotation)
            self.assertTrue(self.containing(errors, 'integer frame indices'))

    def test_letters_the_static_head_has_no_class_for_are_deliberate(self):
        held = held_spans(self.annotation)[0]
        for letter in ('J', 'Z', 'a', 'AE', 'Ó', ''):
            held['letter'] = letter
            errors, _ = validate(self.session, self.annotation)
            self.assertTrue(self.containing(errors, 'is not a static handshape letter'),
                            f'{letter!r} was accepted as a held letter')
        held['letter'] = 'Z'
        errors, _ = validate(self.session, self.annotation)
        self.assertTrue(self.containing(errors, 'J and Z are movement letters'),
                        'a held J or Z should point the annotator at kind=motion')
        motion = next(span for span in self.annotation['spans'] if span['kind'] == 'motion')
        for letter in ('K', 'j'):
            motion['letter'] = letter
            errors, _ = validate(self.session, self.annotation)
            self.assertTrue(self.containing(errors, 'motion spans must hold J or Z'))

    def test_a_motion_span_must_show_movement(self):
        """J and Z have no static class, so a motion span only counts as motion if the track moved."""
        motion = next(span for span in self.annotation['spans'] if span['kind'] == 'motion')
        held = held_spans(self.annotation)[0]
        idle = next(span for span in self.annotation['spans'] if span['kind'] == 'idle')
        self.assertGreaterEqual(movement_score(self.session, motion['start'], motion['end']), MOVEMENT_GATE)
        self.assertLess(movement_score(self.session, held['start'], held['end']), MOVEMENT_GATE)
        motion['start'], motion['end'] = held['start'], held['start'] + 6
        errors, _ = validate(self.session, self.annotation)
        self.assertTrue(self.containing(errors, 'shows no landmark movement'))
        motion['start'], motion['end'] = idle['start'], idle['end']
        errors, _ = validate(self.session, self.annotation)
        self.assertTrue(self.containing(errors, 'without any hand'))

    def test_letters_cannot_be_claimed_where_no_hand_is_visible(self):
        held = held_spans(self.annotation)[0]
        idle = next(span for span in self.annotation['spans'] if span['kind'] == 'idle')
        held['start'], held['end'] = idle['start'], idle['end']
        errors, _ = validate(self.session, self.annotation)
        self.assertTrue(self.containing(errors, 'without any hand'))
        run = next(span for span in self.annotation['spans'] if span['kind'] == 'spelling')
        held['start'], held['end'] = idle['start'], idle['end']
        for span in (held,):
            span['kind'] = 'spelling'
            span['text'] = 'AB'
            span.pop('letter', None)
        errors, _ = validate(self.session, self.annotation)
        self.assertTrue(self.containing(errors, 'missing the hand for over half its frames'))

    def test_travel_may_sit_inside_a_spelling_run_but_not_elsewhere(self):
        travel = next(span for span in self.annotation['spans'] if span['kind'] == 'transition')
        run = next(span for span in self.annotation['spans'] if span['kind'] == 'spelling')
        travel['start'], travel['end'] = run['start'] + 2, run['start'] + 8
        errors, _ = validate(self.session, self.annotation)
        self.assertFalse(self.containing(errors, 'contradictory spans'))
        travel['start'], travel['end'] = run['start'] - 4, run['start'] + 4
        errors, _ = validate(self.session, self.annotation)
        self.assertTrue(self.containing(errors, 'contradictory spans'))

    def test_unreviewed_draft_spans_stay_marked(self):
        annotation = dict(self.annotation, label_provenance='model-suggestion', confirmed=False)
        errors, warnings = validate(self.session, annotation)
        self.assertFalse(errors)
        self.assertTrue([note for note in warnings if note.startswith('DRAFT')])
        annotation['confirmed'] = True
        errors, _ = validate(self.session, annotation)
        self.assertTrue(self.containing(errors, 'must stay confirmed=false'))
        annotation.update(label_provenance='human', confirmed=False)
        errors, _ = validate(self.session, annotation)
        self.assertTrue(self.containing(errors, 'must carry confirmed=true'))
        annotation['label_provenance'] = 'handwritten-on-paper'
        errors, _ = validate(self.session, annotation)
        self.assertTrue(self.containing(errors, 'Unknown label_provenance'))
        annotation.update(label_provenance='fsboard-transcript', confirmed=True)
        errors, warnings = validate(self.session, annotation)
        self.assertFalse(errors)
        self.assertTrue([note for note in warnings if 'no person read this recording' in note])

    def test_consent_and_schema_version_are_checked(self):
        self.assertTrue(self.session['consent']['granted'])
        path = self.directory / SESSION_FILE
        entry = json.loads(path.read_text(encoding='utf-8'))
        entry['consent']['granted'] = False
        path.write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
        with self.assertRaises(SignSessionError) as caught:
            read_session(self.directory)
        self.assertIn('consent', str(caught.exception).lower())
        entry['consent']['granted'] = True
        entry['schema'] = 'asl-shop-sign-session/v0'
        path.write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
        with self.assertRaises(SignSessionError) as caught:
            read_session(self.directory)
        self.assertIn('session schema', str(caught.exception))
        entry['schema'] = 'asl-shop-sign-session/v1'
        path.write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
        write_annotation(self.directory, dict(self.annotation, schema=ANNOTATION_SCHEMA.replace('v1', 'v2')))
        with self.assertRaises(SignSessionError) as caught:
            read_annotation(self.directory)
        self.assertIn('annotation schema', str(caught.exception))
        with self.assertRaises(SignSessionError) as caught:
            write_session(self.directory.parent / 'no-consent', session_id='x', signer_id='y',
                          right=np.zeros((5, 21, 3)), left=np.zeros((5, 21, 3)),
                          present=np.ones((5, 2), dtype=np.uint8), timestamps_ms=np.arange(5) * 33.0,
                          width=640, height=480, fps=30.0, consent={'granted': True})
        self.assertIn('consent', str(caught.exception))

    def test_a_scripted_recording_never_becomes_a_human_label(self):
        directory = self.directory.parent / 'scripted'
        scripted_session(directory, session_id='fixture-scripted')
        annotation = dict(read_annotation(directory), label_provenance='human-reviewed-model-suggestions',
                          confirmed=True)
        errors, warnings = validate(read_session(directory), annotation)
        self.assertFalse(errors)
        self.assertTrue([note for note in warnings if 'not a signer' in note])
        write_annotation(directory, dict(annotation, labels_by='someone-who-should-have-checked'))
        _, _, errors, _ = check_session(directory, require_human=True)
        self.assertTrue(self.containing(errors, 'scripted'),
                        'confirming a scripted recording must not pass --require-human')


class ScriptedScoringTests(unittest.TestCase):
    """The scorer judged on scripted landmark tracks whose answer the test already knows."""

    def setUp(self):
        self.root = scratch(self)

    def test_emitter_reading_the_annotated_letters_scores_one(self):
        fixture = scripted_fixture(self.root / 'perfect')
        _, block = scored(fixture)
        row = at_gate(block)
        self.assertEqual(row['held_spans'], len(fixture['info']['held']))
        self.assertEqual(row['held_spans_correct'], row['held_spans'])
        self.assertEqual(row['held_top1_accuracy'], 1.0)
        self.assertEqual(row['held_top1_accuracy_when_emitted'], 1.0)
        self.assertEqual(row['held_top3_accuracy'], 1.0)
        self.assertGreater(row['coverage'], 0.0)
        self.assertEqual(row['idle_emissions'], 0, 'a perfect emitter must stay quiet on idle spans')
        self.assertEqual(row['idle_false_accept_rate'], 0.0)
        self.assertEqual(row['activity_emissions'], 0, 'unrelated hand use must not be claimed either')
        for span in block['held_letters']['spans_at_app_gate']:
            self.assertTrue(span['correct'])
            self.assertIn(span['letter'], span['top3'])
        self.assertEqual(block['motion_letters']['detected'], len(fixture['info']['motion']))
        runs = block['natural_spelling']['runs']
        self.assertEqual([run['text'] for run in runs], fixture['info']['spelling'])
        self.assertEqual([run['observed'] for run in runs], fixture['info']['spelling'])
        self.assertEqual([run['cer'] for run in runs], [0.0] * len(runs))

    def test_a_gate_that_stays_silent_scores_zero_coverage(self):
        fixture = scripted_fixture(self.root / 'silent')
        fixture['runtime'].letters, fixture['runtime'].sequence = {}, {}
        _, block = scored(fixture)
        row = at_gate(block)
        self.assertEqual(row['coverage'], 0.0)
        self.assertEqual(row['emission_frames'], 0)
        self.assertEqual(row['held_spans_with_emission'], 0)
        self.assertEqual(row['held_top1_accuracy'], 0.0, 'a span the app never accepted counts as wrong')
        self.assertEqual(row['held_top3_spans'], len(fixture['info']['held']),
                         'the gate threw the evidence away, but the curve still pools what the model saw')
        self.assertEqual(row['held_top3_accuracy'], 0.0, 'the pooled evidence never contained the letter')
        self.assertEqual(row['idle_emissions'], 0)
        self.assertEqual(block['motion_letters']['detected'], 0)
        runs = block['natural_spelling']['runs']
        self.assertEqual([run['cer'] for run in runs], [1.0] * len(runs))
        self.assertEqual([run['deletions'] for run in runs], [len(text) for text in fixture['info']['spelling']])
        self.assertEqual(block['held_letters']['confusion_at_app_gate']['cells'], [])
        self.assertEqual(at_gate(block, .0)['coverage'], 0.0, 'every swept threshold stays silent too')

    def test_a_wrong_letter_lands_in_the_confusion_matrix(self):
        fixture = scripted_fixture(self.root / 'wrong')
        for frame in list(fixture['runtime'].letters):
            fixture['runtime'].letters[frame] = ('B', .95)
        _, block = scored(fixture)
        held = held_spans(fixture['annotation'])
        row = at_gate(block)
        self.assertEqual(row['held_top1_accuracy'], round(1 / len(held), 6),
                         'only the held span actually annotated B counts as correct')
        span_rows = block['held_letters']['spans_at_app_gate']
        top3_correct = sum(1 for span_row in span_rows if span_row['letter'] in span_row['top3'])
        self.assertEqual(row['held_top3_correct'], top3_correct)
        self.assertEqual(row['held_top3_accuracy'], round(top3_correct / len(span_rows), 6))
        for span_row in span_rows:
            if span_row['letter'] != 'B':
                self.assertIn('B', span_row['top3'], 'the wrong letter leads the pooled evidence too')
        cells = {(cell['expected'], cell['predicted']): cell['count']
                 for cell in block['held_letters']['confusion_at_app_gate']['cells']}
        wrong = {span['letter']: span['end'] - span['start'] for span in held if span['letter'] != 'B'}
        for letter, frames in wrong.items():
            self.assertEqual(cells.get((letter, 'B')), frames, f'{letter} should be confused with B throughout')
        self.assertNotIn(('nothing', 'B'), cells, 'idle spans stay quiet, so nothing is never a claimed row')
        self.assertEqual(block['held_letters']['confusion_at_app_gate']['top_confusions'][0]['predicted'], 'B')
        self.assertEqual(block['motion_letters']['detected'], len(fixture['info']['motion']),
                         'a wrong static letter must not erase the J/Z motion spans')

    def test_j_and_z_need_movement_evidence(self):
        """The live gate reads movement accumulated over the visible run, so the still case needs a
        session whose hand never moves at all."""
        root = scratch(self)
        still = root / 'still'
        frames = 40
        pose = np.asarray(fixture_pose('M'), dtype=np.float32)
        right = np.zeros((frames, 21, 3), dtype=np.float32)
        right[:, :, :2] = pose
        present = np.zeros((frames, 2), dtype=np.uint8)
        present[:, 0] = 1
        write_session(still, session_id='still-j', signer_id='scripted', right=right,
                      left=np.zeros_like(right), present=present,
                      timestamps_ms=np.arange(frames) * (1000 / 30), width=640, height=480, fps=30.0,
                      consent={'granted': True, 'synthetic': True, 'note': 'scripted still hand'},
                      notes='a hand that never moves, claimed as J')
        write_annotation(still, dict(schema=ANNOTATION_SCHEMA, session_id='still-j', signer_id='scripted',
                                     confirmed=True, label_provenance='synthetic-fixture',
                                     spans=[dict(kind='motion', letter='J', start=8, end=30,
                                                 note='a J claimed over a hand that never moved')]))
        session, annotation = read_session(still), read_annotation(still)
        runtime = ScriptedRuntime()
        runtime.add(session)
        runtime.sequence = {19: 'J'}
        data = eval_held_letters.replay(session, annotation, runtime, runtime.logs)
        data.update(session=session, spans=annotation['spans'], replayed_frames=data['frames'],
                    measured_fps=session['measured_fps'])
        data['kinds'] = eval_held_letters.frame_kinds(data['expected'], data['frames'])
        block = eval_held_letters.score_session(session, annotation, data, eval_held_letters.DEFAULT_THRESHOLDS)
        row = block['motion_letters']['spans'][0]
        self.assertLess(row['movement_score'], MOVEMENT_GATE, 'the annotation itself admits this hand sat still')
        self.assertFalse(row['detected'], 'a J claimed over a still hand is not a J')
        self.assertEqual(row['segment_text'], '', 'the streaming path refused the character entirely')
        self.assertEqual(row['characters_in_span'], [])
        self.assertEqual(block['motion_letters']['detected'], 0)
        moving = scripted_fixture(self.root / 'moving', session_id='moving')
        motion = next(span for span in moving['annotation']['spans'] if span['kind'] == 'motion')
        moving['runtime'].letters = {}
        moving['runtime'].sequence = {(motion['start'] + motion['end']) // 2: motion['letter']}
        _, block = scored(moving)
        self.assertEqual(block['motion_letters']['detected'], 1)
        self.assertEqual(block['motion_letters']['spans'][0]['segment_text'], motion['letter'])

    def test_spelling_cer_splits_into_deletions_substitutions_and_insertions(self):
        fixture = scripted_fixture(self.root / 'spelling')
        run = next(span for span in fixture['annotation']['spans'] if span['kind'] == 'spelling')
        text = run['text']
        positions = [run_position(run, index, len(text)) for index in range(len(text))]
        fixture['runtime'].letters = {}
        substituted = text[0] + 'Q' + text[2:]
        fixture['runtime'].sequence = dict(zip(positions, substituted))
        _, block = scored(fixture)
        row = block['natural_spelling']['runs'][0]
        self.assertEqual(row['observed'], substituted)
        self.assertEqual((row['substitutions'], row['deletions'], row['insertions']), (1, 0, 0))
        self.assertAlmostEqual(row['cer'], 1 / len(text), places=6)
        self.assertEqual(row['exact_sequence_accuracy'], 0.0)
        fixture['runtime'].sequence = {positions[0]: substituted[0]}
        _, block = scored(fixture)
        row = block['natural_spelling']['runs'][0]
        self.assertEqual((row['deletions'], row['substitutions'], row['insertions']), (len(text) - 1, 0, 0))
        fixture['runtime'].sequence = dict(zip(positions, 'Q' * len(text)))
        _, block = scored(fixture)
        self.assertEqual(block['natural_spelling']['runs_summary']['substitutions'], len(text))
        self.assertEqual(block['natural_spelling']['runs_summary']['cer'], 1.0)

    def test_segments_are_cut_on_presence_and_idle_frames_are_counted(self):
        fixture = scripted_fixture(self.root / 'segments')
        data, block = scored(fixture)
        present = fixture['session']['present'].any(axis=1)
        blocks = [segment['block'] for segment in data['segments'] if segment['block']]
        self.assertEqual(len(blocks), len(presence_runs(present)))
        for start, stop in blocks:
            self.assertTrue(present[start:stop].all(), 'a segment must sit inside a visible-hand run')
        self.assertEqual(block['annotated_frames_by_span_kind']['idle'], int((~present).sum()))
        self.assertEqual(at_gate(block)['no_hand_frames'], int((~present).sum()))

    def test_two_sessions_in_one_process_do_not_leak_frame_evidence(self):
        """main() reuses one runtime; each session must still be scored against its own frames."""
        root = scratch(self)
        easy = scripted_session(root / 'first', session_id='first')
        hard = scripted_session(root / 'second', session_id='second', plan='hard')
        self.assertNotEqual(easy['held'], hard['held'])
        runtime = ScriptedRuntime()
        results = []
        for directory, info in ((root / 'first', easy), (root / 'second', hard)):
            session, annotation = read_session(directory), read_annotation(directory)
            runtime.add(session)
            runtime.letters, runtime.sequence = emitter_for(annotation)
            data = eval_held_letters.replay(session, annotation, runtime, runtime.logs)
            data.update(session=session, spans=annotation['spans'], replayed_frames=data['frames'],
                        measured_fps=session['measured_fps'])
            data['kinds'] = eval_held_letters.frame_kinds(data['expected'], data['frames'])
            results.append((info, eval_held_letters.score_session(session, annotation, data,
                                                                  eval_held_letters.DEFAULT_THRESHOLDS)))
        for info, block in results:
            self.assertEqual(at_gate(block)['held_top1_accuracy'], 1.0,
                             'a later session read an earlier session\'s handshape evidence')
            self.assertEqual([row['emitted_letter'] for row in block['held_letters']['spans_at_app_gate']],
                             info['held'])


class ReportTests(unittest.TestCase):
    """The report has to say what it measured and who stood behind the labels."""

    REPORT_KEYS = ('schema', 'generated', 'app_gate', 'thresholds', 'replay', 'replay_limit_frames',
                   'segment_gap_ms', 'models', 'human_labels', 'label_provenances', 'synthetic_recordings',
                   'numbers_from_suggested_labels', 'label_warning', 'claim_limits', 'sessions', 'aggregate',
                   'skipped_sessions', 'sessions_requested', 'sessions_scored')
    SESSION_KEYS = ('session_id', 'signer_id', 'directory', 'frames', 'replayed_frames', 'measured_fps',
                    'label_provenance', 'confirmed', 'human_labels', 'numbers_from_suggested_labels',
                    'label_warning', 'synthetic_recording', 'annotated_frames_by_span_kind', 'held_letters',
                    'motion_letters', 'natural_spelling', 'sequence_gate', 'segments', 'consent',
                    'annotation_warnings')

    def setUp(self):
        self.root = scratch(self)
        self.report = self.root / 'held-letter.report.json'
        self.directory = self.root / 'session'
        scripted_session(self.directory, session_id='report-session')

    def run_eval(self, *argv, directory=None):
        directory = directory or self.directory
        runtime = ScriptedRuntime()
        runtime.add(read_session(directory))
        runtime.letters, runtime.sequence = emitter_for(read_annotation(directory))
        argv = ('eval_held_letters.py', '--session', str(directory), '--report', str(self.report),
                '--threads', '1', *argv)
        with mock.patch.object(sys, 'argv', list(argv)), \
                mock.patch.object(eval_held_letters, 'build_runtime', lambda **kwargs: (runtime, runtime.logs)):
            code, printed = call(eval_held_letters.main)
        return code, json.loads(self.report.read_text(encoding='utf-8')), printed

    def test_report_shape_and_numbers_from_a_scripted_fixture(self):
        code, report, _ = self.run_eval()
        self.assertEqual(code, 0)
        for key in self.REPORT_KEYS:
            self.assertIn(key, report)
        self.assertEqual(report['schema'], eval_held_letters.REPORT_SCHEMA)
        self.assertEqual(report['sessions_requested'], 1)
        self.assertEqual(report['sessions_scored'], 1)
        self.assertEqual(report['skipped_sessions'], [])
        self.assertFalse(report['human_labels'], 'a scripted session is not a human label')
        self.assertEqual(report['synthetic_recordings'], 1)
        self.assertEqual(report['label_provenances'], {'synthetic-fixture': 1})
        self.assertIn('SCRIPTED RECORDINGS', report['label_warning'])
        self.assertTrue(report['claim_limits'])
        block = report['sessions'][0]
        for key in self.SESSION_KEYS:
            self.assertIn(key, block)
        self.assertIn('NOT HUMAN LABELS', block['label_warning'])
        self.assertTrue(block['annotation_warnings'])
        row = next(curve for curve in report['aggregate']['held_curve'] if curve['threshold'] == .8)
        self.assertEqual(row['held_top1_accuracy'], 1.0)
        self.assertEqual(row['idle_emissions'], 0)
        self.assertEqual(len(report['aggregate']['held_curve']), len(report['thresholds']))
        self.assertEqual([curve['threshold'] for curve in report['aggregate']['held_curve']],
                         report['thresholds'])

    def test_numbers_from_a_draft_are_flagged_as_the_model_agreeing_with_itself(self):
        draft = dict(read_annotation(self.directory), label_provenance='model-suggestion', confirmed=False,
                     notice='DRAFT')
        write_annotation(self.directory, draft, draft=True)
        code, report, printed = self.run_eval('--draft')
        self.assertEqual(code, 0)
        self.assertFalse(report['human_labels'])
        self.assertTrue(report['numbers_from_suggested_labels'])
        self.assertEqual(report['label_provenances'], {'model-suggestion': 1})
        self.assertIn('UNCONFIRMED LABELS', report['label_warning'])
        self.assertTrue(report['sessions'][0]['numbers_from_suggested_labels'])
        self.assertIn('scoring unconfirmed model suggestions', printed)

    def test_a_confirmed_draft_on_a_scripted_recording_is_still_not_a_human_label(self):
        write_annotation(self.directory, dict(read_annotation(self.directory),
                                             label_provenance='human-reviewed-model-suggestions',
                                             labels_by='dl'))
        code, report, _ = self.run_eval()
        self.assertEqual(code, 0)
        self.assertFalse(report['human_labels'])
        self.assertEqual(report['sessions'][0]['label_provenance'], 'human-reviewed-model-suggestions')
        self.assertTrue(report['sessions'][0]['confirmed'])
        self.assertFalse(report['sessions'][0]['human_labels'])
        code, report, _ = self.run_eval('--require-human')
        self.assertEqual(code, 1)
        self.assertEqual(report['sessions_scored'], 0)
        self.assertIn('scripted', report['skipped_sessions'][0]['reason'])

    def test_a_human_confirmed_draft_on_a_plain_recording_does_count_as_a_human_label(self):
        directory = self.root / 'plain'
        scripted_session(directory, session_id='person02', synthetic=False)
        write_annotation(directory, dict(read_annotation(directory),
                                         label_provenance='human-reviewed-model-suggestions', labels_by='p02'))
        code, report, _ = self.run_eval('--require-human', directory=directory)
        self.assertEqual(code, 0)
        self.assertTrue(report['human_labels'])
        self.assertIsNone(report['label_warning'])
        self.assertIsNone(report['sessions'][0]['label_warning'])
        self.assertFalse(report['sessions'][0]['numbers_from_suggested_labels'])
        self.assertEqual(report['synthetic_recordings'], 0)

    def test_contradictory_annotations_are_skipped_with_their_problems(self):
        annotation = read_annotation(self.directory)
        first, second = held_spans(annotation)[:2]
        second['start'], second['end'] = first['start'], first['end']
        write_annotation(self.directory, annotation)
        code, report, _ = self.run_eval()
        self.assertEqual(code, 1)
        self.assertEqual(report['sessions_scored'], 0)
        self.assertIsNone(report['aggregate'], 'an empty report is not evidence about the model')
        self.assertIn('contradictory', report['skipped_sessions'][0]['reason'])
        self.assertTrue(self.containing(report['skipped_sessions'][0]['problems'], 'contradictory spans'))
        self.assertIn('No session was scored', report['label_warning'])

    def containing(self, problems, phrase):
        return [problem for problem in problems if phrase in problem]

    def test_worker_replay_without_a_kept_video_says_so(self):
        path = self.directory / SESSION_FILE
        entry = json.loads(path.read_text(encoding='utf-8'))
        for named, expected in ((None, 'kept no video'), ('camera.mp4', 'is not next to')):
            entry['video_file'] = named
            path.write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
            gap = eval_held_letters.video_gap(read_session(self.directory))
            self.assertIsNotNone(gap)
            self.assertIn(expected, gap)
            self.assertIn('--via-worker', gap)
        entry['video_file'] = None
        path.write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
        code, report, printed = self.run_eval('--via-worker')
        self.assertEqual(code, 1)
        self.assertEqual(report['sessions_scored'], 0)
        self.assertIn('kept no video', report['skipped_sessions'][0]['reason'])
        self.assertIn('SKIP', printed)
        entry['video_file'] = 'camera.mp4'
        path.write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
        (self.directory / 'camera.mp4').write_bytes(b'')
        self.assertIsNone(eval_held_letters.video_gap(read_session(self.directory)))
        (self.directory / 'camera.mp4').unlink()


class LabellingAssistTests(unittest.TestCase):
    """The assist may save a person hours, but only if its draft cannot be promoted while it is wrong."""

    def setUp(self):
        self.root = scratch(self)
        self.directory = self.root / 'session'
        scripted_session(self.directory, session_id='draft-session', synthetic=False)
        self.fixture = scripted_fixture(self.root / 'assist', session_id='assist', synthetic=False)

    def propose(self):
        data = replay(self.fixture)
        spans = label_sign_session.propose_spans(self.fixture['session'], data)
        return spans, label_sign_session.draft_for(self.fixture['session'], spans=spans,
                                                   models=dict(handshape='scripted', sequence='scripted',
                                                               app_gate=.8))

    def confirm(self, *argv):
        argv = ['label_sign_session.py', '--session', str(self.directory), '--confirm', *argv]
        with mock.patch.object(sys, 'argv', argv):
            return call(label_sign_session.main)

    def test_the_assist_proposes_spans_that_do_not_contradict_each_other(self):
        spans, draft = self.propose()
        self.assertEqual(draft['label_provenance'], 'model-suggestion')
        self.assertFalse(draft['confirmed'])
        self.assertTrue(draft['notice'])
        errors, _ = validate(self.fixture['session'], draft)
        self.assertEqual(errors, [], f'the assist proposed a draft that does not validate: {errors}')
        claimed = [span for span in spans if span['kind'] != 'idle']
        for left, right in zip(claimed, claimed[1:]):
            self.assertLessEqual(left['end'], right['start'], 'draft spans must not overlap')
        letters = {span['letter'] for span in spans if span['kind'] == 'held'}
        self.assertEqual(letters, set(self.fixture['info']['held']))
        self.assertTrue([span for span in spans if span['kind'] == 'idle'])
        self.assertTrue(all(span.get('note') for span in spans), 'every proposed span says it is a suggestion')

    def test_confirm_refuses_a_draft_that_does_not_validate(self):
        fixture_written = (self.directory / 'annotation.json').read_text(encoding='utf-8')
        draft = dict(read_annotation(self.directory), label_provenance='model-suggestion', confirmed=False)
        first, second = held_spans(draft)[:2]
        original = second['start'], second['end']
        second['start'], second['end'] = first['start'] + 1, first['end'] + 1
        write_annotation(self.directory, draft, draft=True)
        code, printed = self.confirm('--by', 'person01')
        self.assertEqual(code, 1)
        self.assertIn('REFUSED', printed)
        self.assertIn('contradictory spans', printed)
        self.assertEqual((self.directory / 'annotation.json').read_text(encoding='utf-8'), fixture_written,
                         'a refused confirm must leave the existing annotation exactly as it was')
        second['start'], second['end'] = original
        write_annotation(self.directory, draft, draft=True)
        code, printed = self.confirm('--by', 'person01')
        self.assertEqual(code, 0, printed)
        confirmed = read_annotation(self.directory)
        self.assertEqual(confirmed['label_provenance'], 'human-reviewed-model-suggestions')
        self.assertTrue(confirmed['confirmed'])
        self.assertEqual(confirmed['labels_by'], 'person01')
        self.assertNotIn('notice', confirmed)

    def test_confirm_needs_the_name_of_the_person_who_reviewed(self):
        fixture_written = (self.directory / 'annotation.json').read_text(encoding='utf-8')
        write_annotation(self.directory, dict(read_annotation(self.directory),
                                             label_provenance='model-suggestion', confirmed=False), draft=True)
        argv = ['label_sign_session.py', '--session', str(self.directory), '--confirm']
        with mock.patch.object(sys, 'argv', argv):
            with self.assertRaises(SystemExit):
                call(label_sign_session.main)
        self.assertEqual((self.directory / 'annotation.json').read_text(encoding='utf-8'), fixture_written)

    def test_a_confirmed_assist_draft_scores_as_human_labels_on_a_plain_recording(self):
        spans, draft = self.propose()
        write_annotation(self.fixture['directory'], draft, draft=True)
        argv = ['label_sign_session.py', '--session', str(self.fixture['directory']), '--confirm',
                '--by', 'person03']
        with mock.patch.object(sys, 'argv', argv):
            code, printed = call(label_sign_session.main)
        self.assertEqual(code, 0, printed)
        session, annotation = read_session(self.fixture['directory']), read_annotation(self.fixture['directory'])
        self.assertEqual(annotation['label_provenance'], 'human-reviewed-model-suggestions')
        data = eval_held_letters.replay(session, annotation, self.fixture['runtime'], self.fixture['logs'])
        data.update(session=session, spans=annotation['spans'], replayed_frames=data['frames'],
                    measured_fps=session['measured_fps'])
        data['kinds'] = eval_held_letters.frame_kinds(data['expected'], data['frames'])
        block = eval_held_letters.score_session(session, annotation, data,
                                                eval_held_letters.DEFAULT_THRESHOLDS)
        self.assertTrue(block['human_labels'], 'a person confirmed these spans on a real recording')
        self.assertFalse(block['numbers_from_suggested_labels'])
        self.assertIsNone(block['label_warning'])
        self.assertEqual(at_gate(block)['held_top1_accuracy'], 1.0)


def packed_rows(letters=('H', 'U', 'G'), frames_per_letter=6, leading=2, trailing=2):
    """Packed-layout rows made the way the packer makes them: our own hands through the 91-column feature
    contract, with all-zero rows where the hand was not detected."""
    from types import SimpleNamespace
    rows = np.zeros((leading + len(letters) * frames_per_letter + trailing, 91), dtype=np.float32)
    index = leading
    for letter in letters:
        landmarks = [SimpleNamespace(x=float(point[0]), y=float(point[1]), z=0.0)
                     for point in fixture_pose(letter)]
        for _ in range(frames_per_letter):
            rows[index] = frame_features([{'hand': 'Right', 'landmarks': landmarks}], 640, 480, 1000 / 30)
            index += 1
    return rows


class FsboardReconstructionTests(unittest.TestCase):
    """What makes a real FSboard clip usable as a known-good check is the round trip back to features."""

    def setUp(self):
        self.root = scratch(self)
        self.rows = packed_rows()

    def features_of(self, right, left, present, rows):
        """Rebuild the app's own frame features from reconstructed landmark tracks, hand by hand."""
        from types import SimpleNamespace
        rebuilt = np.zeros((len(present), 91), dtype=np.float32)
        for index in range(len(present)):
            hands = []
            for slot, side in enumerate(('Right', 'Left')):
                if not present[index, slot]:
                    continue
                stream = right if slot == 0 else left
                hands.append({'hand': side, 'landmarks': [SimpleNamespace(x=float(point[0]), y=float(point[1]),
                                                                          z=float(point[2]))
                                                          for point in stream[index]]})
            # the packed delta column is capture time since the previous row, which is what the harness
            # recomputes from the session's own clock
            rebuilt[index] = frame_features(hands, 640, 480, float(rows[index, -1]) * 1000)
        return rebuilt

    def test_reconstruction_comes_back_through_the_live_feature_path_unchanged(self):
        right, left, present = reconstruct_landmarks(self.rows, width=640, height=480)
        self.assertEqual(right.shape, (len(self.rows), 21, 3))
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(present.shape, (len(self.rows), 2))
        self.assertTrue((present.sum(axis=1) <= 1).all())
        rebuilt = self.features_of(right, left, present, self.rows)
        self.assertTrue(np.allclose(rebuilt, self.rows, atol=1e-5),
                        f'worst column drift {np.abs(rebuilt - self.rows).max()}')
        self.assertEqual(np.flatnonzero(rebuilt[:, 44] > 0).tolist(),
                         np.flatnonzero(self.rows[:, 44] > 0).tolist())
        self.assertEqual(np.flatnonzero(self.rows[:, 89] > 0).tolist(), [], 'this fixture signs with one hand')

    def test_left_hands_survive_the_mirror(self):
        from types import SimpleNamespace
        rows = np.zeros((4, 91), dtype=np.float32)
        landmarks = [SimpleNamespace(x=float(point[0]), y=float(point[1]), z=0.0) for point in fixture_pose('M')]
        for index in range(len(rows)):
            rows[index] = frame_features([{'hand': 'Left', 'landmarks': landmarks}], 640, 480, 1000 / 30)
        right, left, present = reconstruct_landmarks(rows, width=640, height=480)
        rebuilt = self.features_of(right, left, present, rows)
        self.assertTrue((present[:, 1] > 0).all(), 'the left slot must stay the left slot')
        self.assertTrue(np.allclose(rebuilt, rows, atol=1e-5),
                        'shape_features() mirrors Left hands, so reconstruction has to mirror them back')

    def test_signed_run_cuts_idle_edges_and_keeps_the_word_as_one_span(self):
        self.assertEqual(signed_run(self.rows), (2, 2 + 18))
        spans = spans_of(self.rows, text='HUG')
        self.assertEqual([span['kind'] for span in spans], ['idle', 'spelling', 'idle'])
        spelling = next(span for span in spans if span['kind'] == 'spelling')
        self.assertEqual((spelling['start'], spelling['end']), (2, 20))
        self.assertEqual(spelling['text'], 'HUG')
        self.assertEqual(spans_of(np.zeros_like(self.rows), text='HUG'),
                         [dict(kind='idle', start=0, end=len(self.rows),
                               note='no hand detection anywhere in the clip')])

    def test_a_transcript_session_validates_but_never_reads_as_a_human_label(self):
        directory = self.root / 'fsboard'
        clip = dict(name='ab12_0_20220101_101010_100_200', text='HUG', signer='ab12',
                    session='ab12_0_20220101_101010.mp4', start=0, end=len(self.rows))
        write_probe_session(directory, clip, self.rows, pack=REPO / 'data' / 'openfs-packed-v3',
                            generation=REPO / 'data' / 'openfs-packed-v3' / 'generations' / 'unit-test',
                            split='test')
        session, annotation, errors, warnings = check_session(directory)
        self.assertFalse(errors)
        self.assertEqual(annotation['label_provenance'], 'fsboard-transcript')
        self.assertNotIn(annotation['label_provenance'], HUMAN_PROVENANCES)
        self.assertTrue([note for note in warnings if 'no person read this recording' in note])
        self.assertEqual(session['recorded_at'], '2022-01-01T10:10:10')
        self.assertEqual(held_spans(annotation), [], 'a clip transcript says nothing about held letters')
        _, _, errors, _ = check_session(directory, require_human=True)
        self.assertTrue([problem for problem in errors if 'not a human label' in problem])

    def test_a_packed_fsboard_clip_round_trips_when_the_pack_is_present(self):
        manifest = REPO / 'data' / 'openfs-packed-v3' / 'dataset.json'
        if not manifest.is_file():
            self.skipTest('no packed FSboard generation on this machine')
        description = json.loads(manifest.read_text(encoding='utf-8'))
        generation = manifest.parent / description['generation']
        clips = json.loads((generation / 'test_clips.json').read_text(encoding='utf-8'))
        clip = next(row for row in clips
                    if 4 <= len(row['text']) <= 9 and row['text'].isalpha()
                    and row['end'] - row['start'] >= 10 * len(row['text']))
        rows = np.asarray(np.lib.format.open_memmap(generation / 'test_frames.npy', mode='r')
                          [clip['start']:clip['end']], dtype=np.float32)
        right, left, present = reconstruct_landmarks(rows, width=640, height=480)
        rebuilt = self.features_of(right, left, present, rows)
        self.assertTrue(np.allclose(rebuilt, rows, atol=1e-4),
                        f'real packed rows drift by {np.abs(rebuilt - rows).max()}')
        self.assertGreater(float(present.any(axis=1).mean()), .5, 'the chosen clip is mostly signed')
        first,last=signed_run(rows)
        self.assertGreaterEqual(visible_run(rows[first:last]).mean(), .8,
                                'the signed run keeps a hand through at least 80% of the clip')


class LiveRuntimeSmokeTests(unittest.TestCase):
    """The real live path, when the app's checkpoints are on disk: the harness must keep mirroring it."""

    def test_real_runtime_replays_a_scripted_session(self):
        checkpoints = [REPO / 'data' / 'handshape.pt', REPO / 'data' / 'sequence.pt']
        if not all(path.is_file() for path in checkpoints):
            self.skipTest('the app checkpoints are not present')
        import torch
        torch.set_num_threads(1)
        runtime, logs = eval_held_letters.build_runtime(threads=1)
        self.assertEqual(runtime.errors, {}, 'the live runtime failed to load a checkpoint')
        self.assertEqual(sorted(logs), ['handshape', 'sequence'])
        directory = scratch(self) / 'live'
        scripted_session(directory, session_id='live-session')
        session, annotation = read_session(directory), read_annotation(directory)
        data = eval_held_letters.replay(session, annotation, runtime, logs)
        data.update(session=session, spans=annotation['spans'], replayed_frames=data['frames'],
                    measured_fps=session['measured_fps'])
        data['kinds'] = eval_held_letters.frame_kinds(data['expected'], data['frames'])
        block = eval_held_letters.score_session(session, annotation, data, [.0, .8])
        self.assertFalse(block['human_labels'])
        self.assertTrue(block['synthetic_recording'])
        self.assertEqual(block['label_provenance'], 'synthetic-fixture')
        for row in block['held_letters']['curve']:
            self.assertEqual(row['frames'], session['frame_count'])
            self.assertEqual(row['no_hand_frames'], int((~session['present'].any(axis=1)).sum()))
        self.assertGreater(at_gate(block, .0)['emission_frames'], 0,
                           'the live gate scores every single visible hand, so a swept threshold sees some')
        self.assertEqual(at_gate(block)['idle_emissions'], 0, 'idle frames have no hand to claim')


if __name__ == '__main__':
    unittest.main()
