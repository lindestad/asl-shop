"""Pack generated OpenFS signing-hand sequences into this repo's 91-column app contract.

Reads generation directories written by scripts/generate_openfs_synthetic.py (poses.npy,
frame_labels.npy, clips.json, provenance.json) and writes a pack scripts/train_openfs.py consumes
like any other pack: {split}_frames.npy, {split}_targets.npy, {split}_clips.json and a dataset.json
carrying the contract, labels, signer partitions, a `synthetic` flag on every clip and the full
provenance block (generator sha256, conditioner sha256, word-list hashes, trials, seed).

Research-only. The poses come from a released research checkpoint under the OpenFS Non-Commercial
Research License ("Commercial use of the source code, model checkpoints, or derivative works is
strictly prohibited"), so these rows may never train a model the app ships. Every manifest says so.

What a generated clip cannot represent, stated because the pack still claims the contract:
- one signing hand only. Every synthetic row has the right slot present and the left slot empty with
  its presence flag at 0, so there is no second hand, no two-handed sign and no non-dominant motion.
  The released generator was trained on single signing-hand tracks with left hands mirrored into the
  right-hand convention, so sampled poses already live in that convention and none is applied here.
- no capture geometry. The global wrist columns are wrist position in hand-size units inside the
  generator's own per-frame normalised frame, not measured pixels, and no bounding box or image size
  exists anywhere in the chain.
- no real capture frame rate. The delta column is a nominal 1/--fps step we assert, not a measured
  timestamp: the generator never saw a clock, so any timing read from synthetic rows is ours.
- no detection gaps. Every generated frame is a detection, so synthetic rows cannot teach the app
  what a dropped detection looks like, which is the shape of its deletion failure.
- no depth. The contract is 2-D XY; the generated third coordinate is dropped.
- no idle or background frames. Every generated frame carries a letter, so frame targets are dense,
  unlike the real pack where clean clips label background frames blank and unlabelled clips are
  ignored (-100).

Frame-index convention: rows are written on the app's gap-preserving timeline exactly the way
scripts/prepare_openfs_training.py builds real rows (delta 1/fps from the second row, 0 in the
first). For a gap-free clip scripts.train_openfs_recognizer.timeline_indices is
cumsum(delta)*fps = arange, which is also the `compressed` convention's per-track position and the
frame_idx the generator was trained and sampled with. Writing the app timeline therefore costs
nothing, and the same rows are legal in both offline recipes; the choice is documented, not forced.

Round trip: the packer re-derives the upstream pose normalisation from its own packed rows and
compares against the generated poses (--roundtrip-atol, 1e-5 by default; the run fails if it does
not hold). The divisor that matters is per axis: upstream normalisation splits a frame's x and y
extents, so an isotropic rescale - the bug class fixed last week in
scripts/fsboard_sessions.py::reconstruct_landmarks - only shows up on frames whose x and y extents
differ, which is most of them.

Quality filter: clips whose released-recognizer transcription fails the generation's own match rule
(exact word, or word-prefix when the greedy decoder rambles past a word it read correctly) are
dropped before packing unless --keep-rejected is passed. That filter is model-derived, computed by a
research checkpoint, not a human check, and it biases the kept set toward poses the released
recognizer can already read. Kept/dropped counts, the rule each generation used and the
recognition-on-generated CER are copied into the pack manifest.

Commands (pilot first, then the real corpus; see docs/synthetic-data-2026-10-19.md):

  .venv/bin/python scripts/pack_openfs_synthetic.py --generation data/openfs-synthetic-pilot --output data/openfs-synthetic-pilot-packed --report data/openfs-synthetic-pilot-packed.report.json
  .venv/bin/python scripts/pack_openfs_synthetic.py --generation data/openfs-synthetic-train3 --generation data/openfs-synthetic-letters --output data/openfs-synthetic-packed --split-words --report data/openfs-synthetic-packed.report.json
  .venv/bin/python scripts/pack_openfs_synthetic.py --generation data/openfs-synthetic-train3 --generation data/openfs-synthetic-letters --output data/openfs-synthetic-packed-mix15k --merge-with data/openfs-packed --limit-clips 15000 --report data/openfs-synthetic-packed-mix15k.report.json
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import CONTRACT, HANDSHAPES, VOCAB
from scripts.prepare_openfs_training import UPSTREAM_CHARS, encode_poses
from scripts.train_openfs_recognizer import normalized_box, timeline_indices

OUTPUT_PREFIX = 'openfs-synthetic'
LICENSE_QUOTE = ('Commercial use of the source code, model checkpoints, or derivative works is '
                 'strictly prohibited')
SYNTHETIC_SIGNER = 'synthetic'
WORD_SPLITS = ('train', 'validation', 'test')


def read_generation(directory):
    """Load one generation directory and insist its provenance carries the pinned digests."""
    directory = Path(directory)
    provenance = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
    if provenance.get('research_only') is not True:
        raise ValueError(f'{directory}: provenance.json does not declare research_only=true.')
    for key in ('generator', 'conditioner'):
        if not provenance.get(key, {}).get('artifact_sha256'):
            raise ValueError(f'{directory}: provenance.json is missing the {key} sha256.')
    clips = json.loads((directory / 'clips.json').read_text(encoding='utf-8'))
    return dict(directory=str(directory.resolve()), provenance=provenance, clips=clips,
                poses=np.load(directory / 'poses.npy', mmap_mode='r'),
                labels=np.load(directory / 'frame_labels.npy', mmap_mode='r'))


def frame_targets(labels):
    """Upstream per-frame symbol ids -> this contract's VOCAB indices (index 0 is the CTC blank).

    J and Z are representable here: VOCAB carries all 26 letters, so a movement letter gets a real
    frame target even though the static HANDSHAPES taxonomy deliberately leaves them out.
    """
    targets = np.empty(len(labels), dtype=np.int16)
    for position, identifier in enumerate(labels):
        identifier = int(identifier)
        if identifier == 0:
            targets[position] = 0
            continue
        if not 0 <= identifier < len(UPSTREAM_CHARS):
            raise ValueError(f'upstream frame symbol id {identifier} is outside the released alphabet.')
        letter = UPSTREAM_CHARS[identifier].upper()
        if letter not in VOCAB[1:]:
            raise ValueError(f'generated symbol {letter!r} is not in the app vocabulary.')
        targets[position] = VOCAB.index(letter)
    return targets


def clip_key(clip):
    return (clip['set'], clip['word'], clip['trial'])


def select_clips(candidates, limit, seed):
    """Deterministic prefix of one seeded permutation, so a smaller mix is a subset of a bigger one."""
    ordered = sorted(candidates, key=clip_key)
    if not limit or limit >= len(ordered):
        return ordered
    order = np.random.default_rng(seed).permutation(len(ordered))
    return [ordered[int(index)] for index in order[:int(limit)]]


def pack_rows(generations, clips, *, fps, min_hand_size, max_abs_coordinate, dropped):
    """Generated 21x3 poses -> 91-column contract rows plus dense per-frame frame targets."""
    lookup = {clip['name']: generation for generation in generations for clip in generation['clips']}
    rows, targets, kept = [], [], []
    for clip in clips:
        generation = lookup[clip['name']]
        pose = np.asarray(generation['poses'][clip['start']:clip['end']], dtype=np.float32)
        labels = np.asarray(generation['labels'][clip['start']:clip['end']], dtype=np.int16)
        if not len(pose) or not np.isfinite(pose).all():
            dropped['empty_or_non_finite'] += 1
            continue
        if float(np.linalg.norm(pose[:, 9, :2] - pose[:, 0, :2], axis=1).min()) < min_hand_size:
            dropped['degenerate_hand_scale'] += 1
            continue
        try:
            target = frame_targets(labels)
        except ValueError:
            dropped['unsupported_symbol'] += 1
            continue
        values = encode_poses({0: pose}, {}, np.arange(len(pose), dtype=np.int64),
                              np.array([], dtype=np.int64), fps=fps)
        if not np.isfinite(values).all() or float(np.abs(values[:, :90]).max()) > max_abs_coordinate:
            dropped['coordinate_bound'] += 1
            continue
        clip['generated_frames'] = len(pose)
        clip['letters'] = ''.join(UPSTREAM_CHARS[int(value)] for value in np.unique(labels))
        rows.append(values)
        targets.append(target)
        kept.append(clip)
    return rows, targets, kept


def roundtrip_check(generations, clips, rows, *, atol):
    """Re-derive the upstream normalisation from packed rows and compare it with the sampled poses.

    The same pass checks the timeline claim: because a generated clip has no detection gaps, the
    gap-preserving position index the offline `gaps` recipe builds must come out as arange, which is
    also the `compressed` convention and the frame_idx the generator was sampled with.
    """
    lookup = {clip['name']: generation for generation in generations for clip in generation['clips']}
    worst = 0.0
    timeline_error = 0.0
    for clip, row in zip(clips, rows):
        pose = np.asarray(lookup[clip['name']]['poses'][clip['start']:clip['end'], :, :2], dtype=np.float32)
        worst = max(worst, float(np.abs(normalized_box(row[:, :42].reshape(-1, 21, 2)) - pose).max()))
        timeline_error = max(timeline_error, float(np.abs(timeline_indices(row) - np.arange(len(row))).max()))
    if worst > atol:
        raise ValueError(f'round trip failed: {worst:g} exceeds --roundtrip-atol {atol:g}.')
    return dict(clips_checked=len(clips), maximum_abs_error=worst, atol=atol,
                maximum_timeline_index_error_from_arange=timeline_error,
                note='normalized_box(packed shape channels) reproduces the sampled x,y; the upstream '
                     'divisor is per axis, so an isotropic rescale would miss frames whose x and y '
                     'extents differ')


def hardlink(source, destination):
    """Share a read-only split file with the real pack instead of copying ~1 GB of it."""
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        destination.write_bytes(Path(source).read_bytes())
    same_inode = destination.stat().st_ino == Path(source).stat().st_ino
    return dict(source=str(source), path=str(destination), same_inode=same_inode,
                bytes=destination.stat().st_size)


def real_partition(schema, split):
    for part in schema.get('partitions', []):
        if part.get('split') == split:
            return dict(part)
    raise ValueError(f'the real pack manifest has no {split} partition.')


def word_splits(words, seed, shares=(.8, .1)):
    """Word-disjoint assignment for a standalone synthetic pack; clips inherit their word's split."""
    order = np.random.default_rng(seed).permutation(len(words))
    train_end = int(round(len(words) * shares[0]))
    validation_end = train_end + int(round(len(words) * shares[1]))
    bounds = dict(train=(0, train_end), validation=(train_end, validation_end),
                  test=(validation_end, len(words)))
    assignment = {}
    for split in WORD_SPLITS:
        start, end = bounds[split]
        for index in order[start:end]:
            assignment[words[int(index)]] = split
    return assignment


def write_arrays(directory, split, rows, targets, length):
    frames = np.lib.format.open_memmap(directory / f'{split}_frames.npy', mode='w+', dtype=np.float32,
                                       shape=(length, 91))
    targets_array = np.lib.format.open_memmap(directory / f'{split}_targets.npy', mode='w+', dtype=np.int16,
                                              shape=(length,))
    targets_array[:] = -100
    position = 0
    for row, target in zip(rows, targets):
        frames[position:position + len(row)] = row
        targets_array[position:position + len(target)] = target
        position += len(row)
    frames.flush()
    targets_array.flush()
    del frames, targets_array


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--generation', type=Path, action='append', required=True,
                        help='a scripts/generate_openfs_synthetic.py output directory (repeatable)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--merge-with', type=Path,
                        help='a real packed FSboard dataset: its validation and test splits are kept '
                             'identical and synthetic clips are appended to its train split')
    parser.add_argument('--split-words', action='store_true',
                        help='standalone pack only: assign words to train/validation/test so the pack can '
                             'train on itself (diagnostic only: its held-out text is synthetic too)')
    parser.add_argument('--limit-clips', type=int, help='deterministic subset of kept clips (nested subsets)')
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--fps', type=float, default=30., help='nominal frame rate asserted for gap-free clips')
    parser.add_argument('--min-hand-size', type=float, default=1e-3,
                        help='drop a clip if any frame has a wrist-to-middle-knuckle distance below this '
                             '(the contract divides shape coordinates by it)')
    parser.add_argument('--max-abs-coordinate', type=float, default=32.,
                        help='drop a clip whose packed coordinates exceed this (the app clamps at +-32)')
    parser.add_argument('--keep-rejected', action='store_true',
                        help='ablation: keep clips the model-derived quality filter rejected')
    parser.add_argument('--roundtrip-atol', type=float, default=1e-5)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    if not args.output.name.startswith(OUTPUT_PREFIX):
        raise ValueError(f'--output must sit in a directory named {OUTPUT_PREFIX}*; got {args.output.name}.')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'{args.output} already has content; packs are never written over.')
    if args.merge_with and args.split_words:
        raise ValueError('--split-words is for a standalone synthetic pack; a merged pack keeps the real splits.')
    started = time.perf_counter()

    generations = [read_generation(directory) for directory in args.generation]
    candidates, rejected, generated = [], Counter(), 0
    for generation in generations:
        for clip in generation['clips']:
            generated += 1
            if clip['accepted'] or args.keep_rejected:
                candidates.append(dict(clip))
            else:
                rejected['model_derived_quality_filter'] += 1
    rows, targets, kept = pack_rows(generations, select_clips(candidates, args.limit_clips, args.seed),
                                    fps=args.fps, min_hand_size=args.min_hand_size,
                                    max_abs_coordinate=args.max_abs_coordinate, dropped=rejected)
    if not kept:
        raise ValueError('no synthetic clip survived the filters.')
    roundtrip = roundtrip_check(generations, kept, rows, atol=args.roundtrip_atol)
    names = [clip['name'] for clip in kept]
    if len(set(names)) != len(names):
        raise ValueError('two generations produced the same clip name; pack them separately.')

    assignment = word_splits(sorted({clip['word'] for clip in kept}), args.seed) if args.split_words else {}
    per_split = {split: dict(rows=[], targets=[], clips=[], length=0) for split in WORD_SPLITS}
    for clip, row, target in zip(kept, rows, targets):
        split = assignment.get(clip['word'], 'train')
        clip.update(split=split, synthetic=True, generated_clip=True, text=clip['word'],
                    signer=(f'{SYNTHETIC_SIGNER}-{split}' if args.split_words else SYNTHETIC_SIGNER),
                    name=f"synthetic-{clip['set']}-{clip['word']}-trial{clip['trial']}",
                    start=per_split[split]['length'], end=per_split[split]['length'] + len(row))
        per_split[split]['rows'].append(row)
        per_split[split]['targets'].append(target)
        per_split[split]['clips'].append(clip)
        per_split[split]['length'] = clip['end']

    schema = dict(contract=CONTRACT, labels=VOCAB, static_labels=HANDSHAPES, research_only=True,
                  license=dict(quote=LICENSE_QUOTE, source='data/external/openfs/source/LICENSE',
                               note='these rows were sampled by a released research checkpoint and may never '
                                    'train a model that ships in the app'),
                  fps=f'nominal {args.fps:g} asserted for gap-free generated clips; not a measured capture rate',
                  timeline='gap-preserving app timeline: the delta column is 1/fps from the second row and 0 in '
                           'the first, so for a gap-free clip cumsum(delta)*fps equals arange, which is also '
                           'the compressed convention and the generator training distribution',
                  frame_targets='dense per-frame VOCAB targets from the conditioning labels the generator was '
                                'sampled with; index 0 is the CTC blank, and J/Z carry real targets even though '
                                'the static HANDSHAPES taxonomy excludes them; no blank or background frames',
                  handshape_arrays='not written: the sequence task reads frames/targets/clips; generated frame '
                                   'letter targets live in the targets array',
                  synthetic=dict(kind='generated signing-hand sequences from the released OpenFS diffusion generator',
                                 per_clip_flag='every clip carries "synthetic": true',
                                 generations=[generation['directory'] for generation in generations],
                                 roundtrip=roundtrip,
                                 selection=dict(generated=generated, candidates=len(candidates), packed=len(kept),
                                                limit=args.limit_clips, seed=args.seed, dropped=dict(rejected),
                                                quality_filter='model-derived quality filter: the released recognizer '
                                                               "transcribes each generated clip and the generation's "
                                                               'own match rule keeps it (exact or prefix, recorded '
                                                               'per generation); not a human check',
                                                quality_rules=sorted(generation['provenance']['quality_filter']
                                                                     .get('rule', 'exact')
                                                                     for generation in generations))),
                  provenance=dict(generators=[generation['provenance'] for generation in generations]),
                  partitions=[])

    if args.merge_with:
        real_schema = json.loads((args.merge_with / 'dataset.json').read_text(encoding='utf-8'))
        if real_schema.get('contract') != CONTRACT or real_schema.get('labels') != VOCAB:
            raise ValueError('the real pack contract/vocabulary differs from this repo; prepare it again.')
        args.output.mkdir(parents=True, exist_ok=True)
        shared, partitions = [], []
        real_frames = np.load(args.merge_with / 'train_frames.npy', mmap_mode='r')
        real_targets = np.load(args.merge_with / 'train_targets.npy', mmap_mode='r')
        real_clips = json.loads((args.merge_with / 'train_clips.json').read_text(encoding='utf-8'))
        shift = len(real_frames)
        for split in ('validation', 'test'):
            for suffix in ('frames.npy', 'targets.npy', 'clips.json'):
                shared.append(hardlink(args.merge_with / f'{split}_{suffix}', args.output / f'{split}_{suffix}'))
            partitions.append(real_partition(real_schema, split))
        length = shift + per_split['train']['length']
        frames = np.lib.format.open_memmap(args.output / 'train_frames.npy', mode='w+', dtype=np.float32,
                                           shape=(length, 91))
        targets_array = np.lib.format.open_memmap(args.output / 'train_targets.npy', mode='w+', dtype=np.int16,
                                                  shape=(length,))
        targets_array[:] = -100
        frames[:shift] = real_frames
        targets_array[:shift] = real_targets
        position = shift
        for row, target in zip(per_split['train']['rows'], per_split['train']['targets']):
            frames[position:position + len(row)] = row
            targets_array[position:position + len(target)] = target
            position += len(row)
        frames.flush()
        targets_array.flush()
        del frames, targets_array, real_frames, real_targets
        clips = real_clips + [dict(clip, start=clip['start'] + shift, end=clip['end'] + shift)
                              for clip in per_split['train']['clips']]
        (args.output / 'train_clips.json').write_text(json.dumps(clips), encoding='utf-8')
        train_partition = real_partition(real_schema, 'train')
        train_partition['signers'] = list(dict.fromkeys([*train_partition['signers'], SYNTHETIC_SIGNER]))
        train_partition.update(real_clips=len(real_clips), synthetic_clips=len(per_split['train']['clips']),
                               real_frames=shift, synthetic_frames=per_split['train']['length'],
                               clips=len(clips), frames=length)
        partitions.append(train_partition)
        schema.update(merged_from=str(args.merge_with.resolve()), shared_split_files=shared, partitions=partitions)
        splits = dict(train=length,
                      validation=len(np.load(args.output / 'validation_frames.npy', mmap_mode='r')),
                      test=len(np.load(args.output / 'test_frames.npy', mmap_mode='r')))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for split in WORD_SPLITS:
            member = per_split[split]
            if not member['clips']:
                continue
            write_arrays(args.output, split, member['rows'], member['targets'], member['length'])
            (args.output / f'{split}_clips.json').write_text(json.dumps(member['clips']), encoding='utf-8')
            schema['partitions'].append(dict(split=split, clips=len(member['clips']), frames=member['length'],
                                             words=len({clip['word'] for clip in member['clips']}),
                                             synthetic_clips=len(member['clips']),
                                             signers=[f'{SYNTHETIC_SIGNER}-{split}' if args.split_words
                                                      else SYNTHETIC_SIGNER]))
        splits = {split: per_split[split]['length'] for split in WORD_SPLITS if per_split[split]['clips']}

    synthetic_rows = np.concatenate([row[:, :90] for row in rows])
    summary = dict(pack=str(args.output), synthetic_clips=len(kept),
                   synthetic_frames=per_split['train']['length'] if args.merge_with
                   else sum(per_split[split]['length'] for split in WORD_SPLITS),
                   words=len({clip['word'] for clip in kept}), per_set=dict(Counter(clip['set'] for clip in kept)),
                   seconds=round(time.perf_counter() - started, 2), split_frames=splits,
                   maximum_abs_shape_coordinate=float(np.abs(synthetic_rows[:, :42]).max()),
                   maximum_abs_packed_coordinate=float(np.abs(synthetic_rows).max()),
                   rejected=dict(rejected))
    temporary = args.output / f'dataset.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(schema, indent=2), encoding='utf-8')
    temporary.replace(args.output / 'dataset.json')
    report = args.report or args.output.with_suffix('.report.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(dict(kind='openfs-synthetic-pack', summary=summary, roundtrip=roundtrip,
                                      quality_filter_per_generation=[generation['provenance']['quality_filter']
                                                                    for generation in generations],
                                      manifest=schema), indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=summary, roundtrip=roundtrip, report=str(report))), flush=True)


if __name__ == '__main__':
    main()
