"""Pack FSboard sequences as generation v3: idle frames retained, optional refined frame labels.

Same 91-column feature contract, metadata joins and clean filters as the v2 packer. Adds idle
negatives (frames labelled blank phi by the released refiner, or by upstream clean labels when no
refined file covers the clip) as the nothing class, and an optional per-frame label source produced
by scripts/refine_openfs_frame_labels.py. Every label here stays model-derived: none of it is human
annotation, and none of it is an app asset.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import HANDSHAPES, VOCAB
from scripts.prepare_openfs_training import UPSTREAM_CHARS, clean_indices, encode_poses, metadata_from

CONTRACT_V3 = 'asl-shop-xy-wrist-v3'
SPLITS = ('validation', 'test', 'train')
STATIC_NOTE = ('static_labels is nothing plus 24 static letters; J and Z are movement letters that a '
               'single-frame model cannot recognise, so they stay excluded from static_labels')
FEATURE_NOTE = ('91 columns unchanged from asl-shop-xy-wrist-v2: two 45-wide hand slots (42 shape '
                'coordinates, 2 wrist XY in hand-size units, 1 presence mask) plus one delta column')
ORIGIN_CODES = {'clean': 0, 'refined': 1}


def _table(values, empty):
    """Lookup over upstream class ids, shifted by one so a -1 id stays addressable."""
    result = np.full(len(UPSTREAM_CHARS) + 2, empty, dtype=np.int16)
    for upstream_id, value in enumerate(values):
        result[upstream_id + 1] = value
    return result


def _blank_at_zero(letter):
    return 0 if letter == '{' else (HANDSHAPES.index(letter) if letter in HANDSHAPES[1:] else -1)


UPSTREAM_LETTERS = [UPSTREAM_CHARS[index].upper() for index in range(len(UPSTREAM_CHARS))]
UPSTREAM_HANDSHAPE = _table([_blank_at_zero(letter) for letter in UPSTREAM_LETTERS], -1)
UPSTREAM_TARGET = _table([0 if letter == '{' else (VOCAB.index(letter) if letter in VOCAB[1:] else -1)
                          for letter in UPSTREAM_LETTERS], -1)  # upstream blank phi is our blank token 0
if (UPSTREAM_TARGET[2:len(UPSTREAM_CHARS) + 1] < 0).any():  # index 1 is blank phi, which maps to 0
    raise ValueError('Packed label vocabulary cannot express every upstream non-blank class.')
TARGET_HANDSHAPE = np.full(len(VOCAB), -1, dtype=np.int16)
TARGET_HANDSHAPE[0] = HANDSHAPES.index('nothing')  # a packed blank phi frame is the nothing class
for _index, _label in enumerate(HANDSHAPES[1:], 1):
    TARGET_HANDSHAPE[VOCAB.index(_label)] = _index


def _mapped(table, upstream_ids, empty):
    values = np.asarray(upstream_ids, dtype=np.int64) + 1
    result = table[np.clip(values, 0, len(table) - 1)].astype(np.int64)
    return np.where(result < 0, empty, result)


def target_handshape(packed_targets):
    """Packed frame target → 26-class handshape id. 0 is nothing, -1 is outside the static vocabulary."""
    values = np.clip(np.asarray(packed_targets, dtype=np.int64), 0, len(VOCAB) - 1)
    result = TARGET_HANDSHAPE[values].astype(np.int64)
    return np.where(np.asarray(packed_targets, dtype=np.int64) < 0, -1, result)


def upstream_handshape(upstream_ids):
    """Upstream class id → 26-class handshape id. 0 is blank phi, -1 is outside the static vocabulary."""
    return _mapped(UPSTREAM_HANDSHAPE, upstream_ids, -1)


def upstream_target(upstream_ids):
    """Upstream class id → packed CTC label. 0 is blank phi, -100 is outside our label vocabulary."""
    return _mapped(UPSTREAM_TARGET, upstream_ids, -100)


def refined_lookup(path):
    """Refined labels keyed by clip name: (original frame indices, upstream class ids, signing hand)."""
    with np.load(path, allow_pickle=False) as source:
        missing = {'names', 'offsets', 'frame_index', 'label', 'hand'} - set(source.files)
        if missing:
            raise ValueError(f'Refined label file is missing arrays: {sorted(missing)}')
        names = [str(name) for name in source['names']]
        offsets, index, labels, hands = source['offsets'], source['frame_index'], source['label'], source['hand']
    if len(names) != len(set(names)):
        raise ValueError(f'Duplicate clip names in refined labels: {path}')
    if not names or offsets[0] != 0 or offsets[-1] != len(labels) or len(offsets) != len(names) + 1:
        raise ValueError(f'Refined label offsets do not cover every detection: {path}')
    for position in range(len(names)):
        span = index[offsets[position]:offsets[position + 1]]
        if len(span) > 1 and not np.greater(np.diff(span), 0).all():
            raise ValueError(f'Refined frame indices are not strictly increasing for clip {names[position]}')
    return {name: (index[offsets[position]:offsets[position + 1]], labels[offsets[position]:offsets[position + 1]],
                   int(hands[position])) for position, name in enumerate(names)}


def audit_disjointness(clips_by_split):
    """Fail loudly if a signer, source session or clip name appears in more than one split."""
    splits, counts = list(clips_by_split), {}
    for key in ('signer', 'session', 'name'):
        groups = [{str(row[key]) for row in clips_by_split[name]} for name in splits]
        leaked = {}
        for first in range(len(splits)):
            for second in range(first + 1, len(splits)):
                shared = groups[first] & groups[second]
                if shared:
                    leaked[f'{splits[first]}/{splits[second]}'] = sorted(shared)[:10]
        if leaked:
            raise ValueError(f'{key} overlap across packed splits: ' + json.dumps(leaked))
        counts[key] = dict(per_split={name: len(groups[index]) for index, name in enumerate(splits)},
                           distinct=len(set().union(*groups)) if groups else 0)
    return dict(splits=splits, checked=('signer', 'session', 'name'), overlap='none', counts=counts)


def prepare_split(path, output, metadata, split, refined, *, max_frames=3000, static_per_letter=8, idle_per_letter=4):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=True) as data:
        names, words, signing, coarse, clean, right, left, ri, li = (
            data[key] for key in ("vidnames", "words", "signing_hand", "frame_label", "frame_label_clean",
                                 "right_hand_poses", "left_hand_poses", "right_frame_indices", "left_frame_indices"))
    selected = clean_indices(words, signing, coarse)
    if len(selected) != len(clean):
        raise ValueError(f"Clean filtering mismatch: {len(selected)} != {len(clean)}")
    clean_lookup = {source: j for j, source in enumerate(selected)}
    plans, identities, skipped, labelled = [], set(), Counter(), Counter()
    offsets = [0]
    for i, name in enumerate(names):
        if str(name) not in metadata:
            raise ValueError(f"Missing original identity metadata: {name}")
        row = metadata[str(name)]
        identities.add(row['signerId'])
        text = str(words[i]).upper().strip()
        if text != row['phrase'].upper().strip():
            raise ValueError(f"Transcript mismatch: {name}")
        if not text or any(c not in VOCAB[1:] for c in text):
            skipped['unsupported_transcript'] += 1
            continue
        length = max(int(indices[-1]) + 1 if len(indices) else 0 for indices in (ri[i], li[i]))
        required = len(text) + sum(a == b for a, b in zip(text, text[1:]))
        if length > max_frames or length < required:
            skipped['length'] += 1
            continue
        plans.append((i, text))
        offsets.append(offsets[-1] + length)
    frames = np.lib.format.open_memmap(output/f'{split}_frames.npy', mode='w+', dtype=np.float32,
                                     shape=(offsets[-1], 91))
    auxiliary = np.lib.format.open_memmap(output/f'{split}_targets.npy', mode='w+', dtype=np.int16,
                                        shape=(offsets[-1],))
    auxiliary[:] = -100
    shape_samples, shape_labels, shape_clip_ids, shape_frames, shape_origin, clips = [], [], [], [], [], []
    for position, (i, text) in enumerate(plans):
        values = encode_poses(right[i], left[i], ri[i], li[i])
        frames[offsets[position]:offsets[position+1]] = values
        row = metadata[str(names[i])]
        clips.append(dict(name=str(names[i]), text=text, signer=row['signerId'],
                          session=row['fullVideoFilename'], start=offsets[position], end=offsets[position+1]))
        if i in clean_lookup:
            labels = np.asarray(clean[clean_lookup[i]], dtype=np.int64)
            hand_id = int(signing[i])
            if hand_id not in (0, 1):
                raise ValueError("Unexpected signing person.")
            indices = np.asarray((ri if hand_id == 0 else li)[i], dtype=np.int64)
            if len(labels) != len(indices):
                raise ValueError(f"Clean labels don't match signing detections: {names[i]}")
            origin = 'clean'
            if refined is not None and str(names[i]) in refined:
                refined_index, refined_labels, refined_hand = refined[str(names[i])]
                if int(refined_hand) != hand_id or not np.array_equal(np.asarray(refined_index, dtype=np.int64), indices):
                    raise ValueError(f"Refined labels don't match signing detections: {names[i]}")
                labels = np.asarray(refined_labels, dtype=np.int64)
                origin = 'refined'
                labelled['refined_clips'] += 1
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"Duplicated signing detection indices: {names[i]}")
            auxiliary[offsets[position]:offsets[position+1]] = 0
            targets = upstream_target(labels)
            auxiliary[offsets[position] + indices[targets < 0]] = -100
            for upstream_id in np.unique(labels[targets >= 0]):
                auxiliary[offsets[position] + indices[labels == upstream_id]] = int(UPSTREAM_TARGET[upstream_id + 1])
            base = 0 if hand_id == 0 else 45
            classes = 0
            for upstream_id in np.unique(labels):
                if upstream_id == 0:
                    continue  # blank ({) frames become idle negatives below, never a letter class
                if not 0 <= upstream_id < len(UPSTREAM_CHARS):
                    skipped['unmapped_class_frames'] += int((labels == upstream_id).sum())
                    continue
                letter = UPSTREAM_LETTERS[int(upstream_id)]
                if letter not in HANDSHAPES[1:]:
                    skipped['non_static_letter_frames'] += int((labels == upstream_id).sum())
                    continue
                candidates = np.flatnonzero(labels == upstream_id)
                chosen = candidates[np.linspace(0, len(candidates)-1,
                                                min(static_per_letter, len(candidates))).round().astype(int)]
                shape_samples.append(values[indices[chosen], base:base+42])
                shape_labels.extend([HANDSHAPES.index(letter)]*len(chosen))
                shape_clip_ids.extend([position]*len(chosen))
                shape_frames.extend((offsets[position] + indices[chosen]).tolist())
                shape_origin.extend([ORIGIN_CODES[origin]]*len(chosen))
                classes += 1
            idle = np.flatnonzero(labels == 0)
            if idle.size and classes:
                chosen = idle[np.linspace(0, idle.size-1, min(idle_per_letter*classes, idle.size)).round().astype(int)]
                shape_samples.append(values[indices[chosen], base:base+42])
                shape_labels.extend([HANDSHAPES.index('nothing')]*len(chosen))
                shape_clip_ids.extend([position]*len(chosen))
                shape_frames.extend((offsets[position] + indices[chosen]).tolist())
                shape_origin.extend([ORIGIN_CODES[origin]]*len(chosen))
                labelled['idle_negative_frames'] += len(chosen)
                labelled['idle_source_frames'] += int(idle.size)
            labelled['labelled_clips'] += 1
            labelled[f'{origin}_labelled_clips'] += 1
            labelled[f'{origin}_detections'] += len(labels)
        if (position+1) % 10000 == 0:
            print(json.dumps(dict(split=split, packed=position+1, total=len(plans))), flush=True)
    frames.flush()
    auxiliary.flush()
    shape = np.concatenate(shape_samples) if shape_samples else np.zeros((0,42), dtype=np.float32)
    np.savez(output/f'{split}_handshape.npz', features=shape,
             labels=np.array(shape_labels,dtype=np.int64), clips=np.array(shape_clip_ids,dtype=np.int64),
             frames=np.array(shape_frames,dtype=np.int64), origin=np.array(shape_origin,dtype=np.int8))
    (output/f'{split}_clips.json').write_text(json.dumps(clips), encoding='utf-8')
    idle_frames = labelled['idle_negative_frames']
    result = dict(split=split, source_clips=len(names), clips=len(clips), frames=offsets[-1],
                  signers=sorted(identities), sessions=len({row['session'] for row in clips}),
                  clean_clips=len(selected), handshape_frames=len(shape_labels),
                  static_letter_frames=len(shape_labels)-idle_frames, idle_negative_frames=idle_frames,
                  idle_available_frames=labelled['idle_source_frames'],
                  idle_negative_ratio=round(idle_frames/len(shape_labels),4) if shape_labels else 0.0,
                  labelled_clips=labelled['labelled_clips'], refined_clips=labelled['refined_clips'],
                  clean_detections=labelled['clean_detections'], refined_detections=labelled['refined_detections'],
                  skipped=dict(skipped), seconds=round(time.perf_counter()-started,2))
    print(json.dumps(result), flush=True)
    return result, clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,type=Path,help='Extracted *_proc_clean directory')
    parser.add_argument('--metadata',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path,help='New v3 generation directory; existing packs are never touched')
    parser.add_argument('--refined-labels',type=Path,nargs='+',default=[],
                        help='Refined label files from scripts/refine_openfs_frame_labels.py, or a directory holding <split>.npz')
    parser.add_argument('--splits',nargs='+',choices=list(SPLITS),default=list(SPLITS))
    parser.add_argument('--max-frames',type=int,default=3000)
    parser.add_argument('--static-per-letter',type=int,default=8)
    parser.add_argument('--idle-per-letter',type=int,default=4,
                        help='Idle negatives per clip as a multiple of the static letter classes sampled in that clip')
    parser.add_argument('--trusted-pickle',action='store_true')
    args=parser.parse_args()
    if not args.trusted_pickle:
        parser.error('Official object NPZ input requires --trusted-pickle.')
    if args.output.resolve() == (ROOT/'data/openfs-packed').resolve():
        parser.error('--output must be a new directory; the v2 pack is shared input for other pipelines.')
    metadata=metadata_from(args.metadata)
    args.output.mkdir(parents=True,exist_ok=True)
    generation=Path('generations')/uuid.uuid4().hex
    staging=args.output/generation
    results,rows=[],{}
    for split in args.splits:
        refined=None
        for entry in args.refined_labels:
            candidate=entry/f'{split}.npz' if entry.is_dir() else (entry if entry.stem==split else None)
            if candidate is not None and candidate.is_file():
                refined=refined_lookup(candidate)
                break
        if args.refined_labels and refined is None:
            print(json.dumps(dict(split=split,warning='No refined label file; falling back to upstream blank labels.')),flush=True)
        result,clips=prepare_split(args.input/f'fsboard_{split}_proc_clean.npz',staging,metadata,split,refined,
                                   max_frames=args.max_frames,static_per_letter=args.static_per_letter,
                                   idle_per_letter=args.idle_per_letter)
        results.append(result)
        rows[split]=clips
    summary=dict(contract=CONTRACT_V3,feature_contract=FEATURE_NOTE,labels=VOCAB,static_labels=HANDSHAPES,
                 static_labels_note=STATIC_NOTE,partitions=results,generation=generation.as_posix(),
                 split_hygiene=audit_disjointness(rows),
                 fps='nominal 30; original frame indices retained; not measured sensor timestamps',
                 handshape_labels='model-derived frame annotations; blank (phi) frames retained as idle negatives; '
                                  'no human held-letter annotation exists in this generation',
                 idle_negative_policy=f'evenly spaced blank detections per clip, capped at {args.idle_per_letter} times '
                                      f'the static letter classes sampled in that clip '
                                      f'({args.static_per_letter} frames per letter class)',
                 handshape_origin_encoding={str(value): name for name, value in ORIGIN_CODES.items()},
                 label_source='released upstream OpenFS frame classifier refined labels (model-derived, research-only '
                              'weights) where a --refined-labels file covers the clip, otherwise upstream '
                              'frame_label_clean (model-derived)',
                 clean_alignment='upstream non-null signing-hand and ordered-unique-label filters',
                 frame_label_alignment='one label per signing-hand detection row; refined labels joined on original '
                                       'frame indices, verified against the same indices the packer encodes',
                 license_note='labels and refined labels derive from research-licensed OpenFS upstream annotations and '
                              'checkpoints; not for commercial use and never shipped in the app')
    if not args.refined_labels:
        summary['label_source']='upstream frame_label_clean (model-derived); no refined label source supplied'
    temporary=args.output/f'dataset.{uuid.uuid4().hex}.tmp'
    temporary.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    temporary.replace(args.output/'dataset.json')


if __name__=='__main__':
    main()
