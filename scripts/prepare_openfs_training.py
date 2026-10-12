"""Pack trusted OpenFS FSboard sequences with verified original signer metadata.

Uses original detection indices, not compressed detection order. Clean frame
labels are aligned with the two filters in the upstream released generator.
No image dimensions are guessed: global wrist XY is measured in hand-size units.
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
from model.neural import CONTRACT, HANDSHAPES, VOCAB

UPSTREAM_CHARS = "{ !#$%&'()*+,-./0123456789:;=?@[_abcdefghijklmnopqrstuvwxyz~"


def clean_indices(words, signing, coarse):
    """Reproduce upstream _proc + _proc_frame_label before indexing clean labels."""
    selected = []
    for i, word in enumerate(words):
        if signing[i] is None:
            continue
        ids, first = np.unique(coarse[i], return_index=True)
        ids = ids[np.argsort(first)]
        ids = ids[ids != -1]
        wanted = list(dict.fromkeys(UPSTREAM_CHARS.index(c) for c in str(word) if c in UPSTREAM_CHARS))
        if np.array_equal(ids, wanted):
            selected.append(i)
    return selected


def encode_poses(right, left, right_indices, left_indices, fps=30):
    streams = []
    for base, side, hands, indices in ((0, "Right", right, right_indices),
                                      (45, "Left", left, left_indices)):
        if set(hands) - {0}:
            raise ValueError("FSboard import requires the released single-person schema.")
        poses = np.asarray(hands.get(0, []), dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        if not len(poses):
            if len(indices):
                raise ValueError("Frame indices without poses.")
            continue
        if poses.shape[1:] != (21, 3) or len(poses) != len(indices):
            raise ValueError("Pose/index shape mismatch.")
        if not np.isfinite(poses).all() or np.any(indices < 0) or np.any(np.diff(indices) <= 0):
            raise ValueError("Invalid coordinates/frame indices.")
        streams.append((base, side, poses, indices))
    if not streams:
        raise ValueError("No hands.")
    count = max(int(s[3][-1]) for s in streams) + 1
    frames = np.zeros((count, 91), dtype=np.float32)
    frames[1:, -1] = 1 / fps
    for base, side, poses, indices in streams:
        xy = poses[..., :2]
        relative = xy - xy[:, :1]
        scale = np.linalg.norm(relative[:, 9], axis=1) + 1e-6
        shape = relative / scale[:, None, None]
        if side == "Left":
            shape[:, :, 0] *= -1
        frames[indices, base:base + 42] = shape.reshape(-1, 42)
        frames[indices, base + 42:base + 44] = xy[:, 0] / scale[:, None]
        frames[indices, base + 44] = 1
    return frames


def metadata_from(directory):
    metadata = {}
    for path in sorted(Path(directory).glob('*-*.json')):
        for row in json.loads(path.read_text(encoding="utf-8")):
            name = Path(row["clipFilename"]).stem
            if name in metadata and any(metadata[name][key] != row[key]
                                        for key in ('signerId','phrase','fullVideoFilename')):
                raise ValueError("Conflicting signer/transcript/session metadata.")
            metadata[name] = row
    if not metadata:
        raise ValueError("Missing official FSboard metadata JSONs.")
    return metadata


def prepare_split(path, output, metadata, split, *, max_frames=3000, static_per_letter=8):
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
    plans, identities, skipped = [], set(), Counter()
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
    shape_samples, shape_labels, shape_clip_ids, clips = [], [], [], []
    for position, (i, text) in enumerate(plans):
        values = encode_poses(right[i], left[i], ri[i], li[i])
        frames[offsets[position]:offsets[position+1]] = values
        row = metadata[str(names[i])]
        clips.append(dict(name=str(names[i]), text=text, signer=row['signerId'],
                          session=row['fullVideoFilename'], start=offsets[position], end=offsets[position+1]))
        if i in clean_lookup:
            labels = np.asarray(clean[clean_lookup[i]])
            hand_id = int(signing[i])
            if hand_id not in (0, 1):
                raise ValueError("Unexpected signing person.")
            indices = np.asarray((ri if hand_id == 0 else li)[i], dtype=int)
            if len(labels) != len(indices):
                raise ValueError(f"Clean labels don't match signing detections: {names[i]}")
            auxiliary[offsets[position]:offsets[position+1]] = 0
            for upstream_id in np.unique(labels):
                letter = UPSTREAM_CHARS[int(upstream_id)].upper()
                target = 0 if int(upstream_id) == 0 else VOCAB.index(letter)
                auxiliary[offsets[position] + indices[labels == upstream_id]] = target
            base = 0 if hand_id == 0 else 45
            for upstream_id in np.unique(labels):
                if not 0 <= upstream_id < len(UPSTREAM_CHARS):
                    raise ValueError("Invalid upstream class ID.")
                letter = UPSTREAM_CHARS[int(upstream_id)].upper()
                if letter not in HANDSHAPES[1:]:
                    continue
                candidates = np.flatnonzero(labels == upstream_id)
                chosen = candidates[np.linspace(0, len(candidates)-1,
                                                min(static_per_letter, len(candidates))).round().astype(int)]
                shape_samples.append(values[indices[chosen], base:base+42])
                shape_labels.extend([HANDSHAPES.index(letter)]*len(chosen))
                shape_clip_ids.extend([position]*len(chosen))
        if (position+1) % 10000 == 0:
            print(json.dumps(dict(split=split, packed=position+1, total=len(plans))), flush=True)
    frames.flush()
    auxiliary.flush()
    shape = np.concatenate(shape_samples) if shape_samples else np.zeros((0,42), dtype=np.float32)
    np.savez(output/f'{split}_handshape.npz', features=shape,
             labels=np.array(shape_labels,dtype=np.int64), clips=np.array(shape_clip_ids,dtype=np.int64))
    (output/f'{split}_clips.json').write_text(json.dumps(clips), encoding='utf-8')
    result = dict(split=split, source_clips=len(names), clips=len(clips), frames=offsets[-1],
                  signers=sorted(identities), clean_clips=len(selected), handshape_frames=len(shape_labels),
                  skipped=dict(skipped), seconds=round(time.perf_counter()-started,2))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,type=Path,help='Extracted *_proc_clean directory')
    parser.add_argument('--metadata',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-frames',type=int,default=3000)
    parser.add_argument('--trusted-pickle',action='store_true')
    args=parser.parse_args()
    if not args.trusted_pickle:
        parser.error('Official object NPZ input requires --trusted-pickle.')
    metadata=metadata_from(args.metadata)
    args.output.mkdir(parents=True,exist_ok=True)
    generation=Path('generations')/uuid.uuid4().hex
    staging=args.output/generation
    results=[]
    for split in ('validation','test','train'):
        results.append(prepare_split(args.input/f'fsboard_{split}_proc_clean.npz',staging,metadata,split,
                                     max_frames=args.max_frames))
    for i,a in enumerate(results):
        for b in results[i+1:]:
            if set(a['signers']) & set(b['signers']):
                raise ValueError('Signer overlap in original dataset partitions.')
    summary=dict(contract=CONTRACT,labels=VOCAB,static_labels=HANDSHAPES,partitions=results,generation=generation.as_posix(),
                 fps='nominal 30; original frame indices retained; not measured sensor timestamps',
                 handshape_labels='model-derived clean annotations; blank labels excluded; no idle negatives',
                 clean_alignment='upstream non-null signing-hand and ordered-unique-label filters')
    temporary=args.output/f'dataset.{uuid.uuid4().hex}.tmp'
    temporary.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    temporary.replace(args.output/'dataset.json')


if __name__=='__main__':
    main()
