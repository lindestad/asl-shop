"""Refine FSboard frame labels with the released OpenFS frame classifier, blank (phi) included.

Runs the pinned upstream recognizer encoder over the signing-hand detections of packed clips and
re-labels every detection with the released upstream frame classifier, keeping the blank class that
the v2 packer dropped so held frames can become idle negatives. Refined labels are aligned to the
packed <split>_clips.json order so scripts/prepare_openfs_labels_v3.py can join them by clip name and
original frame index. Research-only: the upstream checkout and both checkpoints carry a
non-commercial research license and never enter app code.
"""
import os

# Set before torch builds its caching allocator: the shared GPU only has limited headroom, and
# expandable segments avoid reserving about twice what the encoder keeps live.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import HANDSHAPES
from scripts.prepare_openfs_labels_v3 import target_handshape, upstream_handshape, upstream_target
from scripts.prepare_openfs_training import UPSTREAM_CHARS, clean_indices

LICENSE_NOTE = ('OpenFS Non-Commercial Research License: upstream source, checkpoints, annotations and any '
                'labels derived from them are research-only, and nothing here ships in the app')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_commit(directory):
    return subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()


def load_models(source, recognizer_dir, classifier_dir, device):
    """Released upstream constructors and weights; no executable targets read from upstream YAML."""
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    modules = {}
    for name, relative in (('lib.models.recognizer', 'lib/models/recognizer.py'),
                           ('lib.models.frame_classifier', 'lib/models/frame_classifier.py')):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (source / relative).resolve():
            raise ValueError(f'Another OpenFS module is already loaded: {module.__file__}')
        modules[name] = module
    recognizer_config = yaml.safe_load((Path(recognizer_dir) / '.hydra/config.yaml').read_text(encoding='utf-8'))
    classifier_config = yaml.safe_load((Path(classifier_dir) / '.hydra/config.yaml').read_text(encoding='utf-8'))
    for name, config in (('recognizer', recognizer_config), ('classifier', classifier_config)):
        if config['decode']['chars'] != UPSTREAM_CHARS:
            raise ValueError(f'{name} checkpoint vocabulary does not match the packed label vocabulary.')
    classifier_kwargs = {key: value for key, value in classifier_config['frame_classifier'].items()
                         if not key.startswith('_') and key != 'ckpt_path'}
    state = torch.load(Path(classifier_dir) / 'best_model.pth', map_location='cpu', weights_only=True)
    classifier = modules['lib.models.frame_classifier'].Classifier(char_size=len(UPSTREAM_CHARS) + 1, **classifier_kwargs)
    classifier.load_state_dict(state, strict=True)
    recognizer_kwargs = {key: value for key, value in recognizer_config['recognizer'].items()
                         if not key.startswith('_') and key != 'ckpt_path'}
    recognizer = modules['lib.models.recognizer'].EncoderDecoderTransformer(char_size=len(UPSTREAM_CHARS) + 1,
                                                                          **recognizer_kwargs)
    recognizer.load_state_dict(torch.load(Path(recognizer_dir) / 'best_model.pth', map_location='cpu',
                                         weights_only=True), strict=True)
    topology = list(state)
    expected = ['linear.0.weight', 'linear.0.bias', 'linear.1.weight', 'linear.1.bias',
                'linear.4.weight', 'linear.4.bias']
    width = classifier_kwargs['d_model']
    shapes = {key: list(state[key].shape) for key in expected}
    if topology != expected or [shapes[key] for key in expected] != [[width, width], [width], [width], [width],
                                                                     [len(UPSTREAM_CHARS) + 1, width],
                                                                     [len(UPSTREAM_CHARS) + 1]]:
        raise ValueError('Frame classifier checkpoint does not match the released Table S11 architecture.')
    for network in (recognizer, classifier):
        network.eval().to(device)
    return recognizer, classifier, dict(width=width, classes=len(UPSTREAM_CHARS) + 1, topology=shapes)


def signing_poses(hands, hand_id):
    """Upstream refiner preprocessing: per-frame box normalization, left-hand mirroring, 2D joints."""
    if set(hands) - {0}:
        raise ValueError('FSboard import requires the released single-person schema.')
    poses = np.asarray(hands.get(0, []), dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (21, 3) or not np.isfinite(poses).all():
        raise ValueError('Expected finite T x 21 x 3 signing-hand poses.')
    xy = poses[..., :2].copy()
    xy -= (xy.min(axis=1, keepdims=True) + xy.max(axis=1, keepdims=True)) * .5
    xy /= np.abs(xy).max(axis=1, keepdims=True) + 1e-6
    xy *= .5
    if hand_id % 2:
        xy[..., 0] *= -1
    return xy


class StagedClips(Dataset):
    """Memory-mapped pose staging so DataLoader workers share pages instead of copying every clip."""

    def __init__(self, poses, offsets, lengths):
        self.poses, self.offsets, self.lengths = poses, offsets, lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        start = int(self.offsets[index])
        return index, np.asarray(self.poses[start:start + int(self.lengths[index])], dtype=np.float32)


def padded_collate(samples):
    """Pad with the upstream sentinels: -2 poses and frame indices padded at 1024, right padded."""
    positions = [int(index) for index, _ in samples]
    width = max(len(values) for _, values in samples)
    poses = np.full((len(samples), width, 21, 2), -2, dtype=np.float32)
    frame_index = np.full((len(samples), width), 1024, dtype=np.int64)
    for row, (_, values) in enumerate(samples):
        poses[row, :len(values)] = values
        frame_index[row, :len(values)] = np.arange(len(values), dtype=np.int64)
    return positions, torch.from_numpy(poses), torch.from_numpy(frame_index)


class LengthBatchSampler:
    """Group similar lengths and cap both token count and quadratic attention work per batch."""

    def __init__(self, lengths, batch_size, token_budget, attention_budget):
        self.groups = []
        group = []
        for index in np.argsort(np.asarray(lengths), kind='stable'):
            span = int(lengths[int(index)])
            oversized = group and (len(group) + 1 > batch_size or (len(group) + 1) * span > token_budget
                                   or (len(group) + 1) * span * span > attention_budget)
            if oversized:
                self.groups.append(group)
                group = []
            group.append(int(index))
        if group:
            self.groups.append(group)

    def __iter__(self):
        return iter(self.groups)

    def __len__(self):
        return len(self.groups)


def transitions(left, right):
    buckets = Counter()
    for values, name in ((left > 0, 'letter'), (left == 0, 'nothing'), (left < 0, 'outside')):
        for others, target in ((right == 0, 'blank'), (right > 0, 'letter'), (right < 0, 'outside')):
            buckets[f'{name}_to_{target}'] = int((values & others).sum())
    return dict(sorted(buckets.items()))


def tally(values, labels):
    outside = len(labels) - 1
    counts = Counter(int(value) if value >= 0 else outside for value in values)
    return {name: counts.get(index, 0) for index, name in enumerate(labels)}


def comparison(split, packed, clips, names, offsets, frame_index, clean_labels, refined_labels):
    """Join refined labels onto packed frames and quantify what the released refiner changes."""
    starts = {row['name']: int(row['start']) for row in clips}
    if any(name not in starts for name in names):
        raise ValueError('Refined clip names are absent from the packed split.')
    lengths = np.diff(offsets)
    packed_offsets = np.repeat(np.asarray([starts[name] for name in names], dtype=np.int64), lengths) + frame_index
    targets = np.load(Path(packed) / f'{split}_targets.npy', mmap_mode='r')
    packed_labels = np.asarray(targets[packed_offsets], dtype=np.int64)
    mismatched = int((upstream_target(clean_labels) != packed_labels).sum())
    if mismatched:
        raise ValueError(f'Refined label join disagrees with packed targets on {mismatched} detections.')
    classes = list(HANDSHAPES) + ['outside_static_vocabulary']
    clean_classes, refined_classes, packed_classes = (upstream_handshape(clean_labels),
                                                     upstream_handshape(refined_labels), target_handshape(packed_labels))
    counts = dict(clean=tally(clean_classes, classes), refined=tally(refined_classes, classes),
                  packed=tally(packed_classes, classes))
    for left, right in (('clean', 'refined'), ('packed', 'refined')):
        counts[f'{right}_minus_{left}'] = {name: counts[right][name] - counts[left][name] for name in classes}
    with np.load(Path(packed) / f'{split}_handshape.npz') as samples:
        sampled_labels, sampled_clips = samples['labels'], samples['clips']
    present = {name: {int(value) for value in refined_classes[offsets[position]:offsets[position + 1]] if value >= 0}
               for position, name in enumerate(names)}
    in_slice = [sample for sample in range(len(sampled_labels))
                if clips[int(sampled_clips[sample])]['name'] in present]
    still = sum(int(sampled_labels[sample] in present[clips[int(sampled_clips[sample])]['name']])
                for sample in in_slice)
    return dict(
        split=split, clips=len(names), detections=int(len(refined_labels)),
        packed_label_source=f'{split}_targets.npy frame labels of the existing pack',
        join_verified=True, join_mismatches=0, join_frames_checked=int(len(packed_labels)),
        refined_vs_clean_agreement=round(float((clean_labels == refined_labels).mean()), 4),
        refined_vs_packed_agreement=round(float((packed_classes == refined_classes).mean()), 4),
        sampled_handshape_frames_in_refined_subset=len(in_slice),
        sampled_handshape_class_still_present=round(still / len(in_slice), 4) if in_slice else None,
        frames_becoming_blank_vs_clean=transitions(clean_classes, refined_classes).get('letter_to_blank', 0),
        frames_becoming_blank_vs_packed=transitions(packed_classes, refined_classes).get('letter_to_blank', 0),
        transitions_vs_clean=transitions(clean_classes, refined_classes),
        transitions_vs_packed=transitions(packed_classes, refined_classes),
        per_class=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Pinned upstream OpenFS checkout')
    parser.add_argument('--input', required=True, type=Path, help='Extracted fsboard_<split>_proc_clean.npz')
    parser.add_argument('--clips', required=True, type=Path, help='Packed <split>_clips.json: clip order and identity')
    parser.add_argument('--recognizer', type=Path, default=ROOT / 'data/openfs-checkpoints/recognizer_fsboard')
    parser.add_argument('--classifier', type=Path, default=ROOT / 'data/openfs-checkpoints/frame_classifier_fsboard')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/openfs-refined')
    parser.add_argument('--report', type=Path, default=ROOT / 'data/openfs-refined/label-comparison.report.json')
    parser.add_argument('--staging', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit-clips', type=int, help='Seeded random subset of the split, for a slice run')
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--max-frames', type=int, default=3000, help='Skip clips longer than this many detections')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--token-budget', type=int, default=16000, help='Max encoder tokens per batch')
    parser.add_argument('--attention-budget', type=int, default=4000000, help='Max batch x length^2 per batch')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--trusted-source', action='store_true',
                        help='Acknowledge running the upstream checkout against official object NPZ input')
    args = parser.parse_args()
    if not args.trusted_source:
        parser.error('Explicitly acknowledge the upstream source checkout and object NPZ input with --trusted-source.')
    if not args.input.name.startswith('fsboard_'):
        parser.error('--input must be an official fsboard_<split>_proc_clean.npz file.')
    split = args.clips.name[:-len('_clips.json')] if args.clips.name.endswith('_clips.json') else args.clips.stem
    clips = json.loads(args.clips.read_text(encoding='utf-8'))
    recognizer, classifier, architecture = load_models(args.source, args.recognizer, args.classifier, args.device)
    torch.set_num_threads(4)
    staging = args.staging or (args.output / '.staging')
    staging.mkdir(parents=True, exist_ok=True)
    staged_path = staging / f'{split}_poses.npy'
    skipped, hand_of, frame_index = Counter(), {}, {}
    reading = time.perf_counter()
    with np.load(args.input, allow_pickle=True) as data:
        names = [str(name) for name in data['vidnames']]
        position_of = {name: index for index, name in enumerate(names)}
        selected = clean_indices(data['words'], data['signing_hand'], data['frame_label'])
        clean_member = data['frame_label_clean']
        if len(selected) != len(clean_member):
            raise ValueError(f'Clean filtering mismatch: {len(selected)} != {len(clean_member)}')
        clean_at = {int(source): order for order, source in enumerate(selected)}
        hand_of = {index: int(value) if value is not None else -1 for index, value in enumerate(data['signing_hand'])}
        candidates = []
        for row in clips:
            source = position_of.get(row['name'])
            if source is None:
                skipped['missing_from_source'] += 1
            elif source not in clean_at:
                skipped['not_clean'] += 1
            elif hand_of[source] not in (0, 1):
                skipped['no_signing_hand'] += 1
            else:
                candidates.append(source)
        right_indices, left_indices = data['right_frame_indices'], data['left_frame_indices']
        for source in candidates:
            frame_index[source] = np.asarray((right_indices if hand_of[source] == 0 else left_indices)[source],
                                             dtype=np.int64)
        del right_indices, left_indices
        split_detections = sum(len(values) for values in frame_index.values())
        usable = []
        for source in candidates:
            if not len(frame_index[source]):
                skipped['no_detections'] += 1
            elif len(frame_index[source]) > args.max_frames:
                skipped['too_long'] += 1
            else:
                usable.append(source)
        chosen = usable
        if args.limit_clips and args.limit_clips < len(chosen):
            keep = np.random.default_rng(args.sample_seed).choice(len(chosen), args.limit_clips, replace=False)
            chosen = [chosen[int(row)] for row in sorted(keep)]
        lengths = np.asarray([len(frame_index[source]) for source in chosen], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        labels = np.zeros(int(offsets[-1]), dtype=np.int8)
        staged = np.lib.format.open_memmap(staged_path, mode='w+', dtype=np.float32, shape=(int(offsets[-1]), 21, 2))
        for key in ('right_hand_poses', 'left_hand_poses'):
            container = data[key]
            for position, source in enumerate(chosen):
                if (key == 'right_hand_poses') != (hand_of[source] == 0):
                    continue
                values = signing_poses(container[source], hand_of[source])
                staged[offsets[position]:offsets[position + 1]] = values
            del container
        staged.flush()
        del staged
        clean_member = data['frame_label_clean']
        clean_labels = {}
        for source in chosen:
            values = np.asarray(clean_member[clean_at[source]], dtype=np.int64)
            if len(values) != len(frame_index[source]):
                raise ValueError(f'Clean labels do not match signing detections: {names[source]}')
            clean_labels[source] = values
        del clean_member
    read_seconds = time.perf_counter() - reading

    sampler = LengthBatchSampler(lengths, args.batch_size, args.token_budget, args.attention_budget)
    poses = np.lib.format.open_memmap(staged_path, mode='r')
    loader = DataLoader(StagedClips(poses, offsets, lengths), batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=padded_collate)
    started, batches, frames = time.perf_counter(), 0, 0
    with torch.inference_mode():
        for positions, inputs, frame_positions in loader:
            inputs, frame_positions = inputs.to(args.device), frame_positions.to(args.device)
            rlh_seg = torch.zeros_like(frame_positions)  # matches upstream save_frame_label_clean.py
            memory = recognizer(inputs, rlh_seg, frame_positions)['encoder_memory']
            predicted = classifier(memory).argmax(-1).cpu().numpy()
            for row, position in enumerate(positions):
                span = int(lengths[position])
                labels[offsets[position]:offsets[position] + span] = predicted[row, :span]
            batches, frames = batches + 1, frames + int(lengths[list(positions)].sum())
            if batches % args.log_every == 0:
                rate = frames / (time.perf_counter() - started)
                print(json.dumps(dict(split=split, batches=batches, total_batches=len(sampler), frames=frames,
                                      detections_per_second=round(rate, 1),
                                      seconds_elapsed=round(time.perf_counter() - started, 1))), flush=True)
    elapsed = time.perf_counter() - started
    rate = frames / elapsed if elapsed else 0.0
    peak_vram = torch.cuda.max_memory_allocated() // 2**20 if str(args.device).startswith('cuda') else 0

    order = [names[source] for source in chosen]
    del poses
    flat_frames = np.concatenate([frame_index[source] for source in chosen]).astype(np.int32) if chosen else np.zeros(0, np.int32)
    flat_clean = np.concatenate([clean_labels[source] for source in chosen]).astype(np.int8) if chosen else np.zeros(0, np.int8)
    hands = np.asarray([hand_of[source] for source in chosen], dtype=np.int8)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f'{split}.npz'
    np.savez(output, names=np.asarray(order), offsets=offsets, frame_index=flat_frames, label=labels,
             clean_label=flat_clean, hand=hands)
    staged_path.unlink(missing_ok=True)

    report = comparison(split, args.clips.parent, clips, order, offsets, flat_frames, flat_clean, labels)
    provenance = dict(
        generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), split=split, license=LICENSE_NOTE,
        upstream_source=str(args.source.resolve()), upstream_commit=source_commit(args.source),
        frame_classifier=dict(path=str((args.classifier / 'best_model.pth').resolve()),
                              sha256=sha256(args.classifier / 'best_model.pth'), architecture=architecture,
                              gdown_id='1QsPuhSBbmu-RFiTUcLhEne5vKIRpQO-u'),
        recognizer=dict(path=str((args.recognizer / 'best_model.pth').resolve()),
                        sha256=sha256(args.recognizer / 'best_model.pth')),
        input=str(args.input.resolve()), clips=str(args.clips.resolve()), output=str(output.resolve()),
        label_id_space='upstream autoreg class ids; 0 is blank phi, exactly the packer UPSTREAM_CHARS order',
        label_provenance='model-derived by upstream released weights; not human annotation; the frame '
                         'classifier was trained on the FSboard train split coarse labels',
        detection_alignment='one clean label per signing-hand detection row, verified against the same frame '
                            'indices the packer encodes',
        clips_refined=len(chosen), split_clean_clips=len(candidates), detections=int(len(labels)),
        split_detections_total=split_detections, skipped=dict(skipped),
        sampled_subset=dict(limit=args.limit_clips, seed=args.sample_seed) if args.limit_clips else None,
        read_seconds=round(read_seconds, 2), inference_seconds=round(elapsed, 2), batches=batches,
        batches_total=len(sampler), detections_per_second=round(rate, 1),
        projected_hours_for_split_total=round(split_detections / rate / 3600, 2) if rate else None,
        peak_cuda_memory_mib=peak_vram, device=args.device,
        token_budget=args.token_budget, attention_budget=args.attention_budget, batch_size=args.batch_size)
    sidecar = args.output / f'{split}.refine.json'
    sidecar.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    document = json.loads(args.report.read_text(encoding='utf-8')) if args.report.is_file() else {}
    document.update(model=provenance['frame_classifier'], license=LICENSE_NOTE,
                    splits={**document.get('splits', {}), split: report})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f'{args.report.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(document, indent=2), encoding='utf-8')
    temporary.replace(args.report)
    print(json.dumps({key: provenance[key] for key in ('split', 'clips_refined', 'detections', 'detections_per_second',
                                                       'projected_hours_for_split_total', 'peak_cuda_memory_mib')}), flush=True)


if __name__ == '__main__':
    main()
