"""Hypothesis-level (late) fusion of independently decoded recogniser checkpoints: every
checkpoint beam-decodes the identical gap-preserving batch on its own, and the member
hypotheses are combined in string space instead of frame-posterior space.  Two late
selections are reported per sample: the median hypothesis (medoid of the per-member MAP
hypotheses under unit-cost Levenshtein) as primary, and a cross-member minimum-Bayes-risk
selection that pools all members' beam hypotheses, weights each member's pool by its own
score softmax (a member that never proposed a hypothesis contributes 0), and returns the
hypothesis minimising expected Levenshtein distance under the averaged pool posterior.
Per-member CER is reported as diagnostic.  This is the hypothesis-space counterpart of the
frame-level geometric fusion in beam_eval_openfs_recognizer.py; no dictionary or language
correction, no checkpoints written.

  python scripts/late_fusion_eval_openfs_recognizer.py --checkpoint A.pt B.pt ... \
      --data data/openfs-packed --splits validation --member-beam 1 --limit 4000 \
      --output <stem>   # writes <stem>.late-fusion-eval-report.json
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import CONTRACT
from model.offline_recognizer import Recognizer, decode_text
from scripts.train_neural import edit_distance
from scripts.train_openfs_recognizer import batches, read_data, summarize
from scripts.beam_eval_openfs_recognizer import beam_decode


def medoid_select(hypotheses):
    """Median hypothesis: among the deduplicated member MAP hypotheses, the one minimising
    the summed Levenshtein distance to all member hypotheses; ties keep the first member's."""
    unique = list(dict.fromkeys(hypotheses))
    best, best_cost = None, None
    for candidate in unique:
        cost = sum(edit_distance(list(candidate), list(other)) for other in hypotheses)
        if best_cost is None or cost < best_cost:
            best, best_cost = candidate, cost
    return best if best is not None else ()


def cross_member_mbr(pools, cap=8):
    """Cross-member MBR: average the per-member pool score softmaxes (uniform member prior;
    a hypothesis absent from a member's pool contributes 0 there) and return the union-set
    hypothesis minimising expected unit-cost Levenshtein distance under that average.  Each
    member contributes at most its top `cap` distinct pool hypotheses by score, bounding the
    quadratic Levenshtein cost."""
    weights = {}
    for pool in pools:
        unique = {}
        for score_value, ids in pool:
            if ids not in unique or score_value > unique[ids]:
                unique[ids] = score_value
        if not unique:
            continue
        keys = sorted(unique, key=unique.get, reverse=True)[:cap]
        scores = [unique[key] for key in keys]
        top = max(scores)
        exponentials = [math.exp(score - top) for score in scores]
        total = sum(exponentials)
        for key, weight in zip(keys, exponentials):
            weights[key] = weights.get(key, 0.0) + weight / total
    if not weights:
        return ()
    total = sum(weights.values())
    candidates = list(weights)
    probabilities = [weights[key] / total for key in candidates]
    best, best_cost = None, None
    for index, candidate in enumerate(candidates):
        cost = sum(probability * edit_distance(list(candidate), list(other))
                   for probability, other in zip(probabilities, candidates))
        if best_cost is None or cost < best_cost - 1e-9:
            best, best_cost = index, cost
    return candidates[best]


@torch.inference_mode()
def score(models, data, representation, member_beam, max_chars, batch_size, device, split, limit):
    for model in models:
        model.eval()
    started = time.perf_counter()
    member_stats = [defaultdict(lambda: dict(errors=0, characters=0, exact=0, clips=0))
                    for _ in models]
    medoid_stats = defaultdict(lambda: dict(errors=0, characters=0, exact=0, clips=0))
    mbr_stats = defaultdict(lambda: dict(errors=0, characters=0, exact=0, clips=0))
    examples = []
    for pose, segments, positions, _, selected in batches(data, batch_size, representation, limit=limit):
        poses, segments, positions = pose.to(device), segments.to(device), positions.to(device)
        member_outputs = []
        member_pools = []
        for model in models:
            decoded, pools = beam_decode([model], [poses], segments, positions, member_beam,
                                         max_chars, 0.0, [1.0], 'logprob', (0.0,), None)
            member_outputs.append(decoded)
            member_pools.append(pools)
        for i, index in enumerate(selected):
            row = data['clips'][index]
            hypotheses = [member_output[i] for member_output in member_outputs]
            for member_index, hypothesis in enumerate(hypotheses):
                actual = decode_text(hypothesis)
                error = edit_distance(row['text'], actual)
                for key in ('overall', row['signer']):
                    stats = member_stats[member_index][key]
                    stats['errors'] += error
                    stats['characters'] += len(row['text'])
                    stats['exact'] += actual == row['text']
                    stats['clips'] += 1
            medoid_actual = decode_text(medoid_select(hypotheses))
            medoid_error = edit_distance(row['text'], medoid_actual)
            for key in ('overall', row['signer']):
                stats = medoid_stats[key]
                stats['errors'] += medoid_error
                stats['characters'] += len(row['text'])
                stats['exact'] += medoid_actual == row['text']
                stats['clips'] += 1
            mbr_actual = decode_text(cross_member_mbr([member_pool[i] for member_pool in member_pools]))
            mbr_error = edit_distance(row['text'], mbr_actual)
            for key in ('overall', row['signer']):
                stats = mbr_stats[key]
                stats['errors'] += mbr_error
                stats['characters'] += len(row['text'])
                stats['exact'] += mbr_actual == row['text']
                stats['clips'] += 1
            if len(examples) < 30:
                examples.append(dict(name=row['name'], truth=row['text'],
                                     members=[decode_text(hypothesis) for hypothesis in hypotheses],
                                     medoid=medoid_actual, cross_member_mbr=mbr_actual))
    metrics = summarize(medoid_stats.pop('overall'))
    metrics['cross_member_mbr'] = summarize(mbr_stats.pop('overall'))
    metrics.update(split=split,
                   per_signer={k: summarize(v) for k, v in medoid_stats.items()},
                   cross_member_mbr_per_signer={k: summarize(v) for k, v in mbr_stats.items()},
                   members=[summarize(stats.pop('overall')) for stats in member_stats],
                   examples=examples,
                   decoder=(f'late fusion: {len(models)} checkpoints each decoded independently with '
                            f'beam width {member_beam}; primary selection is the median hypothesis '
                            f'(medoid under unit-cost Levenshtein over the member MAP hypotheses), '
                            f'cross-member MBR over the pooled beam hypotheses reported alongside; '
                            f'no dictionary or language correction'),
                   seconds=round(time.perf_counter() - started, 3))
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Late (hypothesis-space) fusion evaluation of offline OpenFS recognizer checkpoints.')
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True, help='member checkpoints; decoded independently and combined in string space')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['validation'], choices=['train', 'validation', 'test'])
    parser.add_argument('--member-beam', type=int, default=8, help='independent beam width per member')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-chars', type=int, help='defaults to the checkpoint max_decode (48)')
    parser.add_argument('--output', type=Path, required=True, help='report stem; never loads or writes a checkpoint')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    payloads = [torch.load(path, map_location='cpu', weights_only=True) for path in args.checkpoint]
    for path, payload in zip(args.checkpoint, payloads):
        if payload.get('contract') != CONTRACT:
            raise ValueError(f'Checkpoint contract mismatch: {path}.')
    representations = [payload['representation'] for payload in payloads]
    if len(set(representations)) != 1 or representations[0] not in ('gaps', 'wrist2d'):
        raise ValueError('Late fusion requires all checkpoints to share one gap-preserving representation.')
    labels = payloads[0]['labels']
    if any(payload['labels'] != labels for payload in payloads):
        raise ValueError('Late-fusion members must share the same label vocabulary.')
    max_chars = args.max_chars or payloads[0].get('max_decode', 48)
    models = []
    for payload in payloads:
        model = Recognizer(**payload['architecture']).to(args.device)
        model.load_state_dict(payload['state_dict'], strict=True)
        models.append(model)
    parameters = sum(q.numel() for model in models for q in model.parameters())
    print(json.dumps(dict(task='recognizer', mode='late-fusion', representation=representations[0],
                          parameters=parameters, ensemble_size=len(models),
                          member_beam=args.member_beam,
                          ensemble_architectures=[payload['architecture'] for payload in payloads],
                          torch=torch.__version__, device=args.device,
                          gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)), flush=True)
    evaluations = [score(models, read_data(args.data, name), representations[0], args.member_beam,
                         max_chars, args.batch_size, args.device, name, args.limit)
                   for name in args.splits]
    for metrics in evaluations:
        print(json.dumps({key: metrics[key] for key in ('split', 'cer', 'letter_accuracy',
                                                        'exact_sequence_accuracy', 'clips', 'characters',
                                                        'errors', 'exact', 'seconds')}), flush=True)
        mbr = dict(metrics['cross_member_mbr'])
        mbr['selection'] = 'cross-member MBR'
        print(json.dumps(mbr), flush=True)
        for member_index, member in enumerate(metrics['members']):
            member = dict(member)
            member['selection'] = f'member {member_index}'
            print(json.dumps(member), flush=True)
    peak = torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
    report = dict(task='recognizer', mode='late-fusion', representation=representations[0],
                  parameters=parameters, ensemble_size=len(models), member_beam=args.member_beam,
                  max_chars=max_chars,
                  checkpoints=[str(path.resolve()) for path in args.checkpoint],
                  evaluations=evaluations, dataset=json.loads((args.data / 'dataset.json').read_text()),
                  torch=torch.__version__, peak_vram_mib=round(peak / 2 ** 20),
                  gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None)
    report_path = args.output.with_suffix('.late-fusion-eval-report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(report=str(report_path), peak_vram_mib=report['peak_vram_mib'])), flush=True)


if __name__ == '__main__':
    main()
