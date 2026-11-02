"""Score one sequence checkpoint on one packed split, with the movement-letter hole called out.

scripts/train_openfs.py --evaluate already gives the whole-split CER and per-signer numbers; this
adds the breakdown this research question needs: the same clips tallied through
model.recognition_metrics.ErrorTally, plus the subset whose transcript contains J or Z and the
subset whose transcript is a single letter. Those subsets are where synthetic data could plausibly
help, and an average over 12,521 clips hides them.

Nothing here trains anything, and nothing here imports the generator: it reads a pack and a
checkpoint. It is research tooling because the packs it scores are the mixed ones, whose synthetic
rows were sampled from released OpenFS research checkpoints (commercial use of those checkpoints or
derivative works is strictly prohibited), so a checkpoint that saw them may never ship.

Usage:

  .venv/bin/python scripts/evaluate_synthetic_mix.py --data data/openfs-synthetic-packed-mix15k --checkpoint /home/coder/asl-data/models/synthetic-words/sequence.pt --report data/openfs-synthetic-mix15k-words.evaluate.report.json
  .venv/bin/python scripts/evaluate_synthetic_mix.py --data data/openfs-packed --checkpoint data/sequence.pt --report data/openfs-synthetic-baseline.evaluate.report.json
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import CONTRACT, VOCAB, decode_ctc, load_checkpoint
from model.recognition_metrics import ErrorTally
from scripts.train_openfs import batches, read_data

MOVEMENT_LETTERS = set('JZ')


def clip_groups(clips):
    """Which tallies a clip belongs to: every clip, and the two subsets synthetic data aims at.

    A clip counts as movement-letter evidence when its transcript contains J or Z anywhere; the
    single-letter subset is its own row because one-letter clips are where the app's held-letter
    gating and the missing J/Z handshape classes meet.
    """
    groups = []
    for row in clips:
        text = str(row['text']).upper()
        groups.append(dict(overall=True,
                           movement_letters=bool(MOVEMENT_LETTERS & set(text)),
                           single_letter=len(text) == 1))
    return groups


@torch.inference_mode()
def score(model, data, *, batch_size, device, amp=False):
    model.eval()
    groups, tallies = clip_groups(data['clips']), {}
    for values, _, lengths, _, _, selected in batches(data, batch_size, 'sequence'):
        with torch.autocast('cuda', enabled=bool(amp and str(device).startswith('cuda'))):
            predicted = model(values.to(device)).argmax(-1).cpu().numpy()
        for position, index in enumerate(selected):
            truth = str(data['clips'][index]['text'])
            guess = decode_ctc(predicted[position, :int(lengths[position])])
            for name, active in groups[index].items():
                if active:
                    tallies.setdefault(name, ErrorTally()).add(truth, guess)
    return {name: tally.summary() for name, tally in tallies.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='packed FSboard dataset (real or mixed)')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()

    schema = json.loads((args.data / 'dataset.json').read_text(encoding='utf-8'))
    if schema.get('contract') != CONTRACT or schema.get('labels') != VOCAB:
        raise ValueError('pack contract/vocabulary differs from this repo; prepare it again.')
    if 'synthetic' in args.data.name and 'synthetic' not in args.report.name:
        raise ValueError(f'{args.data} holds synthetic rows; its report must sit in the synthetic '
                         'namespace so a research number cannot be mistaken for an app number.')

    model = load_checkpoint(args.checkpoint, 'sequence').to(args.device)
    checkpoint_payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    groups = score(model, read_data(args.data, args.split, 'sequence'),
                   batch_size=args.batch_size, device=args.device, amp=args.amp)
    document = dict(kind='openfs-synthetic-evaluation',
                    contract=CONTRACT,
                    research_only=bool(schema.get('research_only')),
                    synthetic_rows_in_pack=dict(present='synthetic' in schema,
                                                merged_from=schema.get('merged_from'),
                                                clip_flag='every synthetic clip carries "synthetic": true',
                                                license=schema.get('license')),
                    checkpoint=dict(path=str(args.checkpoint.resolve()),
                                    trained_on=checkpoint_payload.get('dataset'),
                                    experimental=bool(checkpoint_payload.get('experimental')),
                                    hidden=checkpoint_payload.get('hidden'),
                                    dilations=list(checkpoint_payload.get('dilations') or [])),
                    pack=str(args.data.resolve()), split=args.split,
                    decoder='greedy CTC; no dictionary or language correction',
                    gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                    rows=[dict(checkpoint=str(args.checkpoint.resolve()), pack=str(args.data.resolve()),
                               split=args.split, research_only=bool(schema.get('research_only')),
                               **{name: summary for name, summary in groups.items()})])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f'{args.report.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(document, indent=2), encoding='utf-8')
    temporary.replace(args.report)
    print(json.dumps(dict(report=str(args.report),
                          **{name: dict(cer=summary['cer'], exact=summary['exact_sequence_accuracy'],
                                        clips=summary['clips'])
                             for name, summary in groups.items()})), flush=True)


if __name__ == '__main__':
    main()
