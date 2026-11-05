#!/usr/bin/env python3
"""Linear weight-space interpolation of two compatible recognizer checkpoints.

Both checkpoints must share contract, task, labels, representation, max_decode and
architecture; the output carries the newer checkpoint's metadata plus a 'soup' record.
With --alpha 0 the output state_dict is byte-identical to --checkpoint-b.

    python scripts/soup_openfs_recognizer.py --checkpoint-a <a.pt> --checkpoint-b <b.pt> \
        --alpha 0.5 --output <out.pt>
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-a', type=Path, required=True)
    p.add_argument('--checkpoint-b', type=Path, required=True)
    p.add_argument('--alpha', type=float, required=True, help='weight of checkpoint-a; 0 reproduces checkpoint-b exactly')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    a = torch.load(args.checkpoint_a, map_location='cpu')
    b = torch.load(args.checkpoint_b, map_location='cpu')
    for key in ('contract', 'task', 'labels', 'representation', 'max_decode', 'architecture'):
        if a[key] != b[key]:
            raise SystemExit(f'checkpoint mismatch on {key}')
    if set(a['state_dict']) != set(b['state_dict']):
        raise SystemExit('state_dict key mismatch')

    state = {}
    for key in a['state_dict']:
        ta, tb = a['state_dict'][key], b['state_dict'][key]
        state[key] = ta * args.alpha + tb * (1.0 - args.alpha) if ta.is_floating_point() else tb

    out = {key: b[key] for key in b if key != 'state_dict'}
    out['soup'] = {'alpha': args.alpha,
                   'checkpoint_a': str(args.checkpoint_a), 'epoch_a': a.get('epoch'),
                   'checkpoint_b': str(args.checkpoint_b), 'epoch_b': b.get('epoch')}
    out['state_dict'] = state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    identity = args.alpha == 0.0 and all(torch.equal(state[k], b['state_dict'][k]) for k in state)
    print(json.dumps({'output': str(args.output), 'alpha': args.alpha,
                      'params': sum(t.numel() for t in state.values() if t.is_floating_point()),
                      'identity_to_b': identity}))


if __name__ == '__main__':
    main()
