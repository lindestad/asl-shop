"""Score a saved run on validation and the held-out test split (final checkpoints only).

  .venv/bin/python autoresearch/tools/score_test.py autoresearch/runs/NAME.pt [--report out.json]

The checkpoint stores the train.py source it was trained with; the model class is rebuilt from it.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import prepare  # noqa: E402


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module = types.ModuleType("trained")
    module.__file__ = str(HERE / "train.py")
    exec(compile(payload["source"], str(path), "exec"), module.__dict__)
    model = module.Recognizer()
    model.load_state_dict(payload["state_dict"])
    return model.cuda().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    model = load(args.checkpoint)
    out = dict(checkpoint=str(args.checkpoint), params_M=sum(p.numel() for p in model.parameters()) / 1e6)
    for name in ("validation", "test"):
        split = prepare.Split(name)
        out[name] = prepare.evaluate(model, split)
        del split
    out["streaming"] = prepare.check_streaming(model, prepare.Split("validation"))
    out["latency"] = prepare.measure_latency(model, prepare.Split("validation"))
    print(json.dumps(out, indent=2))
    if args.report:
        args.report.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
