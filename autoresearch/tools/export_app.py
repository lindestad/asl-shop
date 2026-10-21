"""Export an autoresearch run for the app (model/stream_net.py), and check it scores the same.

  .venv/bin/python autoresearch/tools/export_app.py autoresearch/runs/NAME.pt assets/models/fsboard-stream.pt

Weights are stored as float16 (half the size); the validation CER of the exported weights is
recomputed and stored in the checkpoint next to the training-time number.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import prepare  # noqa: E402
from score_test import load  # noqa: E402
from model.neural import VOCAB  # noqa: E402
from model import stream_net  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    trained = load(args.run)
    g = type(trained).__init__.__globals__  # the train.py namespace the run was trained with
    config = dict(hidden=g["HIDDEN"], kernel=g["KERNEL"], dilations=list(g["DILATIONS"]), lookahead=g["LOOKAHEAD"])
    state = {k: v.detach().half().cpu() for k, v in trained.state_dict().items()}
    checkpoint = dict(contract=stream_net.CONTRACT, task="sequence", labels=VOCAB, config=config, state_dict=state)
    exported = stream_net.load(checkpoint).cuda()
    assert (exported.history_frames, exported.lookahead_frames) == (trained.history_frames, trained.lookahead_frames)

    val = prepare.Split("validation")
    before, after = prepare.evaluate(trained, val), prepare.evaluate(exported, val)
    streaming = prepare.check_streaming(exported, val)
    latency = prepare.measure_latency(exported, val)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    checkpoint["metrics"] = dict(run=args.run.stem, commit=commit, val_cer=after["cer"], val_jz_cer=after["jz_cer"],
                                 val_exact=after["exact"], val_cer_float32=before["cer"],
                                 history_frames=streaming["history_frames"], lookahead_frames=streaming["lookahead_frames"],
                                 cpu_ms_p50=latency["cpu_ms_p50"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(checkpoint["metrics"])
    if abs(after["cer"] - before["cer"]) > 5e-4:
        sys.exit(f"exported CER {after['cer']:.4f} differs from trained {before['cer']:.4f}")


if __name__ == "__main__":
    main()
