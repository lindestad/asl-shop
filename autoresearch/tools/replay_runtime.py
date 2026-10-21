"""Replay validation clips frame by frame through the app's live runtime (model/live_neural.py).

  .venv/bin/python autoresearch/tools/replay_runtime.py assets/models/fsboard-stream.pt --clips 500

Landmarks go in as MediaPipe would report them for a 1920x1080 camera, at 30 fps, so this checks
slot mapping, z scaling, resampling, incremental decoding and the app's text gating together.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import prepare  # noqa: E402
from model.live_neural import NeuralRuntime  # noqa: E402
from model.stream_net import Z_REFERENCE_WIDTH  # noqa: E402

W, H = 1920, 1080


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--clips", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    clips = json.loads((prepare.PACKED / "validation_clips.json").read_text())
    xyz = np.load(prepare.PACKED / "validation_xyz.npy", mmap_mode="r")
    present = np.load(prepare.PACKED / "validation_present.npy", mmap_mode="r")
    runtime = NeuralRuntime(HERE.parent)
    runtime.paths = {"sequence": args.model, "handshape": HERE / "missing.pt"}
    pick = np.random.default_rng(args.seed).choice(len(clips), size=min(args.clips, len(clips)), replace=False)
    errors = chars = exact = blanked = frames = 0
    started = time.perf_counter()
    for n, k in enumerate(pick):
        clip = clips[int(k)]
        T = clip["end"] - clip["start"]
        result = None
        for t in range(T):
            hands = []
            for slot, side in enumerate(("Right", "Left")):
                if present[clip["start"] + t, slot]:
                    p = xyz[clip["start"] + t, slot]
                    hands.append({"hand": side, "landmarks": [
                        SimpleNamespace(x=float(x) / W, y=float(y) / H, z=float(z) * Z_REFERENCE_WIDTH / W) for x, y, z in p]})
            result = runtime.analyze(hands, W, H, video=True, session_id=f"clip{n}", timestamp_ms=1000.0 + t * 1000 / 30,
                                     finish=t == T - 1)
        text = result["fingerspelling"]["text"]
        assert result["fingerspelling"]["method"] == "fsboard-stream", result["fingerspelling"]
        errors += prepare.edit_distance(clip["text"], text)
        chars += len(clip["text"])
        exact += text == clip["text"]
        blanked += not text
        frames += T
    seconds = time.perf_counter() - started
    print(json.dumps(dict(clips=len(pick), runtime_cer=errors / chars, exact=exact / len(pick), empty_outputs=blanked,
                          ms_per_frame=1000 * seconds / frames)))


if __name__ == "__main__":
    main()
