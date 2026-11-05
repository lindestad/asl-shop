"""Dataset clip replay for the no-camera demo mode.

Emits JSON landmark frames reconstructed from the packed OpenFS timeline
(contract v2, the ASL_PACKED_DIR directory). The inverse of model.neural
shape_features is exact up to float error, so feeding these landmarks through
the worker's landmark path reproduces the packed 91-value model frames, and the
runtime then runs exactly the live pipeline: TCN provisional text plus the
ensemble finish-time correction.

  python scripts/demo_replay.py --list
  python scripts/demo_replay.py --clip INDEX
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

PACKED = Path(os.environ.get("ASL_PACKED_DIR", ROOT / "data" / "openfs-packed"))
# Optional source footage (googleai/fsboard on Kaggle, CC BY 4.0); demo mode plays
# without video when a clip's mp4 is missing.
CLIP_VIDEOS = Path(os.environ.get("ASL_DEMO_VIDEO_DIR", ROOT / "data" / "external" / "openfs" / "demo-clips"))
CANVAS = 1000.0  # square replay canvas; landmarks travel normalized by its size
HAND_SCALE = 200.0  # wrist-to-MCP distance in canvas pixels for reconstructed hands

# fixed demo selection: shortest, median, first both-hands clip, longest, then
# curated real-word spellings (JAMES/ZEBULON test the J/Z motion gates; the rest
# are plain words). Clip indices are positions in the packed test split, which is
# itself deterministic, so the selection is reproducible from the packed data alone.
def pick_indices(clips, x):
    lengths = [c["end"] - c["start"] for c in clips]
    order = sorted(range(len(clips)), key=lambda i: lengths[i])
    both = None
    for i in range(len(clips)):
        seg = x[clips[i]["start"]:clips[i]["end"]]
        if bool(np.any((seg[:, 44] > 0) & (seg[:, 89] > 0))):
            both = i
            break
    words = [6689,  # JAMES FRENCH
             528,   # ZEBULON FELIX
             9321,  # SURPRISE ARIZONA
             5086,  # CRUZ HOUSTON
             4331,  # SEATTLE WASHINGTON
             7993]  # HOLLYWOOD FLORIDA
    indices = [order[0], order[len(order) // 2], both, order[-1]]
    return indices + [i for i in words if i not in indices]


def unshape(shape21, side):
    """model.neural.shape_features inverse: wrist-relative normalized XY back to
    pixel offsets (the x mirror is its own inverse)."""
    points = np.asarray(shape21, dtype=np.float32).reshape(21, 2).copy()
    if side == "Left":
        points[:, 0] *= -1
    return points


def clip_frames(row, x):
    frames = []
    for frame_index, values in enumerate(x[row["start"]:row["end"]]):
        hands = []
        for slot, side in enumerate(("Right", "Left")):
            base = slot * 45
            if values[base + 44] <= 0:
                continue
            wrist = values[base + 42:base + 44] * HAND_SCALE
            shape = unshape(values[base:base + 42], side)
            pixels = wrist + shape * HAND_SCALE
            hands.append({"hand": side,
                          "points": [[float(px / CANVAS), float(py / CANVAS)] for px, py in pixels]})
        frames.append({"t": round(frame_index * 1000 / 30), "hands": hands})
    return frames


def main():
    parser = argparse.ArgumentParser(description="Replay packed dataset clips as landmark frames.")
    parser.add_argument("--list", action="store_true", help="list the fixed demo clip selection")
    parser.add_argument("--clip", type=int, help="index into the demo selection")
    parser.add_argument("--by-index", type=int, dest="by_index",
                        help="replay an arbitrary packed clip index, bypassing the fixed selection")
    args = parser.parse_args()
    if not (PACKED / "test_clips.json").is_file():
        raise SystemExit(f"Packed dataset not found at {PACKED}; set ASL_PACKED_DIR.")
    clips = json.loads((PACKED / "test_clips.json").read_text(encoding="utf-8"))
    x = np.load(PACKED / "test_frames.npy", mmap_mode="r")
    if args.by_index is not None:
        if not 0 <= args.by_index < len(clips):
            raise SystemExit(f"Clip index must be between 0 and {len(clips) - 1}.")
        print(json.dumps(dict(index=args.by_index,
                              **{k: clips[args.by_index][k] for k in ("name", "signer", "text")},
                              video=(CLIP_VIDEOS / f"{clips[args.by_index]['name']}.mp4").name
                              if (CLIP_VIDEOS / f"{clips[args.by_index]['name']}.mp4").is_file() else None,
                              frames=clip_frames(clips[args.by_index], x))))
        return
    indices = pick_indices(clips, x)
    if args.list or args.clip is None:
        print(json.dumps([dict(index=i, name=clips[i]["name"], signer=clips[i]["signer"],
                               frames=clips[i]["end"] - clips[i]["start"],
                               durationMs=round((clips[i]["end"] - clips[i]["start"]) * 1000 / 30),
                               text=clips[i]["text"],
                               video=(CLIP_VIDEOS / f"{clips[i]['name']}.mp4").is_file())
                          for i in indices]))
        return
    if not 0 <= args.clip < len(indices):
        raise SystemExit(f"Demo clip must be between 0 and {len(indices) - 1}.")
    index = indices[args.clip]
    video_name = f"{clips[index]['name']}.mp4"
    print(json.dumps(dict(index=index, **{k: clips[index][k] for k in ("name", "signer", "text")},
                          video=video_name if (CLIP_VIDEOS / video_name).is_file() else None,
                          frames=clip_frames(clips[index], x))))


if __name__ == "__main__":
    main()
