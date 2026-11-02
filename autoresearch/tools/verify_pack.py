"""verify_pack.py [split ...] : check packed landmarks against the raw FSboard archives, frame by frame.
Run after packing and whenever results look off; the CephFS home once lost memmap writes silently."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare as P

ok = True
for split in sys.argv[1:] or ("validation", "test", "train"):
    clips = json.loads((P.PACKED / f"{split}_clips.json").read_text())
    xyz = np.load(P.PACKED / f"{split}_xyz.npy", mmap_mode="r")
    present = np.load(P.PACKED / f"{split}_present.npy", mmap_mode="r")
    with np.load(P.RAW / f"fsboard_{split}.npz", allow_pickle=True) as z:
        names, hands, indices = z["vidnames"], (z["right_hand_poses"], z["left_hand_poses"]), \
            (z["right_frame_indices"], z["left_frame_indices"])
    row = {str(n): i for i, n in enumerate(names)}
    bad_frames = bad_clips = frames = 0
    for c in clips:
        i, bad = row[c["name"]], False
        for slot in (0, 1):
            poses = np.asarray(hands[slot][i].get(0, []), dtype=np.float32)
            ix = np.asarray(indices[slot][i], dtype=np.int64)
            if not len(poses):
                continue
            got = np.asarray(xyz[c["start"] + ix, slot])
            wrong = (np.abs(got - poses).max(axis=(1, 2)) > 1e-4) | ~np.asarray(present[c["start"] + ix, slot])
            bad_frames += int(wrong.sum()); frames += len(ix); bad |= bool(wrong.any())
        bad_clips += bad
    print(f"{split}: {bad_frames}/{frames} hand-frames wrong, {bad_clips}/{len(clips)} clips affected", flush=True)
    ok &= bad_frames == 0
sys.exit(0 if ok else 1)
