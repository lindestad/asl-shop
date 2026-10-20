"""Fixed harness for the FSboard streaming autoresearch loop. DO NOT MODIFY during experiments.

  uv run --extra neural python autoresearch/prepare.py      # one-off: pack the raw OpenFS landmarks

Packs the official OpenFS FSboard landmark archive (raw MediaPipe hand landmarks, original frame
indices) joined with the official FSboard signer metadata, keeping the same clip filter as
scripts/prepare_openfs_training.py so the test split is the same 12,521 clips. No model-derived
labels are stored: supervision is the human transcript only.

Per frame and per hand slot (0 = Right, 1 = Left) we keep raw pixel x, y and MediaPipe z, plus a
presence flag. That is exactly what the app's Hand Landmarker gives at runtime (x*width, y*height,
z, handedness), so any feature computed from it inside the model can be computed in the app.

The ground-truth metric is `evaluate()` (greedy CTC CER on the signer-disjoint validation split).
`check_streaming()` and `measure_latency()` enforce the app constraints: output at frame t may only
depend on frames t-H+1 .. t+L with H <= MAX_HISTORY and L <= MAX_LOOKAHEAD, and recomputing one
window on 4 CPU threads must fit CPU_BUDGET_MS.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import VOCAB  # noqa: E402

RAW = ROOT / "data/external/openfs/extracted/fingerspelling_fsboard_raw"
METADATA = ROOT / "data/external/openfs/metadata"
PACKED = ROOT / "data/autoresearch"

FPS = 30
TIME_BUDGET = 15 * 60          # seconds of training per experiment (excluding startup and eval)
MAX_HISTORY = 300              # frames (10 s) the app may hold for recomputation
MAX_LOOKAHEAD = 9              # frames (0.3 s) of future context allowed
CPU_BUDGET_MS = 15.0           # one window recompute on 4 CPU threads
MAX_FRAMES = 3000
XYZ_SCALE = np.array([4.0, 4.0, 1e4], dtype=np.float32)


# ---------------------------------------------------------------------------- packing

def _metadata():
    rows = {}
    for path in sorted(METADATA.glob("*-*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows[Path(row["clipFilename"]).stem] = row
    return rows


def _pack_split(split, metadata):
    started = time.perf_counter()
    with np.load(RAW / f"fsboard_{split}.npz", allow_pickle=True) as z:  # official archive, trusted
        names, words, right, left, ri, li = (z[k] for k in (
            "vidnames", "words", "right_hand_poses", "left_hand_poses",
            "right_frame_indices", "left_frame_indices"))
    plans, offsets, skipped = [], [0], {}
    for i, name in enumerate(names):
        row = metadata[str(name)]
        text = str(words[i]).upper().strip()
        assert text == row["phrase"].upper().strip(), name
        if not text or any(c not in VOCAB[1:] for c in text):
            skipped["unsupported_transcript"] = skipped.get("unsupported_transcript", 0) + 1
            continue
        length = max(int(ix[-1]) + 1 if len(ix) else 0 for ix in (ri[i], li[i]))
        required = len(text) + sum(a == b for a, b in zip(text, text[1:]))
        if length > MAX_FRAMES or length < required:
            skipped["length"] = skipped.get("length", 0) + 1
            continue
        plans.append((i, text))
        offsets.append(offsets[-1] + length)
    total = offsets[-1]
    xyz = np.lib.format.open_memmap(PACKED / f"{split}_xyz.npy", mode="w+", dtype=np.float32, shape=(total, 2, 21, 3))
    present = np.lib.format.open_memmap(PACKED / f"{split}_present.npy", mode="w+", dtype=np.bool_, shape=(total, 2))
    present[:] = False
    clips = []
    for k, (i, text) in enumerate(plans):
        start = offsets[k]
        for slot, (hands, indices) in enumerate(((right[i], ri[i]), (left[i], li[i]))):
            assert set(hands) <= {0}, "single-person schema expected"
            poses = np.asarray(hands.get(0, []), dtype=np.float32)
            indices = np.asarray(indices, dtype=np.int64)
            if not len(poses):
                continue
            assert poses.shape[1:] == (21, 3) and len(poses) == len(indices)
            xyz[start + indices, slot] = poses
            present[start + indices, slot] = True
        row = metadata[str(names[i])]
        clips.append(dict(name=str(names[i]), text=text, signer=row["signerId"],
                          session=row["fullVideoFilename"], start=start, end=offsets[k + 1]))
    xyz.flush(); present.flush()
    (PACKED / f"{split}_clips.json").write_text(json.dumps(clips), encoding="utf-8")
    info = dict(split=split, clips=len(clips), frames=total, signers=len({c["signer"] for c in clips}),
                skipped=skipped, seconds=round(time.perf_counter() - started, 1))
    print(json.dumps(info), flush=True)
    return info


def pack():
    PACKED.mkdir(parents=True, exist_ok=True)
    metadata = _metadata()
    infos = [_pack_split(split, metadata) for split in ("validation", "test", "train")]
    signers = [{c["signer"] for c in json.loads((PACKED / f"{s}_clips.json").read_text())}
               for s in ("validation", "test", "train")]
    assert not (signers[0] & signers[1] or signers[0] & signers[2] or signers[1] & signers[2]), "signer leak"
    (PACKED / "dataset.json").write_text(json.dumps(dict(
        vocab=VOCAB, fps=FPS, partitions=infos,
        layout="xyz[frame, slot(0=Right,1=Left), joint(21), (x_px, y_px, z_mediapipe)], present[frame, slot]",
        labels="human transcripts only; no model-derived frame labels"), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------- loading

class Split:
    """A packed split held on `device`. xyz: (N,2,21,3) float32, present: (N,2) bool."""

    def __init__(self, split, device="cuda"):
        self.name = split
        self.clips = json.loads((PACKED / f"{split}_clips.json").read_text(encoding="utf-8"))
        # int16 on device: x, y at 1/4 pixel, z at 1e-4 (halves GPU memory, error far below jitter)
        raw = np.load(PACKED / f"{split}_xyz.npy", mmap_mode="r")
        q = np.empty(raw.shape, dtype=np.int16)
        for i in range(0, len(raw), 1 << 22):
            chunk = raw[i:i + (1 << 22)] * XYZ_SCALE
            q[i:i + (1 << 22)] = np.clip(np.round(chunk), -32767, 32767)
        self.xyz = torch.from_numpy(q).to(device)
        self.scale = torch.tensor(1.0 / XYZ_SCALE, dtype=torch.float32, device=device)
        self.present = torch.from_numpy(np.load(PACKED / f"{split}_present.npy")).to(device)
        self.start = torch.tensor([c["start"] for c in self.clips], device=device)
        self.length = torch.tensor([c["end"] - c["start"] for c in self.clips], device=device)
        self.texts = [c["text"] for c in self.clips]
        self.targets = [torch.tensor([VOCAB.index(ch) for ch in t], dtype=torch.long) for t in self.texts]
        self.device = device

    def __len__(self):
        return len(self.clips)

    def batch(self, ids, max_len=None):
        """Right-padded raw batch for clip indices `ids` (tensor/list). Padding = no hands present.
        Returns xyz (B,T,2,21,3), present (B,T,2), lengths (B,)."""
        ids = torch.as_tensor(ids, device=self.device)
        lengths = self.length[ids]
        T = int(lengths.max()) if max_len is None else max_len
        steps = torch.arange(T, device=self.device)
        mask = steps[None] < lengths[:, None]
        rows = (self.start[ids, None] + steps[None]).clamp(max=len(self.xyz) - 1)
        xyz = self.xyz[rows].float() * self.scale * mask[..., None, None, None]
        present = self.present[rows] & mask[..., None]
        return xyz, present, lengths


# ---------------------------------------------------------------------------- metric

def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def greedy_decode(indices):
    out, prev = [], None
    for i in indices:
        if i and i != prev:
            out.append(VOCAB[i])
        prev = i
    return "".join(out)


@torch.inference_mode()
def evaluate(model, split, batch_frames=400_000, return_predictions=False):
    """Greedy-CTC CER over every clip of `split`. model(xyz, present) -> logits (B,T,len(VOCAB))."""
    model.eval()
    order = sorted(range(len(split)), key=lambda i: int(split.length[i]))
    errors = chars = exact = jz_errors = jz_chars = 0
    predictions = {}
    i = 0
    while i < len(order):
        T = int(split.length[order[i]])
        j = i
        while j < len(order) and (j - i + 1) * int(split.length[order[j]]) <= batch_frames:
            j += 1
        j = max(j, i + 1)
        ids = order[i:j]
        xyz, present, lengths = split.batch(ids)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xyz.is_cuda):
            logits = model(xyz, present)
        best = logits.argmax(-1).cpu().numpy()
        for k, idx in enumerate(ids):
            text = split.texts[idx]
            pred = greedy_decode(best[k, :int(lengths[k])].tolist())
            e = edit_distance(text, pred)
            errors += e; chars += len(text); exact += pred == text
            if "J" in text or "Z" in text:
                jz_errors += e; jz_chars += len(text)
            if return_predictions:
                predictions[split.clips[idx]["name"]] = pred
        i = j
    result = dict(cer=errors / chars, exact=exact / len(split), jz_cer=jz_errors / max(jz_chars, 1),
                  clips=len(split), characters=chars)
    if return_predictions:
        result["predictions"] = predictions
    return result


# ---------------------------------------------------------------------------- app constraints

@torch.inference_mode()
def check_streaming(model, split, clips=12, points=8, seed=0, atol=2e-3):
    """Output at t computed on the whole clip must match the output computed from the window
    t-H+1 .. t+L alone (on CPU, float32). H = model.history_frames, L = model.lookahead_frames."""
    H, L = int(model.history_frames), int(model.lookahead_frames)
    assert 1 <= H <= MAX_HISTORY, f"history_frames {H} > {MAX_HISTORY}"
    assert 0 <= L <= MAX_LOOKAHEAD, f"lookahead_frames {L} > {MAX_LOOKAHEAD}"
    cpu = _cpu_copy(model)
    rng = np.random.default_rng(seed)
    long_ids = [i for i in range(len(split)) if int(split.length[i]) > H + 2 * L + 20]
    worst = 0.0
    for idx in rng.choice(long_ids, size=min(clips, len(long_ids)), replace=False):
        xyz, present, lengths = (t.cpu() for t in split.batch([int(idx)]))
        T = int(lengths[0])
        full = cpu(xyz, present)[0]
        for t in rng.choice(T, size=points, replace=False).tolist() + [T - 1]:
            s, e = max(0, t - H + 1), min(T, t + L + 1)
            window = cpu(xyz[:, s:e], present[:, s:e])[0]
            diff = float((full[t] - window[t - s]).abs().max())
            worst = max(worst, diff)
            if diff > atol * max(1.0, float(full[t].abs().max())):
                raise AssertionError(f"streaming violation at t={t}/{T}: max diff {diff:.4g} (H={H}, L={L})")
    return dict(history_frames=H, lookahead_frames=L, max_abs_diff=worst)


@torch.inference_mode()
def measure_latency(model, split, threads=4, repeats=40):
    """Median ms to recompute one H+L frame window on `threads` CPU threads, float32."""
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        cpu = _cpu_copy(model)
        W = int(model.history_frames) + int(model.lookahead_frames)
        idx = max(range(len(split)), key=lambda i: int(split.length[i]) if int(split.length[i]) <= 1000 else 0)
        xyz, present, _ = (t.cpu() for t in split.batch([idx]))
        xyz, present = xyz[:, :W].contiguous(), present[:, :W].contiguous()
        for _ in range(5):
            cpu(xyz, present)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter(); cpu(xyz, present); times.append((time.perf_counter() - t0) * 1000)
        return dict(window=W, cpu_ms_p50=float(np.median(times)), cpu_ms_p90=float(np.percentile(times, 90)),
                    threads=threads, within_budget=float(np.median(times)) <= CPU_BUDGET_MS)
    finally:
        torch.set_num_threads(previous)


def _cpu_copy(model):
    import copy
    cpu = copy.deepcopy(getattr(model, "_orig_mod", model)).float().cpu().eval()
    return cpu


def final_report(model, val, started, extra=None):
    """Prints the summary block the loop greps for. Call once at the end of train.py."""
    metrics = evaluate(model, val)
    streaming = check_streaming(model, val)
    latency = measure_latency(model, val)
    params = sum(p.numel() for p in model.parameters())
    out = dict(val_cer=metrics["cer"], val_exact=metrics["exact"], val_jz_cer=metrics["jz_cer"],
               params_M=params / 1e6, cpu_ms=latency["cpu_ms_p50"], within_budget=latency["within_budget"],
               history=streaming["history_frames"], lookahead=streaming["lookahead_frames"],
               peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
               total_seconds=time.perf_counter() - started, **(extra or {}))
    print("---")
    for k, v in out.items():
        print(f"{k + ':':16s} {v:.6f}" if isinstance(v, float) else f"{k + ':':16s} {v}")
    return out


if __name__ == "__main__":
    pack()
