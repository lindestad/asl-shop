"""Streaming FSboard recognizer from autoresearch: raw MediaPipe landmarks in, CTC letters out.

The network is the one in autoresearch/train.py (exported with autoresearch/tools/export_app.py).
Output at frame t depends only on frames t-history+1 .. t+lookahead, so the app recomputes one
short window per new frame instead of the whole session.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CONTRACT = "asl-shop-raw-xyz-v1"
FPS = 30
# FSboard frames are ~2000 px wide and the network scales MediaPipe z by that width;
# the app feeds z * frame_width / Z_REFERENCE_WIDTH so live z matches the trained scale.
Z_REFERENCE_WIDTH = 2000.0


def raw_frame(hands, width, height):
    """MediaPipe hands -> xyz (2,21,3) [x_px, y_px, z scaled] and present (2,), slot 0 = Right."""
    xyz = np.zeros((2, 21, 3), dtype=np.float32)
    present = np.zeros(2, dtype=bool)
    for slot, side in enumerate(("Right", "Left")):
        matches = [hand for hand in hands if hand["hand"] == side]
        if len(matches) > 1:
            raise ValueError("Use one signer: multiple hands of the same side are ambiguous.")
        if not matches:
            continue
        points = np.array([[p.x, p.y, getattr(p, "z", 0.0)] if hasattr(p, "x") else [p["x"], p["y"], p.get("z", 0.0)]
                           for p in matches[0]["landmarks"]], dtype=np.float32)
        xyz[slot] = points * [width, height, width / Z_REFERENCE_WIDTH]
        present[slot] = True
    return xyz, present


def frame_features(xyz, present):
    """(B,T,2,21,3) + (B,T,2) -> per-slot features (B,T,2,68); per-frame and one frame back only."""
    p = xyz[..., :2]
    wrist = p[..., 0, :]
    rel = p - wrist[..., None, :]
    size = rel[..., 9, :].norm(dim=-1, keepdim=True).clamp_min(1e-3)
    shape = rel / size[..., None]
    shape = shape * torch.tensor([[1.0, 1.0], [-1.0, 1.0]], device=xyz.device)[:, None, :]  # mirror Left x
    shape = shape.flatten(-2).clamp(-4, 4)
    zrel = ((xyz[..., 2] - xyz[..., :1, 2]) * Z_REFERENCE_WIDTH / size).clamp(-4, 4)
    pos = wrist / size
    prev_pos = torch.cat([pos[:, :1], pos[:, :-1]], 1)
    prev_present = torch.cat([present[:, :1], present[:, :-1]], 1)
    vel = (pos - prev_pos) * (present & prev_present)[..., None]
    m = present[..., None].float()
    return torch.cat([shape, zrel, pos.clamp(-32, 32) * 0.1, vel.clamp(-4, 4), torch.ones_like(m)], -1) * m


class CausalBlock(nn.Module):
    def __init__(self, dim, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.pw = nn.Linear(dim, 2 * dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(0.0), nn.Linear(2 * dim, dim))

    def forward(self, x):
        h = self.norm1(x).transpose(1, 2)
        h = self.conv(F.pad(h, (self.pad, 0))).transpose(1, 2)
        x = x + F.glu(self.pw(h), -1)
        return x + self.ff(self.norm2(x))


class StreamNet(nn.Module):
    contract = CONTRACT

    def __init__(self, hidden=256, kernel=5, dilations=(1, 2, 4, 8, 1, 2, 4, 8), lookahead=9, vocab_size=None):
        super().__init__()
        from model.neural import VOCAB
        self.config = dict(hidden=hidden, kernel=kernel, dilations=list(dilations), lookahead=lookahead)
        self.hand = nn.Sequential(nn.Linear(68, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.slot = nn.Parameter(torch.zeros(2, hidden))
        self.blocks = nn.ModuleList(CausalBlock(hidden, kernel, d) for d in dilations)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size or len(VOCAB))
        receptive = 1 + sum((kernel - 1) * d for d in dilations)
        self.lookahead_frames = lookahead
        self.history_frames = max(1, receptive - lookahead)

    def forward(self, xyz, present):
        """xyz (B,T,2,21,3), present (B,T,2) -> logits (B,T,vocab); output t is read at step t+L."""
        L = self.lookahead_frames
        if L:
            xyz = F.pad(xyz, (0, 0, 0, 0, 0, 0, 0, L))
            present = F.pad(present, (0, 0, 0, L))
        f = frame_features(xyz, present)
        x = ((self.hand(f) + self.slot) * present[..., None].to(f.dtype)).sum(2)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))[:, L:]


def load(checkpoint):
    """Build a StreamNet from an exported checkpoint dict (see autoresearch/tools/export_app.py)."""
    from model.neural import VOCAB
    if checkpoint.get("contract") != CONTRACT or checkpoint.get("task") != "sequence":
        raise ValueError(f"Incompatible sequence checkpoint; expected {CONTRACT}.")
    if checkpoint.get("labels") != VOCAB:
        raise ValueError("Checkpoint vocabulary differs from the runtime vocabulary.")
    network = StreamNet(**checkpoint["config"])
    network.load_state_dict({k: v.float() for k, v in checkpoint["state_dict"].items()}, strict=True)
    network.eval()
    network.metrics = checkpoint.get("metrics", {})
    return network


class RawTimeline:
    """30 Hz resampling of raw landmarks with the same rules as model.timeline.TimelineResampler:
    endpoint ticks take the captured frame, ticks in between interpolate a hand only when it is present
    on both sides of a gap of at most max_gap_ms; otherwise the hand is missing at that tick."""

    def __init__(self, fps=FPS, max_gap_ms=250):
        self.period_ms = 1000 / fps
        self.max_gap_ms = max_gap_ms
        self.previous = None
        self.origin = None
        self.next_index = 0

    def feed(self, xyz, present, timestamp_ms):
        if self.previous is None:
            self.previous = (xyz, present, timestamp_ms)
            self.origin, self.next_index = timestamp_ms, 1
            return [(xyz, present)]
        p_xyz, p_present, p_time = self.previous
        interval = timestamp_ms - p_time
        out = []
        while True:
            tick = self.origin + self.next_index * self.period_ms
            if tick > timestamp_ms + 1e-6:
                break
            if abs(tick - timestamp_ms) <= 1e-6:
                out.append((xyz, present))
            else:
                both = p_present & present if interval <= self.max_gap_ms else np.zeros(2, dtype=bool)
                w = (tick - p_time) / interval
                value = (p_xyz * (1 - w) + xyz * w) * both[:, None, None]
                out.append((value.astype(np.float32), both))
            self.next_index += 1
        self.previous = (xyz, present, timestamp_ms)
        return out


class StreamDecoder:
    """Incremental greedy CTC over 30 Hz frames. Frames older than the lookahead are final; the last
    `lookahead` frames are scored with the not-yet-seen future treated as no hands (provisional)."""

    def __init__(self, network):
        self.network = network
        self.xyz, self.present = [], []
        self.final = []          # argmax token per finalized frame
        self.tail = []           # provisional tokens for the newest frames

    def push(self, frames):
        for xyz, present in frames:
            self.xyz.append(xyz)
            self.present.append(present)
        if not self.xyz:
            return
        H, L = self.network.history_frames, self.network.lookahead_frames
        end, done = len(self.xyz), len(self.final)
        start = max(0, done - H + 1)  # oldest frame any not-yet-final output can see
        xyz = torch.from_numpy(np.stack(self.xyz[start:end]))[None]
        present = torch.from_numpy(np.stack(self.present[start:end]))[None]
        with torch.inference_mode():
            tokens = self.network(xyz, present).argmax(-1)[0].tolist()
        ready = max(done, end - L)
        self.final.extend(tokens[done - start:ready - start])
        self.tail = tokens[ready - start:]

    def tokens(self):
        return self.final + self.tail

    def text(self):
        from model.neural import decode_ctc
        return decode_ctc(self.tokens())
