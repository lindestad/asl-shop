"""Streaming fingerspelling recognizer, FSboard, CTC. The one file the autoresearch loop edits.

  uv run --extra neural python autoresearch/train.py > run.log 2>&1
"""
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import TIME_BUDGET, VOCAB, Split, final_report  # noqa: E402

# ---------------------------------------------------------------------------- hyperparameters
SEED = 42
HIDDEN = 256
KERNEL = 5
DILATIONS = (1, 2, 4, 8, 1, 2, 4, 8)
LOOKAHEAD = 9                 # frames of future context (output shift)
FRAMES_PER_BATCH = 160_000    # padded frames per optimizer step
LR = 2e-3
WEIGHT_DECAY = 0.01
WARMUP = 0.03                 # fraction of the time budget
DROPOUT = 0.1
OUT = Path(os.environ.get("RUN_OUT", Path(__file__).resolve().parent / "runs/last.pt"))


# ---------------------------------------------------------------------------- features
def frame_features(xyz, present):
    """Raw (B,T,2,21,3) pixels/z + presence (B,T,2) -> per-slot features (B,T,2,F). Per-frame and
    one-frame-back only, so it is causal and computable in the app."""
    p = xyz[..., :2]
    wrist = p[..., 0, :]
    rel = p - wrist[..., None, :]
    size = rel[..., 9, :].norm(dim=-1, keepdim=True).clamp_min(1e-3)          # (B,T,2,1)
    shape = rel / size[..., None]
    shape = shape * torch.tensor([[1.0, 1.0], [-1.0, 1.0]], device=xyz.device)[:, None, :]  # mirror Left x
    shape = shape.flatten(-2).clamp(-4, 4)                                      # (B,T,2,42)
    pos = wrist / size
    prev_pos = torch.cat([pos[:, :1], pos[:, :-1]], 1)
    prev_present = torch.cat([present[:, :1], present[:, :-1]], 1)
    vel = (pos - prev_pos) * (present & prev_present)[..., None]
    m = present[..., None].float()
    feats = torch.cat([shape, pos.clamp(-32, 32) * 0.1, vel.clamp(-4, 4), torch.ones_like(m)], -1) * m
    return feats


# ---------------------------------------------------------------------------- model
class CausalBlock(nn.Module):
    def __init__(self, dim, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.pw = nn.Linear(dim, 2 * dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):                               # (B,T,C)
        h = self.norm1(x).transpose(1, 2)
        h = self.conv(F.pad(h, (self.pad, 0))).transpose(1, 2)
        x = x + self.drop(F.glu(self.pw(h), -1))
        return x + self.drop(self.ff(self.norm2(x)))


class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        n_in = 42 + 2 + 2 + 1
        self.hand = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.slot = nn.Parameter(torch.zeros(2, HIDDEN))
        self.blocks = nn.ModuleList(CausalBlock(HIDDEN, KERNEL, d) for d in DILATIONS)
        self.norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, len(VOCAB))
        receptive = 1 + sum((KERNEL - 1) * d for d in DILATIONS)
        self.lookahead_frames = LOOKAHEAD
        self.history_frames = max(1, receptive - LOOKAHEAD)

    def forward(self, xyz, present):
        if self.lookahead_frames:                       # output t is read at causal step t+L
            xyz = F.pad(xyz, (0, 0, 0, 0, 0, 0, 0, self.lookahead_frames))
            present = F.pad(present, (0, 0, 0, self.lookahead_frames))
        f = frame_features(xyz, present)
        m = present[..., None].to(f.dtype)
        x = ((self.hand(f) + self.slot) * m).sum(2)     # (B,T,C)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        return logits[:, self.lookahead_frames:]


# ---------------------------------------------------------------------------- augmentation (GPU)
def augment(xyz, present, lengths):
    B, T = xyz.shape[:2]
    dev = xyz.device
    # Mirror the image: swap hand slots and flip x around the frame centre.
    flip = torch.rand(B, device=dev) < 0.5
    xyz = torch.where(flip[:, None, None, None, None], xyz.flip(2) * torch.tensor([-1.0, 1, 1], device=dev), xyz)
    present = torch.where(flip[:, None, None], present.flip(2), present)
    # Per-clip affine on pixel xy: rotation, scale, shear.
    a = torch.randn(B, device=dev) * 0.15
    s = torch.exp(torch.randn(B, device=dev) * 0.1)
    sh = torch.randn(B, device=dev) * 0.1
    M = torch.stack([torch.cos(a) * s, -torch.sin(a) * s + sh, torch.sin(a) * s, torch.cos(a) * s], -1).view(B, 1, 1, 1, 2, 2)
    xy = (M @ xyz[..., :2, None]).squeeze(-1)
    xy = xy + torch.randn(B, T, 2, 21, 2, device=dev) * 2.0     # pixel jitter
    xyz = torch.cat([xy, xyz[..., 2:]], -1)
    # Random frame drops (simulate missed detections / lower capture rate).
    keep = torch.rand(B, T, 1, device=dev) > torch.rand(B, 1, 1, device=dev) * 0.2
    present = present & keep
    return xyz, present


def time_stretch(xyz, present, lengths, texts_len):
    """Resample each clip by a random factor with nearest-frame indexing."""
    B, T = xyz.shape[:2]
    dev = xyz.device
    factor = torch.empty(B, device=dev).uniform_(0.8, 1.25)
    new_len = (lengths.float() * factor).round().long()
    new_len = torch.maximum(new_len, texts_len * 2 + 2).clamp(max=int(T * 1.25) + 1)
    T2 = int(new_len.max())
    steps = torch.arange(T2, device=dev)[None].float()
    src = (steps * (lengths.float() / new_len.float())[:, None]).floor().long()
    src = torch.minimum(src, (lengths - 1)[:, None]).clamp(min=0)
    valid = steps < new_len[:, None]
    bi = torch.arange(B, device=dev)[:, None]
    xyz = xyz[bi, src] * valid[..., None, None, None]
    present = present[bi, src] & valid[..., None]
    return xyz, present, new_len


# ---------------------------------------------------------------------------- training
def batches(split, generator):
    """Length-bucketed random batches under a padded-frame budget, forever."""
    lengths = split.length.cpu()
    while True:
        perm = torch.randperm(len(split), generator=generator)
        for chunk in perm.split(4096):
            chunk = chunk[lengths[chunk].argsort()]
            groups, cur, cur_max = [], [], 0
            for idx in chunk.tolist():
                L = int(lengths[idx])
                if cur and max(cur_max, L) * (len(cur) + 1) > FRAMES_PER_BATCH:
                    groups.append(cur); cur, cur_max = [], 0
                cur.append(idx); cur_max = max(cur_max, L)
            if cur:
                groups.append(cur)
            for g in torch.randperm(len(groups), generator=generator).tolist():
                yield groups[g]


def main():
    started = time.perf_counter()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train, val = Split("train"), Split("validation")
    model = Recognizer().cuda()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M history {model.history_frames} "
          f"lookahead {model.lookahead_frames} train clips {len(train)}", flush=True)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0)],
                            lr=LR, betas=(0.9, 0.98), fused=True)
    gen = torch.Generator().manual_seed(SEED)
    stream = batches(train, gen)
    t0 = time.perf_counter()
    step, smooth = 0, None
    while True:
        elapsed = time.perf_counter() - t0
        progress = elapsed / TIME_BUDGET
        if progress >= 1:
            break
        lr = LR * (progress / WARMUP if progress < WARMUP else 0.5 * (1 + math.cos(math.pi * (progress - WARMUP) / (1 - WARMUP))))
        for g in opt.param_groups:
            g["lr"] = lr
        ids = next(stream)
        model.train()
        xyz, present, lengths = train.batch(ids)
        tlen = torch.tensor([len(train.texts[i]) for i in ids], device="cuda")
        xyz, present, lengths = time_stretch(xyz, present, lengths, tlen)
        xyz, present = augment(xyz, present, lengths)
        targets = torch.cat([train.targets[i] for i in ids]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xyz, present)
        logp = logits.float().log_softmax(-1).transpose(0, 1)
        loss = F.ctc_loss(logp, targets, lengths, tlen, blank=0, reduction="mean", zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        smooth = float(loss) if smooth is None else 0.98 * smooth + 0.02 * float(loss) if step % 10 == 0 else smooth
        if step % 200 == 0:
            print(f"step {step} {elapsed:.0f}s lr {lr:.2e} loss {smooth:.4f}", flush=True)
    train_seconds = time.perf_counter() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), source=Path(__file__).read_text()), OUT)
    final_report(model, val, started, extra=dict(steps=step, train_seconds=train_seconds))


if __name__ == "__main__":
    main()
