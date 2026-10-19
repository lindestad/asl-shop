"""Versioned neural models. No upstream weights are compatible with this contract."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CONTRACT = "asl-shop-xy-wrist-v2"
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"  # J/Z require a sequence.
HANDSHAPES = ["nothing", *LETTERS]
VOCAB = ["", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !#$%&'()*+,-./:;=?@[_~"]
SEQUENCE_SIZE = 91


def shape_features(points, side):
    """Pixel XY → mirrored, wrist-relative, hand-size-normalized XY."""
    points = np.asarray(points, dtype=np.float32).reshape(21, 2).copy()
    points -= points[0]
    points /= np.linalg.norm(points[9]) + 1e-6
    if side == "Left":
        points[:, 0] *= -1
    return points.reshape(42)


def frame_features(hands, width, height, delta_ms):
    """Stable right/left slots, explicit missing masks and global wrist trajectory."""
    result = np.zeros(SEQUENCE_SIZE, dtype=np.float32)
    for slot, side in enumerate(("Right", "Left")):
        matches = [hand for hand in hands if hand["hand"] == side]
        if len(matches) > 1:
            raise ValueError("Use one signer: multiple hands of the same side are ambiguous.")
        if not matches:
            continue
        points = np.array([[p.x, p.y] if hasattr(p, "x") else [p["x"], p["y"]]
                           for p in matches[0]["landmarks"]], dtype=np.float32)
        base = slot * 45
        pixels = points * [width, height]
        scale = np.linalg.norm(pixels[9] - pixels[0]) + 1e-6
        result[base:base + 42] = shape_features(pixels, side)
        result[base + 42:base + 44] = pixels[0] / scale
        result[base + 44] = 1
    result[-1] = min(max(delta_ms, 0), 1000) / 1000
    return result


class HandshapeNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(42, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                     nn.Dropout(0.15), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, len(HANDSHAPES)))

    def forward(self, frames):
        return self.network(frames.clamp(-4, 4))


class CausalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.padding = 4 * dilation
        self.conv = nn.Conv1d(hidden, hidden, 5, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        value = self.conv(F.pad(x, (self.padding, 0)))
        value = self.norm(value.transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(F.gelu(value))


class FingerspellingNet(nn.Module):
    """Causal temporal CNN; the dilation stack sets how many frames of history each output sees
    (1 + sum(4 * dilation) for these 5-tap causal blocks, i.e. 61 frames for the default stack).
    `dilations` is additive and defaults to the shipped stack, so old checkpoints are unchanged."""
    def __init__(self, hidden=128, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.hand_encoder = nn.Sequential(*list(HandshapeNet(hidden).network.children())[:-1])
        self.hand_gate = nn.Linear(hidden, 1)
        self.input = nn.Linear(hidden + 7, hidden)
        self.blocks = nn.Sequential(*[CausalBlock(hidden, dilation) for dilation in self.dilations])
        self.output = nn.Linear(hidden, len(VOCAB))

    def forward(self, frames):
        right = self.hand_encoder(frames[..., :42])
        left = self.hand_encoder(frames[..., 45:87])
        hands = torch.stack((right, left), dim=-2)
        presence = torch.stack((frames[..., 44], frames[..., 89]), dim=-1)
        attention = self.hand_gate(hands).squeeze(-1).masked_fill(presence == 0, -1000).softmax(-1)
        shape = (hands * presence.unsqueeze(-1) * attention.unsqueeze(-1)).sum(-2)
        trajectory = torch.cat((frames[..., 42:45], frames[..., 87:91]), dim=-1).clamp(-32, 32)
        encoded = self.input(torch.cat((shape, trajectory), dim=-1)).transpose(1, 2)
        return self.output(self.blocks(encoded).transpose(1, 2))


def decode_ctc(indices):
    result, previous = [], None
    for index in indices:
        index = int(index)
        if index and index != previous:
            result.append(VOCAB[index])
        previous = index
    return "".join(result)


def load_checkpoint(path, task):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("contract") != CONTRACT or checkpoint.get("task") != task:
        raise ValueError(f"Incompatible {task} checkpoint; expected {CONTRACT}.")
    labels = HANDSHAPES if task == "handshape" else VOCAB
    if checkpoint.get("labels") != labels:
        raise ValueError("Checkpoint vocabulary differs from the runtime vocabulary.")
    if task == "handshape":
        network = HandshapeNet(checkpoint["hidden"])
    else:
        network = FingerspellingNet(checkpoint["hidden"], checkpoint.get("dilations") or (1, 2, 4, 8))
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.eval()
    return network
