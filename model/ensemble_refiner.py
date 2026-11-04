"""Final-text refinement by the local four-member OpenFS-trained recogniser ensemble.

The checkpoints are the research/auto-research ensemble (3-layer cosine+EMA epoch-62,
6-layer step epoch-82, 6-layer cosine+EMA soup, 9-layer trajectory soup; 70,039,280
parameters total). Unlike the optional upstream OpenFS adapter this model lives here,
so refinement needs no external checkout; the model directory holds the .pt files.
Decode mirrors scripts/beam_eval_openfs_recognizer.py: beam search over the
length-normalized... same sum-of-log-probabilities score, equal ensemble weights,
fusion as the weighted mean of per-model frame log-posteriors, no dictionary or
language correction (argmax-score MAP output only).
"""
import math
import os
from pathlib import Path

import numpy as np


def timeline_tokens(values):
    """Live 91-value model frames to (pose, segments, positions), matching
    scripts.train_openfs_recognizer.timeline_tokens for the gaps representation."""
    right, left = values[:, 44] > 0, values[:, 89] > 0
    dominant = 0 if right.sum() >= left.sum() else 1
    segments = np.full(len(values), 2, dtype=np.int64)
    segments[right] = 0
    segments[left] = 1
    segments[right & left] = dominant
    positions = np.cumsum(values[:, -1], dtype=np.float64) * 30
    return values.astype(np.float32), segments, positions


class EnsembleRefiner:
    """Env-gated exactly like OpenFSRefiner: set ASL_ENSEMBLE_MODEL_DIR to a directory
    containing the ensemble .pt checkpoints; they load lazily on the first refinement."""

    def __init__(self, model_directory=None):
        self.directory = model_directory or os.environ.get("ASL_ENSEMBLE_MODEL_DIR")
        self.enabled = bool(self.directory)
        self.models = None
        self.max_chars = None
        self.device = None
        self.error = None

    def readiness(self):
        if not self.enabled:
            return {"enabled": False, "status": "disabled", "available": False}
        error = self.error
        checkpoints = sorted(Path(self.directory).glob("*.pt")) if self.directory else []
        if not checkpoints:
            error = "ASL_ENSEMBLE_MODEL_DIR contains no .pt checkpoints."
        return {"enabled": True, "available": self.models is not None and error is None,
                "members": len(checkpoints),
                "status": "error" if error else "ready" if self.models is not None else "configured",
                **({"error": error} if error else {})}

    def _load(self):
        import torch
        from model.neural import CONTRACT
        from model.offline_recognizer import Recognizer
        checkpoints = sorted(Path(self.directory).glob("*.pt"))
        payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in checkpoints]
        for path, payload in zip(checkpoints, payloads):
            if payload.get("contract") != CONTRACT:
                raise ValueError(f"Checkpoint contract mismatch: {path}.")
            if payload.get("representation") != "gaps":
                raise ValueError(f"Ensemble members must use the gaps timeline: {path}.")
        labels = payloads[0]["labels"]
        if any(payload["labels"] != labels for payload in payloads):
            raise ValueError("Ensemble members must share the same label vocabulary.")
        self.max_chars = payloads[0].get("max_decode", 48)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []
        for payload in payloads:
            model = Recognizer(**payload["architecture"]).to(self.device)
            model.load_state_dict(payload["state_dict"], strict=True)
            self.models.append(model.eval())

    def refine(self, model_frames):
        import torch
        from model.offline_recognizer import decode_text
        values = np.asarray(model_frames, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 91 or not np.isfinite(values).all():
            raise ValueError("Ensemble refinement expects the 91-value model-frame timeline.")
        if not 1 <= len(values) <= 900:
            raise ValueError("Ensemble refinement is limited to the bounded captured segment.")
        status = self.readiness()
        if status.get("error"):
            raise ValueError(status["error"])
        try:
            if self.models is None:
                self._load()
            pose, segments, positions = timeline_tokens(values)
            pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
            segments = torch.from_numpy(segments).unsqueeze(0).to(self.device)
            positions = torch.from_numpy(positions).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                decoded = _beam_decode(self.models, pose, segments, positions, 8, self.max_chars)
            return decode_text(decoded[0])
        except Exception as error:
            self.error = str(error)
            raise


def _beam_decode(models, poses, segments, positions, beam_size, max_chars):
    """Single-segment MAP beam search ported line-for-line from
    scripts/beam_eval_openfs_recognizer.py (n=1, equal weights, logprob fusion,
    length penalty 0). Rotations, LM shallow-fusion and the MBR pool are eval-harness
    features and stay out of the runtime; shapes are kept identical to the harness so
    tie-breaking during top-k stays bit-compatible."""
    import torch
    from torch.nn import functional as F
    from model.offline_recognizer import BOS, EOS, PAD_CHAR

    n = 1
    encoded = []
    for model in models:
        memory, padding = model.encode(poses, segments, positions)
        encoded.append((memory.repeat_interleave(beam_size, dim=0),
                        padding.repeat_interleave(beam_size, dim=0)))
    device = poses.device
    samples = torch.arange(n, device=device).repeat_interleave(beam_size)
    prefixes = torch.full((n * beam_size, 1), BOS, dtype=torch.long, device=device)
    scores = torch.zeros(n, beam_size, device=device)
    active = torch.ones(n, beam_size, dtype=torch.bool, device=device)
    finished_scores = torch.full((n,), -float("inf"), device=device)
    completed_prefixes = torch.full((n, max_chars + 1), PAD_CHAR, dtype=torch.long, device=device)
    completed_lengths = torch.zeros(n, dtype=torch.long, device=device)
    for _ in range(max_chars):
        length = prefixes.shape[1]
        logp = None
        for model, (memory, padding) in zip(models, encoded):
            logits, _ = model.decode(memory, padding, prefixes)
            model_logp = F.log_softmax(logits[:, -1].float(), dim=-1)
            logp = model_logp if logp is None else logp + model_logp
        logp = (logp / len(models)).view(n, beam_size, -1)
        candidate = (scores.unsqueeze(-1)
                     + logp.masked_fill(~active.unsqueeze(-1), -float("inf"))).flatten(1)
        top, indices = torch.topk(candidate, beam_size, dim=-1)
        vocab = candidate.shape[-1] // beam_size
        source = indices // vocab
        token = (indices % vocab).reshape(-1)
        source_flat = source.reshape(-1)
        src_rows = torch.add(samples * beam_size, source_flat)
        score_flat = top.reshape(-1)
        valid = active.gather(1, source) & torch.isfinite(score_flat).reshape(n, beam_size)
        valid_flat = valid.reshape(-1)
        cand_prefixes = torch.full((n * beam_size, length + 1), PAD_CHAR, dtype=torch.long, device=device)
        if valid_flat.any():
            valid_rows = valid_flat.nonzero(as_tuple=True)[0]
            cand_prefixes[valid_rows, :length] = prefixes[src_rows[valid_rows]]
            cand_prefixes[valid_rows, length] = token[valid_rows]
        cand_scores = score_flat.reshape(n, beam_size)
        eos_mask = valid & (token.reshape(n, beam_size) == EOS)
        eos_scores = torch.where(eos_mask, cand_scores, torch.full_like(cand_scores, -float("inf")))
        best_eos_scores, best_eos = eos_scores.max(dim=-1)
        improved = best_eos_scores > finished_scores
        if improved.any():
            rows = improved.nonzero(as_tuple=True)[0]
            completed_prefixes[rows, :length + 1] = cand_prefixes[rows * beam_size + best_eos[rows]]
            completed_lengths[rows] = length + 1
            finished_scores[rows] = best_eos_scores[rows]
        active_mask = valid & (token.reshape(n, beam_size) != EOS)
        new_prefixes = torch.full((n * beam_size, length + 1), PAD_CHAR, dtype=torch.long, device=device)
        new_scores = torch.full((n, beam_size), -float("inf"), device=device)
        new_active = torch.zeros(n, beam_size, dtype=torch.bool, device=device)
        active_flat = active_mask.reshape(-1)
        if active_flat.any():
            active_rows = active_flat.nonzero(as_tuple=True)[0]
            keys = torch.cat((cand_prefixes[active_rows], samples[active_rows].unsqueeze(-1)), dim=1)
            unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
            unique_scores = torch.full((unique_keys.shape[0],), -float("inf"), device=device)
            unique_scores.scatter_reduce_(0, inverse, score_flat[active_flat], reduce="amax", include_self=True)
            unique_samples = unique_keys[:, -1]
            unique_prefixes = unique_keys[:, :-1]
            for sample in unique_samples.unique().tolist():
                selected = unique_samples == sample
                selected_indices = selected.nonzero(as_tuple=True)[0]
                count = min(int(selected_indices.numel()), beam_size)
                selected_scores, selected_local = torch.topk(unique_scores[selected_indices], k=count)
                picked = selected_indices[selected_local]
                row_start = sample * beam_size
                new_prefixes[row_start:row_start + count, :length + 1] = unique_prefixes[picked]
                new_scores[sample, :count] = selected_scores
                new_active[sample, :count] = True
        prefixes, scores, active = new_prefixes, new_scores, new_active
        if not bool(active.any()):
            break
    completed_length = int(completed_lengths[0])
    best_score = float(finished_scores[0])
    best = tuple(completed_prefixes[0, 1:completed_length].tolist()) if completed_length else None
    prefix_rows = prefixes.tolist()
    active_rows = active[0].tolist()
    score_rows = scores[0].tolist()
    for j, is_active in enumerate(active_rows):
        if not is_active or not math.isfinite(score_rows[j]):
            continue
        if score_rows[j] > best_score:
            best_score = score_rows[j]
            best = tuple(prefix_rows[j][1:])
    return [best if best is not None else ()]
