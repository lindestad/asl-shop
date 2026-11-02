"""Collection manifests preserve source, identity, time and label provenance."""
import json
from pathlib import Path
import uuid

import numpy as np

KINDS = {"held", "spelling", "transition", "idle"}


def write_clip(directory, frames, timestamps_ms, *, signer_id, session_id, kind,
               transcript, source, label_provenance, split="train", hand=None):
    from model.neural import CONTRACT, HANDSHAPES, VOCAB
    frames = np.asarray(frames, dtype=np.float32)
    times = np.asarray(timestamps_ms, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 91 or not len(frames) or not np.isfinite(frames).all():
        raise ValueError("Expected non-empty finite T × 91 features.")
    if times.shape != (len(frames),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Capture timestamps must strictly increase and match the frames.")
    if not signer_id or not session_id or not source or not label_provenance:
        raise ValueError("Signer/session IDs and provenance are required.")
    if kind not in KINDS or split not in {"train", "validation", "test"}:
        raise ValueError("Invalid clip kind or split.")
    transcript = transcript.upper()
    if any(char not in VOCAB[1:] for char in transcript):
        raise ValueError("Transcript contains an unsupported character.")
    if kind in {"transition", "idle"} and transcript:
        raise ValueError("Idle and transition clips require an empty transcript.")
    if kind == "held" and (transcript not in HANDSHAPES[1:] or hand not in {"Left", "Right"}):
        raise ValueError("Held clips need a static letter and explicit signing hand; collect J/Z as spelling.")
    if kind == "spelling" and not transcript:
        raise ValueError("Spelling clips need a transcript.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    clip_id = uuid.uuid4().hex
    filename = f"{clip_id}.npz"
    np.savez_compressed(directory / filename, frames=frames, timestamps_ms=times)
    entry = dict(clip_id=clip_id, file=filename, contract=CONTRACT, signer_id=str(signer_id),
                 session_id=str(session_id), kind=kind, transcript=transcript, source=source,
                 label_provenance=label_provenance, split=split, hand=hand)
    with (directory / "manifest.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")
    return entry


def read_manifest(path):
    from model.neural import CONTRACT
    path = Path(path)
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError("Manifest is empty.")
    splits = {}
    for entry in entries:
        if entry.get("contract") != CONTRACT:
            raise ValueError("Manifest preprocessing contract is incompatible.")
        for key in ("signer_id", "session_id", "source", "label_provenance"):
            if not entry.get(key):
                raise ValueError(f"Missing {key}.")
        signer = entry["signer_id"]
        splits.setdefault(signer, set()).add(entry["split"])
        if len(splits[signer]) > 1:
            raise ValueError(f"Signer {signer} occurs in multiple splits; unseen-signer evaluation would leak.")
        target = (path.parent / entry["file"]).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError("Clip path leaves the manifest directory.")
        entry["path"] = target
    return entries
