"""Convert trusted OpenFS NPZs to the app contract, using explicit identity/timing metadata.

NPZ dictionaries contain pickle objects. Only import the official trusted download.
This is NOT an OpenFS checkpoint adapter: these features train our own models.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import HANDSHAPES, shape_features
from model.sessions import write_clip


def convert(right, left, meta, assume_contiguous=False, frame_indices=None):
    fps = float(meta.get("fps") or 30)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Positive finite FPS is required.")
    streams = []
    assumed = False
    for side, poses in (("Right", right), ("Left", left)):
        poses = np.asarray(poses, dtype=np.float32)
        raw_indices = (frame_indices or {}).get(side)
        if raw_indices is None:
            raw_indices = meta.get(f"{side.lower()}_frame_indices")
        if isinstance(raw_indices, str):
            raw_indices = json.loads(raw_indices) if raw_indices else None
        if not len(poses):
            if raw_indices is not None and len(raw_indices):
                raise ValueError("Frame indices without matching hand detections.")
            continue
        if poses.ndim != 3 or poses.shape[1:] != (21, 3) or not np.isfinite(poses).all():
            raise ValueError("Expected finite T × 21 × 3 pixel-coordinate poses.")
        if raw_indices is not None:
            source_indices = np.asarray(raw_indices)
            if source_indices.dtype.kind not in "iuf" or not np.isfinite(source_indices).all() or np.any(source_indices != np.floor(source_indices)):
                raise ValueError("Original frame indices must be finite integers.")
            indices = source_indices.astype(np.int64)
        elif assume_contiguous:
            indices = np.arange(len(poses))
            assumed = True
        else:
            raise ValueError("Original frame indices missing. Supply metadata or explicitly --assume-contiguous.")
        if indices.ndim != 1 or len(indices) != len(poses) or np.any(indices < 0) or np.any(np.diff(indices) <= 0):
            raise ValueError("Invalid original frame indices.")
        streams.append((side, poses, indices))
    if not streams:
        raise ValueError("No hand poses in clip.")
    if assumed and len(streams) == 2 and len(right) != len(left):
        raise ValueError("Unequal hand lengths cannot be aligned without original frame indices.")
    count = max(int(indices[-1]) for _, _, indices in streams) + 1
    frames = np.zeros((count, 91), dtype=np.float32)
    frames[1:, -1] = 1 / fps
    for side, poses, indices in streams:
        base = 0 if side == "Right" else 45
        for points, index in zip(poses, indices):
            frames[index, base:base + 42] = shape_features(points[:, :2], side)
            scale = np.linalg.norm(points[9, :2] - points[0, :2]) + 1e-6
            frames[index, base + 42:base + 44] = points[0, :2] / scale
            frames[index, base + 44] = 1
    return frames, np.arange(count, dtype=np.float64) * 1000 / fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path,
                        help="CSV: video_id,signer_id,session_id,split,person_id; optional fps and fallback frame indices")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trusted-pickle", action="store_true", help="Acknowledge trusted object NPZ loading")
    parser.add_argument("--assume-contiguous", action="store_true",
                        help="Synthesize indices when the upstream NPZ discarded original timing")
    parser.add_argument("--clean-label-map", type=Path,
                        help="JSON mapping upstream numeric frame IDs to letters; clean NPZ only")
    args = parser.parse_args()
    if not args.trusted_pickle:
        parser.error("Object NPZ input requires --trusted-pickle for the verified official source.")
    with args.metadata.open(encoding="utf-8", newline="") as file:
        metadata = {}
        for row in csv.DictReader(file):
            if row["video_id"] in metadata:
                raise ValueError(f"Duplicate video identity metadata: {row['video_id']}")
            metadata[row["video_id"]] = row
    mapping = json.loads(args.clean_label_map.read_text()) if args.clean_label_map else None
    with np.load(args.input, allow_pickle=True) as archive:
        required = {"vidnames", "right_hand_poses", "left_hand_poses", "words"}
        if not required.issubset(archive.files):
            parser.error(f"Expected OpenFS keys: {sorted(required)}")
        # Materialize each object array once; indexing an NPZ member repeatedly
        # decompresses the full member on every iteration.
        names, rights, lefts, words = (archive[key] for key in
                                      ("vidnames", "right_hand_poses", "left_hand_poses", "words"))
        clean = archive["frame_label_clean"] if mapping and "frame_label_clean" in archive.files else None
        signing = archive["signing_hand"] if mapping and "signing_hand" in archive.files else None
        clean_lookup = {}
        if mapping:
            if clean is None or signing is None or "frame_label" not in archive.files:
                raise ValueError("Clean annotation import requires frame_label_clean, signing_hand and coarse frame_label.")
            from scripts.prepare_openfs_training import clean_indices
            coarse = archive["frame_label"]
            if len(signing) != len(names) or len(coarse) != len(names):
                raise ValueError("Signing/coarse label arrays don't match the full clip array.")
            selected = clean_indices(words, signing, coarse)
            if len(selected) != len(clean):
                raise ValueError("Clean label filtering/count mismatch; unsupported archive schema.")
            clean_lookup = {source: cleaned for cleaned, source in enumerate(selected)}
        right_indices = archive["right_frame_indices"] if "right_frame_indices" in archive.files else None
        left_indices = archive["left_frame_indices"] if "left_frame_indices" in archive.files else None
        for values in (rights, lefts, words, right_indices, left_indices):
            if values is not None and len(values) != len(names):
                raise ValueError("OpenFS arrays have different clip counts.")
        for index, name in enumerate(names):
            name = str(name)
            if name not in metadata:
                raise ValueError(f"Missing verified signer/timing metadata for {name}; never infer signer IDs from row numbers.")
            meta = metadata[name]
            right, left = rights[index], lefts[index]
            keys = set(right) | set(left)
            person = next((key for key in keys if str(key) == str(meta["person_id"])), None)
            if person is None:
                raise ValueError(f"Missing person_id {meta['person_id']} for {name}.")
            embedded = {}
            for side, values in (("Right", right_indices), ("Left", left_indices)):
                if values is not None:
                    embedded[side] = values[index]
            frames, times = convert(right.get(person, []), left.get(person, []), meta,
                                    args.assume_contiguous, embedded)
            provenance = "human-transcript; fps-derived-time"
            if not meta.get("fps"):
                provenance += "; nominal-30-fps"
            assumed = args.assume_contiguous and any(
                side not in embedded and not meta.get(f"{side.lower()}_frame_indices")
                for side, poses in (("Right", right.get(person, [])), ("Left", left.get(person, []))) if len(poses))
            if assumed:
                provenance += "; assumed-contiguous-detections"
            common = dict(signer_id=meta["signer_id"], session_id=meta["session_id"],
                          split=meta["split"], source=f"OpenFS:{args.input.name}:{name}")
            write_clip(args.output, frames, times, kind="spelling", transcript=str(words[index]),
                       label_provenance=provenance, **common)
            if index in clean_lookup:
                signing_hand = signing[index]
                if signing_hand is None:
                    continue
                if int(signing_hand) // 2 != int(person):
                    raise ValueError("Clean annotation signing person differs from metadata.")
                side = "Right" if int(signing_hand) % 2 == 0 else "Left"
                labels = np.asarray(clean[clean_lookup[index]])
                poses = right.get(person, []) if side == "Right" else left.get(person, [])
                if len(labels) != len(poses):
                    raise ValueError("Clean labels don't align with signing-hand detections.")
                original = embedded.get(side)
                if original is None:
                    original = meta.get(f"{side.lower()}_frame_indices")
                    original = json.loads(original) if original else None
                indices = np.asarray(original, dtype=int) if original is not None else np.arange(len(poses))
                start = 0
                while start < len(labels):
                    end = start + 1
                    while end < len(labels) and labels[end] == labels[start]:
                        end += 1
                    letter = mapping.get(str(int(labels[start])))
                    if letter in HANDSHAPES[1:]:
                        selected = indices[start:end]
                        write_clip(args.output, frames[selected], times[selected], kind="held", transcript=letter,
                                   hand=side, label_provenance=provenance.replace("human-transcript", "model-derived-clean-frame-label"), **common)
                    start = end
    print(f"Wrote manifest to {args.output / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
