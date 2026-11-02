"""CPU runtime latency on public packed test landmarks; no camera/detector/IPC timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.live_neural import NeuralRuntime


def replay_hands(frame):
    hands = []
    for base, side in ((0, "Right"), (45, "Left")):
        if not frame[base + 44]:
            continue
        xy = frame[base:base + 42].reshape(21, 2).copy()
        if side == "Left":
            xy[:, 0] *= -1
        # Any positive pixel hand scale reconstructs equivalent v2 geometry.
        xy = (xy + frame[base + 42:base + 44]) * 20
        hands.append(dict(hand=side, landmarks=[SimpleNamespace(x=float(x) / 640, y=float(y) / 480)
                                               for x, y in xy]))
    return hands


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", required=True, type=Path)
    parser.add_argument("--handshape", type=Path, default=ROOT / "data/handshape.pt")
    parser.add_argument("--sequence", type=Path, default=ROOT / "data/sequence.pt")
    parser.add_argument("--threads", type=int, nargs="+", default=[4])
    parser.add_argument("--frames", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--allow-cycle", action="store_true", help="Explicitly cycle source landmarks for longer stress segments")
    parser.add_argument("--finish", action="store_true", help="Also time final-text analysis; OpenFS paths come from explicit environment variables")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if any(not 1 <= thread <= 64 for thread in args.threads) or any(not 1 <= size <= 900 for size in args.frames):
        parser.error("Use 1..64 threads and 1..900 frames.")
    schema = json.loads((args.packed / "dataset.json").read_text())
    packed = (args.packed / schema.get("generation", "")).resolve()
    if not packed.is_relative_to(args.packed.resolve()):
        raise ValueError("Prepared generation leaves the dataset directory.")
    clips = json.loads((packed / "test_clips.json").read_text())
    row = max(clips, key=lambda clip: clip["end"] - clip["start"])
    archive = np.load(packed / "test_frames.npy", mmap_mode="r")
    source = np.array(archive[row["start"]:row["end"]], copy=True)
    if max(args.frames) > len(source) and not args.allow_cycle:
        parser.error(f"Longest public test clip has {len(source)} frames; explicitly --allow-cycle for a longer stress segment.")
    providers = [replay_hands(source[index % len(source)]) for index in range(max(args.frames))]
    runtime = NeuralRuntime(ROOT)
    runtime.paths = {"handshape": args.handshape, "sequence": args.sequence}
    runtime.load()
    if set(runtime.models) != {"handshape", "sequence"}:
        raise ValueError(f"Both neural checkpoints must load: {runtime.errors}")
    report = dict(scope="CPU NeuralRuntime with feature conversion and decoding; excludes MediaPipe, camera, IPC",
                  source_clip=row["name"], source_frames=len(source), fps="nominal 30",
                  torch=torch.__version__, model_hashes={task: model_hash(model) for task, model in runtime.models.items()},
                  measurements=[])
    for threads in args.threads:
        os.environ["ASL_TORCH_THREADS"] = str(threads)
        torch.set_num_threads(threads)
        for _ in range(20):
            runtime.reset()
            runtime.analyze(providers[0], 640, 480, video=True, session_id="warmup", timestamp_ms=0)
        for size in args.frames:
            runtime.reset()
            times = []
            for index, hands in enumerate(providers[:size]):
                start = time.perf_counter_ns()
                result = runtime.analyze(hands, 640, 480, video=True, session_id=f"{threads}-{size}",
                                         timestamp_ms=index * 1000 / 30)
                times.append((time.perf_counter_ns() - start) / 1e6)
            tail = times[-min(100, size):]
            measurement = dict(threads=threads, segment_frames=size, source_cycled=size > len(source),
                               p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                               tail100_p50_ms=float(np.median(tail)), tail100_p95_ms=float(np.percentile(tail, 95)),
                               runtime_frames=result["fingerspelling"]["frames"])
            report["measurements"].append(measurement)
            print(json.dumps(measurement), flush=True)
            if args.finish:
                finishes = []
                for repeat in range(4):
                    start = time.perf_counter_ns()
                    runtime.finish_text()
                    finishes.append((time.perf_counter_ns() - start) / 1e6)
                measurement['finish'] = dict(first_ms=finishes[0], warmed_p50_ms=float(np.median(finishes[1:])),
                                             method=runtime.method, context=runtime.context_status,
                                             error=runtime.refinement_error,
                                             scope='Final-text model only; first call includes lazy checkpoint load')
                print(json.dumps(measurement['finish']), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
