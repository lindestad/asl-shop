"""Train/evaluate neural models from signer-disjoint landmark clip manifests."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

# Set before torch initializes its caching allocator; see scripts/train_openfs.py.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.neural import CONTRACT, HANDSHAPES, VOCAB, HandshapeNet, FingerspellingNet, decode_ctc
from model.sessions import read_manifest


class Clips(Dataset):
    def __init__(self, entries, task, augment=False):
        self.entries = [e for e in entries if task == "sequence" or e["kind"] != "spelling"]
        self.task, self.augment = task, augment
        if not self.entries:
            raise ValueError(f"No usable {task} clips in this split.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        with np.load(entry["path"], allow_pickle=False) as archive:
            frames = archive["frames"].astype(np.float32)
        if frames.ndim != 2 or frames.shape[1] != 91 or not np.isfinite(frames).all():
            raise ValueError(f"Invalid features: {entry['path']}")
        if self.task == "handshape":
            label = HANDSHAPES.index(entry["transcript"]) if entry["kind"] == "held" else 0
            sides = [entry["hand"]] if entry["kind"] == "held" else ["Right", "Left"]
            samples = [frame[base:base + 42] for frame in frames for side in sides
                       for base in [0 if side == "Right" else 45] if frame[base + 44] > 0]
            if not samples:
                if label:
                    raise ValueError(f"No detected target hands: {entry['path']}")
                samples = [np.zeros(42, dtype=np.float32)]
            # One random frame per clip per epoch prevents long recordings dominating.
            sample = np.array([samples[random.randrange(len(samples))]] if self.augment else samples, dtype=np.float32)
            if self.augment:
                sample = sample + np.random.normal(0, .015, sample.shape).astype(np.float32)
            return torch.from_numpy(sample.copy()), torch.full((len(sample),), label, dtype=torch.long)
        text = entry["transcript"]
        required = len(text) + sum(a == b for a, b in zip(text, text[1:]))
        if len(frames) < max(1, required):
            raise ValueError(f"Clip is too short for CTC alignment: {entry['path']}")
        if self.augment:
            # Preserve order and repeated-character capacity when changing speed.
            original_length = len(frames)
            length = max(required, 1, round(original_length * random.uniform(.85, 1.15)))
            indices = np.linspace(0, len(frames) - 1, length).round().astype(int)
            frames = frames[indices].copy()
            for base in (0, 45):
                valid = frames[:, base + 44:base + 45]
                frames[:, base:base + 42] += np.random.normal(0, .015, (length, 42)).astype(np.float32) * valid
            frames[:, -1] *= original_length / length
        target = torch.tensor([VOCAB.index(char) for char in text], dtype=torch.long)
        return torch.from_numpy(frames), target


def collate_sequence(batch):
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(value) for value in inputs], dtype=torch.long)
    target_lengths = torch.tensor([len(value) for value in targets], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return padded, torch.cat(targets), lengths, target_lengths


def collate_handshape(batch):
    inputs, targets = zip(*batch)
    return torch.cat(inputs), torch.cat(targets)


def edit_distance(expected, actual):
    row = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        next_row = [i]
        for j, b in enumerate(actual, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


@torch.inference_mode()
def evaluate(model, loader, task, device):
    model.eval()
    errors = total = exact = clips = idle_false = idle_count = 0
    for batch in loader:
        if task == "handshape":
            inputs, targets = batch
            predicted = model(inputs.to(device)).argmax(-1).cpu()
            errors += int((predicted != targets).sum())
            total += len(targets)
            idle_false += int(((targets == 0) & (predicted != 0)).sum())
            idle_count += int((targets == 0).sum())
        else:
            inputs, targets, lengths, target_lengths = batch
            predicted = model(inputs.to(device)).argmax(-1).cpu()
            offset = 0
            for i, count in enumerate(target_lengths):
                expected = "".join(VOCAB[token] for token in targets[offset:offset + count])
                actual = decode_ctc(predicted[i, :lengths[i]])
                offset += count
                errors += edit_distance(expected, actual)
                total += len(expected)
                exact += expected == actual
                clips += 1
                if not expected:
                    idle_count += 1
                    idle_false += bool(actual)
    result = {"accuracy": 1 - errors / total} if task == "handshape" else {
        "cer": errors / total if total else None, "exact_sequence_accuracy": exact / clips}
    result["idle_false_positive_rate"] = idle_false / idle_count if idle_count else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["handshape", "sequence"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision; CTC stays float32")
    parser.add_argument("--allow-weak-evaluation", action="store_true",
                        help="Explicitly allow model-derived frame labels in handshape evaluation")
    parser.add_argument("--evaluate", type=Path, help="Evaluate checkpoint on test split without training")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    entries = read_manifest(args.manifest)
    def loader(split, training=False):
        selected = [entry for entry in entries if entry["split"] == split]
        if args.task == "handshape" and not training and not args.allow_weak_evaluation:
            if any("model-derived" in entry["label_provenance"] for entry in selected):
                raise ValueError("Use human frame labels for evaluation, or explicitly --allow-weak-evaluation.")
        dataset = Clips(selected, args.task, training)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                          collate_fn=collate_sequence if args.task == "sequence" else collate_handshape)
    if args.evaluate:
        from model.neural import load_checkpoint
        model = load_checkpoint(args.evaluate, args.task).to(args.device)
        print(json.dumps(evaluate(model, loader("test"), args.task, args.device)))
        return
    training, validation = loader("train", True), loader("validation")
    model = (HandshapeNet if args.task == "handshape" else FingerspellingNet)(args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    amp_enabled = args.amp and str(args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.
        for batch in training:
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if amp_enabled else "cpu", enabled=amp_enabled):
                inputs, targets = batch[:2]
                output = model(inputs.to(args.device))
            if args.task == "handshape":
                loss = F.cross_entropy(output.float(), targets.to(args.device))
            else:
                lengths, target_lengths = batch[2:]
                logits = output.float().log_softmax(-1).transpose(0, 1)
                loss = ctc(logits, targets.to(args.device), lengths, target_lengths)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; check input lengths and labels.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach())
        metrics = evaluate(model, validation, args.task, args.device)
        print(json.dumps({"epoch": epoch + 1, "loss": loss_total / len(training), **metrics}), flush=True)
        score = 1 - metrics["accuracy"] if args.task == "handshape" else metrics["cer"]
        if score is None:
            raise ValueError("Sequence validation must include non-empty transcripts.")
        if score < best:
            best = score
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"contract": CONTRACT, "task": args.task, "hidden": args.hidden,
                        "labels": HANDSHAPES if args.task == "handshape" else VOCAB,
                        "state_dict": model.cpu().state_dict(), "metrics": metrics,
                        "seed": args.seed, "manifest": str(args.manifest.resolve())}, args.output)
            model.to(args.device)


if __name__ == "__main__":
    main()
