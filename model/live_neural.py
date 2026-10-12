"""Independent handshape and whole-segment temporal recognition, with explicit readiness."""
import os
import math
from pathlib import Path

import numpy as np
from model.timeline import TimelineResampler
from model.openfs_refiner import OpenFSRefiner


class NeuralRuntime:
    def __init__(self, root):
        self.paths = {
            "handshape": Path(os.environ.get("ASL_HANDSHAPE_MODEL", Path(root) / "data" / "handshape.pt")),
            "sequence": Path(os.environ.get("ASL_SEQUENCE_MODEL", Path(root) / "data" / "sequence.pt")),
        }
        self.models = {}
        self.errors = {}
        self.loaded = False
        self.refiner = OpenFSRefiner()
        self.reset()

    def load(self):
        if self.loaded:
            return
        self.loaded = True
        for task, path in self.paths.items():
            if not path.is_file():
                continue
            try:
                import torch
                # Small live batches suffer from large thread-pool overhead.
                threads = int(os.environ.get("ASL_TORCH_THREADS", "4"))
                if not 1 <= threads <= 64:
                    raise ValueError("ASL_TORCH_THREADS must be an integer between 1 and 64.")
                torch.set_num_threads(threads)
                from model.neural import load_checkpoint
                self.models[task] = load_checkpoint(path, task)
            except Exception as error:
                self.errors[task] = str(error)

    def reset(self, session_id=None):
        self.session_id = session_id
        self.frames = []
        self.timestamps = []
        self.model_frames = []
        self.model_timestamps = []
        self.resampler = TimelineResampler()
        self.raw_hands = {"Right": [], "Left": []}
        self.text = ""
        self.finished = False
        self.limit_reached = False
        self.method = "causal-tcn"
        self.refinement_error = None
        self.context_status = "collecting"
        self.context_message = None

    def spelling_context(self):
        captured = np.array(self.frames)
        usable = np.any(captured[:, [44, 89]] > 0, axis=1) if len(captured) else np.array([], dtype=bool)
        times = np.array(self.timestamps)[usable]
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0
        movement = False
        for stream in self.raw_hands.values():
            if len(stream) >= 3:
                pixels = np.array(stream)
                scale = np.median(np.linalg.norm(pixels[:, 9] - pixels[:, 0], axis=-1)) + 1e-6
                movement |= bool(np.max(np.ptp(pixels, axis=0)) / scale > .12)
        return int(usable.sum()), duration, movement, min(128, max(2, math.floor(duration / 1000 * 20) + 2))

    def gate_text(self, text):
        count, duration, movement, maximum = self.spelling_context()
        if not movement:
            text = text.replace("J", "").replace("Z", "")
            if len(text.strip()) > 1:
                text = ""
        if len(text) > maximum:
            text = ""
        if count < 6 or duration < 500:
            self.context_status = "insufficient-context"
            self.context_message = "For kort sekvens for sluttanalyse av staving."
        elif not movement:
            self.context_status = "held-letter"
            self.context_message = "Holdt bokstav. Stav med bevegelse for å få tekst."
        else:
            self.context_status, self.context_message = "spelling", None
        return text

    def finish_text(self):
        self.finished = True
        self.text = self.gate_text(self.text)
        count, duration, movement, maximum = self.spelling_context()
        if self.refiner.enabled and count >= 6 and duration >= 500 and movement:
            try:
                text = self.refiner.refine({0: self.raw_hands["Right"]}, {0: self.raw_hands["Left"]})
                if len(text) > maximum:
                    raise ValueError("OpenFS output is too long for the observed spelling duration.")
                self.text = text
                self.method = "openfs"
            except Exception as error:
                self.refinement_error = str(error)

    def sequence_metadata(self):
        return {"method": self.method, "refiner": self.refiner.readiness(),
                "contextStatus": self.context_status,
                **({"contextMessage": self.context_message} if self.context_message else {}),
                **({"refinementError": self.refinement_error} if self.refinement_error else {})}

    def analyze(self, hands, width, height, *, video=False, session_id=None, timestamp_ms=None, finish=False):
        self.load()
        prediction = None
        if "handshape" in self.models and hands:
            import torch
            from model.neural import HANDSHAPES, shape_features
            # The static head abstains when selecting the signing hand is ambiguous.
            if len(hands) == 1:
                hand = hands[0]
                points = np.array([[point.x * width, point.y * height] for point in hand["landmarks"]])
                features = shape_features(points, hand["hand"])
                with torch.inference_mode():
                    probabilities = self.models["handshape"](torch.from_numpy(features).unsqueeze(0)).softmax(-1)[0]
                winner = int(probabilities.argmax())
                if winner and float(probabilities[winner]) >= .8:
                    prediction = {"letter": HANDSHAPES[winner], "confidence": float(probabilities[winner]),
                                  "method": "neural"}
        sequence = {"available": "sequence" in self.models, "text": "", "status": "unavailable",
                    "provisional": True, "frames": 0, "modelFrames": 0, **self.sequence_metadata()}
        if self.errors.get("sequence"):
            sequence["error"] = self.errors["sequence"]
        if video and "sequence" in self.models:
            if not isinstance(session_id, str) or not session_id or len(session_id) > 100:
                raise ValueError("Video recognition requires a bounded session ID.")
            if not isinstance(timestamp_ms, (float, int)) or not np.isfinite(timestamp_ms):
                raise ValueError("Video recognition requires a finite capture timestamp.")
            if session_id != self.session_id:
                self.reset(session_id)
            if self.timestamps and timestamp_ms <= self.timestamps[-1]:
                raise ValueError("Capture timestamps must increase within a session.")
            if (len(self.frames) >= 900 or len(self.model_frames) >= 900
                    or (self.timestamps and timestamp_ms - self.timestamps[0] >= 30000)):
                self.limit_reached = True
            if not self.finished and not self.limit_reached and self.timestamps and timestamp_ms - self.timestamps[-1] > 1000:
                self.reset(session_id)
            if self.finished:
                sequence.update(text=self.text, status="finished", provisional=False, frames=len(self.frames), modelFrames=len(self.model_frames))
            elif self.limit_reached or len(self.frames) >= 900 or (self.timestamps and timestamp_ms - self.timestamps[0] >= 30000):
                self.limit_reached = True
                if finish:
                    self.finish_text()
                    sequence.update(self.sequence_metadata())
                sequence.update(text=self.text, status="finished" if finish else "limit", provisional=not finish,
                                frames=len(self.frames), modelFrames=len(self.model_frames))
            else:
                import torch
                from model.neural import frame_features, VOCAB
                delta = timestamp_ms - self.timestamps[-1] if self.timestamps else 0
                ambiguous = any(sum(hand["hand"] == side for hand in hands) > 1 for side in ("Right", "Left"))
                if ambiguous:
                    sequence["error"] = "Vis én person for tekstgjenkjenning. Håndvalget er uklart."
                usable = [] if ambiguous else hands
                features = frame_features(usable, width, height, delta)
                for hand in usable:
                    self.raw_hands[hand["hand"]].append(np.array([[point.x * width, point.y * height]
                                                                 for point in hand["landmarks"]], dtype=np.float32))
                self.frames.append(features)
                self.timestamps.append(timestamp_ms)
                generated = self.resampler.feed(features, timestamp_ms)
                capacity = 900 - len(self.model_frames)
                if len(generated) > capacity:
                    self.limit_reached = True
                for tick, value in generated[:capacity]:
                    self.model_timestamps.append(tick)
                    self.model_frames.append(value)
                if not generated:
                    self.text = self.gate_text(self.text)
                    if finish:
                        self.finish_text()
                    sequence.update(self.sequence_metadata())
                    sequence.update(text=self.text, status="finished" if finish else "recognizing",
                                    provisional=not finish, frames=len(self.frames), modelFrames=len(self.model_frames))
                    return {"handshapeAvailable": "handshape" in self.models, "handshapeError": self.errors.get("handshape"),
                            "prediction": prediction, "fingerspelling": sequence}
                values = torch.from_numpy(np.array(self.model_frames)).unsqueeze(0)
                with torch.inference_mode():
                    indices = self.models["sequence"](values).argmax(-1)[0]
                # J/Z must have local motion evidence in the preceding second.
                # This is a necessary gate, not a geometric classifier of their paths.
                text, previous = [], None
                stacked = np.array(self.model_frames)
                for frame_index, token in enumerate(indices.tolist()):
                    if not (stacked[frame_index, 44] or stacked[frame_index, 89]):
                        previous = token
                        continue
                    if token and token != previous:
                        char = VOCAB[token]
                        movement = char not in {"J", "Z"}
                        if not movement:
                            # Slice by absolute frame index before selecting the time window.
                            local = stacked[:frame_index + 1][np.array(self.model_timestamps[:frame_index + 1]) >= self.model_timestamps[frame_index] - 1000]
                            for base in (0, 45):
                                visible = local[local[:, base + 44] > 0]
                                if len(visible) >= 3:
                                    movement |= bool(np.max(np.ptp(visible[:, base:base + 44], axis=0)) > .12)
                        if movement:
                            text.append(char)
                    previous = token
                self.text = self.gate_text("".join(text))
                if finish:
                    self.finish_text()
                sequence.update(self.sequence_metadata())
                sequence.update(text=self.text, status="limit" if self.limit_reached else "finished" if finish else "recognizing",
                                provisional=not finish, frames=len(self.frames), modelFrames=len(self.model_frames))
        return {"handshapeAvailable": "handshape" in self.models, "handshapeError": self.errors.get("handshape"),
                "prediction": prediction, "fingerspelling": sequence}
