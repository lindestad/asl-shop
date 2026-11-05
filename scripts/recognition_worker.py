"""Persistent newline-delimited JSON adapter for hand and letter recognition.

Stdout is reserved for protocol messages. MediaPipe diagnostics go to stderr.
"""

import base64
import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "model" / "trained_models"
DEFAULT_MODEL = "neural"
classifiers = {}

# The KNN was fitted on a DataFrame, but the worker's features use the same column order.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from model.live_neural import NeuralRuntime
neural = NeuralRuntime(ROOT)

video_detector = None
video_timestamp = -1
video_scan_timestamp = -1
video_capacity = 0


def load_knn():
    with open(MODEL_DIR / "knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    if model.n_features_in_ != 63:
        raise ValueError("The KNN classifier uses an incompatible feature format.")
    return model.classes_, lambda features: model.predict_proba(features.reshape(1, -1))[0]


def load_cnn():
    # Imported here so that torch's slow startup only affects people who pick the CNN.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Must match the CNN in model/cnn-landmarks.py; load_state_dict rejects any drift.
    class CNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    with open(MODEL_DIR / "cnn-landmarks.pkl", "rb") as f:
        checkpoint = pickle.load(f)
    classes = checkpoint["classes"]
    model = CNN(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def predict_proba(features):
        # (63,) -> (1, 3, 21): x/y/z are the channels, as in training.
        batch = torch.from_numpy(features).float().reshape(1, 21, 3).permute(0, 2, 1)
        with torch.no_grad():
            return torch.softmax(model(batch), dim=1)[0].numpy()

    return classes, predict_proba


LOADERS = {"knn": load_knn, "cnn": load_cnn}


def get_classifier(name=DEFAULT_MODEL):
    if name not in LOADERS:
        raise ValueError(f"Unknown model: {name}")
    if name not in classifiers:
        try:
            classifiers[name] = LOADERS[name]()
        except (FileNotFoundError, ImportError) as error:
            # Hand tracking still works, and the other model can still be selected.
            print(f"Could not load the {name} model: {error}", file=sys.stderr, flush=True)
            return None
    return classifiers[name]


def predict_letter(hands, classifier, name):
    if not hands or classifier is None:
        return None
    import numpy as np

    classes, predict_proba = classifier
    hand = max(hands, key=lambda item: item["score"])
    probabilities = predict_proba(hand["features"])
    winner = int(np.argmax(probabilities))
    if probabilities[winner] < 0.6:
        return None
    return {"letter": str(classes[winner]), "confidence": float(probabilities[winner]), "method": name}


def track(frame):
    global video_detector, video_timestamp, video_scan_timestamp, video_capacity
    from handTracker.hand_features import create_detector, detector, extractHands
    from mediapipe.tasks.python import vision

    # A worker-wide clock stays increasing across camera stops/restarts and
    # interleaved still-image requests, regardless of the camera's clock.
    video_timestamp = max(video_timestamp + 1, time.monotonic_ns() // 1_000_000)
    if video_detector is None or video_timestamp - video_scan_timestamp >= 250:
        hands = extractHands(frame, detector)
        video_scan_timestamp = video_timestamp
        capacity = max(1, len(hands))
        if capacity != video_capacity:
            if video_detector is not None:
                video_detector.close()
            video_detector = create_detector(num_hands=capacity, running_mode=vision.RunningMode.VIDEO)
            video_capacity = capacity
        return hands

    # MediaPipe reruns palm detection every frame if fewer than num_hands are
    # visible. Match capacity to the visible count so ordinary frames can use
    # tracking, and use the separate eight-hand scan to find newcomers promptly.
    return extractHands(frame, video_detector, video_timestamp)


def analyze(data_url, video=False, session_id=None, timestamp_ms=None, finish=False,
            model_name=DEFAULT_MODEL):
    import cv2
    import numpy as np
    from handTracker.hand_features import detector, extractHands

    header, encoded = data_url.split(",", 1)
    if header not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
        raise ValueError("Unsupported image format.")
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not read this image.")
    height, width = frame.shape[:2]
    if width * height > 16_777_216:
        raise ValueError("Use an image smaller than 16 megapixels.")
    if video:
        hands = track(frame)
    else:
        hands = extractHands(frame, detector)
    outputs = neural.analyze(hands, width, height, video=video, session_id=session_id,
                             timestamp_ms=timestamp_ms, finish=finish)
    # The picker owns the static letter head; the sequence model always runs.
    # "neural" falls back to the KNN when the handshape checkpoint is missing.
    classifier = None
    model_used = "neural"
    if model_name != "neural" or not outputs["handshapeAvailable"]:
        requested = model_name if model_name in LOADERS else "knn"
        classifier = get_classifier(requested)
        outputs["prediction"] = predict_letter(hands, classifier, requested) if classifier else None
        model_used = requested if classifier else model_name
    return {
        "width": width,
        "height": height,
        "model": model_used,
        "classifierAvailable": outputs["handshapeAvailable"] or classifier is not None,
        **outputs,
        "hands": [
            {"hand": hand["hand"], "score": float(hand["score"]),
             "features": hand["features"].tolist(),
             "landmarks": [{"x": point.x, "y": point.y, "z": point.z} for point in hand["landmarks"]]}
            for hand in hands
        ],
    }


DEMO_CANVAS = 1000.0


def analyze_landmarks(hands, session_id=None, timestamp_ms=None, finish=False):
    """Demo replay path: landmark frames produced by scripts/demo_replay.py feed the
    runtime directly, skipping image decode and MediaPipe; identical downstream pipeline."""
    from types import SimpleNamespace

    prepared = [
        {"hand": hand["hand"], "score": 1.0,
         "landmarks": [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=0.0) for p in hand["points"]]}
        for hand in hands
    ]
    outputs = neural.analyze(prepared, DEMO_CANVAS, DEMO_CANVAS, video=True, session_id=session_id,
                             timestamp_ms=timestamp_ms, finish=finish)
    return {
        "width": DEMO_CANVAS,
        "height": DEMO_CANVAS,
        "model": "neural",
        "classifierAvailable": outputs["handshapeAvailable"],
        **outputs,
        "hands": [
            {"hand": hand["hand"], "score": hand["score"], "features": [],
             "landmarks": [{"x": p.x, "y": p.y, "z": p.z} for p in hand["landmarks"]]}
            for hand in prepared
        ],
    }


def main():
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("landmarks") is not None:
                response = {"id": request["id"], "result": analyze_landmarks(
                    request["landmarks"], request.get("sessionId"),
                    request.get("timestampMs"), request.get("finish", False))}
            else:
                response = {"id": request["id"], "result": analyze(
                    request["dataUrl"], request.get("video", False), request.get("sessionId"),
                    request.get("timestampMs"), request.get("finish", False),
                    request.get("model") or DEFAULT_MODEL)}
        except Exception as error:
            response = {"id": request.get("id"), "error": str(error)}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
