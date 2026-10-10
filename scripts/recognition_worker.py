"""Persistent newline-delimited JSON adapter for hand and letter recognition.

Stdout is reserved for protocol messages. MediaPipe diagnostics go to stderr.
"""

import base64
import json
from pathlib import Path
import sys
import time

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
CLASSIFIER_PATH = ROOT / "model" / "asl_knn.npz"
FEATURE_SCHEMA = "wrist_relative_mirrored_scaled_63_v1"
classifier = None


video_detector = None
video_timestamp = -1
video_scan_timestamp = -1
video_capacity = 0


def get_classifier():
    global classifier
    if classifier is None and CLASSIFIER_PATH.exists():
        import numpy as np

        with np.load(CLASSIFIER_PATH, allow_pickle=False) as saved:
            features = saved["features"]
            labels = saved["labels"]
            neighbors = int(saved["neighbors"])
            schema = str(saved["schema"])
        if schema != FEATURE_SCHEMA or features.ndim != 2 or features.shape[1] != 63:
            raise ValueError("The ASL classifier uses an incompatible feature format.")
        if len(features) != len(labels) or not 1 <= neighbors <= len(features):
            raise ValueError("The ASL classifier file is incomplete.")
        classifier = (features, labels, neighbors)
    return classifier


def predict_letter(hands, model):
    if not hands or model is None:
        return None
    import numpy as np

    features, labels, neighbors = model
    hand = max(hands, key=lambda item: item["score"])
    delta = features - hand["features"].astype(np.float32)
    distances = np.einsum("ij,ij->i", delta, delta)
    closest = np.argpartition(distances, neighbors - 1)[:neighbors]
    letters, counts = np.unique(labels[closest], return_counts=True)
    winner = int(np.argmax(counts))
    if counts[winner] < 3:
        return None
    return {"letter": str(letters[winner]), "confidence": float(counts[winner] / neighbors)}


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


def analyze(data_url, video=False):
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
    model = get_classifier()
    return {
        "width": width,
        "height": height,
        "classifierAvailable": model is not None,
        "prediction": predict_letter(hands, model),
        "hands": [
            {"hand": hand["hand"], "score": float(hand["score"]),
             "features": hand["features"].tolist(),
             "landmarks": [{"x": point.x, "y": point.y, "z": point.z} for point in hand["landmarks"]]}
            for hand in hands
        ],
    }


def main():
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            response = {"id": request["id"], "result": analyze(request["dataUrl"], request.get("video", False))}
        except Exception as error:
            response = {"id": request.get("id"), "error": str(error)}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
