from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark

model_path = Path(__file__).resolve().parents[1] / 'data' / 'hand_landmarker.task'

def create_detector(num_hands=8, running_mode=vision.RunningMode.IMAGE):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=num_hands,
        running_mode=running_mode,
    ))


detector = create_detector()

def frameToMpImage(frame):
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

def landmarks_to_features(hand, handedness_label, width, height):
    pts = np.array([[lm.x * width, lm.y * height, lm.z * width] for lm in hand])
    pts -= pts[0]                                  # relativt til håndledd
    if handedness_label == 'Left':
        pts[:, 0] *= -1                            # speil til felles side
    pts /= np.linalg.norm(pts[9]) + 1e-6           # skaler med håndstørrelse
    return pts.flatten()                           # vektor med 63 verdier


def extractHands(frame, detector, timestamp_ms=None):
    height, width, _ = frame.shape # mp returnerer verdier mellom 0 og 1 og ikke koordinater
    image = frameToMpImage(frame)
    result = (detector.detect(image) if timestamp_ms is None
              else detector.detect_for_video(image, timestamp_ms))

    hands = []
    for hand, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name
        hands.append({
            "hand": label,
            "score": handedness[0].score,
            "features": landmarks_to_features(hand, label, width, height),
            "landmarks": hand,
        })
    return hands

if __name__ == "__main__":
    # overlay
    overlay_path = Path(__file__).resolve().parents[1] / 'linux.png'
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)  # beholder alfakanal
    if overlay is None:
        raise FileNotFoundError(overlay_path)
    overlay = cv2.resize(overlay, (150, 150))
    oh, ow, _ = overlay.shape
    alpha = overlay[:, :, 3:] / 255.0              # 0 = gjennomsiktig, 1 = helt synlig
    overlay_bgr = overlay[:, :, :3]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    x = 50
    y = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, _ = frame.shape # mp returnerer verdier mellom 0 og 1 og ikke koordinater
        
        hands = extractHands(frame, detector)
        frame = cv2.flip(frame, 1)
        
        for h in hands:
            print(h["hand"], h["features"].shape)
            for lm in h["landmarks"]:
                px = int((1 - lm.x) * width)
                py = int(lm.y * height)
                cv2.circle(frame, (px,py), 5, (0, 255, 255), 2)

        # set overlay on frame (after detection so it doesn't hide hands)
        roi = frame[y:y+oh, x:x+ow]
        frame[y:y+oh, x:x+ow] = (alpha * overlay_bgr + (1 - alpha) * roi).astype(np.uint8)

        cv2.imshow("Webcam feed", frame)
        # print(detectionResult)
        # print(frame.shape)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
