import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark

cap = cv2.VideoCapture(0)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

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


def extractHands(frame, detector):
    height, width, _ = frame.shape # mp returnerer verdier mellom 0 og 1 og ikke koordinater
    result = detector.detect(frameToMpImage(frame))

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
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape # mp returnerer verdier mellom 0 og 1 og ikke koordinater

        for h in extractHands(frame, detector):
            print(h["hand"], h["features"].shape)
            for lm in h["landmarks"]:
                px = int(lm.x * width)
                py = int(lm.y * height)
                cv2.circle(frame, (px,py), 5, (0, 255, 255), 2)

        cv2.imshow("Webcam feed", frame)
        # print(detectionResult)
        # print(frame.shape)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
