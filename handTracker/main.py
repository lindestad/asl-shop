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



while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape # mp returnerer verdier mellom 0 og 1 og ikke koordinater

    mpFrame = frameToMpImage(frame)
    detectionResult = detector.detect(mpFrame)

    for hand in detectionResult.hand_landmarks:
        for landmark in hand:
            px = int(landmark.x * width)
            py = int(landmark.y * height)
            # print(px, py) # pos til hånd
            cv2.circle(frame, (px,py), 5,(0,255,0), 1)
            cv2.line(frame, )

    cv2.imshow('Live Webcam Feed', frame)
    # print(detectionResult)
    # print(frame.shape)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
