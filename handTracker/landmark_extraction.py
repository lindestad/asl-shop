from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2  # må importeres etter mediapipe, ellers blir prosessen drept (exit 137)
from tqdm import tqdm
from hand_features import extractHands, landmarks_to_features, frameToMpImage

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / 'data'
train_path = data_path / 'Train'
test_path = data_path / 'Test'
model_path = data_path / 'hand_landmarker.task'
output_path = data_path / 'extractedLandmarks.csv'

base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Collect (label, path) pairs up front so tqdm knows the total
images = [
    (class_dir.name, img_path)
    for class_dir in train_path.iterdir() if class_dir.is_dir()
    for img_path in class_dir.glob('*.jpg')
]

with open(output_path, "w", encoding="utf-8") as file:
    for label, img_path in tqdm(images, desc="Extracting landmarks", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"Could not read {img_path}")
            continue
        hands = extractHands(img, detector)
        for hand in hands:
            row = [label] + hand["features"].tolist()
            file.write(",".join(map(str, row)) + "\n")