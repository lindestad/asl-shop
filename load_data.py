from pathlib import Path
import cv2
from sklearn.utils import shuffle
from handTracker.hand_features import extractHands, detector

data_path = Path(__file__).resolve().parent / 'data'
train_path = data_path / 'asl_alphabet_train' / 'asl_alphabet_train'
test_path = data_path / 'asl_alphabet_test' / 'asl_alphabet_test'

with open("extactedLandmarks.csv", "w", encoding="utf-8") as file:
    for class_dir in train_path.iterdir():
        label = class_dir.name
        for img_path in class_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            hands = extractHands(img, detector)
            for hand in hands:
                row = [label] + hand["features"].tolist()
                file.write(",".join(map(str, row)) + "\n")
