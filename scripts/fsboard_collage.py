"""Build a face-blurred collage of FSboard video frames for presenting the dataset.

Picks clips from different signers in the FSboard test split, downloads only those clips from Kaggle
(no account needed), grabs a frame from the middle of each signed span, and blurs every face before
anything is written. FSboard ships faces unblurred, so a frame where no face is found is skipped rather
than shown: the collage never contains a frame the detector could not clear. Look over the tiles anyway
before putting them on a slide.

    uv run python scripts/fsboard_collage.py
    uv run python scripts/fsboard_collage.py --count 8 --columns 4 --seed 3 --captions
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/googleai/fsboard/"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
FACE_MODEL = ROOT / "data/blaze_face_short_range.tflite"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_MODEL = ROOT / "data/face_detection_yunet_2023mar.onnx"
CACHE = ROOT / "data/external/fsboard-collage"
SPLIT = "daun_v3-test"
# Where in the signed span to look for a frame, best first: mid-word shows a hand shape, not a rest pose.
FRAME_POSITIONS = (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8)


def download(url, target):
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urlopen(Request(url, headers={"User-Agent": "asl-shop"}), timeout=120) as response, \
            partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    partial.replace(target)
    return target


def dataset_file(path):
    return download(DATASET_URL + quote(path, safe=""), CACHE / path)


def choose_clips(metadata, count, seed):
    """One clip per signer, never one flagged sensitive, preferring short all-letter phrases for captions."""
    clips = [clip for clip in metadata if clip["signingType"] == "fs" and not clip["sensitiveContent"]]
    readable = [clip for clip in clips if clip["phrase"].replace(" ", "").isalpha() and len(clip["phrase"]) <= 24]
    rng = random.Random(seed)
    rng.shuffle(readable)
    by_signer = {}
    for clip in readable:
        by_signer.setdefault(clip["signerId"], clip)
    chosen = list(by_signer.values())
    rng.shuffle(chosen)
    return chosen


class FaceFinder:
    """MediaPipe BlazeFace plus OpenCV's YuNet: a face either one finds gets blurred."""

    def __init__(self, min_confidence):
        from mediapipe.tasks.python import BaseOptions, vision

        download(FACE_MODEL_URL, FACE_MODEL)
        options = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
                                             min_detection_confidence=min_confidence)
        self.blazeface = vision.FaceDetector.create_from_options(options)
        download(YUNET_URL, YUNET_MODEL)
        self.yunet = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), min_confidence)

    def __call__(self, image):
        import mediapipe as mp

        # Both detectors work on small inputs, so detect on a copy at most 640 px wide and scale back.
        scale = min(1.0, 640 / image.shape[1])
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        boxes = [(box.origin_x, box.origin_y, box.width, box.height)
                 for box in (detection.bounding_box for detection in self.blazeface.detect(rgb).detections)]
        self.yunet.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.yunet.detect(small)
        boxes += [tuple(face[:4]) for face in (faces if faces is not None else [])]
        return [tuple(int(value / scale) for value in box) for box in boxes]


def blur_faces(image, boxes, padding):
    """Blend a heavily blurred copy over one padded ellipse per face, fully opaque inside the ellipse.

    Each face gets a blur sized to itself, keeping about 8 samples across it: too few for anyone to be
    recognized at any display size, whether it is the signer or a small face in a photo on the wall.
    Where boxes from the two detectors overlap, the region is simply blurred again. The soft edge only
    grows outward from the opaque ellipse, never into it."""
    height, width = image.shape[:2]
    result = image.copy()
    for x, y, w, h in boxes:
        # Extra room above the box covers the forehead and hairline, which BlazeFace leaves out.
        center_x, center_y = x + w / 2, y + h / 2 - h * padding * 0.25
        radius_x, radius_y = w * (0.5 + padding), h * (0.5 + padding * 1.25)
        sample = max(2, int(w) // 8)
        margin = 2 * sample
        left, right = max(0, int(center_x - radius_x) - margin), min(width, int(center_x + radius_x) + margin)
        top, bottom = max(0, int(center_y - radius_y) - margin), min(height, int(center_y + radius_y) + margin)
        if right <= left or bottom <= top:
            continue
        region = result[top:bottom, left:right]
        region_height, region_width = region.shape[:2]
        smeared = cv2.resize(region, (max(1, region_width // sample), max(1, region_height // sample)),
                             interpolation=cv2.INTER_AREA)
        smeared = cv2.GaussianBlur(cv2.resize(smeared, (region_width, region_height), interpolation=cv2.INTER_LINEAR),
                                   (0, 0), sample / 2)
        mask = np.zeros((region_height, region_width), dtype=np.uint8)
        cv2.ellipse(mask, (int(center_x) - left, int(center_y) - top), (int(radius_x), int(radius_y)),
                    0, 0, 360, 255, -1)
        edge = cv2.GaussianBlur(cv2.dilate(mask, np.ones((sample | 1, sample | 1), np.uint8)), (0, 0), sample / 2)
        alpha = (np.maximum(mask, edge).astype(np.float32) / 255)[..., None]
        result[top:bottom, left:right] = (smeared * alpha + region * (1 - alpha)).astype(np.uint8)
    return result


def frame_at(video, seconds):
    video.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = video.read()
    return frame if ok else None


def blurred_frame(path, clip, find_faces, padding, keep_original=False):
    """The first candidate frame with a face found, blurred, or None when no frame clears the detector.

    keep_original also returns the unblurred frame, for analysis that never ends up in the picture."""
    video = cv2.VideoCapture(str(path))
    try:
        start = clip["annotationStartTimeS"] - clip["clipStartTimeS"]
        span = clip["annotationEndTimeS"] - clip["annotationStartTimeS"]
        for position in FRAME_POSITIONS:
            frame = frame_at(video, max(0.0, start + span * position))
            if frame is None:
                continue
            boxes = find_faces(frame)
            if boxes:
                blurred = blur_faces(frame, boxes, padding)
                return (blurred, frame) if keep_original else blurred
    finally:
        video.release()
    return (None, None) if keep_original else None


def fit_tile(image, width, height, vertical=0.5):
    """Crop to the tile's aspect ratio, then resize. vertical places a tall crop: 0 keeps the top, 0.5 the middle."""
    image_height, image_width = image.shape[:2]
    target = width / height
    if image_width / image_height > target:
        crop = int(image_height * target)
        image = image[:, (image_width - crop) // 2:(image_width + crop) // 2]
    else:
        crop = int(image_width / target)
        top = int((image_height - crop) * vertical)
        image = image[top:top + crop]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def caption(tile, text):
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, tile.shape[1] / 600, max(1, tile.shape[1] // 240)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    band = text_height + baseline + 24
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, tile.shape[0] - band), (tile.shape[1], tile.shape[0]), (0, 0, 0), -1)
    tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)
    cv2.putText(tile, text, ((tile.shape[1] - text_width) // 2, tile.shape[0] - baseline - 12), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return tile


def collage(tiles, columns, gap, background):
    rows = -(-len(tiles) // columns)
    height, width = tiles[0].shape[:2]
    canvas = np.full((rows * height + (rows + 1) * gap, columns * width + (columns + 1) * gap, 3), background,
                     dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        top, left = gap + row * (height + gap), gap + column * (width + gap)
        canvas[top:top + height, left:left + width] = tile
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=6, help="Number of pictures in the collage.")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--tile-height", type=int, default=640)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--background", default="ffffff", help="Gap color as hex RGB.")
    parser.add_argument("--seed", type=int, default=0, help="Change this for a different set of signers.")
    parser.add_argument("--captions", action="store_true", help="Write each clip's phrase under its picture.")
    parser.add_argument("--padding", type=float, default=0.35, help="Blur margin around a face, as box fraction.")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="BlazeFace detection threshold.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/fsboard-collage.jpg")
    args = parser.parse_args()

    metadata = json.loads(dataset_file(f"daun_v3/metadata/{SPLIT}.json").read_text(encoding="utf-8"))
    find_faces = FaceFinder(args.min_confidence)
    tiles_dir = args.output.with_name(args.output.stem + "-tiles")
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for clip in choose_clips(metadata, args.count, args.seed):
        if len(tiles) == args.count:
            break
        path = dataset_file(f"daun_v3/video_clips/{SPLIT}/{clip['clipFilename']}")
        if hashlib.md5(path.read_bytes()).hexdigest() != clip["clipFileMd5"]:
            path.unlink()
            print(f"skip {clip['clipFilename']}: checksum mismatch, deleted the download")
            continue
        frame = blurred_frame(path, clip, find_faces, args.padding)
        if frame is None:
            print(f"skip {clip['clipFilename']}: no face found to blur")
            continue
        tile = fit_tile(frame, args.tile_width, args.tile_height)
        if args.captions:
            tile = caption(tile, clip["phrase"].upper())
        cv2.imwrite(str(tiles_dir / f"{len(tiles):02d}-{clip['signerId']}.jpg"), tile)
        tiles.append(tile)
        print(f"tile {len(tiles)}: signer {clip['signerId']}, \"{clip['phrase']}\"")
    if not tiles:
        raise SystemExit("No usable frames found.")
    background = tuple(int(args.background[i:i + 2], 16) for i in (4, 2, 0))  # hex RGB to BGR
    cv2.imwrite(str(args.output), collage(tiles, args.columns, args.gap, background), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {args.output} ({len(tiles)} pictures) and the single tiles in {tiles_dir}")


if __name__ == "__main__":
    main()
