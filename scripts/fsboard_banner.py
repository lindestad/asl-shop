"""Render a wide halftone banner of FSboard signers for a slide header.

Uses the same clips, downloads and face blurring as fsboard_collage.py: faces are blurred on the full
frame before anything else, and a clip with no face found is skipped. MediaPipe's selfie segmenter cuts
each signer out, the background becomes a soft gradient, and the signer is printed as a two-color
halftone, with ink dots on a paper tone.

    uv run python scripts/fsboard_banner.py
    uv run python scripts/fsboard_banner.py --count 3 --ink 3f63c8 --paper f6caca --seed 2
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fsboard_collage import (CACHE, ROOT, SPLIT, FaceFinder, blurred_frame, choose_clips, dataset_file,
                             download, fit_tile)

SEGMENTER_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/"
                 "float16/latest/selfie_segmenter.tflite")
SEGMENTER_MODEL = ROOT / "data/selfie_segmenter.tflite"


def bgr(hex_rgb):
    return np.array([int(hex_rgb[i:i + 2], 16) for i in (4, 2, 0)], dtype=np.float32)


class PersonMask:
    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision

        download(SEGMENTER_URL, SEGMENTER_MODEL)
        options = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
                                               output_confidence_masks=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def __call__(self, image):
        import mediapipe as mp

        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = self.segmenter.segment(rgb).confidence_masks[0].numpy_view()
        return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def halftone(darkness, cell, angle=45):
    """Ink coverage per pixel for a clustered-dot screen: dots grow with darkness, like print halftone."""
    height, width = darkness.shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    theta = np.deg2rad(angle)
    u = (x * np.cos(theta) + y * np.sin(theta)) * (2 * np.pi / cell)
    v = (-x * np.sin(theta) + y * np.cos(theta)) * (2 * np.pi / cell)
    screen = (np.cos(u) + np.cos(v)) / 4 + 0.5  # 0..1, peaks at the dot centers
    # A pixel-wide soft edge instead of a hard threshold keeps the dots round when the banner is scaled down.
    return np.clip((darkness - (1 - screen)) * cell / 2 + 0.5, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=5, help="Signers across the banner.")
    parser.add_argument("--width", type=int, default=3910)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--cell", type=int, default=14, help="Halftone dot spacing in pixels.")
    parser.add_argument("--ink", default="4466cc", help="Dot color as hex RGB.")
    parser.add_argument("--paper", default="f5cccc", help="Color under the signers as hex RGB.")
    parser.add_argument("--top", default="b3d1f5", help="Background gradient color at the top.")
    parser.add_argument("--bottom", default="f5c7b8", help="Background gradient color at the bottom.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vertical", type=float, default=0.45,
                        help="Where the crop sits in the tall frame: 0 is the top, 0.5 the middle.")
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=ROOT / "data/fsboard-banner.png")
    args = parser.parse_args()

    metadata = json.loads(dataset_file(f"daun_v3/metadata/{SPLIT}.json").read_text(encoding="utf-8"))
    find_faces, person_mask = FaceFinder(0.3), PersonMask()
    slot = args.width // args.count
    gradient = np.linspace(0, 1, args.height, dtype=np.float32)[:, None, None]
    banner = np.repeat(bgr(args.top) * (1 - gradient) + bgr(args.bottom) * gradient, args.width, axis=1)
    placed = 0
    for clip in choose_clips(metadata, args.count, args.seed):
        if placed == args.count:
            break
        path = dataset_file(f"daun_v3/video_clips/{SPLIT}/{clip['clipFilename']}")
        frame, original = blurred_frame(path, clip, find_faces, args.padding, keep_original=True)
        if frame is None:
            print(f"skip {clip['clipFilename']}: no face found to blur")
            continue
        tile = fit_tile(frame, slot, args.height, args.vertical)
        # The segmenter loses people whose face is blurred, so the cutout comes from the unblurred frame.
        # Only the mask is taken from it; every printed pixel comes from the blurred tile.
        confidence = person_mask(fit_tile(original, slot, args.height, args.vertical))
        alpha = cv2.GaussianBlur(np.clip((confidence - 0.4) * 5, 0, 1), (0, 0), 2)
        # Fade toward the slot's sides so a signer cut by the frame edge doesn't end in a straight seam.
        fade = np.clip(np.minimum(np.arange(slot), np.arange(slot)[::-1]) / (slot * 0.08), 0, 1)
        alpha = (alpha * fade[None, :])[..., None]
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        # Stretch contrast per signer so dim and bright rooms print with the same density.
        low, high = np.percentile(gray[alpha[..., 0] > 0.5], (2, 98)) if (alpha > 0.5).any() else (0, 1)
        darkness = 1 - np.clip((gray - low) / max(high - low, 1e-3), 0, 1)
        ink = halftone(darkness * 0.9 + 0.05, args.cell)[..., None]
        signer = bgr(args.paper) * (1 - ink) + bgr(args.ink) * ink
        left = placed * slot
        region = banner[:, left:left + slot]
        banner[:, left:left + slot] = signer * alpha + region * (1 - alpha)
        placed += 1
        print(f"signer {placed}: {clip['signerId']}")
    if not placed:
        raise SystemExit("No usable frames found.")
    cv2.imwrite(str(args.output), banner.astype(np.uint8))
    print(f"wrote {args.output} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
