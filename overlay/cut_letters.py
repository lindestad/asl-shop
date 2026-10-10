from pathlib import Path
import string

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).parent
out = HERE / "letters"
out.mkdir(exist_ok=True)

sheet = Image.open(HERE / "letters.png").convert("RGBA")

# Ignore faint background noise (alpha 1-128) that otherwise becomes thousands of 1x1 "sprites"
mask = np.array(sheet)[:, :, 3] > 128
# Grow the mask so the dots on i/j join their letter body
merged = ndimage.binary_dilation(mask, iterations=40)
labels, _ = ndimage.label(merged)

# Drop small leftovers (the stray triangles on the sheet)
boxes = [s for s in ndimage.find_objects(labels)
         if (s[0].stop - s[0].start) * (s[1].stop - s[1].start) > 150 * 150]

# Reading order: row by vertical centre, then left to right
boxes.sort(key=lambda s: ((s[0].start + s[0].stop) // 2 // 800, s[1].start))

for letter, (y, x) in zip(string.ascii_lowercase, boxes):
    # Tighten the box back to the real (undilated) pixels
    ys, xs = np.nonzero(mask[y, x])
    box = (x.start + xs.min(), y.start + ys.min(), x.start + xs.max() + 1, y.start + ys.max() + 1)
    sheet.crop(box).save(out / f"{letter}.png")

print(f"saved {len(boxes)} letters to {out}")
