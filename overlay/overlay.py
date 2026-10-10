from PIL import Image
from pathlib import Path
from scipy import ndimage
import numpy as np

out = Path("sprites")
out.mkdir(exist_ok=True)

sheet = Image.open("overlay/letters.png")
alpha = np.array(sheet)[:, :, 3] > 0
labels, n = ndimage.label(alpha)

for i, sl in enumerate(ndimage.find_objects(labels)):
    y, x = sl
    sheet.crop((x.start, y.start, x.stop, y.stop)).save(out / f"sprite_{i}.png")