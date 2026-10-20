"""diagnose.py RUN.pt : validation CER by clip length bucket + worst examples."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare
from score_test import load
model = load(sys.argv[1])
val = prepare.Split("validation")
r = prepare.evaluate(model, val, return_predictions=True)
rows = []
for c in val.clips:
    p = r["predictions"][c["name"]]
    rows.append((c["end"] - c["start"], len(c["text"]), prepare.edit_distance(c["text"], p), len(p), c["text"], p))
rows = np.array(rows, dtype=object)
L = rows[:, 0].astype(int)
for lo, hi in [(0, 150), (150, 300), (300, 600), (600, 1200), (1200, 4000)]:
    m = (L >= lo) & (L < hi)
    if m.any():
        print(f"frames {lo}-{hi}: clips {m.sum()} cer {rows[m,2].sum()/rows[m,1].sum():.4f} pred/ref len {rows[m,3].sum()/rows[m,1].sum():.3f}")
print("overall", r["cer"])
for row in sorted(rows.tolist(), key=lambda x: -x[2])[:8]:
    print(row[0], row[2], repr(row[4]), "->", repr(row[5][:120]))
