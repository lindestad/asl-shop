"""Plot the trained landmark examples in two principal components."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

MODEL_DIR = Path(__file__).resolve().parent

with np.load(MODEL_DIR / "asl_knn.npz", allow_pickle=False) as saved:
    features = saved["features"]
    labels = saved["labels"]

pca = PCA(n_components=2)
points = pca.fit_transform(features)
explained = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(11, 8))
palette = plt.colormaps["tab20"]
for index, letter in enumerate(sorted(set(labels))):
    group = points[labels == letter]
    color = palette(index % 20)
    ax.scatter(group[:, 0], group[:, 1], s=6, alpha=0.35, color=color)
    center = np.median(group, axis=0)
    ax.text(center[0], center[1], letter, ha="center", va="center", fontweight="bold", color=color)

ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
ax.set_title("ASL hand landmarks by letter")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(MODEL_DIR / "pca_letters.png", dpi=120)
