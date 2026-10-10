"""PCA av håndlandmarks -> 2D-scatter med én farge og etikett per bokstav."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("data/mock_asl_landmarks.csv")
df = df.drop(columns=["person"], errors="ignore")   # funker både med og uten person-kolonnen

y = df["label"].to_numpy()
X = df.drop(columns="label").to_numpy()             # (n, 63)

# PCA til 2 dimensjoner. Ingen standardisering: alle features er koordinater
# i samme skala, så det er greit å bruke dem rått (samme avstand som KNN ser).
pca = PCA(n_components=2)
Z = pca.fit_transform(X)
ev = pca.explained_variance_ratio_
print(f"Forklart varians: PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}, sum {ev.sum():.1%}")

labels = sorted(set(y))
colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist() + plt.cm.Dark2(np.linspace(0, 1, 8)).tolist()

fig, ax = plt.subplots(figsize=(11, 8))
for i, L in enumerate(labels):
    pts = Z[y == L]
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.35, color=colors[i % len(colors)])
    cx, cy = np.median(pts, axis=0)
    ax.text(cx, cy, L, fontsize=13, fontweight="bold", ha="center", va="center",
            color=colors[i % len(colors)],
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

ax.set_xlabel(f"PC1 ({ev[0]:.1%} av variansen)")
ax.set_ylabel(f"PC2 ({ev[1]:.1%} av variansen)")
ax.set_title("PCA av håndlandmarks, farget etter bokstav")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("model/pca_letters.png", dpi=120)
plt.show()