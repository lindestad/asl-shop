from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm 


project_root = Path(__file__).resolve().parents[1]
data_path = project_root / 'data'
models = Path(__file__).resolve().parent
model_path = models / 'trained_models'

# Load extracted landmarks
cols = ["label"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]
df = pd.read_csv(data_path / "extractedLandmarks.csv", header=None, names=cols) 

X = df.drop(columns="label")
y = df["label"]

# Cross validation testing 
k_range = range(1, 51)
cv_scores = []

# Evaluate each k using 5-fold cross-validation
for k in tqdm(k_range, desc="Evaluating k"):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    mean_cv_score = scores.mean()
    cv_scores.append(mean_cv_score)
    if k == 1:
        print(f"k=1 loss: {mean_cv_score}")
    if k == 28:
            print(f"k=28 loss: {mean_cv_score}")

# Plot accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_scores, marker='o')
plt.title("k-NN Cross-Validation accuracy vs k")
plt.xlabel("Number of Neighbors: k")
plt.ylabel("Cross-Validated accuracy")
plt.grid(True)
plt.show()

# best k 
best_k = k_range[np.argmax(cv_scores)]
print(f"Best k from cross-validation: {best_k}")