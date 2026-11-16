from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm 


from landmarks import load_landmarks

df = load_landmarks()

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
        print(f"k=1 accuracy: {mean_cv_score}")
    if k == 28:
        print(f"k=28 accuracy: {mean_cv_score}")

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