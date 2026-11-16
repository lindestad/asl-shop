from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle
import numpy as np 

from landmarks import load_landmarks

models = Path(__file__).resolve().parent
model_path = models / 'trained_models'

df = load_landmarks()

X = df.drop(columns="label")
y = df["label"]

# 60% train, 20% validation, 20% test.
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

svm = LinearSVC(random_state=42, tol=1e-05)
svm.fit(X_train, y_train)

print(f"Validation Accuracy: {accuracy_score(y_val, svm.predict(X_val)):.2%}")
print(f"Test Accuracy: {accuracy_score(y_test, svm.predict(X_test)):.2%}")

with open(model_path / "svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)