from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
import pickle
import numpy as np 

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / 'data'
models = Path(__file__).resolve().parent
model_path = models / 'trained_models'

# Load extracted landmarks
cols = ["label"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]
df = pd.read_csv(data_path / "extractedLandmarks.csv", header=None, names=cols) 

X = df.drop(columns="label")
y = df["label"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm = LinearSVC(random_state=42, tol=1e-05)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

with open(model_path / "svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)