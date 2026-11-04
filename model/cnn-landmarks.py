from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
WEIGHT_DECAY = 1e-4

project_root = Path(__file__).resolve().parents[3]
data_path = project_root / 'data'

# Load extracted landmarks
cols = ["label"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]
df = pd.read_csv(data_path / "extractedLandmarks.csv", header=None, names=cols)

X = df.drop(columns="label").to_numpy(dtype="float32")

# CrossEntropyLoss needs integer class indices, not letters
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
num_classes = len(label_encoder.classes_)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


class LandmarksDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        # (N, 63) -> (N, 3, 21): x/y/z are the channels, the 21 landmarks are the sequence
        self.features = torch.from_numpy(features).reshape(-1, 21, 3).permute(0, 2, 1)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


train_set = LandmarksDataset(X_train, y_train)
test_set = LandmarksDataset(X_test, y_test)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 21 landmarks -> 10 -> 5 after the two pooling layers
        self.fc1 = nn.Linear(64 * 5, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = CNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=WEIGHT_DECAY)


for epoch in range(EPOCHS):
    model.train()
    runningLoss = 0.0
    for landmarks, labels in trainloader:
        landmarks, labels = landmarks.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        runningLoss += loss.item() * landmarks.size(0)

    epoch_loss = runningLoss / len(train_set)
    print("Epoch ", epoch + 1, "of ", EPOCHS)
    print("Loss: ", epoch_loss)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for landmarks, labels in testLoader:
        landmarks, labels = landmarks.to(device), labels.to(device)
        outputs = model(landmarks)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy: ", 100 * correct / total)


model.to("cpu")
with open(Path(__file__).resolve().parent / "trained_models" / "cnn-landmarks.pkl", "wb") as f:
    pickle.dump({"state_dict": model.state_dict(), "classes": list(label_encoder.classes_)}, f)