import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load scaled features and binary labels from preprocessed files
X_train = np.load("X_train_scaled.npy")
y_train = np.load("y_train_binary.npy")
X_test = np.load("X_test_scaled.npy")
y_test = np.load("y_test_binary.npy")

# Build and train the RandomForest model
forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

forest.fit(X_train, y_train)

rf_preds = forest.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"RandomForest test acc: {rf_acc:.4f}")

# Save the trained RandomForest model for use later in the pipeline
joblib.dump(forest, "rf_model.joblib")

# MLP surrogate model
# used to simulate attacker knowledge during adversarial crafting
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),  # a little regularization
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2)  # binary output
        )

    def forward(self, x):
        return self.net(x)

# Check if we have a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]
mlp = SurrogateMLP(input_dim).to(device)

# Prepare training data for PyTorch
batch_size = 256
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Simple training loop
epochs = 20
for epoch in range(epochs):
    mlp.train()
    epoch_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out = mlp(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    avg_loss = epoch_loss / len(train_loader.dataset)

    # Quick evaluation on test set
    mlp.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_logits = mlp(x_test_tensor)
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        test_acc = accuracy_score(y_test, test_preds)

    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, test_acc={test_acc:.4f}")

# Save model weights for use in downstream attack simulation
torch.save(mlp.state_dict(), "surrogate_mlp.pth")
print("Saved surrogate_mlp.pth and rf_model.joblib")
