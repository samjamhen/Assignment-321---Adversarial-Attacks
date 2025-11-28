from pathlib import Path
import numpy as np
import joblib
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier, SklearnClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from sklearn.metrics import accuracy_score

# Locate model/data files relative to project root
BASE = Path(__file__).resolve().parent.parent

# Load test data (float32 is required for PyTorch)
X_test = np.load(BASE / "X_test_scaled.npy").astype(np.float32)
y_test = np.load(BASE / "y_test_binary.npy")

# Load the trained RandomForest model (this is our real-world target)
rf_model = joblib.load(BASE / "rf_model.joblib")
rf_art = SklearnClassifier(model=rf_model)

# Reconstruct surrogate MLP used to craft adversarial examples
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_test.shape[1]

surrogate = SurrogateMLP(input_dim).to(device)
surrogate.load_state_dict(torch.load(BASE / "surrogate_mlp.pth", map_location=device))
surrogate.eval()

# Wrap surrogate in ART's PyTorch classifier
loss_fn = nn.CrossEntropyLoss()
x_min = float(X_test.min())
x_max = float(X_test.max())

surrogate_art = PyTorchClassifier(
    model=surrogate,
    loss=loss_fn,
    input_shape=(input_dim,),
    nb_classes=2,
    clip_values=(x_min, x_max)
)

# Baseline performance
clean_preds = rf_model.predict(X_test)
baseline_acc = accuracy_score(y_test, clean_preds)
print(f"RandomForest accuracy on clean test set: {baseline_acc:.4f}")

# Define evasion attacks using the surrogate
attacks = {
    "FGSM_eps_0.075": FastGradientMethod(estimator=surrogate_art, eps=0.075),
    "FGSM_eps_0.125": FastGradientMethod(estimator=surrogate_art, eps=0.125),
    "PGD_eps_0.075": ProjectedGradientDescent(estimator=surrogate_art, eps=0.075, max_iter=20, eps_step=0.005),
    "PGD_eps_0.125": ProjectedGradientDescent(estimator=surrogate_art, eps=0.125, max_iter=40, eps_step=0.01),
}

batch_size = 2048
results = {}

# Loop through each attack config and evaluate impact on RF model
for name, attack in attacks.items():
    print(f"\nRunning attack: {name}")
    X_adv = np.zeros_like(X_test, dtype=np.float32)

    # Craft adversarial examples in chunks (batching helps memory)
    for i in range(0, X_test.shape[0], batch_size):
        xb = X_test[i:i + batch_size]
        xb_adv = attack.generate(x=xb)
        X_adv[i:i + batch_size] = xb_adv.astype(np.float32)

    # Save the generated adversarial samples
    np.save(BASE / f"X_test_adv_{name}.npy", X_adv)

    # Evaluate attack success against RandomForest
    adv_preds = rf_model.predict(X_adv)
    adv_acc = accuracy_score(y_test, adv_preds)

    # Calculate ASR: how many malicious samples fooled the model?
    mask_malicious = y_test == 1

    if mask_malicious.sum() > 0:
        clean_correct = (clean_preds[mask_malicious] == 1)
        adv_incorrect = (adv_preds[mask_malicious][clean_correct] == 0)
        if clean_correct.sum() > 0:
            asr = adv_incorrect.sum() / clean_correct.sum()
        else:
            asr = 0.0
    else:
        asr = None  # No attacks in the dataset

    print(f"{name} → RF accuracy on adversarial: {adv_acc:.4f} | ASR: {asr}")
    results[name] = {"acc_adv_rf": adv_acc, "ASR": asr}

# Print a simple summary
print("\nAdversarial attack summary:")
for attack_name, metrics in results.items():
    print(f"{attack_name}: {metrics}")
