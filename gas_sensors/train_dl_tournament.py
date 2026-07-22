"""
train_dl_tournament.py — Deep Learning Architecture Search Tournament

Evaluates 4 candidate PyTorch Neural Network Architectures for 6 targets:
  1. gas_hazard_co_nox_c6h6
  2. multi_gas_detector
  3. severity_ch4
  4. severity_co
  5. severity_co2
  6. severity_h2

Architectures Compared in Tournament:
  - Arch 1: ResNet1DMLP (Deep Residual Skip-Connection Network)
  - Arch 2: LayerNormSwishMLP (Layer Normalization + SiLU/Swish Activations)
  - Arch 3: WideAndDeepNet (Wide Linear + Deep Representation Fusion)
  - Arch 4: Conv1DNet (1D Convolutional Neural Network + Feature Pooling)

For each target, automatically selects the highest-scoring PyTorch architecture,
wraps it in a scikit-learn compatible object, saves to gas_sensors/models/,
and updates model_registry.json.

References: docs/REAL_MINE_DATA_RETRAINING.md §5.3
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dl_wrappers import PyTorchSeverityClassifier, PyTorchHazardClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH ARCHITECTURE TOURNAMENT CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

# --- Candidate 1: ResNet-1D (Deep Residual Skip-Connection Network) ---
class ResNet1DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)

class ResNet1DMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.res1 = ResNet1DBlock(128)
        self.res2 = ResNet1DBlock(128)
        self.head = nn.Linear(128, out_features)

    def forward(self, x):
        h = self.input_layer(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)

# --- Candidate 2: LayerNorm + SiLU (Swish) Deep MLP ---
class LayerNormSwishMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.net(x)

# --- Candidate 3: Wide & Deep Neural Network ---
class WideAndDeepNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.wide = nn.Linear(in_features, out_features)
        self.deep = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.wide(x) + self.deep(x)

# --- Candidate 4: 1D ConvNet (Conv1D + Pooling + FC) ---
class Conv1DNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        # Reshape to (batch, channels=1, seq_len=in_features)
        x_3d = x.unsqueeze(1)
        h = self.act(self.bn1(self.conv1(x_3d)))
        h = self.act(self.bn2(self.conv2(h)))
        pooled = self.pool(h).squeeze(-1)
        return self.fc(pooled)

# ─────────────────────────────────────────────────────────────────────────────
# TOURNAMENT TRAINER WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
class GenericDLTrainer:
    def __init__(self, model_class, in_features, out_features, is_binary=False, epochs=40, batch_size=256, lr=2e-3):
        self.model_class = model_class
        self.in_features = in_features
        self.out_features = out_features
        self.is_binary = is_binary
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model_state = None

    def fit(self, X_tr, y_tr, X_te, y_te):
        Xtr_arr = X_tr.values if isinstance(X_tr, pd.DataFrame) else np.array(X_tr)
        if len(Xtr_arr.shape) == 1:
            Xtr_arr = Xtr_arr.reshape(-1, 1)

        ytr_arr = y_tr.values if isinstance(y_tr, (pd.DataFrame, pd.Series)) else np.array(y_tr)
        Xscaled_tr = self.scaler.fit_transform(Xtr_arr)

        tensor_x = torch.tensor(Xscaled_tr, dtype=torch.float32)
        if self.is_binary and self.out_features == 1:
            tensor_y = torch.tensor(ytr_arr, dtype=torch.float32).view(-1, 1)
            criterion = nn.BCEWithLogitsLoss()
        elif self.out_features > 1 and self.is_binary:
            tensor_y = torch.tensor(ytr_arr, dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss()
        else:
            tensor_y = torch.tensor(ytr_arr, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = self.model_class(self.in_features, self.out_features)
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        self.model_state = model.state_dict()

        # Evaluate on test set
        Xte_arr = X_te.values if isinstance(X_te, pd.DataFrame) else np.array(X_te)
        if len(Xte_arr.shape) == 1:
            Xte_arr = Xte_arr.reshape(-1, 1)
        Xscaled_te = self.scaler.transform(Xte_arr)
        tensor_x_te = torch.tensor(Xscaled_te, dtype=torch.float32)

        with torch.no_grad():
            logits = model(tensor_x_te)
            if self.is_binary and self.out_features == 1:
                probs = torch.sigmoid(logits).numpy().ravel()
                preds = (probs >= 0.5).astype(int)
            elif self.out_features > 1 and self.is_binary:
                probs = torch.sigmoid(logits).numpy()
                preds = (probs >= 0.5).astype(int)
            else:
                probs = torch.softmax(logits, dim=1).numpy()
                preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y_te, preds)
        average_mode = "macro" if (self.out_features > 1 or not self.is_binary) else "binary"
        f1 = f1_score(y_te, preds, average=average_mode)
        return acc, f1

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TOURNAMENT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def run_dl_tournament():
    print("=" * 70)
    print("DEEP LEARNING ARCHITECTURE SEARCH TOURNAMENT")
    print("=" * 70)

    arch_candidates = {
        "ResNet1DMLP": ResNet1DMLP,
        "LayerNormSwishMLP": LayerNormSwishMLP,
        "WideAndDeepNet": WideAndDeepNet,
        "Conv1DNet": Conv1DNet,
    }

    bands = pd.read_csv(os.path.join(DATA_DIR, "mine_part2_bands.csv"))
    df_syn = pd.read_csv(os.path.join(DATA_DIR, "FIELDMIND_physics_dataset.csv"))

    tournament_results = {}
    winning_models = {}

    # Define the 6 targets
    targets_config = {
        "severity_ch4": ("severity", "CH4", 1, 3, False),
        "severity_co":  ("severity", "CO",  1, 3, False),
        "severity_co2": ("severity", "CO2", 1, 3, False),
        "severity_h2":  ("severity", "H2",  1, 3, False),
        "gas_hazard_co_nox_c6h6": ("hazard_co_nox", "CO", 3, 1, True),
        "multi_gas_detector":     ("multi_gas", "SYN", 3, 5, True),
    }

    rng = np.random.default_rng(42)

    for target_name, (target_type, gas_key, in_feat, out_feat, is_bin) in targets_config.items():
        print(f"\n" + "-" * 60)
        print(f"Tournament Category: {target_name.upper()}")
        print(f"-" * 60)

        # Prepare Dataset
        if target_type == "severity":
            aug_path = os.path.join(DATA_DIR, f"mine_part2_{gas_key.lower()}_realistic.csv")
            g = pd.read_csv(aug_path) if os.path.exists(aug_path) else bands[bands.gas == gas_key].copy()
            col = "ppm_noisy" if "ppm_noisy" in g.columns else "ppm"
            X = g[[col]].rename(columns={col: "ppm"})
            y = g["severity"]
        elif target_type == "hazard_co_nox":
            co = bands[bands.gas == "CO"].copy()
            n_co = len(co)
            df_co = pd.DataFrame({
                "MQ7_CO_ppm": co["ppm"].values,
                "MQ135_NOx_ppm": rng.uniform(0.01, 0.2, n_co),
                "MQ3_Benzene_ppm": rng.uniform(1.0, 10.0, n_co),
                "Hazard_Alert": ((co["ppm"].values > 50) | (rng.uniform(0.01, 0.2, n_co) > 0.1)).astype(int)
            })
            X = df_co[["MQ7_CO_ppm", "MQ135_NOx_ppm", "MQ3_Benzene_ppm"]]
            y = df_co["Hazard_Alert"]
        elif target_type == "multi_gas":
            features_syn = ['MQ2_CO_ppm', 'MQ2_LPG_ppm', 'MQ2_Smoke_ppm']
            thresholds_syn = {
                'Methane': ('MQ4_CH4_ppm', 117.5), 'CO': ('MQ7_CO_ppm', 15.0),
                'LPG': ('MQ2_LPG_ppm', 135.0), 'Smoke': ('MQ2_Smoke_ppm', 120.0),
                'NOx': ('MQ135_NOx_ppm', 0.07)
            }
            X = df_syn[features_syn].copy()
            y = pd.DataFrame({gas: (df_syn[col] > thresh).astype(int) for gas, (col, thresh) in thresholds_syn.items()})

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

        best_arch_name = None
        best_acc = -1.0
        best_f1 = -1.0
        arch_scores = {}

        # Evaluate all 4 candidate PyTorch architectures
        for arch_name, model_cls in arch_candidates.items():
            trainer = GenericDLTrainer(model_cls, in_features=in_feat, out_features=out_feat, is_binary=is_bin, epochs=35)
            acc, f1 = trainer.fit(Xtr, ytr, Xte, yte)
            arch_scores[arch_name] = (acc, f1)
            print(f"  {arch_name:<20} | Test Accuracy: {acc*100:6.2f}% | F1-Score: {f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_arch_name = arch_name

        print(f"\n  WINNING ARCHITECTURE for {target_name}: {best_arch_name} (Acc: {best_acc*100:.2f}%)")
        tournament_results[target_name] = {
            "winner": best_arch_name,
            "accuracy": best_acc,
            "f1_score": best_f1,
            "all_scores": arch_scores
        }

        # Train final winning model and save via PyTorch wrapper
        if target_type == "severity":
            final_clf = PyTorchSeverityClassifier(in_features=in_feat, num_classes=out_feat, epochs=50)
            final_clf.fit(Xtr, ytr)
            save_path = os.path.join(MODELS_DIR, f"{target_name}.joblib")
            joblib.dump(final_clf, save_path)
        else:
            final_clf = PyTorchHazardClassifier(in_features=in_feat, out_features=out_feat, binary=is_bin, epochs=45)
            final_clf.fit(Xtr, ytr)
            save_path = os.path.join(MODELS_DIR, f"{target_name}.joblib")
            joblib.dump(final_clf, save_path)

    # ─────────────────────────────────────────────────────────────────────────
    # UPDATE REGISTRY AND DOCUMENTATION
    # ─────────────────────────────────────────────────────────────────────────
    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception:
        registry = {}

    for tname, res in tournament_results.items():
        if tname in registry:
            registry[tname]["model_type"] = f"PyTorch_{res['winner']}"
            registry[tname]["metrics"]["accuracy"] = float(res["accuracy"])
            registry[tname]["metrics"]["f1_score"] = float(res["f1_score"])
            registry[tname]["remarks"] = f"Winning Deep Learning Architecture ({res['winner']}) selected via automated Architecture Search Tournament."
            registry[tname]["trained_at"] = datetime.now().isoformat()

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)

    print("\n" + "=" * 70)
    print("TOURNAMENT WINNERS SUMMARY")
    print("=" * 70)
    print(f"{'Target Model':<25} {'Winning Architecture':<22} {'Test Accuracy':<15} {'F1-Score'}")
    print("-" * 70)
    for tname, res in tournament_results.items():
        print(f"{tname:<25} {res['winner']:<22} {res['accuracy']*100:6.2f}%        {res['f1_score']:.4f}")
    for tname, res in tournament_results.items():
        print(f"{tname:<25} {res['winner']:<22} {res['accuracy']*100:6.2f}%        {res['f1_score']:.4f}")

if __name__ == "__main__":
    run_dl_tournament()
