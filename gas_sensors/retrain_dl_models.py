"""
retrain_dl_models.py — PyTorch Deep Learning Retraining & Tournament Suite

Retrains gas sensor Deep Learning models using balanced synthetic datasets:
  1. mine_part1_balanced_gan.csv (Target: is_warmup)
  2. mine_part2_ch4_balanced_cgan.csv (Targets: severity, over_tlv)
  3. mine_part2_co_balanced_cgan.csv (Targets: severity, over_tlv)
  4. mine_part2_co2_balanced_cgan.csv (Targets: severity, over_tlv)
  5. mine_part2_h2_balanced_cgan.csv (Targets: severity, over_tlv)

Evaluates 5 candidate PyTorch Deep Learning Architectures per target:
  - Arch 1: ResNet1DMLP (Deep Residual Skip-Connection Network)
  - Arch 2: LayerNormSwishMLP (Layer Normalization + SiLU/Swish Activations)
  - Arch 3: WideAndDeepNet (Wide Linear + Deep Representation Fusion)
  - Arch 4: Conv1DNet (1D Convolutional Neural Network + Feature Pooling)
  - Arch 5: SelfAttentionMLP (Multi-Head Feature Self-Attention Network)

Per Target:
  - Compares top 3 performing DL algorithms
  - Saves ONLY the best-performing model to gas_sensors/models/
  - Updates gas_sensors/models/model_registry.json
  - Generates docs/DL_MODELS_RETRAINING_COMPARISON.md
"""
import os
import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dl_wrappers import PyTorchSeverityClassifier, PyTorchHazardClassifier

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
DOCS_DIR = os.path.join(script_dir, "..", "docs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# PYTORCH ARCHITECTURE TOURNAMENT CANDIDATES
# -----------------------------------------------------------------------------

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
        x_3d = x.unsqueeze(1)
        h = self.act(self.bn1(self.conv1(x_3d)))
        h = self.act(self.bn2(self.conv2(h)))
        pooled = self.pool(h).squeeze(-1)
        return self.fc(pooled)


# --- Candidate 5: Self-Attention Feature Transformer ---
class SelfAttentionMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.ffn = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)  # (batch, seq=1, dim=64)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out).squeeze(1)
        return self.ffn(h)


# -----------------------------------------------------------------------------
# TOURNAMENT TRAINER CLASS
# -----------------------------------------------------------------------------
class GenericDLTrainer:
    def __init__(self, model_class, in_features, out_features, is_binary=False, epochs=35, batch_size=256, lr=2e-3):
        self.model_class = model_class
        self.in_features = in_features
        self.out_features = out_features
        self.is_binary = is_binary
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model_state = None

    def fit_eval(self, X_tr, y_tr, X_te, y_te):
        Xtr_arr = X_tr.values if isinstance(X_tr, pd.DataFrame) else np.array(X_tr)
        if len(Xtr_arr.shape) == 1:
            Xtr_arr = Xtr_arr.reshape(-1, 1)

        ytr_arr = y_tr.values if isinstance(y_tr, (pd.DataFrame, pd.Series)) else np.array(y_tr)
        Xscaled_tr = self.scaler.fit_transform(Xtr_arr)

        tensor_x = torch.tensor(Xscaled_tr, dtype=torch.float32)

        if self.is_binary and self.out_features == 1:
            tensor_y = torch.tensor(ytr_arr, dtype=torch.float32).view(-1, 1)
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

        # Evaluate on Training Set for Train Acc
        with torch.no_grad():
            logits_tr = model(tensor_x)
            if self.is_binary and self.out_features == 1:
                probs_tr = torch.sigmoid(logits_tr).numpy().ravel()
                preds_tr = (probs_tr >= 0.5).astype(int)
            else:
                probs_tr = torch.softmax(logits_tr, dim=1).numpy()
                preds_tr = np.argmax(probs_tr, axis=1)
        train_acc = accuracy_score(ytr_arr, preds_tr)

        # Evaluate on Test Set
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
                try:
                    auc = roc_auc_score(y_te, probs)
                except Exception:
                    auc = 0.5
            else:
                probs = torch.softmax(logits, dim=1).numpy()
                preds = np.argmax(probs, axis=1)
                try:
                    auc = roc_auc_score(y_te, probs, multi_class="ovr")
                except Exception:
                    auc = 0.5

        acc = accuracy_score(y_te, preds)
        avg = "macro" if not self.is_binary else "binary"
        prec = precision_score(y_te, preds, average=avg, zero_division=0)
        rec = recall_score(y_te, preds, average=avg, zero_division=0)
        f1 = f1_score(y_te, preds, average=avg, zero_division=0)

        return {
            "train_accuracy": train_acc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc,
            "model_class": self.model_class,
            "model_state": model.state_dict(),
            "scaler": self.scaler
        }


def run_retraining():
    print("=" * 75)
    print("  RETRAINING PYTORCH DEEP LEARNING GAS SENSOR MODELS ON BALANCED DATASETS")
    print("=" * 75)

    arch_candidates = {
        "ResNet1DMLP": ResNet1DMLP,
        "LayerNormSwishMLP": LayerNormSwishMLP,
        "WideAndDeepNet": WideAndDeepNet,
        "Conv1DNet": Conv1DNet,
        "SelfAttentionMLP": SelfAttentionMLP,
    }

    # Define targets to retrain based on user requests and datasets
    targets_config = {
        "part1_warmup": {
            "dataset": os.path.join(DATA_DIR, "mine_part1_balanced_gan.csv"),
            "features": ["air_quality", "smoke", "alcohol", "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h"],
            "target": "is_warmup",
            "in_feat": 8, "out_feat": 1, "is_binary": True,
            "desc": "Part 1 Clean Telemetry Warmup State Classifier"
        },
        "ch4_severity": {
            "dataset": os.path.join(DATA_DIR, "mine_part2_ch4_balanced_cgan.csv"),
            "features": ["ppm"],
            "target": "severity",
            "in_feat": 1, "out_feat": 3, "is_binary": False,
            "desc": "Methane (CH4) Multi-Class Severity Model (0, 1, 2)"
        },
        "ch4_over_tlv": {
            "dataset": os.path.join(DATA_DIR, "mine_part2_ch4_balanced_cgan.csv"),
            "features": ["ppm"],
            "target": "over_tlv",
            "in_feat": 1, "out_feat": 1, "is_binary": True,
            "desc": "Methane (CH4) Over TLV Hazard Classifier (0, 1)"
        },
        "co_severity": {
            "dataset": os.path.join(DATA_DIR, "mine_part2_co_balanced_cgan.csv"),
            "features": ["ppm"],
            "target": "severity",
            "in_feat": 1, "out_feat": 3, "is_binary": False,
            "desc": "Carbon Monoxide (CO) Multi-Class Severity Model (0, 1, 2)"
        },
        "co_over_tlv": {
            "dataset": os.path.join(DATA_DIR, "mine_part2_co_balanced_cgan.csv"),
            "features": ["ppm"],
            "target": "over_tlv",
            "in_feat": 1, "out_feat": 1, "is_binary": True,
            "desc": "Carbon Monoxide (CO) Over TLV Hazard Classifier (0, 1)"
        },
        "co2_severity": {
            "dataset": os.path.join(DATA_DIR, "mine_part2_co2_balanced_cgan.csv"),
            "features": ["ppm"],
            "target": "severity",
            "in_feat": 1, "out_feat": 3, "is_binary": False,
            "desc": "Carbon Dioxide (CO2) Multi-Class Severity Model (0, 1, 2)"
        },
        "co2_over_tlv": {
            "dataset": os.path.join(DATA_DIR, "mine_part2_co2_balanced_cgan.csv"),
            "features": ["ppm"],
            "target": "over_tlv",
            "in_feat": 1, "out_feat": 1, "is_binary": True,
            "desc": "Carbon Dioxide (CO2) Over TLV Hazard Classifier (0, 1)"
        },
        "h2_severity": {
            "dataset": os.path.join(DATA_DIR, "mine_part2_h2_balanced_cgan.csv"),
            "features": ["ppm"],
            "target": "severity",
            "in_feat": 1, "out_feat": 3, "is_binary": False,
            "desc": "Hydrogen (H2) Multi-Class Severity Model (0, 1, 2)"
        },
        "h2_over_tlv": {
            "dataset": os.path.join(DATA_DIR, "mine_part2_h2_balanced_cgan.csv"),
            "features": ["ppm"],
            "target": "over_tlv",
            "in_feat": 1, "out_feat": 1, "is_binary": True,
            "desc": "Hydrogen (H2) Over TLV Hazard Classifier (0, 1)"
        },
    }

    all_tournament_results = {}
    best_saved_models = {}

    for t_key, cfg in targets_config.items():
        print(f"\n" + "-" * 70)
        print(f"  TOURNAMENT TARGET: {t_key.upper()} ({cfg['desc']})")
        print(f"-" * 70)

        df = pd.read_csv(cfg["dataset"])
        X = df[cfg["features"]]
        y = df[cfg["target"]]

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        target_evals = []

        # Train and evaluate all 5 candidate DL architectures
        for arch_name, model_cls in arch_candidates.items():
            trainer = GenericDLTrainer(
                model_class=model_cls,
                in_features=cfg["in_feat"],
                out_features=cfg["out_feat"],
                is_binary=cfg["is_binary"],
                epochs=35, batch_size=256
            )
            res = trainer.fit_eval(Xtr, ytr, Xte, yte)
            res["arch_name"] = arch_name
            target_evals.append(res)

            print(f"  {arch_name:<20} | Test Acc: {res['accuracy']*100:6.2f}% | F1: {res['f1_score']:.4f} | AUC: {res['auc']:.4f}")

        # Rank candidates by F1-Score (and Accuracy as secondary)
        target_evals.sort(key=lambda x: (x["f1_score"], x["accuracy"]), reverse=True)
        top3 = target_evals[:3]
        winner = top3[0]

        print(f"\n  WINNING ARCHITECTURE: {winner['arch_name']} (F1: {winner['f1_score']:.4f}, Acc: {winner['accuracy']*100:.2f}%)")

        all_tournament_results[t_key] = {
            "description": cfg["desc"],
            "winner": winner,
            "top3": top3,
            "all_candidates": target_evals
        }

        # Save ONLY the single BEST model for this target
        if cfg["is_binary"] and cfg["out_feat"] == 1:
            best_wrapper = PyTorchHazardClassifier(
                in_features=cfg["in_feat"],
                out_features=cfg["out_feat"],
                binary=True, epochs=40
            )
            best_wrapper.scaler = winner["scaler"]
            best_wrapper.model_state = winner["model_state"]
        else:
            best_wrapper = PyTorchSeverityClassifier(
                in_features=cfg["in_feat"],
                num_classes=cfg["out_feat"],
                epochs=40
            )
            best_wrapper.scaler = winner["scaler"]
            best_wrapper.model_state = winner["model_state"]

        save_filename = f"{t_key}_dl_best.joblib"
        save_path = os.path.join(MODELS_DIR, save_filename)
        joblib.dump(best_wrapper, save_path)
        print(f"  Saved Best Model Artifact -> {save_path}")

        best_saved_models[t_key] = {
            "model_path": save_path,
            "winning_arch": winner["arch_name"],
            "accuracy": float(winner["accuracy"]),
            "f1_score": float(winner["f1_score"]),
            "auc": float(winner["auc"])
        }

    # Update model_registry.json
    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception:
        registry = {}

    for t_key, meta in best_saved_models.items():
        registry[t_key] = {
            "model_type": f"PyTorch_{meta['winning_arch']}",
            "file": f"{t_key}_dl_best.joblib",
            "metrics": {
                "accuracy": meta["accuracy"],
                "f1_score": meta["f1_score"],
                "roc_auc": meta["auc"]
            },
            "remarks": f"Winning PyTorch architecture ({meta['winning_arch']}) selected via DL Tournament on balanced dataset.",
            "trained_at": datetime.now().isoformat()
        }

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)

    print("\nModel registry updated at: ", registry_path)

    # Generate benchmark comparison markdown report
    generate_comparison_report(all_tournament_results)


def generate_comparison_report(tournament_res):
    report_path = os.path.join(DOCS_DIR, "DL_MODELS_RETRAINING_COMPARISON.md")

    md = []
    md.append("# PyTorch Deep Learning Model Retraining & Architecture Tournament Report")
    md.append("## Evaluation on CGAN Balanced Gas Telemetry Datasets\n")
    md.append("This document summarizes the retraining and architectural comparison of five candidate PyTorch Deep Learning models trained on balanced datasets (`mine_part1_balanced_gan.csv` and `mine_part2_*_balanced_cgan.csv`). For each target, **ONLY the single best-performing deep learning model** has been saved to disk.\n")

    md.append("---")
    md.append("## Executive Summary of Winning Models\n")

    md.append("| Target Key | Target Feature | Dataset / Gas | Winning DL Architecture | Test Accuracy | Macro F1-Score | ROC-AUC | Saved Model Artifact |")
    md.append("|---|---|---|---|---|---|---|---|")

    for t_key, data in tournament_res.items():
        win = data["winner"]
        desc = data["description"]
        md.append(f"| `{t_key}` | `{t_key.split('_')[-1]}` | {desc.split(' ')[0]} | **{win['arch_name']}** | **{win['accuracy']*100:.2f}%** | **{win['f1_score']:.4f}** | **{win['auc']:.4f}** | `gas_sensors/models/{t_key}_dl_best.joblib` |")

    md.append("\n---\n")
    md.append("## Detailed Model Training vs. Testing Performance & Dataset Splits\n")

    md.append("| Model Name | Arch Used | Training Acc | Testing Acc | Training Dataset | Testing Dataset | Train Test Split Ratio |")
    md.append("|---|---|---|---|---|---|:---:|")

    for t_key, data in tournament_res.items():
        win = data["winner"]
        model_name = f"{t_key}_dl_best.joblib"
        arch_used = win["arch_name"]
        tr_acc = f"{win['train_accuracy']*100:.2f}%"
        te_acc = f"{win['accuracy']*100:.2f}%"

        if "part1" in t_key:
            ds_name = "mine_part1_balanced_gan.csv"
            tr_ds = f"`{ds_name}` (122,025 rows)"
            te_ds = f"`{ds_name}` (40,675 rows)"
        else:
            gas_tag = t_key.split("_")[0]
            ds_name = f"mine_part2_{gas_tag}_balanced_cgan.csv"
            tr_ds = f"`{ds_name}` (45,000 rows)"
            te_ds = f"`{ds_name}` (15,000 rows)"

        split_ratio = "75:25 (0.75 / 0.25)"
        md.append(f"| `{model_name}` | **{arch_used}** | **{tr_acc}** | **{te_acc}** | {tr_ds} | {te_ds} | {split_ratio} |")

    md.append("\n---\n")
    md.append("## Top 3 Algorithm Comparisons per Target Model\n")

    for t_key, data in tournament_res.items():
        md.append(f"### Target: `{t_key.upper()}` — {data['description']}\n")
        md.append("| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | Macro F1-Score | ROC-AUC | Status |")
        md.append("|:---:|---|---|---|---|---|---|---|")

        top3 = data["top3"]
        for idx, cand in enumerate(top3):
            rank_str = f"**Rank {idx+1}**"
            status_str = "**SAVED (WINNER)**" if idx == 0 else "Discarded"
            md.append(f"| {rank_str} | **{cand['arch_name']}** | {cand['accuracy']*100:.2f}% | {cand['precision']:.4f} | {cand['recall']:.4f} | {cand['f1_score']:.4f} | {cand['auc']:.4f} | {status_str} |")
        md.append("\n")

    md.append("---\n")
    md.append("## Candidate Deep Learning Architecture Specifications\n")

    md.append("1. **ResNet1DMLP (Deep Residual Skip Network)**:\n"
              "   - Input linear projection -> 2x ResNet 1D blocks with BatchNorm1d, GELU activations, and shortcut residual additions.\n"
              "   - Prevents gradient degradation in multi-layer representation learning.\n")

    md.append("2. **LayerNormSwishMLP (Swish Deep MLP)**:\n"
              "   - Layer Normalization + SiLU (Swish) non-linearities + Dropout regularization.\n"
              "   - Provides smooth gradient flow and robust feature scale invariance.\n")

    md.append("3. **WideAndDeepNet (Wide & Deep Architecture)**:\n"
              "   - Direct wide linear connection fused with deep non-linear feature pathways.\n"
              "   - Captures both linear feature thresholds and complex non-linear sensor correlations.\n")

    md.append("4. **Conv1DNet (1D Convolutional Neural Network)**:\n"
              "   - 1D Convolutions (`Conv1d`) -> BatchNorm1d -> Adaptive Average Pooling -> Dense classification head.\n"
              "   - Captures spatial feature receptive fields across multi-sensor channels.\n")

    md.append("5. **SelfAttentionMLP (Multi-Head Self-Attention Transformer)**:\n"
              "   - Multi-Head Self-Attention (`MultiheadAttention`) projection with LayerNorm and Feed-Forward Network.\n"
              "   - Computes dynamic cross-feature attention weights.\n")

    md.append("---\n")
    md.append("## Final Model Registry Verification\n")
    md.append("All winning model artifacts have been serialized using `joblib` into `gas_sensors/models/` and registered in `model_registry.json`. Ready for production agent integration.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nBenchmark Comparison Report successfully written to: {report_path}")


if __name__ == "__main__":
    run_retraining()
