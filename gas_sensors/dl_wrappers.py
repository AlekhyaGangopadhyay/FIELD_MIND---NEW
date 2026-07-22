"""
dl_wrappers.py — Reusable PyTorch Deep Learning Model Wrappers

Provides scikit-learn compatible wrapper classes for PyTorch Deep Neural Networks:
  - PyTorchSeverityClassifier
  - PyTorchHazardClassifier

Ensures seamless serialization via joblib and clean inference (.predict(), .predict_proba())
across gas_agent.py, eval_dl_models.py, and detector_wrappers.py.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. PyTorch Deep Severity Model & Wrapper
# -----------------------------------------------------------------------------
class DeepSeverityNet(nn.Module):
    def __init__(self, in_features=1, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.GELU(),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class PyTorchSeverityClassifier:
    def __init__(self, in_features=1, num_classes=3, epochs=50, batch_size=256, lr=2e-3):
        self.in_features = in_features
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model_state = None
        self.classes_ = np.array([0, 1, 2])
        
    def _create_model(self):
        return DeepSeverityNet(self.in_features, self.num_classes)

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(-1, 1)
            
        y_arr = np.array(y, dtype=np.int64)
        X_scaled = self.scaler.fit_transform(X_arr)
        
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        tensor_y = torch.tensor(y_arr, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        model = self._create_model()
        model.train()
        
        criterion = nn.CrossEntropyLoss()
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
        return self

    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(-1, 1)
            
        X_scaled = self.scaler.transform(X_arr)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        
        model = self._create_model()
        model.load_state_dict(self.model_state)
        model.eval()
        
        with torch.no_grad():
            logits = model(tensor_x)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# -----------------------------------------------------------------------------
# 2. PyTorch Deep Hazard Model & Wrapper
# -----------------------------------------------------------------------------
class DeepHazardNet(nn.Module):
    def __init__(self, in_features, out_features, binary=True):
        super().__init__()
        self.binary = binary
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Linear(64, out_features)
        )
        
    def forward(self, x):
        return self.net(x)

class PyTorchHazardClassifier:
    def __init__(self, in_features, out_features=1, binary=True, epochs=40, batch_size=256, lr=2e-3):
        self.in_features = in_features
        self.out_features = out_features
        self.binary = binary
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model_state = None
        self.classes_ = np.array([0, 1])

    def _create_model(self):
        return DeepHazardNet(self.in_features, self.out_features, self.binary)

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(-1, 1)

        y_arr = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else np.array(y)
        X_scaled = self.scaler.fit_transform(X_arr)
        
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        if self.binary and self.out_features == 1:
            tensor_y = torch.tensor(y_arr, dtype=torch.float32).view(-1, 1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            tensor_y = torch.tensor(y_arr, dtype=torch.float32 if self.out_features > 1 else torch.long)
            criterion = nn.BCEWithLogitsLoss() if self.out_features > 1 else nn.CrossEntropyLoss()
            
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = self._create_model()
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
        return self

    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_scaled = self.scaler.transform(X_arr)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)

        model = self._create_model()
        model.load_state_dict(self.model_state)
        model.eval()

        with torch.no_grad():
            logits = model(tensor_x)
            if self.binary and self.out_features == 1:
                probs_pos = torch.sigmoid(logits).numpy().ravel()
                probs = np.column_stack([1.0 - probs_pos, probs_pos])
            elif self.out_features > 1:
                probs = torch.sigmoid(logits).numpy()
            else:
                probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.binary and self.out_features == 1:
            return (probs[:, 1] >= 0.5).astype(int)
        elif self.out_features > 1:
            return (probs >= 0.5).astype(int)
        else:
            return np.argmax(probs, axis=1)
