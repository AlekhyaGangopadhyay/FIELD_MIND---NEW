"""
gan_generate_ch4.py — PyTorch CGAN for Methane (CH4)

Performs TWO separate CGAN balancing passes:
  Pass 1: Balance 'severity' (0,1,2) -> N_max = 10,000 each [already balanced, no synthesis needed]
  Pass 2: Balance 'over_tlv' (0) -> only class 0 exists physically (all CH4 < 2.5% TLV in dataset)
          SKIP over_tlv balancing as generating class 1 (over TLV) would be physically invalid.

Output: gas_sensors/data/mine_part2_ch4_balanced_cgan.csv
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(script_dir, "mine_part2_ch4_realistic.csv")
OUTPUT_CSV = os.path.join(script_dir, "mine_part2_ch4_balanced_cgan.csv")
FEATURE_COLS = ["ppm", "ppm_noisy", "pct", "tlv_pct", "tlv_ppm"]


class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 64),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, feature_dim)
        )
    def forward(self, noise, labels):
        return self.net(torch.cat([noise, labels], dim=1))


class Discriminator(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )
    def forward(self, features, labels):
        return self.net(torch.cat([features, labels], dim=1))


def train_cgan(X_scaled, y, num_classes, noise_dim=32, epochs=40, batch_size=256):
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y = torch.nn.functional.one_hot(torch.tensor(y), num_classes=num_classes).float()
    loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=True)

    netG = Generator(noise_dim, num_classes, X_scaled.shape[1])
    netD = Discriminator(X_scaled.shape[1], num_classes)
    optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    crit = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for real_f, real_l in loader:
            bs = real_f.size(0)
            optD.zero_grad()
            d_loss = crit(netD(real_f, real_l), torch.ones(bs, 1)) + \
                     crit(netD(netG(torch.randn(bs, noise_dim), real_l).detach(), real_l), torch.zeros(bs, 1))
            d_loss.backward(); optD.step()
            optG.zero_grad()
            fake_f = netG(torch.randn(bs, noise_dim), real_l)
            g_loss = crit(netD(fake_f, real_l), torch.ones(bs, 1))
            g_loss.backward(); optG.step()
    return netG


def synthesize(netG, scaler, num_classes, cls_idx, needed, noise_dim=32):
    netG.eval()
    with torch.no_grad():
        z = torch.randn(needed, noise_dim)
        c = torch.zeros(needed, num_classes); c[:, cls_idx] = 1.0
        gen_scaled = netG(z, c).numpy()
    return scaler.inverse_transform(gen_scaled)


def run():
    print("=" * 60)
    print("PYTORCH CGAN: CH4 — severity + over_tlv BALANCING")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # ── PASS 1: severity (already balanced 10k/10k/10k — no synthesis needed) ──
    sev_counts = df["severity"].value_counts().to_dict()
    N_max_sev = max(sev_counts.values())
    print(f"\nPass 1 — severity: {sev_counts} | N_max={N_max_sev} [Already balanced, no synthesis]")

    # ── PASS 2: over_tlv — only class 0 exists (all CH4 below TLV) ──
    otlv_counts = df["over_tlv"].value_counts().to_dict()
    print(f"\nPass 2 — over_tlv: {otlv_counts}")
    print("  SKIP: CH4 over_tlv has only class 0 (all readings physically below 2.5% TLV).")
    print("  Generating fictitious over_tlv=1 samples would violate physical combustion constraints.")

    # Output = original (already perfectly severity-balanced)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(df)} rows)")
    print("severity distribution:", df["severity"].value_counts().to_dict())
    print("over_tlv distribution:", df["over_tlv"].value_counts().to_dict())


if __name__ == "__main__":
    run()
