"""
gan_generate_co.py — PyTorch CGAN for Carbon Monoxide (CO)

Performs TWO separate CGAN balancing passes:
  Pass 1: Balance 'severity' (0,1,2) -> N_max = 10,000 each [already balanced, no synthesis needed]
  Pass 2: Balance 'over_tlv' (0,1) -> N_max = 20,004
          Class 0: 20,004 samples (majority)
          Class 1: 9,996 samples -> synthesize 10,008 CGAN samples

Output: gas_sensors/data/mine_part2_co_balanced_cgan.csv
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
INPUT_CSV = os.path.join(script_dir, "mine_part2_co_realistic.csv")
OUTPUT_CSV = os.path.join(script_dir, "mine_part2_co_balanced_cgan.csv")
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


def train_cgan(X_scaled, y_mapped, num_classes, noise_dim=32, epochs=40, batch_size=256):
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y = torch.nn.functional.one_hot(torch.tensor(y_mapped), num_classes=num_classes).float()
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
            noise = torch.randn(bs, noise_dim)
            fake_f = netG(noise, real_l)
            d_loss = crit(netD(real_f, real_l), torch.ones(bs, 1)) + \
                     crit(netD(fake_f.detach(), real_l), torch.zeros(bs, 1))
            d_loss.backward(); optD.step()
            optG.zero_grad()
            fake_f2 = netG(torch.randn(bs, noise_dim), real_l)
            g_loss = crit(netD(fake_f2, real_l), torch.ones(bs, 1))
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
    print("PYTORCH CGAN: CO — severity + over_tlv BALANCING")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    all_dfs = [df.copy()]

    # ── PASS 1: severity (already balanced 10k/10k/10k) ──
    sev_counts = df["severity"].value_counts().to_dict()
    N_max_sev = max(sev_counts.values())
    print(f"\nPass 1 — severity: {sev_counts} | N_max={N_max_sev} [Already balanced]")

    # ── PASS 2: over_tlv — N_max = 20,004 ──
    otlv_counts = df["over_tlv"].value_counts().to_dict()
    N_max_otlv = max(otlv_counts.values())
    print(f"\nPass 2 — over_tlv: {otlv_counts} | N_max={N_max_otlv}")

    # Train CGAN on over_tlv classes
    X_raw = df[FEATURE_COLS].values.astype(float)
    y_otlv = df["over_tlv"].values.astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    present = sorted(set(y_otlv))
    class_map = {c: i for i, c in enumerate(present)}
    inv_map = {i: c for i, c in enumerate(present)}
    y_mapped = np.array([class_map[c] for c in y_otlv])
    num_classes = len(present)

    print(f"  Training CGAN on over_tlv ({num_classes} classes)...")
    netG = train_cgan(X_scaled, y_mapped, num_classes)
    print("  CGAN training complete.")

    noise_dim = 32
    for i, orig_cls in inv_map.items():
        needed = N_max_otlv - otlv_counts.get(orig_cls, 0)
        if needed > 0:
            print(f"  Synthesizing {needed} samples for over_tlv={orig_cls}...")
            gen_raw = synthesize(netG, scaler, num_classes, i, needed, noise_dim)
            df_gen = pd.DataFrame(gen_raw, columns=FEATURE_COLS)
            df_gen["gas"] = "CO"
            df_gen["over_tlv"] = orig_cls
            # Assign severity based on pct vs TLV
            df_gen["ppm"] = np.clip(df_gen["ppm"], 0, None)
            df_gen["ppm_noisy"] = np.clip(df_gen["ppm_noisy"], 0, None)
            df_gen["pct"] = df_gen["ppm"] / 10000.0
            # Severity mapping for CO: L1=0 (<= tlv), L2=1 (mid), L3=2 (> tlv)
            df_gen["severity"] = np.where(df_gen["pct"] <= df_gen["tlv_pct"] * 0.75, 0,
                                  np.where(df_gen["pct"] <= df_gen["tlv_pct"], 1, 2))
            df_gen["level"] = df_gen["severity"].map({0: "L1", 1: "L2", 2: "L3"})
            all_dfs.append(df_gen)

    df_balanced = pd.concat(all_dfs, ignore_index=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}  ({len(df_balanced):,} rows)")
    print("severity distribution:", df_balanced["severity"].value_counts().to_dict())
    print("over_tlv distribution:", df_balanced["over_tlv"].value_counts().to_dict())


if __name__ == "__main__":
    run()
