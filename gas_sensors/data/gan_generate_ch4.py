"""
gan_generate_ch4.py — PyTorch CGAN for Methane (CH4) Dataset Balancing

Performs PyTorch CGAN telemetry synthesis conditioned on 'severity' (0, 1, 2),
preserving strict physical concentration monotonicity across levels (L1 < L2 < L3).
Computes over_tlv deterministically from physical thresholding (pct >= 2.5% / ppm >= 25000.0).

Total output dataset: 60,000 rows (20,000 rows per severity class).
Output file: gas_sensors/data/mine_part2_ch4_balanced_cgan.csv
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(script_dir, "mine_part2_ch4_realistic.csv")
OUTPUT_CSV = os.path.join(script_dir, "mine_part2_ch4_balanced_cgan.csv")

FEATURE_COLS = ["pct", "ppm", "ppm_noisy"]
TLV_PCT_CONST = 2.5
TLV_PPM_CONST = 25000.0


class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, feature_dim),
        )

    def forward(self, noise, labels):
        return self.net(torch.cat([noise, labels], dim=1))


class Discriminator(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, features, labels):
        return self.net(torch.cat([features, labels], dim=1))


def train_cgan(X_scaled, y_sev, num_classes=3, noise_dim=32, epochs=60, batch_size=256):
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y = torch.nn.functional.one_hot(torch.tensor(y_sev, dtype=torch.long), num_classes=num_classes).float()
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    feature_dim = X_scaled.shape[1]
    netG = Generator(noise_dim, num_classes, feature_dim)
    netD = Discriminator(feature_dim, num_classes)

    optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    crit = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for real_f, real_l in loader:
            bs = real_f.size(0)

            # Train Discriminator
            optD.zero_grad()
            noise = torch.randn(bs, noise_dim)
            fake_f = netG(noise, real_l)
            d_real_loss = crit(netD(real_f, real_l), torch.ones(bs, 1))
            d_fake_loss = crit(netD(fake_f.detach(), real_l), torch.zeros(bs, 1))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()

            # Train Generator
            optG.zero_grad()
            fake_f2 = netG(torch.randn(bs, noise_dim), real_l)
            g_loss = crit(netD(fake_f2, real_l), torch.ones(bs, 1))
            g_loss.backward()
            optG.step()

    return netG


def synthesize(netG, scaler, sev_idx, num_classes, count, noise_dim=32):
    netG.eval()
    with torch.no_grad():
        z = torch.randn(count, noise_dim)
        c = torch.zeros(count, num_classes)
        c[:, sev_idx] = 1.0
        gen_scaled = netG(z, c).numpy()
    return scaler.inverse_transform(gen_scaled)


def run():
    print("=" * 70)
    print("  PYTORCH CGAN: METHANE (CH4) PHYSICAL SEVERITY BALANCING")
    print("=" * 70)

    if os.path.exists(INPUT_CSV):
        df_real = pd.read_csv(INPUT_CSV)
    else:
        bands_csv = os.path.join(script_dir, "mine_part2_bands.csv")
        df_all = pd.read_csv(bands_csv)
        df_real = df_all[df_all["gas"] == "CH4"].copy()
        if "ppm_noisy" not in df_real.columns:
            df_real["ppm_noisy"] = df_real["ppm"] + np.random.normal(0, 100, len(df_real))

    df_real["tlv_pct"] = TLV_PCT_CONST
    df_real["tlv_ppm"] = TLV_PPM_CONST
    df_real["over_tlv"] = np.where(df_real["pct"] >= TLV_PCT_CONST, 1, 0)

    print(f"Loaded {len(df_real):,} real rows")

    X_train_raw = df_real[FEATURE_COLS].values.astype(float)
    y_train_sev = df_real["severity"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    print("Training PyTorch CGAN conditioned on 3 severity classes...")
    netG = train_cgan(X_train_scaled, y_train_sev, num_classes=3, noise_dim=32, epochs=60, batch_size=256)
    print("CGAN training complete.")

    all_dfs = [df_real.copy()]
    target_per_sev = 20000

    for sev in [0, 1, 2]:
        existing_count = (df_real["severity"] == sev).sum()
        needed = target_per_sev - existing_count
        if needed > 0:
            print(f"Synthesizing {needed:,} CGAN samples for Severity {sev} (Level L{sev+1})...")
            gen_raw = synthesize(netG, scaler, sev, num_classes=3, count=needed, noise_dim=32)
            df_gen = pd.DataFrame(gen_raw, columns=FEATURE_COLS)

            # Enforce physical monotonicity per severity level
            if sev == 0:
                df_gen["pct"] = np.clip(df_gen["pct"], 0.0, 1.25)
            elif sev == 1:
                df_gen["pct"] = np.clip(df_gen["pct"], 1.25, 1.875)
            else:
                df_gen["pct"] = np.clip(df_gen["pct"], 1.875, 2.50)

            df_gen["ppm"] = df_gen["pct"] * 10000.0
            noise_std = np.std(df_real["ppm_noisy"] - df_real["ppm"])
            df_gen["ppm_noisy"] = np.clip(df_gen["ppm"] + np.random.normal(0, max(noise_std, 10.0), len(df_gen)), 0, None)

            df_gen["gas"] = "CH4"
            df_gen["level"] = f"L{sev+1}"
            df_gen["severity"] = sev
            df_gen["tlv_pct"] = TLV_PCT_CONST
            df_gen["tlv_ppm"] = TLV_PPM_CONST
            df_gen["over_tlv"] = np.where(df_gen["pct"] >= TLV_PCT_CONST, 1, 0)

            df_gen = df_gen[df_real.columns]
            all_dfs.append(df_gen)

    df_balanced = pd.concat(all_dfs, ignore_index=True)
    df_balanced["tlv_pct"] = TLV_PCT_CONST
    df_balanced["tlv_ppm"] = TLV_PPM_CONST
    df_balanced["over_tlv"] = np.where(df_balanced["pct"] >= TLV_PCT_CONST, 1, 0)

    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved balanced dataset to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df_balanced):,}")
    print("Severity distribution:\n", df_balanced["severity"].value_counts().to_dict())
    print("Over_TLV distribution:\n", df_balanced["over_tlv"].value_counts().to_dict())


if __name__ == "__main__":
    run()
