"""
gan_generate_h2s.py — PyTorch CGAN for H2S Severity Dataset Balancing & Expansion

Performs PyTorch CGAN telemetry synthesis conditioned on 'severity' (0, 1, 2),
injecting a 5% noise overlap zone around severity thresholds (L1 < 10 ppm, L2 < 20 ppm, L3 >= 20 ppm).

Total output dataset: 60,000 rows (20,000 rows per severity class).
Output file: gas_sensors/data/mine_part2_h2s_balanced_cgan.csv
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
INPUT_CSV = os.path.join(script_dir, "mine_part2_h2s_bands.csv")
OUTPUT_CSV = os.path.join(script_dir, "mine_part2_h2s_balanced_cgan.csv")

FEATURE_COLS = ["ppm"]
L1_L2_THRESHOLD = 10.0
L2_L3_THRESHOLD = 20.0


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
            if bs < 2:
                continue

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


def synthesize(netG, scaler, label_idx, num_classes, count, noise_dim=32):
    netG.eval()
    with torch.no_grad():
        z = torch.randn(count, noise_dim)
        c = torch.zeros(count, num_classes)
        c[:, label_idx] = 1.0
        gen_scaled = netG(z, c).numpy()
    return scaler.inverse_transform(gen_scaled)


def run():
    print("=" * 70)
    print("  PYTORCH CGAN: HYDROGEN SULFIDE (H2S) PHYSICAL SEVERITY BALANCING")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input dataset not found at {INPUT_CSV}")
        return

    df_real = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df_real):,} real rows from {INPUT_CSV}")

    X_train_raw = df_real[FEATURE_COLS].values.astype(float)
    y_train_sev = df_real["severity"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    print("Training PyTorch CGAN conditioned on 3 severity classes...")
    netG = train_cgan(X_train_scaled, y_train_sev, num_classes=3, noise_dim=32, epochs=60, batch_size=256)
    print("CGAN training complete.")

    all_dfs = [df_real.copy()]
    target_per_class = 20000

    for sev in [0, 1, 2]:
        existing_count = (df_real["severity"] == sev).sum()
        needed = target_per_class - existing_count
        if needed > 0:
            print(f"Synthesizing {needed:,} CGAN samples for Severity class {sev}...")
            gen_raw = synthesize(netG, scaler, sev, num_classes=3, count=needed, noise_dim=32)
            df_gen = pd.DataFrame(gen_raw, columns=FEATURE_COLS)

            # Enforce physical severity boundary matching with 5% uncertainty noise overlap
            vals = df_gen["ppm"].values
            if sev == 0:
                vals = np.clip(vals, 0.0, L1_L2_THRESHOLD - 0.01)
            elif sev == 1:
                vals = np.clip(vals, L1_L2_THRESHOLD, L2_L3_THRESHOLD - 0.01)
            else:
                vals = np.clip(vals, L2_L3_THRESHOLD, 100.0)

            # Inject noise on 5% of samples to create boundary overlap
            noise_mask = np.random.random(len(vals)) < 0.05
            vals[noise_mask] += np.random.uniform(-1.5, 1.5, size=noise_mask.sum())
            df_gen["ppm"] = np.clip(vals, 0.0, 100.0)

            # Keep other columns aligned with mine_part2_bands.csv format
            df_gen["gas"] = "H2S"
            df_gen["level"] = f"L{sev+1}"
            df_gen["pct"] = df_gen["ppm"] / 10000.0
            df_gen["tlv_pct"] = 10.0 / 10000.0
            df_gen["tlv_ppm"] = 10.0
            df_gen["over_tlv"] = (df_gen["ppm"] >= 10.0).astype(int)
            df_gen["severity"] = sev

            df_gen = df_gen[df_real.columns]
            all_dfs.append(df_gen)

    df_balanced = pd.concat(all_dfs, ignore_index=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved balanced dataset to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df_balanced):,}")
    print("Severity distribution:\n", df_balanced["severity"].value_counts().to_dict())


if __name__ == "__main__":
    run()
