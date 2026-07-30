"""
gan_generate_nh3.py — PyTorch CGAN for Ammonia (NH3) Dataset Balancing & Expansion

Performs PyTorch CGAN telemetry synthesis conditioned on 'Hazard_Alert' (0, 1),
preserving physical concentration boundaries (Safe < 25 ppm, Hazard >= 25 ppm).

Total output dataset: 60,000 rows (30,000 rows per Hazard class).
Output file: gas_sensors/data/nh3_hazard_balanced_cgan.csv
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
INPUT_CSV = os.path.join(script_dir, "nh3_hazard_dataset.csv")
OUTPUT_CSV = os.path.join(script_dir, "nh3_hazard_balanced_cgan.csv")

FEATURE_COLS = ["MQ135_NH3_ppm"]
THRESHOLD_PPM = 25.0


class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, feature_dim),
        )

    def forward(self, noise, labels):
        return self.net(torch.cat([noise, labels], dim=1))


class Discriminator(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, features, labels):
        return self.net(torch.cat([features, labels], dim=1))


def train_cgan(X_scaled, y_hazard, num_classes=2, noise_dim=16, epochs=50, batch_size=64):
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y = torch.nn.functional.one_hot(torch.tensor(y_hazard, dtype=torch.long), num_classes=num_classes).float()
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    feature_dim = X_scaled.shape[1]
    netG = Generator(noise_dim, num_classes, feature_dim)
    netD = Discriminator(feature_dim, num_classes)

    optG = optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=1e-3, betas=(0.5, 0.999))
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


def synthesize(netG, scaler, label_idx, num_classes, count, noise_dim=16):
    netG.eval()
    with torch.no_grad():
        z = torch.randn(count, noise_dim)
        c = torch.zeros(count, num_classes)
        c[:, label_idx] = 1.0
        gen_scaled = netG(z, c).numpy()
    return scaler.inverse_transform(gen_scaled)


def run():
    print("=" * 70)
    print("  PYTORCH CGAN: AMMONIA (NH3) TELEMETRY SYNTHESIS")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input dataset not found at {INPUT_CSV}")
        return

    df_real = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df_real):,} real rows from {INPUT_CSV}")

    X_train_raw = df_real[FEATURE_COLS].values.astype(float)
    y_train_hazard = df_real["Hazard_Alert"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    print("Training PyTorch CGAN conditioned on 2 hazard classes...")
    netG = train_cgan(X_train_scaled, y_train_hazard, num_classes=2, noise_dim=16, epochs=60, batch_size=32)
    print("CGAN training complete.")

    all_dfs = [df_real.copy()]
    target_per_class = 30000

    for hazard in [0, 1]:
        existing_count = (df_real["Hazard_Alert"] == hazard).sum()
        needed = target_per_class - existing_count
        if needed > 0:
            print(f"Synthesizing {needed:,} CGAN samples for Hazard class {hazard}...")
            gen_raw = synthesize(netG, scaler, hazard, num_classes=2, count=needed, noise_dim=16)
            df_gen = pd.DataFrame(gen_raw, columns=FEATURE_COLS)

            # Enforce physical boundary matching with 5% uncertainty noise overlap
            if hazard == 0:
                # Safe: baseline less than 25 ppm
                vals = np.clip(df_gen["MQ135_NH3_ppm"].values, 0.0, THRESHOLD_PPM - 0.01)
            else:
                # Hazard: baseline >= 25 ppm
                vals = np.clip(df_gen["MQ135_NH3_ppm"].values, THRESHOLD_PPM, 100.0)
            
            # Inject noise on 5% of samples to create boundary overlap
            noise_mask = np.random.random(len(vals)) < 0.05
            vals[noise_mask] += np.random.uniform(-3.0, 3.0, size=noise_mask.sum())
            df_gen["MQ135_NH3_ppm"] = np.clip(vals, 0.0, 100.0)

            df_gen["Hazard_Alert"] = hazard
            df_gen = df_gen[df_real.columns]
            all_dfs.append(df_gen)

    df_balanced = pd.concat(all_dfs, ignore_index=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved balanced dataset to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df_balanced):,}")
    print("Hazard Alert distribution:\n", df_balanced["Hazard_Alert"].value_counts().to_dict())


if __name__ == "__main__":
    run()
