"""
gan_generate_co2.py — PyTorch CGAN for Carbon Dioxide (CO2) Balancing over_tlv & severity

Performs CGAN dataset balancing across target features: 'over_tlv' (0,1) and 'severity' (0,1,2).
Joint classes (severity, over_tlv):
  (0, 0): Real 4,087 samples -> Synthesize 5,913 CGAN samples
  (0, 1): Real 5,913 samples -> Synthesize 4,087 CGAN samples
  (1, 0): Real 0 samples -> Synthesize 10,000 CGAN samples
  (1, 1): Real 10,000 samples
  (2, 0): Real 0 samples -> Synthesize 10,000 CGAN samples
  (2, 1): Real 10,000 samples

Total output dataset: 60,000 rows (perfectly balanced: 10,000 per joint class).
Output file: gas_sensors/data/mine_part2_co2_balanced_cgan.csv
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
INPUT_CSV = os.path.join(script_dir, "mine_part2_co2_realistic.csv")
OUTPUT_CSV = os.path.join(script_dir, "mine_part2_co2_balanced_cgan.csv")

FEATURE_COLS = ["pct", "ppm", "ppm_noisy"]
TARGET_COLS = ["severity", "over_tlv"]

# Constant properties for CO2 TLV
TLV_PCT_CONST = 0.02
TLV_PPM_CONST = 200.0


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


def create_joint_class(df):
    """
    Map (severity, over_tlv) into 6 distinct joint classes:
      0: (0, 0)
      1: (1, 0)
      2: (2, 0)
      3: (0, 1)
      4: (1, 1)
      5: (2, 1)
    """
    return df["severity"] + df["over_tlv"] * 3


def train_cgan(X_scaled, y_joint, num_classes=6, noise_dim=32, epochs=60, batch_size=256):
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y = torch.nn.functional.one_hot(torch.tensor(y_joint, dtype=torch.long), num_classes=num_classes).float()
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


def synthesize(netG, scaler, cls_idx, num_classes, count, noise_dim=32):
    netG.eval()
    with torch.no_grad():
        z = torch.randn(count, noise_dim)
        c = torch.zeros(count, num_classes)
        c[:, cls_idx] = 1.0
        gen_scaled = netG(z, c).numpy()
    return scaler.inverse_transform(gen_scaled)


def run():
    print("=" * 70)
    print("  PYTORCH CGAN: CO2 SYNTHESIS FOR over_tlv & severity BALANCING")
    print("=" * 70)

    df_real = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df_real):,} rows from {INPUT_CSV}")

    # Enforce constant tlv_pct (0.02) and tlv_ppm (200.0) in real data
    df_real["tlv_pct"] = TLV_PCT_CONST
    df_real["tlv_ppm"] = TLV_PPM_CONST

    # Verify over_tlv in real data (0 if pct < 0.02 else 1)
    df_real["over_tlv"] = np.where(np.round(df_real["pct"], 6) >= TLV_PCT_CONST, 1, 0)
    df_real["joint_cls"] = create_joint_class(df_real)

    print("\nInitial joint class distribution (severity, over_tlv):")
    counts = df_real["joint_cls"].value_counts().to_dict()
    for jc in range(6):
        sev = jc % 3
        otlv = jc // 3
        print(f"  Class {jc} (severity={sev}, over_tlv={otlv}): {counts.get(jc, 0):,} rows")

    N_max = max(counts.values())
    print(f"\nMaximum class count N_max = {N_max:,}. Target count per class = {N_max:,}.")

    # Construct representative seeds for missing joint classes (1,0) and (2,0) so CGAN can learn all conditions
    seed_dfs = [df_real.copy()]
    noise_std = np.std(df_real["ppm_noisy"] - df_real["ppm"])

    # Seed for Class 1 (severity 1, over_tlv 0): pct in [0.010, 0.0199]
    n_seed = 2000
    pct_c1 = np.random.uniform(0.010, 0.0199, n_seed)
    ppm_c1 = pct_c1 * 10000.0
    seed_dfs.append(pd.DataFrame({
        "gas": "CO2", "level": "L2", "pct": pct_c1, "ppm": ppm_c1,
        "tlv_pct": TLV_PCT_CONST, "tlv_ppm": TLV_PPM_CONST,
        "over_tlv": 0, "severity": 1, "ppm_noisy": ppm_c1 + np.random.normal(0, noise_std, n_seed),
        "joint_cls": 1
    }))

    # Seed for Class 2 (severity 2, over_tlv 0): pct in [0.015, 0.0199]
    pct_c2 = np.random.uniform(0.015, 0.0199, n_seed)
    ppm_c2 = pct_c2 * 10000.0
    seed_dfs.append(pd.DataFrame({
        "gas": "CO2", "level": "L3", "pct": pct_c2, "ppm": ppm_c2,
        "tlv_pct": TLV_PCT_CONST, "tlv_ppm": TLV_PPM_CONST,
        "over_tlv": 0, "severity": 2, "ppm_noisy": ppm_c2 + np.random.normal(0, noise_std, n_seed),
        "joint_cls": 2
    }))

    df_train_base = pd.concat(seed_dfs, ignore_index=True)

    X_train_raw = df_train_base[FEATURE_COLS].values.astype(float)
    y_train_joint = df_train_base["joint_cls"].values.astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    print("\nTraining PyTorch CGAN on all 6 joint classes...")
    netG = train_cgan(X_train_scaled, y_train_joint, num_classes=6, noise_dim=32, epochs=60, batch_size=256)
    print("CGAN training complete.")

    # Synthesize underrepresented / missing classes
    all_dfs = [df_real.copy()]

    for jc in range(6):
        sev = jc % 3
        otlv = jc // 3
        existing = counts.get(jc, 0)
        needed = N_max - existing

        if needed > 0:
            print(f"Synthesizing {needed:,} CGAN samples for Class {jc} (severity={sev}, over_tlv={otlv})...")
            gen_raw = synthesize(netG, scaler, jc, num_classes=6, count=needed, noise_dim=32)

            df_gen = pd.DataFrame(gen_raw, columns=FEATURE_COLS)

            if otlv == 1:
                # Ensure physical consistency for over_tlv = 1 (pct >= 0.02 / ppm >= 200.0)
                df_gen["pct"] = np.clip(df_gen["pct"], 0.0200, 0.1000)
                df_gen["ppm"] = df_gen["pct"] * 10000.0
                df_gen["ppm_noisy"] = np.clip(df_gen["ppm_noisy"], df_gen["ppm"] - 50, df_gen["ppm"] + 50)
                df_gen["over_tlv"] = 1
            else:
                # Ensure physical consistency for over_tlv = 0 (pct < 0.02 / ppm < 200.0)
                df_gen["pct"] = np.clip(df_gen["pct"], 0.0001, 0.0199)
                df_gen["ppm"] = df_gen["pct"] * 10000.0
                df_gen["ppm_noisy"] = np.clip(df_gen["ppm_noisy"], 0, None)
                df_gen["over_tlv"] = 0

            df_gen["gas"] = "CO2"
            df_gen["severity"] = sev
            df_gen["level"] = f"L{sev+1}"
            df_gen["tlv_pct"] = TLV_PCT_CONST
            df_gen["tlv_ppm"] = TLV_PPM_CONST
            df_gen["joint_cls"] = jc

            # Order columns strictly to match original schema
            df_gen = df_gen[df_real.columns]
            all_dfs.append(df_gen)

    df_balanced = pd.concat(all_dfs, ignore_index=True).drop(columns=["joint_cls"])

    # Double check tlv constants and over_tlv index calculation
    df_balanced["tlv_pct"] = TLV_PCT_CONST
    df_balanced["tlv_ppm"] = TLV_PPM_CONST
    df_balanced["over_tlv"] = np.where(np.round(df_balanced["pct"], 6) >= TLV_PCT_CONST, 1, 0)

    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccessfully saved balanced dataset to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df_balanced):,}")

    print("\nFinal Dataset Distributions:")
    print("Severity distribution:\n", df_balanced["severity"].value_counts().to_dict())
    print("Over_TLV distribution:\n", df_balanced["over_tlv"].value_counts().to_dict())
    print("Joint (severity, over_tlv) distribution:\n", df_balanced.groupby(["severity", "over_tlv"]).size().to_dict())
    print("tlv_pct constant count:\n", df_balanced["tlv_pct"].value_counts().to_dict())
    print("tlv_ppm constant count:\n", df_balanced["tlv_ppm"].value_counts().to_dict())


if __name__ == "__main__":
    run()
