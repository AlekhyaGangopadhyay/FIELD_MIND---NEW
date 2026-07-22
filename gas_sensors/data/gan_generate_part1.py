"""
gan_generate_part1.py — PyTorch Conditional GAN (CGAN) for ESP32 Sensor Telemetry Class Balancing

Loads: gas_sensors/data/mine_part1_clean.csv
Target Class: is_warmup (0: Steady-State Baseline, 1: Warmup/Transient State)

Calculates N_max = max count across state classes.
Synthesizes minority class samples (Warmup/Transient) using a trained PyTorch CGAN until both classes have N_max rows.

Outputs: gas_sensors/data/mine_part1_balanced_gan.csv
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
INPUT_CSV = os.path.join(script_dir, "mine_part1_clean.csv")
OUTPUT_CSV = os.path.join(script_dir, "mine_part1_balanced_gan.csv")

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
            nn.Linear(128, feature_dim)
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, features, labels):
        x = torch.cat([features, labels], dim=1)
        return self.net(x)

def run_part1_gan():
    print("=" * 60)
    print("PYTORCH CONDITIONAL GAN: PART 1 ESP32 TELEMETRY CLASS BALANCING")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded input data: {INPUT_CSV} ({len(df)} rows)")

    feature_cols = ["air_quality", "smoke", "alcohol", "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h"]
    X_raw = df[feature_cols].values
    y_raw = df["is_warmup"].values.astype(int)

    num_classes = 2
    noise_dim = 32
    feature_dim = len(feature_cols)

    class_counts = pd.Series(y_raw).value_counts().to_dict()
    N_max = max(class_counts.values())
    print("\nOriginal Class Distribution:")
    for cls in range(num_classes):
        cls_name = "Steady State (0)" if cls == 0 else "Warmup/Transient (1)"
        print(f"  Class {cls} ({cls_name}): {class_counts.get(cls, 0)} samples")
    print(f"Target Count per Class (N_max): {N_max} samples")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y_onehot = torch.nn.functional.one_hot(torch.tensor(y_raw), num_classes=num_classes).float()

    dataset = TensorDataset(tensor_x, tensor_y_onehot)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    netG = Generator(noise_dim, num_classes, feature_dim)
    netD = Discriminator(feature_dim, num_classes)

    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    epochs = 50
    print(f"\nTraining PyTorch CGAN for {epochs} epochs...")

    for epoch in range(epochs):
        for real_feats, real_labels in dataloader:
            batch_size = real_feats.size(0)

            optimizerD.zero_grad()
            real_target = torch.ones(batch_size, 1)
            fake_target = torch.zeros(batch_size, 1)

            d_real_out = netD(real_feats, real_labels)
            d_real_loss = criterion(d_real_out, real_target)

            noise = torch.randn(batch_size, noise_dim)
            fake_feats = netG(noise, real_labels)
            d_fake_out = netD(fake_feats.detach(), real_labels)
            d_fake_loss = criterion(d_fake_out, fake_target)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizerD.step()

            optimizerG.zero_grad()
            g_fake_out = netD(fake_feats, real_labels)
            g_loss = criterion(g_fake_out, real_target)
            g_loss.backward()
            optimizerG.step()

    print("PyTorch CGAN training complete.")

    synthetic_dfs = [df.copy()]

    netG.eval()
    with torch.no_grad():
        for cls in range(num_classes):
            current_count = class_counts.get(cls, 0)
            needed = N_max - current_count
            if needed > 0:
                print(f"Synthesizing {needed} CGAN samples for Class {cls}...")
                z = torch.randn(needed, noise_dim)
                class_tensor = torch.zeros(needed, num_classes)
                class_tensor[:, cls] = 1.0

                gen_scaled = netG(z, class_tensor).numpy()
                gen_raw = scaler.inverse_transform(gen_scaled)

                df_gen = pd.DataFrame(gen_raw, columns=feature_cols)
                df_gen["is_warmup"] = cls
                df_gen["air_quality"] = np.clip(df_gen["air_quality"], 0, None)
                df_gen["smoke"] = np.clip(df_gen["smoke"], 0, None)
                df_gen["alcohol"] = np.clip(df_gen["alcohol"], 0, None)
                df_gen["flamable_gas"] = np.clip(df_gen["flamable_gas"], 0, None)
                df_gen["MQ136_raw"] = np.clip(df_gen["MQ136_raw"], 0, None)
                df_gen["MQ7_raw"] = np.clip(df_gen["MQ7_raw"], 0, None)

                synthetic_dfs.append(df_gen)

    df_balanced = pd.concat(synthetic_dfs, ignore_index=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print(f"BALANCED DATASET CREATED: {OUTPUT_CSV}")
    print("=" * 60)
    print("Final Class Distribution:")
    final_counts = df_balanced["is_warmup"].value_counts().to_dict()
    for cls in range(num_classes):
        cls_name = "Steady State (0)" if cls == 0 else "Warmup/Transient (1)"
        print(f"  Class {cls} ({cls_name}): {final_counts.get(cls, 0)} samples")
    print(f"Total Rows: {len(df_balanced)}")

if __name__ == "__main__":
    run_part1_gan()
