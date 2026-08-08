"""
gan_generate_part1.py — PyTorch Conditional GAN (CGAN) for ESP32 Sensor Telemetry Class Balancing

Loads: gas_sensors/data/mine_part1_clean.csv
Target Class: is_warmup (False: Steady-State Baseline, True: Warmup/Transient State)

Calculates N_max = max count across state classes (1721).
Synthesizes minority class samples (is_warmup=True) using a trained PyTorch CGAN until both classes have N_max rows.
Fills ALL 12 columns for all 3442 rows with zero NaNs and accurate domain bounds.

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

    feature_cols = ["air_quality_ppm", "smoke_ppm", "alcohol_ppm", "flamable_gas_ppm", "MQ136_ppm", "MQ7_ppm", "t", "h"]
    for col in feature_cols:
        df[col] = df.groupby("is_warmup")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    if "dt_s" in df.columns:
        df["dt_s"] = df["dt_s"].fillna(0.0)

    X_raw = df[feature_cols].values
    y_raw = df["is_warmup"].astype(int).values

    num_classes = 2
    noise_dim = 32
    feature_dim = len(feature_cols)

    class_counts = pd.Series(y_raw).value_counts().to_dict()
    N_max = max(class_counts.values())
    print("\nOriginal Class Distribution:")
    for cls in range(num_classes):
        cls_name = "Steady State (False)" if cls == 0 else "Warmup/Transient (True)"
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

    epochs = 150
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
                df_gen["is_warmup"] = (cls == 1)

                # Physical bounds enforcement
                df_gen["air_quality_ppm"] = np.clip(df_gen["air_quality_ppm"], 0, None)
                df_gen["smoke_ppm"] = np.clip(df_gen["smoke_ppm"], 0, None)
                df_gen["alcohol_ppm"] = np.clip(df_gen["alcohol_ppm"], 0, None)
                df_gen["flamable_gas_ppm"] = np.clip(df_gen["flamable_gas_ppm"], 0, None)
                df_gen["MQ136_ppm"] = np.clip(df_gen["MQ136_ppm"], 0, None)
                df_gen["MQ7_ppm"] = np.clip(df_gen["MQ7_ppm"], 0, None)
                df_gen["t"] = np.clip(df_gen["t"], 15.0, 45.0)
                df_gen["h"] = np.clip(df_gen["h"], 20.0, 95.0)

                # Back-calculate raw columns from generated ppm
                temp = df_gen["t"]
                hum = df_gen["h"]
                cf = 1.0 - 0.005 * (temp - 20.0) - 0.002 * (hum - 65.0)
                cf = np.clip(cf, 0.1, 5.0)

                temp_real = df["t"].fillna(df["t"].median())
                hum_real = df["h"].fillna(df["h"].median())
                cf_real = 1.0 - 0.005 * (temp_real - 20.0) - 0.002 * (hum_real - 65.0)
                cf_real = np.clip(cf_real, 0.1, 5.0)

                calibration_params = {
                    "air_quality":  {"RL": 10.0, "is_mv": False, "clean_ratio": 3.6,   "a": 110.47, "b": -2.862, "out_col": "air_quality_ppm"},
                    "smoke":        {"RL": 10.0, "is_mv": False, "clean_ratio": 9.8,   "a": 613.9,  "b": -2.07,   "out_col": "smoke_ppm"},
                    "alcohol":      {"RL": 10.0, "is_mv": False, "clean_ratio": 60.0,  "a": 0.4,    "b": -2.5,    "out_col": "alcohol_ppm"},
                    "flamable_gas": {"RL": 20.0, "is_mv": True,  "clean_ratio": 4.4,   "a": 1012.7, "b": -2.78,   "out_col": "flamable_gas_ppm"},
                    "MQ136_raw":    {"RL": 10.0, "is_mv": True,  "clean_ratio": 1.0,   "a": 28.5,   "b": -1.95,   "out_col": "MQ136_ppm"},
                    "MQ7_raw":      {"RL": 10.0, "is_mv": False, "clean_ratio": 27.0,  "a": 99.04,  "b": -1.518,  "out_col": "MQ7_ppm"},
                }

                for col, params in calibration_params.items():
                    ppm = df_gen[params["out_col"]]
                    ratio = (np.clip(ppm, 1e-8, None) / params["a"]) ** (1.0 / params["b"])
                    
                    val = df[col].interpolate(limit=3)
                    if params["is_mv"]:
                        vout = np.clip(val / 1000.0, 0.001, 4.999)
                    else:
                        vout = np.clip(val * 5.0 / 1023.0, 0.001, 4.999)
                    rs = params["RL"] * (5.0 - vout) / vout
                    rs_corrected = rs / cf_real
                    steady_state_rs = rs_corrected[~df["is_warmup"]].dropna()
                    if len(steady_state_rs) > 0:
                        rs_clean = np.median(steady_state_rs)
                    else:
                        rs_clean = params["RL"] * 2.0
                    r0 = rs_clean / params["clean_ratio"]

                    rs_gen = r0 * ratio * cf
                    rs_gen = np.clip(rs_gen, 0.001, 1e6)
                    vout_gen = (5.0 * params["RL"]) / (rs_gen + params["RL"])

                    if params["is_mv"]:
                        df_gen[col] = vout_gen * 1000.0
                    else:
                        df_gen[col] = vout_gen * 1023.0 / 5.0

                # Metadata columns
                real_dt = df[df["is_warmup"] == (cls == 1)]["dt_s"].dropna().values
                if len(real_dt) == 0 or np.all(real_dt == 0):
                    real_dt = np.array([6.0, 7.0])
                synthetic_dt = np.random.choice(real_dt[real_dt > 0], size=needed)
                df_gen["dt_s"] = synthetic_dt

                real_elapsed = df[df["is_warmup"] == (cls == 1)]["elapsed_s"].values
                min_el, max_el = (real_elapsed.min(), real_elapsed.max()) if len(real_elapsed) > 0 else (0, 600)
                df_gen["elapsed_s"] = np.random.randint(min_el, max_el + 1, size=needed)

                base_time = pd.to_datetime("3/20/2023 16:55")
                gen_timestamps = [ (base_time + pd.Timedelta(seconds=int(es))).strftime("%m/%d/%Y %H:%M") for es in df_gen["elapsed_s"] ]
                df_gen["timestamp"] = gen_timestamps

                df_gen = df_gen[df.columns]
                synthetic_dfs.append(df_gen)

    df_balanced = pd.concat(synthetic_dfs, ignore_index=True)
    df_balanced.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print(f"BALANCED DATASET CREATED: {OUTPUT_CSV}")
    print("=" * 60)
    print("Final Class Distribution:")
    final_counts = df_balanced["is_warmup"].value_counts().to_dict()
    for cls in [False, True]:
        print(f"  Class {cls}: {final_counts.get(cls, 0)} samples")
    print(f"Total Rows: {len(df_balanced)}")
    print(f"Total NaNs across all columns: {df_balanced.isna().sum().sum()}")

if __name__ == "__main__":
    run_part1_gan()
