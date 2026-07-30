import os
import pandas as pd
import numpy as np

# Paths
DATA_DIR = r"c:\Users\Student\Desktop\FIELD_MIND - NEW\gas_sensors\data"
H2S_RAW = os.path.join(DATA_DIR, "H2s essential", "2024-393-1", "data", "DATASET01.csv")
POULTRY_RAW = os.path.join(DATA_DIR, "poultry_dataset.csv.xls")

print("--- Preparing Datasets ---")

# 1. Prepare NH3 Hazard Dataset from Poultry Dataset
if os.path.exists(POULTRY_RAW):
    df_nh3 = pd.read_csv(POULTRY_RAW)
    print(f"Loaded poultry dataset: {df_nh3.shape}")
    
    # Map NH3 to MQ135_NH3_ppm and create binary Hazard_Alert
    # OSHA STEL is 35 ppm, NIOSH REL is 25 ppm. Let's use 25 ppm as hazard threshold.
    df_nh3_processed = pd.DataFrame({
        "MQ135_NH3_ppm": df_nh3["NH3"].astype(float),
        "Hazard_Alert": (df_nh3["NH3"] >= 25).astype(int)
    })
    
    nh3_out = os.path.join(DATA_DIR, "nh3_hazard_dataset.csv")
    df_nh3_processed.to_csv(nh3_out, index=False)
    print(f"Saved NH3 dataset to {nh3_out} with {len(df_nh3_processed)} rows. Positives: {df_nh3_processed['Hazard_Alert'].sum()}")
else:
    print(f"Error: Poultry dataset not found at {POULTRY_RAW}")

# 2. Prepare H2S Severity Dataset from DATASET01.csv
if os.path.exists(H2S_RAW):
    print("Loading large H2S dataset...")
    df_h2s = pd.read_csv(H2S_RAW)
    print(f"Loaded H2S dataset: {df_h2s.shape}")
    
    # Filter for H2S target gas (Labels == 2) or Air (Labels == 0)
    h2s_only = df_h2s[df_h2s["Labels"] == 2].copy()
    air_only = df_h2s[df_h2s["Labels"] == 0].copy()
    
    # Use True_concentration[ppm] as ppm
    h2s_only["ppm"] = h2s_only["True_concentration[ppm]"]
    air_only["ppm"] = air_only["True_concentration[ppm]"]
    
    # Map H2S ppm to severity levels (MSHA Action limit: 10 ppm, OSHA Ceiling: 20 ppm)
    # L1: < 10 ppm
    # L2: 10 <= ppm < 20 ppm
    # L3: >= 20 ppm
    h2s_only["severity"] = h2s_only["ppm"].apply(lambda x: 0 if x < 10.0 else (1 if x < 20.0 else 2))
    air_only["severity"] = 0
    
    # Combine H2S and Air
    combined = pd.concat([h2s_only, air_only], ignore_index=True)
    
    # Sample exactly 10,000 rows per severity class to match the balance of other gases
    df_sev0 = combined[combined["severity"] == 0].sample(n=10000, random_state=42)
    df_sev1 = combined[combined["severity"] == 1].sample(n=10000, random_state=42)
    df_sev2 = combined[combined["severity"] == 2].sample(n=10000, random_state=42)
    
    balanced_h2s = pd.concat([df_sev0, df_sev1, df_sev2], ignore_index=True)
    
    # Map to format identical to mine_part2_bands.csv
    # Column list: ['gas', 'level', 'pct', 'ppm', 'tlv_pct', 'tlv_ppm', 'over_tlv', 'severity']
    h2s_processed = pd.DataFrame({
        "gas": "H2S",
        "level": balanced_h2s["severity"].map({0: "L1", 1: "L2", 2: "L3"}),
        "pct": balanced_h2s["ppm"] / 10000.0,
        "ppm": balanced_h2s["ppm"],
        "tlv_pct": 10.0 / 10000.0,  # TLV = 10 ppm
        "tlv_ppm": 10.0,
        "over_tlv": (balanced_h2s["ppm"] >= 10.0).astype(int),
        "severity": balanced_h2s["severity"]
    })
    
    h2s_out = os.path.join(DATA_DIR, "mine_part2_h2s_bands.csv")
    h2s_processed.to_csv(h2s_out, index=False)
    print(f"Saved balanced H2S dataset to {h2s_out} with {len(h2s_processed)} rows (10k per severity).")
else:
    print(f"Error: H2S dataset not found at {H2S_RAW}")
