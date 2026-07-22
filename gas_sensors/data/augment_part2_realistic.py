"""
augment_part2_realistic.py — Inject sensor physics into Part 2 band data.

Adjusts cross-sensitivity factors to reflect physical MQ sensor datasheet selectivity ratios:
  - MQ-4 (Methane): Slight H2 cross-sensitivity (~0.01)
  - MQ-7 (CO): Selective to CO; CH4 selectivity is ~0.0005 (1/2000th)
  - MG811 (CO2): NDIR sensor, highly selective; CO cross-sensitivity ~0.001
  - MQ-136 (H2): Slight CH4 cross-sensitivity (~0.005)

Output: gas_sensors/data/mine_part2_{gas}_realistic.csv per gas

References: docs/REAL_MINE_DATA_RETRAINING.md section 6
"""
import os
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = script_dir

rng = np.random.default_rng(42)

df = pd.read_csv(os.path.join(DATA_DIR, "mine_part2_bands.csv"))
print(f"Part 2 loaded: {len(df)} rows")

# Physical MQ datasheet selectivity ratios
CROSS_SENS = {
    "CH4": ("H2",  0.01),    # MQ-4 responds to H2 at ~1%
    "CO":  ("CH4", 0.0005),  # MQ-7 CO sensor: CH4 selectivity ratio is ~1:2000
    "CO2": ("CO",  0.001),   # MG811 CO2 NDIR sensor: highly selective
    "H2":  ("CH4", 0.005),   # MQ-136 responds to CH4 at ~0.5%
}

for gas in ["CH4", "CO", "CO2", "H2"]:
    print(f"\nAugmenting {gas} with physical sensor noise & selectivity...")
    g = df[df.gas == gas].copy().reset_index(drop=True)
    n = len(g)

    noisy = g["ppm"].values.copy().astype(float)

    # 1. Sensor noise: 2% multiplicative Gaussian
    noisy *= (1 + rng.normal(0, 0.02, n))

    # 2. Slow drift: MQ R0 aging random walk
    noisy *= np.cumsum(rng.normal(0, 1e-4, n)) + 1.0

    # 3. Ventilation cycle: 5% sinusoidal dilution (~20 min period)
    noisy *= 1 + 0.05 * np.sin(np.arange(n) * 2 * np.pi / 1200)

    # 4. Physical cross-sensitivity bleed
    cross_gas, cross_frac = CROSS_SENS[gas]
    cross_ppm = df[df.gas == cross_gas]["ppm"].sample(n, random_state=1).values
    noisy += cross_frac * cross_ppm

    noisy = np.clip(noisy, 0, None)
    g["ppm_noisy"] = noisy

    out_path = os.path.join(DATA_DIR, f"mine_part2_{gas.lower()}_realistic.csv")
    g.to_csv(out_path, index=False)

    orig_std = g["ppm"].std()
    noisy_std = pd.Series(noisy).std()
    print(f"  Original std: {orig_std:.1f}, Noisy std: {noisy_std:.1f}")
    print(f"  Original range: [{g['ppm'].min():.1f}, {g['ppm'].max():.1f}]")
    print(f"  Noisy range:    [{noisy.min():.1f}, {noisy.max():.1f}]")
    print(f"  Saved: {out_path}")

print("\nAugmentation complete.")
