"""
ingest_mine_part1.py — MINE DATA_Part1.xlsx -> tidy CSV.

Handles: label:value parsing, UART bit-flip corruption, line-merged records,
range-gating of impostor values, warm-up flagging.

Output: gas_sensors/data/mine_part1_clean.csv

References: docs/REAL_MINE_DATA_RETRAINING.md sections 1-2
"""
import os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Source file lives in "ma'am data" directory at project root
SRC  = os.path.join(HERE, "..", "..", "ma'am data", "MINE DATA_Part1.xlsx")
OUT  = os.path.join(HERE, "mine_part1_clean.csv")

# Deliberately neutral names for the two mislabelled channels (see doc section 1.1).
# MQ-136 prints "h2:" but is an H2S sensor; MQ-7 prints "flame:" but is a CO sensor.
# User confirmed: "no issue" — proceed with neutral names to avoid silent mislabelling.
COLS = ["ts", "air_quality", "smoke", "alcohol",
        "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h", "_junk"]

# Physically plausible ADC / physical ranges. Anything outside == UART corruption.
# User confirmed: hardware constants are ADC values.
GATES = {
    "air_quality":  (0, 1024), "smoke":    (0, 1024), "alcohol": (0, 1024),
    "flamable_gas": (0, 5000), "MQ136_raw":(0, 5000), "MQ7_raw": (0, 1024),
    "t":            (5, 60),   "h":        (10, 100),
}

TS_RE  = re.compile(r"(\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{2})")
REC_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{2})"           # timestamp
    r".*?air_quality\D*?([\d.]+)"  r".*?smoke\D*?([\d.]+)"
    r".*?alcohol\D*?([\d.]+)"      r".*?flamable_gas\D*?([\d.]+)"
    r".*?h2\D*?([\d.]+)"           r".*?flame\D*?([\d.]+)"
    r".*?t:\s*([\d.]+)"            r".*?h:\s*([\d.]+)"
)

def load_raw() -> pd.DataFrame:
    raw = pd.read_excel(SRC)
    raw.columns = COLS
    return raw

def parse(raw: pd.DataFrame) -> pd.DataFrame:
    """Pass 1: strict regex over the whole joined row -> recovers line-merged records."""
    joined = raw.astype(str).apply(lambda r: " | ".join(map(str, r)), axis=1)

    rows = []
    for line in joined:
        for m in REC_RE.finditer(line):          # finditer -> catches BOTH merged records
            ts, *vals = m.groups()
            rows.append([ts] + [float(v) for v in vals])

    df = pd.DataFrame(rows, columns=["ts", "air_quality", "smoke", "alcohol",
                                     "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h"])

    # Pass 2: rows the strict regex missed but that still carry a valid timestamp.
    # Recover them field-by-field so a single corrupt channel doesn't kill the record.
    got = set(df["ts"])
    loose = []
    for line in joined:
        tm = TS_RE.search(line)
        if not tm or tm.group(1) in got:
            continue
        rec = {"ts": tm.group(1)}
        for name, pat in [("air_quality", r"air_q\S*?\D([\d.]+)"), ("smoke", r"\S?m\S?ke\D*?([\d.]+)"),
                          ("alcohol", r"a\S?coh\S*?([\d.]+)"),     ("flamable_gas", r"f\S?a\S?able_g\S*?([\d.]+)"),
                          ("MQ136_raw", r"h2\D*?([\d.]+)"),        ("MQ7_raw", r"flame\D*?([\d.]+)"),
                          ("t", r"\bt:\s*([\d.]+)"),               ("h", r"\bh:\s*([\d.]+)")]:
            mm = re.search(pat, line)
            rec[name] = float(mm.group(1)) if mm else np.nan
        loose.append(rec)

    if loose:
        df = pd.concat([df, pd.DataFrame(loose)], ignore_index=True)
    return df

CALIBRATION_PARAMS = {
    "air_quality":  {"RL": 10.0, "is_mv": False, "clean_ratio": 3.6,   "a": 110.47, "b": -2.862, "out_col": "air_quality_ppm"},
    "smoke":        {"RL": 10.0, "is_mv": False, "clean_ratio": 9.8,   "a": 613.9,  "b": -2.07,   "out_col": "smoke_ppm"},
    "alcohol":      {"RL": 10.0, "is_mv": False, "clean_ratio": 60.0,  "a": 0.4,    "b": -2.5,    "out_col": "alcohol_ppm"},
    "flamable_gas": {"RL": 20.0, "is_mv": True,  "clean_ratio": 4.4,   "a": 1012.7, "b": -2.78,   "out_col": "flamable_gas_ppm"},
    "MQ136_raw":    {"RL": 10.0, "is_mv": True,  "clean_ratio": 1.0,   "a": 28.5,   "b": -1.95,   "out_col": "MQ136_ppm"},
    "MQ7_raw":      {"RL": 10.0, "is_mv": False, "clean_ratio": 27.0,  "a": 99.04,  "b": -1.518,  "out_col": "MQ7_ppm"},
}

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["ts"], format="%H:%M:%S %d/%m/%y", errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop(columns=["ts"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # Range-gate: NaN out impostor values from UART digit corruption.
    for col, (lo, hi) in GATES.items():
        df.loc[~df[col].between(lo, hi), col] = np.nan

    df["dt_s"] = df["timestamp"].diff().dt.total_seconds()

    # Warm-up flag: MQ heaters stabilising. See doc section 1.5 -- do NOT train on these.
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["is_warmup"] = df["elapsed_s"] < 600          # first 10 minutes

    # Calibration logic (ADC -> ppm)
    # Compute temperature and humidity correction factors
    temp = df["t"].fillna(df["t"].median())
    hum = df["h"].fillna(df["h"].median())
    cf = 1.0 - 0.005 * (temp - 20.0) - 0.002 * (hum - 65.0)
    cf = np.clip(cf, 0.1, 5.0)

    for col, params in CALIBRATION_PARAMS.items():
        val = df[col].interpolate(limit=3)
        if params["is_mv"]:
            vout = np.clip(val / 1000.0, 0.001, 4.999)
        else:
            vout = np.clip(val * 5.0 / 1023.0, 0.001, 4.999)

        rs = params["RL"] * (5.0 - vout) / vout
        rs_corrected = rs / cf

        # Calibrate R0 from steady-state clean air
        steady_state_rs = rs_corrected[~df["is_warmup"]].dropna()
        if len(steady_state_rs) > 0:
            rs_clean = np.median(steady_state_rs)
        else:
            rs_clean = params["RL"] * 2.0  # fallback estimation
        
        r0 = rs_clean / params["clean_ratio"]

        # Calculate ratio and ppm
        ratio = rs_corrected / r0
        df[params["out_col"]] = params["a"] * (ratio ** params["b"])

    return df.reset_index(drop=True)

if __name__ == "__main__":
    df = clean(parse(load_raw()))
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  rows={len(df)}  warmup={int(df.is_warmup.sum())}")
    print(df.drop(columns=["timestamp"]).describe().T)
