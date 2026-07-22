"""
ingest_mine_part2.py — Mine_Data_Part2.xlsx -> tidy long-format CSV.

Unpacks the 3-block-per-sheet layout into (gas, level, pct, ppm) rows
and converts percent -> ppm via ppm = pct * 10_000.

Output: gas_sensors/data/mine_part2_bands.csv   (120,000 rows)

User clarifications applied:
- CO2: TLV is 0.03%, L1 edge 0.04% — both correct (TLV=0.03%, L1 extends beyond TLV).
- Part 2 data is MEASURED (not generated), per user confirmation.

References: docs/REAL_MINE_DATA_RETRAINING.md sections 3-3.4
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Source file lives in "ma'am data" directory at project root
SRC  = os.path.join(HERE, "..", "..", "ma'am data", "Mine_Data_Part2.xlsx")
OUT  = os.path.join(HERE, "mine_part2_bands.csv")

PCT_TO_PPM = 10_000
SHEETS     = ["CH4", "CO", "CO2", "H2"]
BLOCKS     = {"L1": 0, "L2": 10, "L3": 20}      # column offset of each severity block

# TLV per gas, in percent, as printed in each sheet's title row.
# CO2 TLV = 0.03% confirmed by user (L1 edge 0.04% is also correct — L1 straddles TLV).
TLV_PCT = {"CH4": 2.5, "CO": 0.005, "CO2": 0.03, "H2": 2.0}

def load() -> pd.DataFrame:
    out = []
    for gas in SHEETS:
        d = pd.read_excel(SRC, sheet_name=gas, header=None)
        body = d.iloc[2:]                                    # skip title + band-header rows
        for level, start in BLOCKS.items():
            blk = body.iloc[:, start:start + 10].apply(pd.to_numeric, errors="coerce")
            pct = blk.values.ravel()
            pct = pct[~pd.isna(pct)]
            out.append(pd.DataFrame({
                "gas":   gas,
                "level": level,
                "pct":   pct,
                "ppm":   pct * PCT_TO_PPM,
            }))
    df = pd.concat(out, ignore_index=True)

    df["tlv_pct"]     = df["gas"].map(TLV_PCT)
    df["tlv_ppm"]     = df["tlv_pct"] * PCT_TO_PPM
    df["over_tlv"]    = (df["pct"] > df["tlv_pct"]).astype(int)
    df["severity"]    = df["level"].map({"L1": 0, "L2": 1, "L3": 2})
    return df

if __name__ == "__main__":
    df = load()
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  rows={len(df)}")
    print(df.groupby(["gas", "level"])["ppm"].agg(["count", "min", "max", "mean"]))
