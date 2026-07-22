# Using the Real Mine Data (Part 1 + Part 2) — Ingest, Calibration & Retraining Guide

**Scope:** how to turn our two original field-captured Excel files into training-ready
artefacts, and exactly where they plug into every FIELD-MIND module — including the
unified demo.

**Source files** (project root, one level above `FIELD_MIND---NEW/`):

| File | What it is | Shape |
|---|---|---|
| `MINE DATA_Part1.xlsx` | Raw ESP32 serial log — MQ-2 / MQ-4 / MQ-136 / MQ-7 / DHT22 | 1832 rows × 10 cols, 1 sheet |
| `Mine_Data_Part2.xlsx` | Hazard-band reference dataset — CH4 / CO / CO2 / H2 in **%** | 4 sheets × 1000 rows × 30 cols |

Everything below was verified by actually parsing the files, not assumed. Where a
working assumption turned out **not** to hold, it is called out explicitly in
[§7 Findings that change the plan](#7-findings-that-change-the-plan). **Read §7 before
writing any ingest code** — one of our stated premises about Part 1 is not supported by
the data.

---

## 1. Part 1 — Raw Sensor Log (`MINE DATA_Part1.xlsx`)

### 1.1 Physical layout

The file is a flat dump of ESP32 serial lines. Only the first cell of each row carries a
real header; pandas reads the rest as `Unnamed: N`. Every cell is a `label:value` string.

```
Data @ 16:55:46 20/03/23 | air_quality:160.00 | smoke:774.00 | alcohol:25.00
                         | flamable_gas:4062.00 | h2:4921.00 | flame:122.00
                         | t:27.10 | h:76.50
```

Column → sensor channel mapping (from the spreadsheet's header row):

| Col idx | Excel header | Field label | Sensor | Meaning |
|---|---|---|---|---|
| 0 | `Unnamed: 0` | `Data @ HH:MM:SS DD/MM/YY` | — | timestamp |
| 1 | `MQ2` | `air_quality` | MQ-2 | raw ADC, general combustible |
| 2 | `Unnamed: 2` | `smoke` | MQ-2 | raw ADC, smoke channel |
| 3 | `Unnamed: 3` | `alcohol` | MQ-2 | raw ADC, alcohol/VOC channel |
| 4 | `MQ4` | `flamable_gas` | MQ-4 | raw ADC, **methane / CNG** |
| 5 | `MQ136` | `h2` | MQ-136 | raw ADC — see warning below |
| 6 | `MQ7` | `flame` | MQ-7 | raw ADC — see warning below |
| 7 | `DHT22` | `t` | DHT22 | temperature, **°C, already physical** |
| 8 | `Unnamed: 8` | `h` | DHT22 | relative humidity, **%, already physical** |
| 9 | `Unnamed: 9` | — | — | junk / trailing serial fragment (3 non-null cells, all garbage) |

> ⚠️ **Two channel labels contradict the sensor part numbers.** MQ-136 is an **H₂S**
> sensor but the firmware prints `h2:`; MQ-7 is a **CO** sensor but the firmware prints
> `flame:`. Either the firmware string constants were copy-pasted wrong, or the
> breadboard wiring differs from the header. **Confirm against the firmware source
> before assigning any chemical meaning to columns 5 and 6.** Until then, treat them as
> `MQ136_raw` and `MQ7_raw` — the ingest script below deliberately renames them that way
> so a wrong label can never silently leak into a trained model.

### 1.2 Units — Part 1 is **raw ADC, not ppm**

The `× 10,000` percentage→ppm rule **applies to Part 2 only**. Part 1 columns 1–6 are raw
10-bit/12-bit ADC counts off the MQ analog pins. Converting them to ppm requires the
MQ datasheet Rs/R0 curve, not a linear scale factor. See [§4](#4-calibrating-part-1-adc--ppm).

Only `t` (°C) and `h` (%RH) are directly usable physical values.

### 1.3 Session structure — verified

| Property | Value |
|---|---|
| Session start | 2023-03-20 **16:55:46** |
| Session end | 2023-03-20 **20:13:09** |
| Duration | 197.4 min (~3 h 17 m) |
| Recoverable records | 1815 of 1832 rows |
| Sample interval | 6 s (1018×) / 7 s (768×) — ESP32 loop jitter |
| Largest gap | 13 s (23 occurrences, each = one corrupted record dropped) |
| Distinct timestamps | **1815 / 1815 — every timestamp is unique** |

### 1.4 Serial corruption (~1 % of rows)

About 17–20 rows are mangled by UART bit-flips. Three distinct failure modes:

1. **Bit-flipped labels, value intact** — `air_q|al9ty:10.00`, `{moke:255.00`,
   `fla�able_gas:100.00`. Recoverable: the numeric suffix survives.
2. **Digit corruption** — `afcoh�l925�00` (should be `alcohol:25.00`). The value `925` is
   a corrupted `25`. **Not** safely recoverable → drop.
3. **Line-merge** — a dropped newline glues two records into one cell:
   `...p_x001C_{#�&_!�\nData @ 17:55:53 20/03/23,air_quality":10.00 | smoke:255.00 | ...`
   The second record is fully intact and **is** recoverable by regex.

The impostor values in mode 2 are exactly why the file's summary stats look wrong:
`alcohol` max is 965 and `h2` max is 128 in a stream whose steady state is 25 and 2.
Range-gating (§2) removes them.

### 1.5 Signal regimes

The log has two clearly distinct phases — this matters for training:

| Channel | Warm-up (rows 0–100) | Steady state (rows 100+) | Steady σ |
|---|---|---|---|
| `air_quality` | 17.99 | 10.56 | 22.1 |
| `smoke` | 269.4 | 245.5 | 38.9 |
| `alcohol` | 25.00 | 32.29 | 73.3 |
| `flamable_gas` | **513.4** | 98.4 | 13.7 |
| `MQ136_raw` (`h2`) | 2.00 | 2.96 | 10.1 |
| `MQ7_raw` (`flame`) | 120.5 | 119.8 | 7.5 |
| `t` (°C) | 28.23 | 30.43 | 4.08 |
| `h` (%RH) | 72.24 | 61.92 | 3.13 |

The first ~100 samples are the **MQ heater warm-up transient** — `flamable_gas` decays
4062 → ~100, `h2` decays 4921 → ~2. This is sensor physics, not a gas event.

> **Never train a hazard classifier on the warm-up rows.** They look exactly like a
> massive gas release and will teach the model that a decaying ramp = hazard. Drop the
> first 100 rows, or use them *only* as a labelled `warmup` class for a startup detector.

After warm-up, the channels are near-constant (`air_quality` is 10 in 1585/1832 rows;
`flamable_gas` is 100 in 1568). **This session contains no gas event.** Its correct role
is a **clean-air baseline / R0 reference**, not a hazard training set. See §4.

---

## 2. Part 1 → Clean CSV (runnable ingest script)

Save as `FIELD_MIND---NEW/gas_sensors/data/ingest_mine_part1.py`.

```python
"""
ingest_mine_part1.py — MINE DATA_Part1.xlsx -> tidy CSV.

Handles: label:value parsing, UART bit-flip corruption, line-merged records,
range-gating of impostor values, warm-up flagging.

Output: gas_sensors/data/mine_part1_clean.csv
"""
import os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "..", "..", "..", "MINE DATA_Part1.xlsx")   # project root
OUT  = os.path.join(HERE, "mine_part1_clean.csv")

# Deliberately neutral names for the two mislabelled channels (see doc section 1.1).
COLS = ["ts", "air_quality", "smoke", "alcohol",
        "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h", "_junk"]

# Physically plausible ADC / physical ranges. Anything outside == UART corruption.
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

    return df.reset_index(drop=True)

if __name__ == "__main__":
    df = clean(parse(load_raw()))
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  rows={len(df)}  warmup={int(df.is_warmup.sum())}")
    print(df.drop(columns=["timestamp"]).describe().T)
```

Expected output: ~1815 rows, ~95 flagged `is_warmup`, and post-gating column stats where
`alcohol` max is 25 and `MQ136_raw` max is ~4921 (warm-up) / ~6 (steady).

**Validate before using:**

```bash
python -c "
import pandas as pd; d=pd.read_csv('gas_sensors/data/mine_part1_clean.csv')
assert d.timestamp.is_unique and d.dt_s.dropna().between(5,15).all()
print('OK', len(d), 'rows,', d.isna().sum().sum(), 'gated cells')"
```

---

## 3. Part 2 — Hazard-Band Reference (`Mine_Data_Part2.xlsx`)

### 3.1 Layout

Four sheets, one per gas. Each sheet is **three side-by-side 10-column blocks**, one per
hazard severity level:

```
row 0:  <GAS NAME>   TLV = <threshold>
row 1:  L1 (%) [...]        L2 (%) [...]        L3 (%) [...]     <- band headers at cols 0,10,20
row 2+: 1000 rows x 10 cols per block
```

So each gas yields **3 bands × 1000 rows × 10 cols = 30,000 samples**, 120,000 total.
Read with `header=None`; the block structure is positional, not named.

### 3.2 Verified bands and the ppm conversion (`ppm = % × 10,000`)

| Gas | TLV | Band | Range (%) | Range (ppm) | n |
|---|---|---|---|---|---|
| **CH₄** | 2.5 % | L1 | 0 – 1.25 | 0 – 12,500 | 10,000 |
| | | L2 | 1.25 – 1.875 | 12,500 – 18,750 | 10,000 |
| | | L3 | 1.875 – 2.5 | 18,750 – 25,000 | 10,000 |
| **CO** | 0.005 % | L1 | 0 – 0.00375 | 0 – 37.5 | 10,000 |
| | | L2 | 0.00375 – 0.005 | 37.5 – 50 | 10,000 |
| | | L3 | 0.005 – 1.0 | **50 – 10,000** | 10,000 |
| **CO₂** | 0.03 % | L1 | 0 – 0.04 | 0 – 400 | 10,000 |
| | | L2 | 0.04 – 0.1 | 400 – 1,000 | 10,000 |
| | | L3 | 0.1 – 0.5 | 1,000 – 5,000 | 10,000 |
| **H₂** | 2 % | L1 | 0 – 1.8 | 0 – 18,000 | 10,000 |
| | | L2 | 1.8 – 2.5 | 18,000 – 25,000 | 10,000 |
| | | L3 | 2.5 – 3.8 | 25,000 – 38,000 | 10,000 |

Zero NaNs, zero non-numeric cells across all 12 blocks.

> ⚠️ **Two header defects — the data is right, the labels are wrong.**
> - **CO L3** header reads `[0.05% - 1%]`, but the actual values span `0.005 – 1.0 %`.
>   `0.05` is a typo for `0.005` (which is also exactly the CO TLV, and exactly where L2
>   ends). Trust the data; fix the label.
> - **CO₂ L1** header reads `[0 - 0.04%]` while the sheet title states `TLV 0.03%`, so
>   band L1 straddles its own TLV. Either the title or the band edge is wrong.
>   **Ask whoever built the sheet which is authoritative before using CO₂ L1 as a
>   "safe" label.**

### 3.3 What this data actually is

Every one of the 12 blocks has `min` exactly on the band's lower edge, `max` exactly on
the upper edge, and `mean` within 1 % of the band midpoint (e.g. CH4 L2: min 1.25000,
max 1.87500, mean 1.56120 vs midpoint 1.5625). That is the signature of **uniform random
sampling within each band**, not of a recorded sensor trace.

**Consequence:** Part 2 has no temporal structure, no autocorrelation, no sensor noise
model, and no cross-gas correlation. Use it for what it is:

- ✅ **Threshold/label calibration** — the authoritative L1/L2/L3 → severity mapping.
- ✅ **Decision-boundary testing** — verify a classifier's cut-points land on the TLVs.
- ✅ **Class-balanced augmentation** for the severity head.
- ❌ **Not** for time-series models, drift detection, or anything using lag/rolling
  features (`combined_gases_regressor` uses lag-1..5 — Part 2 cannot feed it).

### 3.4 Part 2 ingest script

Save as `FIELD_MIND---NEW/gas_sensors/data/ingest_mine_part2.py`.

```python
"""
ingest_mine_part2.py — Mine_Data_Part2.xlsx -> tidy long-format CSV.

Unpacks the 3-block-per-sheet layout into (gas, level, pct, ppm) rows
and converts percent -> ppm via ppm = pct * 10_000.

Output: gas_sensors/data/mine_part2_bands.csv   (120,000 rows)
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "..", "..", "..", "Mine_Data_Part2.xlsx")
OUT  = os.path.join(HERE, "mine_part2_bands.csv")

PCT_TO_PPM = 10_000
SHEETS     = ["CH4", "CO", "CO2", "H2"]
BLOCKS     = {"L1": 0, "L2": 10, "L3": 20}      # column offset of each severity block

# TLV per gas, in percent, as printed in each sheet's title row.
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
```

---

## 4. Calibrating Part 1 (ADC → ppm)

Part 1 gives raw ADC; Part 2 gives ppm bands. Neither alone lets us put Part 1 on a ppm
axis — but Part 1's clean-air steady state gives us **R0**, which is the missing constant
in the standard MQ curve.

The MQ transfer function is a power law in `Rs/R0`:

```
Rs   = ((Vc / Vout) - 1) * RL          # sensor resistance from the divider
ppm  = a * (Rs / R0) ** b              # per-gas curve constants from the datasheet
```

Procedure:

1. **Extract R0** from Part 1's steady state (rows where `is_warmup == False`). The
   session is clean air, so median steady-state `Rs` **is** `R0` by definition.
2. **Look up `a`, `b`** per sensor/gas from the datasheet log-log curve
   (MQ-4/CH₄, MQ-7/CO, MQ-2/smoke, MQ-136/H₂S).
3. **Apply temperature/humidity correction** using Part 1's own `t` and `h` columns —
   MQ response drifts materially across the 27–31 °C, 60–77 %RH range this session covers.
4. **Sanity-check against Part 2**: convert a Part-1-style ADC sweep to ppm and confirm
   the resulting CH₄ values cross 12,500 / 18,750 / 25,000 ppm at sane ADC counts. If
   they don't, `a`/`b`/`R0` are wrong.

> **This step is blocked on two unknowns we do not have in the repo:** the divider
> resistance `RL` and the ADC reference `Vc` used on our board, plus the ADC bit depth
> (`air_quality`/`smoke` cap near 1024 → 10-bit, but `flamable_gas` hits 4062 → 12-bit;
> the channels may not share a scale). **Get these from the hardware build notes before
> writing the calibration.** Until then, keep Part 1 in ADC space and use
> `StandardScaler`, which is what §5.1 does.

---

## 5. Retraining Recipes

### 5.0 Where the existing models stand

Current registry (`gas_sensors/models/model_registry.json`) — all trained on synthetic or
public data:

| Model | Trained on | Reported metric | Real-data risk |
|---|---|---|---|
| `mq4_gas_classifier` | `Methane_MQ4/` batches, 128 features | acc 0.780 | features are opaque `feature_1..128` |
| `smoke_fire_alarm` | `smoke.csv` | acc 0.874 | needs 36 engineered cols we can't build from Part 1 |
| `gas_hazard_lpg_cng` | 50k synthetic | test acc 1.0, **precision 0.0, recall 0.0, f1 0.0** | ⚠️ see below |
| `gas_hazard_co_nox_c6h6` | 50k synthetic | acc 0.918, **precision 0.247** | heavy false-alarm rate |
| `gas_hazard_smoke_env` | 50k synthetic | **fire_alarm acc 0.307, recall 0.051** | ⚠️ effectively non-functional |
| `multi_gas_detector` | 50k synthetic | NOx test acc **0.312**, Methane 0.810 | NOx head is worse than chance-weighted |
| `air_quality_regressor` | UCI | R² 0.9998 | suspiciously high; verify split |
| `combined_gases_regressor` | combined CSV | R² 0.9981 | lag features → chronological leakage check needed |

> ⚠️ **`gas_hazard_lpg_cng` reports accuracy 1.0 with precision, recall and f1 all 0.0 on
> an 84-row test set.** Those numbers are only mutually consistent if the test set
> contains **zero positive samples** — the model is being scored on a set with nothing to
> detect. Its "100 % accuracy" is meaningless. This is the single most important thing
> Part 2 fixes: it gives us a real, class-balanced positive set. Same caution applies to
> `gas_hazard_smoke_env`, whose fire-alarm head has 5 % recall.

**Bottom line: the real data's first job is honest evaluation, not more training.** Run
§5.3 before §5.1.

### 5.1 Baseline / drift model from Part 1

Part 1 has **no gas event**, so there is nothing to supervise a hazard classifier with.
Its correct use is a **one-class clean-air baseline** — the same role
`temperature_humidity/` already uses IsolationForest for.

```python
# gas_sensors/train_mine_baseline.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("gas_sensors/data/mine_part1_clean.csv", parse_dates=["timestamp"])
df = df[~df.is_warmup]                      # section 1.5 -- warm-up is not clean air
FEATS = ["air_quality", "smoke", "alcohol", "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h"]
X = df[FEATS].interpolate(limit=3).dropna()

# contamination is low: this session is a clean baseline by construction.
model = make_pipeline(StandardScaler(), IsolationForest(contamination=0.01, random_state=42))
model.fit(X)
joblib.dump({"model": model, "features": FEATS}, "gas_sensors/models/mine_baseline_iforest.joblib")
print("baseline anomaly rate:", (model.predict(X) == -1).mean())
```

This gives FIELD-MIND something it currently lacks: an anomaly detector fitted to **our
actual hardware's noise floor**, which is the only way to catch a sensor fault as distinct
from a gas event.

### 5.2 Severity classifier from Part 2

Part 2 is the labelled set. Train the **severity head** on it:

```python
# gas_sensors/train_mine_severity.py
import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("gas_sensors/data/mine_part2_bands.csv")

for gas in ["CH4", "CO", "CO2", "H2"]:
    g = df[df.gas == gas]
    X, y = g[["ppm"]], g["severity"]
    # stratify: bands are exactly balanced (10k each), keep it that way in the split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    print(f"\n=== {gas} ===")
    print(classification_report(yte, clf.predict(Xte), digits=4))
    joblib.dump(clf, f"gas_sensors/models/severity_{gas.lower()}.joblib")
```

> **Expect ~1.0 accuracy, and do not report that as a result.** The bands are
> non-overlapping intervals on a single feature — the task is a lookup table, and a
> decision tree solves it exactly. The *point* of training it is to confirm the learned
> split points land on the TLVs, and to have the mapping as a serialised artefact the
> agents can call. If you want a defensible number, report the learned thresholds against
> the §3.2 table, not accuracy.
>
> For a genuinely non-trivial model, add noise/cross-sensitivity first (§6).

### 5.3 Honest re-evaluation of the existing models

This is the highest-value use of the real data. Part 2 gives a class-balanced positive set
that the synthetic test sets lacked.

```python
# gas_sensors/eval_on_real_data.py
import pandas as pd, joblib, numpy as np
from sklearn.metrics import classification_report, confusion_matrix

bands = pd.read_csv("gas_sensors/data/mine_part2_bands.csv")

# gas_hazard_lpg_cng expects [MQ2_LPG_ppm, MQ4_CH4_ppm]. Drive the CH4 axis from real
# banded data; hold LPG at the clean-air baseline so CH4 is the only varying signal.
ch4 = bands[bands.gas == "CH4"]
X = pd.DataFrame({"MQ2_LPG_ppm": 80.0, "MQ4_CH4_ppm": ch4["ppm"].values})

# Ground truth: OSHA/MSHA methane action level is 1000 ppm (10% LEL) -- the same
# threshold generate_dataset.py uses. Every L2/L3 sample is far above it.
y_true = (ch4["ppm"].values > 1000).astype(int)

m = joblib.load("gas_sensors/models/gas_hazard_lpg_cng.joblib")
y_pred = m.predict(X)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
print("positives in test set:", y_true.sum(), "of", len(y_true))   # the number the old eval lacked
```

Run the analogous check for `multi_gas_detector` (CO head vs the CO sheet) and
`gas_hazard_co_nox_c6h6`. **Record whatever comes out, including if it is bad** — a model
that fails on real banded data is a finding, and it is the finding that justifies the
retrain. Write results to `docs/REAL_DATA_EVAL.md` with the date and the git SHA.

### 5.4 Feeding the self-learning agents

`sensor_agents/gas_agent.py` seeds its replay buffer from
`gas_sensors/data/FIELDMIND_physics_dataset.csv`, reading `MQ2_LPG_ppm` / `MQ4_CH4_ppm`
and the `Hazard_Alert` label, retraining a fresh RF every 200 samples.

To let the agent learn from real data, emit a **drop-in replacement with the same schema**
rather than changing agent code:

```python
# gas_sensors/data/make_agent_replay_from_real.py
import pandas as pd, numpy as np

bands = pd.read_csv("gas_sensors/data/mine_part2_bands.csv")
base  = pd.read_csv("gas_sensors/data/mine_part1_clean.csv", parse_dates=["timestamp"])
base  = base[~base.is_warmup]

ch4 = bands[bands.gas == "CH4"].sample(frac=1, random_state=42).reset_index(drop=True)
n   = len(ch4)

# Real Part-1 humidity/temp cycled to length -- gives the agent our true env envelope.
env = base[["t", "h"]].dropna()
env = env.iloc[np.arange(n) % len(env)].reset_index(drop=True)

out = pd.DataFrame({
    "Timestamp":     pd.date_range("2024-01-01 06:00:00", periods=n, freq="1s"),
    "MQ4_CH4_ppm":   ch4["ppm"],
    "MQ2_LPG_ppm":   80.0 + np.random.default_rng(42).normal(0, 5, n),  # clean-air baseline
    "Temp_C":        env["t"].values,
    "Humidity_pct":  env["h"].values,
    "Hazard_Alert":  (ch4["ppm"] > 1000).astype(int),     # 10% LEL, per generate_dataset.py
})
out.to_csv("gas_sensors/data/FIELDMIND_real_replay.csv", index=False)
print(out["Hazard_Alert"].value_counts())
```

Then point the agent at it — one line in `gas_agent.py`:

```python
dataset_path = os.path.join(workspace_root, "gas_sensors", "data",
                            "FIELDMIND_real_replay.csv")   # was FIELDMIND_physics_dataset.csv
```

Better: make it a constructor kwarg (`dataset_name="FIELDMIND_physics_dataset.csv"`) so
`demo_agents.py` can A/B synthetic vs real without editing source. That change is worth
making before the demo — being able to run both back to back is the whole story.

> ⚠️ The synthetic replay is 50,000 rows at 1 Hz with **blast events and ventilation
> cycles**; the real replay is shuffled i.i.d. band samples with **no temporal
> structure**. The agent's `memory_window=30` rolling features will behave completely
> differently. Expect the refit accuracy to move, and do not read a change as
> improvement or regression without checking which effect caused it.

---

## 6. Making Part 2 realistic (optional, recommended before any headline claim)

Uniform-in-band samples are too easy (§3.3, §5.2). To get a model that means something,
inject the physics the bands lack — the machinery for this already exists in
`gas_sensors/generate_dataset.py` (Gaussian plume, sensor drift, ventilation cycle).

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

df = pd.read_csv("gas_sensors/data/mine_part2_bands.csv")
ch4 = df[df.gas == "CH4"].copy().reset_index(drop=True)
n = len(ch4)

# 1. Sensor noise, scaled from Part 1's measured steady-state sigma (section 1.5).
ch4["ppm_noisy"] = ch4["ppm"] * (1 + rng.normal(0, 0.03, n))

# 2. Slow drift: MQ R0 ages. Random walk, same order as generate_dataset.py.
ch4["ppm_noisy"] *= np.cumsum(rng.normal(0, 2e-4, n)) + 1.0

# 3. Ventilation cycle: sinusoidal dilution, ~20 min period.
ch4["ppm_noisy"] *= 1 + 0.10 * np.sin(np.arange(n) * 2 * np.pi / 1200)

# 4. Cross-sensitivity: MQ-4 responds to H2. Bleed the H2 sheet in at ~5%.
h2 = df[df.gas == "H2"]["ppm"].sample(n, random_state=1).values
ch4["ppm_noisy"] += 0.05 * h2

ch4.to_csv("gas_sensors/data/mine_part2_ch4_realistic.csv", index=False)
```

Now retrain §5.2 on `ppm_noisy`. Accuracy will drop below 1.0 — **that is the point**, and
the resulting number is one we can actually defend.

---

## 7. Findings that change the plan

Verified against the files. Each of these contradicts or qualifies a stated assumption.

**7.1 — Part 1 has no duplicate timestamps, so the "same timestamp = two locations" rule
does not apply to it.**

The working assumption was that repeated timestamps indicate simultaneous readings from
two different locations. Parsing all 1815 recoverable records:

- **1815 timestamps, 1815 unique. Zero collisions.**
- Strictly monotonic from 16:55:46 to 20:13:09, no reordering.
- Inter-sample gaps are only 6 s (1018×), 7 s (768×), 12 s (5×), 13 s (23×) — the 6/7 s
  split is ESP32 loop jitter, and the 12/13 s gaps are exactly where a corrupted record
  was dropped.

The two apparent duplicate groups in a naive `value_counts()` are **not real**: their
"timestamps" are corruption artefacts like `5�.3_x001C_` that collide as identical
garbage strings. Once corruption is handled, they vanish.

**Part 1 is a single continuous 3 h 17 m log from one sensor node.** There is no second
location in this file. Do not build a `location_id` splitter for it — it would produce
one group, or worse, split on noise.

If a genuine two-location capture exists, it is in a file we have not seen. **Please point
us at it**, and note that a same-timestamp collision is a fragile way to encode location —
a `node_id` column in the firmware output is worth adding for any future capture.

**7.2 — Part 2 is synthetic, not recorded.** Every band's min/max sit exactly on the
interval edges and every mean is within 1 % of the midpoint — uniform random sampling
(§3.3). It is a valid *label/threshold reference*, but it is not a field measurement, and
should not be described as one in the paper or the proposal.

**7.3 — The two files do not join.** No shared key: Part 1 has timestamps and raw ADC;
Part 2 has ppm and severity bands with no time axis. They are complementary, not
mergeable — Part 1 supplies the noise/baseline model, Part 2 supplies the labels. Any
"combined dataset" is a construction (§5.4), and must be documented as such.

**7.4 — Part 1 contains no hazard event.** Steady state is flat (`air_quality` = 10 in
87 % of rows). It cannot supervise a hazard classifier; it can only provide a baseline
(§5.1).

**7.5 — Channel labels contradict part numbers.** MQ-136 (H₂S) prints `h2:`; MQ-7 (CO)
prints `flame:` (§1.1). **Blocked on the firmware source.**

**7.6 — Two header defects in Part 2.** CO L3 reads `0.05%` where the data says `0.005%`;
CO₂ L1's upper edge (0.04 %) exceeds the sheet's own stated TLV (0.03 %) (§3.2).

**7.7 — `gas_hazard_lpg_cng`'s reported metrics are internally inconsistent** —
accuracy 1.0 with precision/recall/f1 all 0.0 implies a test set with no positives
(§5.0). Its headline number should not be quoted anywhere until §5.3 is run.

**7.8 — ADC → ppm calibration is blocked** on `RL`, `Vc`, and the per-channel ADC bit
depth, none of which are in the repo. Channels appear to disagree on scale
(`smoke` caps ~1024, `flamable_gas` reaches 4062) (§4).

---

## 8. Suggested execution order

```
1.  python gas_sensors/data/ingest_mine_part1.py       # -> mine_part1_clean.csv
2.  python gas_sensors/data/ingest_mine_part2.py       # -> mine_part2_bands.csv
3.  python gas_sensors/eval_on_real_data.py            # honest baseline; write docs/REAL_DATA_EVAL.md
4.  python gas_sensors/train_mine_baseline.py          # clean-air IsolationForest
5.  python gas_sensors/train_mine_severity.py          # per-gas severity heads
6.  (optional) section 6 realism injection, then re-run 5
7.  python gas_sensors/data/make_agent_replay_from_real.py
8.  python sensor_agents/demo_agents.py                # agents now learn from real replay
9.  python -X utf8 unified_demo/streaming_safety_simulation.py --fast
```

Steps 1–3 are safe and read-only w.r.t. existing models. **Steps 4–5 write new
`.joblib` files into `gas_sensors/models/`** — new filenames, so nothing existing is
overwritten, but back up the directory and commit `model_registry.json` before running
them. Step 7 writes a new CSV; step 8's agent retrain mutates in-memory models only.

---

## 9. Wiring the real data into the unified demo

`unified_demo/interactive_safety_hub.py` currently prompts for values with hardcoded
defaults (`ch4=400.0`, `co=12.0`, `temp=22.0`, `humidity=55.0`, `min_dist=2.5`). Two
upgrades, in increasing order of effort:

**9.1 — Real defaults (5 minutes, high value).** Replace the invented defaults with our
measured Part 1 steady-state medians and Part 2 band edges, so every prompt shows a real
number:

```python
# unified_demo/interactive_safety_hub.py
# Defaults sourced from real capture -- see docs/REAL_MINE_DATA_RETRAINING.md sections 1.5, 3.2
temp     = get_float_input("  Ambient Temperature in °C",  default=30.3)   # Part 1 steady median
humidity = get_float_input("  Relative Humidity %",        default=61.6)   # Part 1 steady median
ch4      = get_float_input("  Methane (CH4) in ppm",       default=6250.0) # Part 2 CH4 L1 midpoint
co       = get_float_input("  Carbon Monoxide (CO) in ppm",default=18.75)  # Part 2 CO  L1 midpoint
```

**9.2 — Replay mode (the demo worth showing).** Add a `--replay` flag to
`streaming_safety_simulation.py` that streams `mine_part1_clean.csv` at its true 6–7 s
cadence instead of generating random drift, then escalates by splicing in Part 2 L2 → L3
CH₄ samples. That turns the current synthetic-drift story into: *"this is 3 h 17 m of real
underground capture from our own node, and here is the system detecting a real threshold
crossing."*

Sketch:

```python
def replay_source(path="gas_sensors/data/mine_part1_clean.csv", escalate_after=0.6):
    """Yield real Part-1 samples; after `escalate_after` of the run, splice Part 2 CH4 L2/L3."""
    base  = pd.read_csv(path, parse_dates=["timestamp"])
    base  = base[~base.is_warmup].reset_index(drop=True)
    bands = pd.read_csv("gas_sensors/data/mine_part2_bands.csv")
    ch4_hi = bands[(bands.gas == "CH4") & (bands.level.isin(["L2", "L3"]))]["ppm"].values

    cut = int(len(base) * escalate_after)
    for i, row in base.iterrows():
        sample = row.to_dict()
        if i >= cut:                      # escalation phase: real bands, real severity
            sample["MQ4_CH4_ppm"] = float(ch4_hi[(i - cut) % len(ch4_hi)])
        yield sample
```

> Keep the existing synthetic path as the default and make `--replay` opt-in. The two
> modes exercise different code (the synthetic path has blast + ultrasonic drift that
> Part 1 has no data for), and losing the synthetic path would lose demo coverage of the
> vibration and navigation agents.

---

## 10. Open questions for the team

Answer these before the numbers go into the paper:

1. **Is there a second-location capture file?** §7.1 shows Part 1 is single-node. If a
   two-location dataset exists, we need it; if not, the multi-node claim needs rewording.
2. **Firmware source for the MQ-136/MQ-7 label mismatch** (§7.5). Blocks any chemical
   interpretation of two of the six gas channels.
3. **Hardware constants — `RL`, `Vc`, ADC bit depth per channel** (§7.8). Blocks ADC→ppm.
4. **CO₂ authoritative value: is the TLV 0.03 % or is L1's edge 0.04 % correct?** (§7.6).
5. **Was Part 2 generated or measured?** §7.2 says generated; if it was in fact measured
   with a rounding/binning step, we need the raw capture instead.
6. **Provenance of Part 1:** which mine, which heading, ventilation state during the
   3 h 17 m window? Needed to justify calling it a clean-air baseline (§5.1).
