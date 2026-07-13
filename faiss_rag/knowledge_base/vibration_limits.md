# Blast Vibration Safety — PPV Standards & Protocols

## Overview

Blast-induced ground vibration is a critical safety and structural concern in underground mining. The primary metric is **Peak Particle Velocity (PPV)** measured in mm/s. FIELD-MIND uses a Gradient Boosting Regressor and a Random Forest Classifier to predict and classify PPV hazards from seismic sensor arrays.

---

## PPV Safety Standards

### IS 6922:1973 (Bureau of Indian Standards)

Indian Standard 6922 is the primary regulatory framework for blast vibration in Indian mining operations.

| PPV (mm/s) | Structure Type | Risk Level |
|------------|----------------|------------|
| < 5        | Industrial structures | Safe |
| < 10       | Reinforced concrete structures | Safe |
| < 12.5     | Residential structures | Safe |
| 12.5 – 25  | Residential structures | Marginal risk |
| > 25       | Any structure | High damage risk |
| > 50       | Any structure | Severe damage / structural failure |

**FIELD-MIND threshold**: `vibration_hazard = 1` if `PPV > 1.0 mm/s` (conservative early-warning threshold for underground tunnels).

---

### ISEE (International Society of Explosives Engineers) Guidelines

| Frequency Range | Safe PPV Limit (mm/s) |
|-----------------|----------------------|
| < 40 Hz         | 19.0 |
| 40 – 100 Hz     | 38.0 |
| > 100 Hz        | 50.0 |

Note: Higher frequency vibrations cause less structural damage for the same PPV magnitude.

---

### USBM RI 8507 (US Bureau of Mines)

The USBM scaled distance approach is used to predict safe blast-to-structure distances.

**Scaled Distance (USBM)**:
```
SD_USBM = D / sqrt(W)
```
Where D = distance (m), W = maximum charge per delay (kg).

**Langefors-Kihlstrom Scaled Distance**:
```
SD_LK = D / W^(2/3)
```

FIELD-MIND calculates both `scaled_distance_usbm` and `scaled_distance_langefors` as input features for the PPV regression model.

---

## Vibration Effects on Personnel

| PPV (mm/s) | Human Perception |
|------------|-----------------|
| 0.1 – 0.5  | Barely perceptible |
| 0.5 – 1.0  | Clearly perceptible |
| 1.0 – 6.3  | Strongly felt, annoying |
| 6.3 – 63   | Uncomfortable, potential harm |
| > 63       | Injury threshold |

---

## Pre-Blast Planning Requirements

1. **Blast design**: Calculate maximum charge per delay to keep predicted PPV below safe limits.
2. **Vibration monitoring**: Place seismic sensors at structures within blast influence radius (minimum 2× predicted damage radius).
3. **Personnel exclusion zone**: All personnel must be outside blast exclusion zone; re-entry only after gas checks.
4. **Delay interval**: Minimum 8 ms between delay intervals to prevent constructive interference.
5. **Stemming**: Proper stemming (length ≥ 20× hole diameter) prevents airblast overpressure.

---

## Post-Blast Protocol

1. Wait minimum **30 minutes** after last detonation before re-entry.
2. Ventilate until CO < 25 ppm, NO2 < 3 ppm.
3. Inspect blast site for:
   - Misfires (undetonated charges)
   - New roof cracking / spalling
   - PPV readings from nearest seismic sensor
4. Document: charge weight, delay pattern, measured PPV, atmospheric readings.

---

## FIELD-MIND Vibration Sensor System

### Models

| Model | Type | Input Features | Target |
|-------|------|----------------|--------|
| `best_random_forest_classifier.joblib` | RF Classifier | 14 features | vibration_hazard (0/1) |
| `best_gradient_boosting_regressor.joblib` | GB Regressor | 17 features | log(PPV) → PPV mm/s |

### Feature Sets

**14-feature classifier set**:
offset, max_charge, total_charge, num_holes, detonator_code, trid_12, trid_13, trid_14, gx, gy, gelev, sx, sy, selev

**17-feature regressor set** (adds):
scaled_distance_usbm, scaled_distance_langefors, elevation_diff

### VibrationSensorAgent

The `VibrationSensorAgent` runs both models each tick:
- Classifier → binary `vibration_hazard` flag
- Regressor → predicted `PPV` in mm/s
- ALERT raised when confidence ≥ 0.5 for 2+ consecutive ticks
- PPV > 10 mm/s triggers CRITICAL severity alert
- Agent self-learns from `vibration_features.csv` via 200-sample experience replay buffer

### EKG Integration

Blast events stored as `BlastEvent` nodes; seismic readings as `VibrationEvent` nodes linked by `CAUSED_BY` edges. Risk profiling queries aggregate PPV history per tunnel segment.
