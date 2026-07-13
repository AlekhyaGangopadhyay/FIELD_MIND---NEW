# Gas Safety — Underground Mining Reference

## Overview

Underground mining operations face significant risks from hazardous gas accumulation. This document covers the critical gas thresholds, detection principles, and emergency response protocols for the gases monitored by FIELD-MIND.

---

## Monitored Gases and Safety Thresholds

### Methane (CH4) — MQ-4 Sensor

Methane is the primary combustible hazard in underground coal and hard rock mines. It accumulates in roof cavities and poorly ventilated headings.

| Alert Level | Concentration | Action Required |
|-------------|---------------|-----------------|
| Warning     | > 0.5% (5,000 ppm) | Increase ventilation, alert personnel |
| Evacuation  | > 1.0% (10,000 ppm) | Immediate evacuation of section |
| Explosive   | 5% – 15% (LEL–UEL) | Full site shutdown, no ignition sources |

- **Lower Explosive Limit (LEL)**: 5% v/v
- **Upper Explosive Limit (UEL)**: 15% v/v
- **OSHA PEL**: 1,000 ppm (TWA)
- **IDLH (Immediately Dangerous to Life or Health)**: 10% LEL = 5,000 ppm
- **Detection method**: FIELD-MIND uses a hybrid SVM + MLP Voting Classifier on MQ-4 sensor arrays to detect methane build-up with 128 engineered features.

**Emergency Protocol**: Upon detection above 1% CH4, immediately activate ventilation fans, halt all blasting operations, and evacuate the affected tunnel segment.

---

### Carbon Monoxide (CO) — MQ-7 Sensor

CO is produced by diesel engines, blasting, fires, and spontaneous combustion. It is odourless and colourless.

| Alert Level | Concentration | Action Required |
|-------------|---------------|-----------------|
| Caution     | > 25 ppm      | Investigate source, increase ventilation |
| Warning     | > 50 ppm      | Evacuate non-essential personnel |
| Danger      | > 200 ppm     | Immediate full evacuation |
| IDLH        | 1,200 ppm     | Immediately life-threatening |

- **OSHA PEL**: 50 ppm (TWA over 8 hours)
- **NIOSH REL**: 35 ppm
- **ACGIH TLV-STEL**: 100 ppm (15-minute short-term exposure)
- **Physiological effects**: Headache at 200 ppm, unconsciousness at 400–500 ppm, death within 2–3 hours at 1,000 ppm.

**Post-blast re-entry**: Minimum 30 minutes must elapse after blasting before re-entry; CO must be below 25 ppm.

---

### Hydrogen Sulphide (H2S) — MQ-136 Sensor

H2S is produced in sulphide ore bodies and can be released during drilling or blasting. It has a characteristic "rotten egg" smell at low concentrations but causes olfactory fatigue at dangerous levels.

| Alert Level | Concentration | Physiological Effect |
|-------------|---------------|----------------------|
| Odour threshold | 0.5 ppm  | Detectable smell |
| Warning     | > 10 ppm      | Eye irritation |
| Danger      | > 50 ppm      | Pulmonary oedema risk |
| IDLH        | 100 ppm       | Loss of consciousness, death |

- **OSHA PEL**: 20 ppm (ceiling) / 50 ppm (10-min peak)
- **NIOSH IDLH**: 100 ppm

---

### Nitrogen Oxides (NOx) — MQ-135 Sensor

NOx (NO, NO2) are produced by diesel engines and blasting operations. They can cause pulmonary oedema hours after exposure.

- **NO2 TWA**: 3 ppm (NIOSH REL)
- **NO2 STEL**: 5 ppm
- **NO TWA**: 25 ppm
- Dangerous because symptoms of overexposure may be delayed 4–24 hours.

---

### Liquefied Petroleum Gas / Compressed Natural Gas (LPG / CNG) — MQ-2 Sensor

LPG is used in mining equipment (forklifts, light vehicles) and can accumulate in low-lying areas.

| Parameter | Value |
|-----------|-------|
| LEL | 2.1% (LPG propane) |
| UEL | 9.5% |
| OSHA PEL | 1,000 ppm |
| Detection feature | MQ2_LPG_ppm, MQ4_CH4_ppm (2-feature RF classifier) |

---

### Dust and Particulates (PM2.5) — Optical Sensor

Silica dust (crystalline SiO2) causes silicosis — a fatal irreversible lung disease.

- **OSHA PEL for respirable crystalline silica**: 0.05 mg/m³ (50 μg/m³)
- **Action level**: 0.025 mg/m³ (25 μg/m³)
- FIELD-MIND monitors PM2.5 concentration as a proxy for total dust load.

---

## Ventilation Requirements

1. **Minimum air velocity**: 0.5 m/s at face; 0.3 m/s in general workings
2. **Air quantity per person**: Minimum 3 m³/min per person underground
3. **Diesel equipment**: 0.06 m³/min of fresh air per kW of rated power
4. **After blasting**: Sufficient air to dilute fumes to permissible levels within 30 minutes

---

## FIELD-MIND Gas Sensor Inference Pipeline

The gas sensor AI agent (`GasSensorAgent`) monitors 6 parallel models:
- `mq4_gas_classifier.joblib` — Methane (128-feature SVM+MLP)
- `smoke_fire_alarm_model.joblib` — Smoke/Fire (36 features)
- `gas_hazard_lpg_cng.joblib` — LPG/CNG (2-feature RF)
- `gas_hazard_co_nox_c6h6.joblib` — CO/NOx/Benzene (3 features)
- `gas_hazard_smoke_env.joblib` — Smoke+Env composite (3 features)
- `air_quality_regressor.joblib` — Overall air quality score (7 features)

A hazard ALERT is raised when confidence ≥ 0.5 for two consecutive ticks. The agent learns from new observations using an experience replay buffer (200 samples) and periodic model refit.
