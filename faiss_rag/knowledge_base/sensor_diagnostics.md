# Sensor Diagnostics — Cross-Sensitivity & Environmental Drift

## Overview

Electrochemical and metal-oxide semiconductor (MOS / MQ-series) gas sensors in underground mining are subject to physical environmental interference, cross-sensitivities, and humidity condensation drift. FIELD-MIND uses multimodal cross-correlation (combining environmental DHT22/BMP280 readings, historical EKG blast records, and LangGraph self-reflection) to distinguish genuine hazard events from sensor artifacts.

---

## MQ-Series Sensor Cross-Sensitivities & Physical Drift

### 1. MQ-7 Carbon Monoxide (CO) — Humidity Condensation Drift
- **Mechanism**: The SnO2 sensing layer on MQ-7 sensors adsorbs water molecules in humid environments (> 80% RH). When relative humidity spikes or during mine water-spraying operations, water condensation lowers surface resistance, causing false elevated CO readings (e.g., 25–45 ppm CO).
- **Physical Verification Rule**:
  - If `MQ7_CO_ppm > 25.0` AND `humidity > 80%` AND no blasting record exists in EKG history AND temperature is stable:
  - **Diagnosis**: *Sensor Moisture Condensation Drift (False Alarm)*.
  - **Action**: Do not halt production. Trigger self-learning reflection loop, apply humidity compensation factor, and inspect sensor heater cycle.

### 2. MQ-4 Methane (CH4) — Temperature & Alcohol Cross-Sensitivity
- **Mechanism**: MQ-4 is highly sensitive to methane (CH4) but exhibits secondary cross-sensitivity to elevated ethanol vapors and extreme ambient temperatures (> 40°C).
- **Physical Verification Rule**:
  - Methane buildup in underground mines is accompanied by ventilation pocket stagnation (low air velocity) or localized heading excavation.
  - Real methane combustion hazards require oxygen presence and trigger multi-gas displacement.

### 3. MQ-135 / MQ-136 Multi-Gas Interference
- MQ-135 responds to NH3, NOx, alcohol, benzene, smoke, and CO2.
- MQ-136 responds to H2S and sulfur compounds.
- In blasting zones, NOx spikes must be cross-referenced with recent detonator blast timestamps in EKG memory within the 30-minute post-blast clearance window.

---

## Sensor Calibration & Feasibility Verification Matrix

| Observed Telemetry Pattern | Preliminary Model Output | Multimodal Ground Truth | Feasibility Classification | Corrective Protocol |
|----------------------------|--------------------------|-------------------------|----------------------------|---------------------|
| CO = 32 ppm, RH = 88%, Temp = 22°C, No Blasts | CO Hazard Alert | Water spraying operation active | **PHYSICAL DISCREPANCY / DRIFT** | Update Experience Replay with True Label = 0; apply moisture offset. |
| CO = 55 ppm, NOx = 4.2 ppm, Blast recorded 12 min ago | Gas Alert | Post-blast fume dispersion | **FEASIBLE COMBUSTION / BLAST** | Enforce 30-min post-blast re-entry wait; maintain exhaust ventilation. |
| PPV = 14.2 mm/s, No Detonator recorded in EKG | Vibration Hazard Alert | Heavy continuous miner tramming | **PHYSICAL DISCREPANCY / SEISMIC** | Flag localized machinery noise; inspect geophone coupling. |
| Ultrasonic Min Dist = 0.15 m, Velocity = 0 m/s | Collision Alarm | Stationary drill rod obstruction | **FEASIBLE OBSTACLE** | Command robotic platform backup manoeuvre; update SLAM map. |

---

## Online Retraining & Experience Replay Buffer

When a physical discrepancy is verified:
1. **LangGraph Reflection Engine** generates a domain-specific corrective rule.
2. The rule is embedded into **FAISS RAG** memory (`[SELF-LEARNED RULE]`).
3. The event is linked to the `TunnelSegment` node in the **Expedition Knowledge Graph**.
4. The verified sample `(features, true_label)` is appended to the 200-sample **Experience Replay Buffer** of the corresponding Sensor Agent for atomic model refitting.
