# 🚀 Implementation Plan: Jetson Headless Configuration & Full Model Retraining

**Project**: FIELD-MIND — Offline Multimodal Agentic AI for Underground Mining  
**Target Platform**: NVIDIA Jetson Orin Nano (8GB Unified Memory, JetPack 6.x / Ubuntu 22.04)  
**Objective**: Transition the Jetson Orin Nano to a 100% headless daemon, retrain all ML/DL models in place to resolve `joblib` version warnings, and optimize memory headroom for edge deployment.

---

## Executive Summary & Workflow Overview

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      FIELD-MIND IMPLEMENTATION PIPELINE                      │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
    ├── Phase 1: Environment Model Retraining (In-Place Overwrite)
    │   ├── Gas DL Tournament Models        : retrain_dl_models.py
    │   ├── Gas Severity Classifier Heads   : retrain_severity_models.py
    │   ├── Gas Presence & Hazard Detectors : train_gas_detector.py & train_multi_gas_detector.py
    │   ├── Environmental Telemetry Models : temperature_humidity/src/train.py
    │   └── Ultrasonic Sonar Models         : ultrasonic_sensors/train_models.py
    │
    ├── Phase 2: Artifact Integrity & Cleanup
    │   ├── Run System Check                : python check_artifacts.py
    │   └── Quarantine Redundant Models     : Archive *_over_tlv_dl_best.joblib files
    │
    ├── Phase 3: Jetson Headless Deployment
    │   ├── Set Multi-User System Target    : sudo systemctl set-default multi-user.target
    │   ├── Lock Clocks & Max Power (15W)   : sudo nvpmodel -m 0 && sudo jetson_clocks
    │   └── Enable Boot Service             : sudo systemctl enable fieldmind.service
    │
    └── Phase 4: End-to-End Edge Verification
        ├── Anomaly-Triggered Reasoning     : python atr_activation/demo_atr.py
        ├── Multi-Agent Telemetry           : python sensor_agents/demo_agents.py --ticks 100
        └── Safety Assistant Hub            : python unified_demo/interactive_safety_hub.py
```

---

## Phase 1: Environment-Aligned Model Retraining (In-Place Overwrite)

To resolve all `WARN | Found but failed joblib.load` warnings, all production models will be retrained directly in the target Python environment. Every command below **overwrites existing `.joblib` files in place**.

### Step 1.1: Retrain Gas Deep Learning Tournament & Severity Models
```bash
# 1. Retrain PyTorch Deep Learning Tournament Models (Overwrites *_dl_best.joblib)
python gas_sensors/retrain_dl_models.py

# 2. Retrain 3-Class Gas Severity Heads (Overwrites severity_*.joblib)
python gas_sensors/retrain_severity_models.py
```

### Step 1.2: Retrain Gas Baseline & Multi-Gas Hazard Detectors
```bash
# 3. Retrain Random Forest Gas Detector & Baseline Isolation Forest
python gas_sensors/train_gas_detector.py

# 4. Retrain Production 8-Channel PyTorch DeepHazardNet (Overwrites multi_gas_detector.joblib)
python gas_sensors/train_multi_gas_detector.py --epochs 40
```

### Step 1.3: Retrain Ultrasonic & Environmental Domain Models
```bash
# 5. Retrain Ultrasonic Sonar Ring Models (Overwrites best_ultrasonic_*.joblib)
python ultrasonic_sensors/train_models.py

# 6. Retrain Environmental IoT & Occupancy Models (Overwrites isolation_forest_iot.joblib & random_forest.joblib)
python temperature_humidity/src/train.py
```

---

## Phase 2: System Artifact Integrity Verification & Cleanup

### Step 2.1: Run Artifact Verification Checker
Run [check_artifacts.py](file:///e:/FIELD_MIND/FIELD_MIND---NEW/check_artifacts.py) to verify that all 32 Python modules and 16 core models load cleanly:
```bash
python check_artifacts.py
```
*Expected Output*: `SUMMARY: Python Modules: 32/32 | Core Models: 16/16` with zero `WARN` messages.

### Step 2.2: Quarantine Redundant Experimental DL Files
Archive non-essential tournament evaluation files to free up disk space:
```bash
# Create experimental folder
mkdir -p gas_sensors/models/experimental/

# Move redundant over-TLV & warmup files
mv gas_sensors/models/*_over_tlv_dl_best.joblib gas_sensors/models/experimental/ 2>/dev/null || true
mv gas_sensors/models/part1_is_warmup_dl_best.joblib gas_sensors/models/experimental/ 2>/dev/null || true
```

---

## Phase 3: Jetson Orin Nano Headless Deployment Setup

### Step 3.1: Disable Desktop GUI & Save ~1.5 GB Unified RAM
Execute on Jetson terminal via SSH:
```bash
# 1. Set multi-user command-line default boot target
sudo systemctl set-default multi-user.target

# 2. Stop desktop display manager immediately
sudo systemctl stop gdm3
```

### Step 3.2: Maximize Compute & Clock Frequencies
```bash
# Set 15W Max Performance Mode
sudo nvpmodel -m 0

# Lock CPU, GPU, and EMC memory bus clocks to maximum
sudo jetson_clocks
```

### Step 3.3: Configure Systemd Daemon (`fieldmind.service`)
Create `/etc/systemd/system/fieldmind.service`:
```ini
[Unit]
Description=FIELD-MIND Offline Edge AI Daemon
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/FIELD_MIND/FIELD_MIND---NEW
ExecStart=/home/jetson/FIELD_MIND/FIELD_MIND---NEW/venv/bin/python unified_demo/interactive_safety_hub.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fieldmind.service
sudo systemctl start fieldmind.service
```

---

## Phase 4: End-to-End System & ATR Verification

Run the verification suite to ensure all multi-agent pipelines and reasoning core loops function without errors:

```bash
# 1. Hardware & System Preflight
python jetson_preflight.py --allow-missing

# 2. Anomaly-Triggered Reasoning (ATR) Pipeline Test
python atr_activation/demo_atr.py

# 3. Streaming Multi-Agent Telemetry Test (50 Ticks)
python sensor_agents/demo_agents.py --ticks 50

# 4. Autonomous Self-Learning & Reasoning Loop
python reasoning_core/demo_self_learning.py
```

---

## Verification & Expected Milestones

| Milestone | Check Command | Success Criteria |
| :--- | :--- | :--- |
| **Model Re-serialization** | `python check_artifacts.py` | 16/16 models show `PASS` with **0 warnings**. |
| **Headless RAM Optimization** | `free -h` | **> 1.0 GB RAM headroom** available before LLM load. |
| **ATR Triggering** | `python atr_activation/demo_atr.py` | Tier-1 monitors detect simulated gas spikes cleanly. |
| **Multi-Agent Execution** | `python sensor_agents/demo_agents.py` | 5 sensor agents process streaming telemetry at 10+ Hz. |
