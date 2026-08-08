# Implementation Plan — Formal Ablation Study

This plan details the implementation steps to design, write, and execute a **Formal Ablation Head-to-Head Study** comparing the different architectural configurations of the FIELD-MIND safety supervisor. 

This study provides empirical validation of the system's power-awareness, false-alarm mitigation, and reaction speed across different sensor categories.

---

## Trial Configurations

The study compares four ablated configurations in a head-to-head execution sweep over the simulated mine sensor stream:

1.  **Trial 1: Baseline Thresholds System (Ablated: No CMCR, No VoI)**
    *   Uses only static fixed thresholds ($S \ge 0.30 \rightarrow$ Active Reasoning; $S \ge 0.60 \rightarrow$ Emergency).
    *   CMCR triggers are disabled.
2.  **Trial 2: CMCR Trigger Enabled, Fixed Thresholds (Ablated: No VoI)**
    *   Cross-Modal Coherence Residual (CMCR) anomaly triggers ACTIVE_REASONING.
    *   Fixed thresholds determine emergency escalation.
3.  **Trial 3: VoI Gate Enabled, No CMCR (Ablated: No CMCR)**
    *   Decision-theoretic VoI gate manages wakenings and emergencies based on confidence scores.
    *   CMCR is disabled.
4.  **Trial 4: Full FIELD-MIND System (CMCR + VoI Gate Active)**
    *   Pairwise coherence residuals act as primary anomaly triggers.
    *   VoI gate manages the decision-theoretic escalation and wakenings.

---

## Comparison Metrics

For each 100-tick trial, the script will record and report:
*   **Total Ticks**: Total simulation steps run.
*   **Active reasoning activations (LLM Wakes)**: Count of transitions into `ACTIVE_REASONING`.
*   **Emergency evacuations**: Count of transitions into `EMERGENCY`.
*   **Total Energy consumption (Simulated)**:
    *   IDLE state power draw = 5 Watts
    *   ACTIVE_REASONING (LLM loaded) power draw = 15 Watts
    *   EMERGENCY (LLM loaded + alarm) power draw = 18 Watts
    *   $E_{\text{total}} = \sum (\text{power} \times \Delta t)$ (Lower is better).
*   **Trigger latency (ticks)**: Average delay in ticks from the start of an injected hazard to the system triggering `ACTIVE_REASONING` or `EMERGENCY`.
*   **False Alarm Rate (FAR)**: Number of triggers/alerts generated during safe periods (ticks with no injected hazards).

---

## Proposed Changes

### Component 1 — Ablation Study Runner

#### [NEW] [ablation_study.py](file:///e:/FIELD_MIND/FIELD_MIND---NEW/sensor_agents/ablation_study.py)
* Create an execution script that:
  1. Instantiates a test suite using the `DatasetSensorSimulator` and `AgentBus` (modeled after `demo_agents.py`).
  2. Runs 4 consecutive trials (each 100 ticks) with identical random seeds and hazard injection schedules.
  3. Controls Orchestrator flags (`orchestrator.use_voi` and the CMCR trigger flags) for each trial.
  4. Accumulates metrics for state transitions, latency, FAR, and simulated energy consumption.
  5. Formats and prints a comparison table to the console.

---

## Verification Plan

### Automated Tests
* Run the ablation study:
  ```powershell
  python e:\FIELD_MIND\FIELD_MIND---NEW\sensor_agents\ablation_study.py
  ```
* Verify the script prints the head-to-head metrics comparison table without error.
* Confirm that the Full FIELD-MIND system achieves the lowest simulated energy footprint and the lowest False Alarm Rate compared to the baseline thresholds system.
