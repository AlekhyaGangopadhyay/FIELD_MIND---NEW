# Implementation Plan — SciSense Novelty Update: Cross-Modal Coherence Residual (CMCR)

This plan details how we will update the `FIELD_MIND` SciSense protocol to meet the project's novelty and patent requirements. We will implement **Cross-Modal Coherence Residual (CMCR)**, transforming the SciSense embeddings from a passive logging role into the primary, load-bearing trigger for the Anomaly-Triggered Reasoning (ATR) system.

---

## User Review Required

> [!IMPORTANT]
> **Shift from Simple Thresholds to Coherence Triggers**:
> *   *Current Flow*: The orchestrator triggers based on simple sensor threshold checks (`if Methane > 1.5%` or `vibration_hazard == 1`). If triggered, it projects the SciSense embeddings but does not use them for decision-making.
> *   *New Flow*: The orchestrator projects all active modalities into the 4096-D SciSense space **on every tick**. It calculates a pairwise correlation matrix and checks the **Coherence Residual** (deviation from the historical normal baseline). If the correlation between sensors shifts anomalously, it triggers the active reasoning state. This cannot be duplicated by simple `if/else` checks.

---

## Proposed Changes

### Component 1 — Coherence Residual Tracker

#### [NEW] [coherence.py](file:///e:/FIELD_MIND/FIELD_MIND---NEW/scisense_protocol/coherence.py)
Create a new module in the SciSense directory to:
1. Track the running Exponential Moving Average (EMA) of the pairwise cosine similarity matrix of gas, environmental, vibration, and ultrasonic embeddings during normal periods.
2. Calculate the **Frobenius norm** of the difference between the current similarity matrix and the baseline similarity matrix:
   $$R_t = \sqrt{\sum_{i,j} (C_{t, i, j} - C_{\text{baseline}, i, j})^2}$$
3. Maintain a sliding history of these residuals to compute a running mean and standard deviation ($\sigma$).
4. Implement a threshold function `is_anomaly(residual)` which returns `True` if the current residual deviates by more than 3 standard deviations ($> \text{Mean} + 3\sigma$).

---

### Component 2 — Orchestrator Integration

#### [MODIFY] [orchestrator.py](file:///e:/FIELD_MIND/FIELD_MIND---NEW/atr_activation/orchestrator.py)
1. Import `SciSenseCoherenceTracker` from `scisense_protocol.coherence`.
2. Initialize the tracker inside `ATROrchestrator.__init__()`.
3. In `process_stream_frame()`, perform the following steps **on every tick**:
   - Run the four SciSense encoders (`gas_encoder`, `env_encoder`, `vib_encoder`, `ultra_encoder`) on the normalized input features to generate four 4096-D embeddings.
   - Pass these embeddings to the coherence tracker to calculate the residual $R_t$.
   - Check if `tracker.is_anomaly(R_t)` is `True`. If `True`, set `is_triggered = True` and initiate transition to `ACTIVE_REASONING` state.
   - If `False` (safe baseline), use the tick to update the tracker's running baseline matrix.
4. Return the calculated $R_t$, baseline matrix stats, and active state to the calling simulation.

---

### Component 3 — UI & Simulation Demonstration

#### [MODIFY] [streaming_safety_simulation.py](file:///e:/FIELD_MIND/FIELD_MIND---NEW/unified_demo/streaming_safety_simulation.py)
* Update the live console output to print the calculated **Coherence Residual ($R_t$)** and the trigger threshold on every simulation step.
* This visualizes the multi-modal math in real-time, showing the transition from `IDLE` to `ACTIVE_REASONING` happening when the correlation residual spikes.

---

## Verification Plan

### Automated Tests
1. Verify the code compiles and runs the end-to-end multi-agent streaming simulation:
   ```powershell
   python e:\FIELD_MIND\FIELD_MIND---NEW\unified_demo\streaming_safety_simulation.py
   ```
2. Verify that the console displays the running **Coherence Residual ($R_t$)** values on each tick.
3. Simulate an anomaly (e.g. inject high vibration and gas concentrations simultaneously) and verify that:
   - The Coherence Residual spikes.
   - The system triggers the state transition `IDLE -> ACTIVE_REASONING`.
   - The LLM boots and runs active reasoning.
