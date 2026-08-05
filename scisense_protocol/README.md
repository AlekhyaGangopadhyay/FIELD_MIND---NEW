# SciSense Protocol

SciSense is FIELD-MIND's cross-modal alignment and coherence layer. It converts heterogeneous gas, environmental, vibration, and ultrasonic readings into modality-specific vectors in a shared 4,096-dimensional space. ATR uses those projections to monitor changes in relationships between sensor domains and to trigger higher-level reasoning when the relationships become anomalous.

## Current data flow

```text
Raw sensor streams or synchronized frames
              |
              v
TemporalAligner (optional 1-second windows, averaging, forward-fill)
              |
              v
Bounded modality normalization (tanh(value / practical scale))
              |
              v
Gas / Environment / Vibration / Ultrasonic encoder
              |
              v
L2-normalized 4,096-dimensional embeddings
              |
              v
Pairwise cosine-similarity matrix
              |
              v
CMCR residual against an EMA normal baseline
              |
              v
ATR trigger and state transition
```

The temporal aligner is used by the standalone demo. The ATR orchestrator receives one synchronized frame at a time, keeps a bounded history, projects all four modalities, and sends the embeddings directly to the coherence tracker.

## Modality encoders

| Encoder | Default input | Hidden width | Output |
| --- | ---: | ---: | ---: |
| `GasEncoder` | 6 values: CH4, CO, LPG, smoke, NOx, CO2 | 128 | 4,096-D unit vector |
| `EnvironmentalEncoder` | 4 values: temperature, humidity, pressure, occupancy | 128 | 4,096-D unit vector |
| `VibrationEncoder` | 15 blast/structural features | 256 | 4,096-D unit vector |
| `UltrasonicEncoder` | 24 distance sensors | 256 | 4,096-D unit vector |

Each encoder is a small feed-forward PyTorch network followed by `SciSenseProjection`. The projection applies a linear layer, layer normalization, and L2 normalization. The resulting vectors have unit norm, so their dot product is cosine similarity.

The encoders currently provide the projection architecture and are instantiated with their default PyTorch initialization. No trained SciSense encoder checkpoint is loaded by `ATROrchestrator`; therefore the current embeddings are suitable for pipeline integration and coherence mechanics, but they are not yet a trained semantic cross-modal representation.

## CMCR coherence tracking

`SciSenseCoherenceTracker` maintains a fixed modality order:

```text
(gas, env, vibration, ultrasonic)
```

For each frame it:

1. Builds the pairwise cosine-similarity matrix from the available embeddings.
2. Calculates the Cross-Modal Coherence Residual (CMCR), `R_t`, as the Frobenius distance from the EMA baseline matrix.
3. Estimates an anomaly threshold as `mean(residuals) + 3 * std(residuals)` after at least five prior observations.
4. Updates the EMA baseline only for normal frames when the caller permits baseline updates.

The first frame initializes the baseline. During warm-up, anomaly decisions are disabled until `min_history` observations are available. In ATR, frames already flagged by Tier 1 hazard monitors are not allowed to update the normal baseline, preventing known hazard states from contaminating the reference state.

`normalize_modal_vector` converts heterogeneous numeric channels into finite, bounded values using practical full-scale values and a clipped `tanh` transform. NaN and infinite inputs become zero. This prevents high-ppm gas readings or large spatial coordinates from dominating the projection input.

## ATR integration

[`atr_activation/orchestrator.py`](../atr_activation/orchestrator.py) combines SciSense with the existing Tier 1 monitors:

1. Tier 1 evaluates gas, environmental, vibration, and ultrasonic safety conditions.
2. The current frame is normalized and projected into four SciSense embeddings.
3. CMCR is evaluated against the learned normal relationship between modalities.
4. A Tier 1 hazard or CMCR anomaly sets `triggered=True` and moves the device from `IDLE` to `ACTIVE_REASONING`.
5. Clean frames can return the device to `IDLE`; hazard frames do not update the coherence baseline.

The returned dictionary includes the Tier 1 evaluations plus `coherence_residual`, `coherence_threshold`, `coherence_anomaly`, baseline status, the similarity matrices, and `aligned_embeddings` for downstream inspection.

## Core files

- [`encoders.py`](encoders.py): modality encoders and the shared 4,096-D projection.
- [`alignment.py`](alignment.py): timestamp-based windowing, within-window averaging, and forward-fill alignment.
- [`coherence.py`](coherence.py): input normalization, similarity matrices, EMA baseline, residual history, and the 3-sigma gate.
- [`demo_alignment.py`](demo_alignment.py): standalone heterogeneous-stream alignment and projection demo.
- [`atr_activation/orchestrator.py`](../atr_activation/orchestrator.py): production-style Tier 1 plus SciSense orchestration.
- [`unified_demo/streaming_safety_simulation.py`](../unified_demo/streaming_safety_simulation.py): multi-node streaming simulation with per-node CMCR state.

## Run the standalone demo

From the `FIELD_MIND---NEW` repository root:

```bash
python scisense_protocol/demo_alignment.py
```

The demo creates ten seconds of synthetic streams, aligns them into one-second epochs, projects available modalities, prints embedding shapes and norms, and reports gas-to-ultrasonic cosine similarity.

## Minimal coherence example

```python
from scisense_protocol.coherence import (
    SciSenseCoherenceTracker,
    normalize_modal_vector,
)

tracker = SciSenseCoherenceTracker()

gas = normalize_modal_vector([1200, 8, 20, 2, 1, 450], [10000, 50, 1000, 100, 5, 5000])
result = tracker.update({"gas": gas})

print(result["residual"])
print(result["is_anomaly"])
```

When a modality is unavailable, its embedding may be omitted or set to `None`; the tracker leaves the corresponding similarities at zero. All callers should keep modality names and ordering stable across frames.

## Dependencies

The protocol uses Python, PyTorch, NumPy, and pandas. The standalone demo only requires the SciSense protocol modules and those Python dependencies; the ATR orchestrator additionally loads FIELD-MIND's Tier 1 sensor models.
