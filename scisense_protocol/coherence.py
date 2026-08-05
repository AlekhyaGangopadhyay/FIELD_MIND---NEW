"""Cross-modal coherence residual tracking for the SciSense protocol.

The tracker turns the four modality embeddings into a small, interpretable
signal that can be used by ATR without retaining the full embedding history.
During normal operation the pairwise cosine-similarity matrix is updated with
an exponential moving average.  A frame is anomalous when its Frobenius
distance from that baseline is greater than the rolling ``mean + 3 sigma``
threshold.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


def normalize_modal_vector(values: object, scales: object) -> np.ndarray:
    """Convert heterogeneous sensor values to a finite, bounded input vector.

    ``scales`` represents a practical full-scale value for each native sensor
    channel.  The bounded tanh transform prevents high-ppm gas or coordinate
    values from dominating the randomly initialized projection layers while
    preserving sign for channels such as spatial coordinates.
    """

    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    scale_array = np.asarray(scales, dtype=np.float32).reshape(-1)
    if vector.size != scale_array.size:
        raise ValueError("values and scales must have the same length")
    scale_array = np.where(np.abs(scale_array) < 1e-8, 1.0, np.abs(scale_array))
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return np.tanh(np.clip(vector / scale_array, -5.0, 5.0)).astype(np.float32)


class SciSenseCoherenceTracker:
    """Track residual changes in cross-modal embedding coherence.

    Parameters
    ----------
    modalities:
        Stable ordering used for the similarity matrix.  Keeping this fixed
        makes the residual comparable from frame to frame.
    ema_alpha:
        Weight given to a new normal similarity matrix.
    history_size:
        Maximum number of residuals retained for threshold estimation.
    min_history:
        Number of prior observations required before anomaly decisions are
        enabled.  This prevents the first few startup frames from firing.
    sigma_multiplier:
        Number of standard deviations used by the anomaly threshold.
    """

    def __init__(
        self,
        modalities: Sequence[str] = ("gas", "env", "vibration", "ultrasonic"),
        ema_alpha: float = 0.1,
        history_size: int = 50,
        min_history: int = 5,
        sigma_multiplier: float = 3.0,
    ) -> None:
        if not modalities:
            raise ValueError("at least one modality is required")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if history_size < 1 or min_history < 1:
            raise ValueError("history_size and min_history must be positive")
        if sigma_multiplier < 0.0:
            raise ValueError("sigma_multiplier must be non-negative")

        self.modalities = tuple(modalities)
        self.ema_alpha = float(ema_alpha)
        self.sigma_multiplier = float(sigma_multiplier)
        self.min_history = int(min_history)
        self._baseline: Optional[np.ndarray] = None
        self._baseline_updates = 0
        self._residual_history: deque[float] = deque(maxlen=int(history_size))

    @property
    def baseline_matrix(self) -> Optional[np.ndarray]:
        """Return a copy of the current EMA baseline, if initialized."""

        return None if self._baseline is None else self._baseline.copy()

    @property
    def residual_history(self) -> tuple[float, ...]:
        return tuple(self._residual_history)

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None

    @property
    def threshold(self) -> float:
        """Current threshold based on the residuals seen so far."""

        if len(self._residual_history) < self.min_history:
            return float("inf")
        values = np.asarray(self._residual_history, dtype=np.float64)
        return float(values.mean() + self.sigma_multiplier * values.std())

    @property
    def residual_mean(self) -> float:
        if not self._residual_history:
            return 0.0
        return float(np.mean(self._residual_history))

    @property
    def residual_std(self) -> float:
        if not self._residual_history:
            return 0.0
        return float(np.std(self._residual_history))

    def _vectorize(self, value: object) -> Optional[np.ndarray]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0.0 else None

    def similarity_matrix(self, embeddings: Mapping[str, object]) -> np.ndarray:
        """Build the fixed-order pairwise cosine-similarity matrix."""

        vectors = {name: self._vectorize(embeddings.get(name)) for name in self.modalities}
        size = len(self.modalities)
        matrix = np.zeros((size, size), dtype=np.float64)
        for i, left_name in enumerate(self.modalities):
            left = vectors[left_name]
            if left is None:
                continue
            matrix[i, i] = 1.0
            for j in range(i + 1, size):
                right = vectors[self.modalities[j]]
                if right is None:
                    continue
                # Encoders normally share a 4096-D space. Padding also makes
                # the tracker safe for small unit-test vectors.
                width = max(left.size, right.size)
                left_pad = np.pad(left, (0, width - left.size))
                right_pad = np.pad(right, (0, width - right.size))
                similarity = float(np.clip(np.dot(left_pad, right_pad), -1.0, 1.0))
                matrix[i, j] = matrix[j, i] = similarity
        return matrix

    def calculate_residual(self, embeddings: Mapping[str, object]) -> float:
        """Calculate the current residual without changing tracker state."""

        current = self.similarity_matrix(embeddings)
        if self._baseline is None:
            return 0.0
        return float(np.linalg.norm(current - self._baseline, ord="fro"))

    # Backwards-friendly alias for callers that use the shorter name.
    residual = calculate_residual

    def is_anomaly(self, residual: float) -> bool:
        """Return whether a residual exceeds the learned 3-sigma threshold."""

        if self._baseline is None or len(self._residual_history) < self.min_history:
            return False
        value = float(residual)
        return bool(np.isfinite(value) and value > self.threshold)

    def update_baseline(self, similarity_matrix: np.ndarray) -> None:
        """Update the EMA baseline with a matrix from a normal frame."""

        matrix = np.asarray(similarity_matrix, dtype=np.float64)
        expected = (len(self.modalities), len(self.modalities))
        if matrix.shape != expected:
            raise ValueError(f"similarity matrix must have shape {expected}")
        if self._baseline is None:
            self._baseline = matrix.copy()
        else:
            self._baseline = (
                (1.0 - self.ema_alpha) * self._baseline
                + self.ema_alpha * matrix
            )
        self._baseline_updates += 1

    def update(
        self,
        embeddings: Mapping[str, object],
        update_baseline: bool = True,
    ) -> Dict[str, object]:
        """Observe one frame and return residual/threshold diagnostics.

        The threshold is captured *before* adding the current residual to the
        history, so a spike is judged against the preceding normal window.
        An anomalous frame never contaminates the EMA baseline.
        """

        matrix = self.similarity_matrix(embeddings)
        residual = self.calculate_residual(embeddings)
        threshold_before = self.threshold
        anomaly = self.is_anomaly(residual)
        self._residual_history.append(float(residual))

        baseline_updated = False
        if self._baseline is None:
            self.update_baseline(matrix)
            baseline_updated = True
        elif update_baseline and not anomaly:
            self.update_baseline(matrix)
            baseline_updated = True

        return {
            "residual": float(residual),
            "threshold": float(threshold_before),
            "is_anomaly": bool(anomaly),
            "similarity_matrix": matrix,
            "baseline_matrix": self.baseline_matrix,
            "baseline_updated": baseline_updated,
            "baseline_ready": self.baseline_ready,
            "baseline_updates": self._baseline_updates,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "sample_count": len(self._residual_history),
        }

    # ``observe`` reads naturally at call sites and is useful for integrations.
    observe = update
