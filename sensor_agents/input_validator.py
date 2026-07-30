from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

@dataclass
class SensorHealthReport:
    is_valid: bool = True
    clamped_value: float = 0.0
    is_stuck: bool = False
    is_dead: bool = False
    is_spike: bool = False
    status: str = "OK"  # "OK", "STUCK", "DEAD", "SPIKE", "OUT_OF_RANGE"

# Validation ranges and dead thresholds:
VALID_RANGES = {
    "MQ2_LPG_ppm": {"min": 0.0, "max": 10000.0, "dead_thresh": 1.0},
    "MQ2_Smoke_ppm": {"min": 0.0, "max": 10000.0, "dead_thresh": 1.0},
    "MQ3_Alcohol_ppm": {"min": 0.0, "max": 500.0, "dead_thresh": 0.1},
    "MQ3_Benzene_ppm": {"min": 0.0, "max": 500.0, "dead_thresh": 0.1},
    "MQ4_CH4_ppm": {"min": 0.0, "max": 50000.0, "dead_thresh": 5.0},
    "MQ7_CO_ppm": {"min": 0.0, "max": 5000.0, "dead_thresh": 0.5},
    "MQ135_NOx_ppm": {"min": 0.0, "max": 10.0, "dead_thresh": 0.001},
    "MQ135_NH3_ppm": {"min": 0.0, "max": 500.0, "dead_thresh": None},
    "MQ136_H2S_ppm": {"min": 0.0, "max": 500.0, "dead_thresh": None},
    "MG811_CO2_ppm": {"min": 300.0, "max": 50000.0, "dead_thresh": 350.0},
    "PM25_Dust_ugm3": {"min": 0.0, "max": 5000.0, "dead_thresh": None},
    "Temp_C": {"min": -40.0, "max": 80.0, "dead_thresh": None},
    "Humidity_pct": {"min": 0.0, "max": 100.0, "dead_thresh": None},
}

def validate_range(sensor_name: str, value: float) -> Tuple[bool, float]:
    """Check range and clamp value."""
    if sensor_name not in VALID_RANGES:
        return True, value
    cfg = VALID_RANGES[sensor_name]
    min_val = cfg["min"]
    max_val = cfg["max"]
    if np.isnan(value):
        return False, min_val
    if value < min_val:
        return False, min_val
    if value > max_val:
        return False, max_val
    return True, value

def detect_stuck_sensor(history: List[float], window: int = 10) -> bool:
    """True if the sensor value has been identical for the last `window` ticks."""
    if len(history) < window:
        return False
    recent = history[-window:]
    # Check if all elements are exactly equal (within small epsilon)
    return all(abs(x - recent[0]) < 1e-7 for x in recent)

def detect_dead_sensor(sensor_name: str, value: float) -> bool:
    """True if the value is at or below the dead threshold for analog sensors."""
    if sensor_name not in VALID_RANGES:
        return False
    dead_thresh = VALID_RANGES[sensor_name].get("dead_thresh")
    if dead_thresh is None:
        return False
    if np.isnan(value):
        return False
    return value <= dead_thresh

def detect_spike(history: List[float], value: float, sigma_threshold: float = 5.0) -> bool:
    """Detect sudden spikes using rolling standard deviation."""
    if len(history) < 5:
        return False
    valid_hist = [x for x in history if not np.isnan(x)]
    if len(valid_hist) < 5:
        return False
    mean = np.mean(valid_hist)
    std = np.std(valid_hist)
    if std < 1e-4:
        return False
    return abs(value - mean) > sigma_threshold * std

def validate_sensor(sensor_name: str, value: float, history: List[float]) -> SensorHealthReport:
    """Complete sensor health report check."""
    report = SensorHealthReport()
    
    # 1. Range check and clamping
    in_range, clamped = validate_range(sensor_name, value)
    report.clamped_value = clamped

    # 2. Dead sensor check
    if detect_dead_sensor(sensor_name, value):
        report.is_dead = True
        report.status = "DEAD"
        report.is_valid = False
        return report
        
    # 3. Stuck sensor check
    if detect_stuck_sensor(history, window=10):
        report.is_stuck = True
        report.status = "STUCK"
        report.is_valid = False
        return report

    # 4. Spike anomaly check
    if detect_spike(history, value, sigma_threshold=5.0):
        report.is_spike = True
        report.status = "SPIKE"
        report.is_valid = False
        return report

    # 5. Out of range check
    if not in_range:
        report.is_valid = False
        report.status = "OUT_OF_RANGE"
        
    return report

