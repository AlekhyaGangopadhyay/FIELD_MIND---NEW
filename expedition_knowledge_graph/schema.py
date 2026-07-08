"""
Expedition Knowledge Graph (EKG) - Property Graph Schema
=========================================================
Defines the typed node and edge labels for the FIELD-MIND mine knowledge graph.
Each node type is a dataclass with mandatory and optional properties.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import time
import uuid


def _uid() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Node schemas
# ---------------------------------------------------------------------------

@dataclass
class TunnelSegment:
    """A physical section of the underground mine."""
    label: str = "TunnelSegment"
    segment_id: str = field(default_factory=_uid)
    name: str = ""
    depth_m: float = 0.0
    length_m: float = 0.0
    gx: float = 0.0          # easting coordinate
    gy: float = 0.0          # northing coordinate
    gelev: float = 0.0       # elevation
    status: str = "active"   # active | sealed | collapsed

    def to_dict(self):
        return asdict(self)


@dataclass
class SensorNode:
    """A sensor hardware installation."""
    label: str = "SensorNode"
    sensor_id: str = field(default_factory=_uid)
    sensor_type: str = ""    # gas | env | vibration | ultrasonic
    location_segment_id: str = ""
    install_date: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class BlastEvent:
    """A drill-and-blast operation record."""
    label: str = "BlastEvent"
    event_id: str = field(default_factory=_uid)
    blast_number: int = 0
    timestamp: float = 0.0
    total_charge: float = 0.0
    max_charge: float = 0.0
    num_holes: int = 0
    detonator_code: int = 0
    segment_id: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class VibrationEvent:
    """An individual vibration measurement linked to a blast."""
    label: str = "VibrationEvent"
    event_id: str = field(default_factory=_uid)
    timestamp: float = 0.0
    ppv: float = 0.0
    scaled_distance_usbm: float = 0.0
    scaled_distance_langefors: float = 0.0
    elevation_diff: float = 0.0
    hazard_flag: int = 0
    blast_id: str = ""
    segment_id: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class GasAnomaly:
    """A gas hazard event detected by the Tier 1 monitors."""
    label: str = "GasAnomaly"
    event_id: str = field(default_factory=_uid)
    timestamp: float = 0.0
    timestamp_str: str = ""
    gas_readings: dict = field(default_factory=dict)  # {gas_name: ppm}
    hazard_alert: int = 0
    blast_correlated: int = 0
    segment_id: str = ""

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class EnvironmentalReading:
    """A temperature / humidity snapshot from the IoT telemetry."""
    label: str = "EnvironmentalReading"
    reading_id: str = field(default_factory=_uid)
    timestamp: float = 0.0
    temp: float = 0.0
    humidity: float = 0.0
    anomaly_flag: int = 0
    segment_id: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class NavigationEvent:
    """An ultrasonic sensor navigation decision record."""
    label: str = "NavigationEvent"
    event_id: str = field(default_factory=_uid)
    timestamp: float = 0.0
    command: str = ""           # Move-Forward | Sharp-Right-Turn | Slight-Right-Turn | Slight-Left-Turn
    sensor_readings: dict = field(default_factory=dict)  # {US1: val, US2: val, ...}
    min_distance: float = 0.0   # closest obstacle distance across all sensors
    collision_risk: int = 0     # 1 if any sensor < 0.5m threshold
    segment_id: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Equipment:
    """A piece of tracked underground machinery."""
    label: str = "Equipment"
    equipment_id: str = field(default_factory=_uid)
    equipment_type: str = ""   # drill_rig | loader | conveyor | ventilation_fan
    name: str = ""
    location_segment_id: str = ""
    status: str = "operational"  # operational | maintenance | decommissioned

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Edge / relationship type constants
# ---------------------------------------------------------------------------

LOCATED_IN      = "LOCATED_IN"       # SensorNode / Equipment → TunnelSegment
RECORDED_BY     = "RECORDED_BY"      # Event → SensorNode
OCCURRED_IN     = "OCCURRED_IN"      # Event → TunnelSegment
CAUSED_BY       = "CAUSED_BY"        # VibrationEvent → BlastEvent
CORRELATED_WITH = "CORRELATED_WITH"  # GasAnomaly → BlastEvent
OPERATES_IN     = "OPERATES_IN"      # Equipment → TunnelSegment
PRECEDED_BY     = "PRECEDED_BY"      # Event → Event (temporal chain)
