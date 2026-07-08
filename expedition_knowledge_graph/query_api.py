"""
Expedition Knowledge Graph (EKG) - Query API
=============================================
High-level query functions designed for the downstream
Scientific Reasoning Core (Layer 3 / LangGraph agent).
"""

from collections import defaultdict
from graph_store import MineKnowledgeGraph


def get_recent_anomalies(graph: MineKnowledgeGraph, max_count: int = 20) -> list[dict]:
    """
    Return the most recent gas and environmental anomaly events
    across all tunnel segments.
    """
    gas_anomalies = graph.query_by_label("GasAnomaly", filters={"hazard_alert": 1})
    env_anomalies = graph.query_by_label("EnvironmentalReading", filters={"anomaly_flag": 1})

    all_anomalies = []
    for a in gas_anomalies:
        a["anomaly_type"] = "gas"
        all_anomalies.append(a)
    for a in env_anomalies:
        a["anomaly_type"] = "environmental"
        all_anomalies.append(a)

    # Sort by timestamp descending and limit
    all_anomalies.sort(key=lambda x: x.get("timestamp", 0.0), reverse=True)
    return all_anomalies[:max_count]


def get_blast_history(graph: MineKnowledgeGraph, segment_id: str = None) -> list[dict]:
    """
    Return all blast events, optionally filtered to a specific tunnel segment.
    """
    if segment_id:
        blasts = graph.query_by_label("BlastEvent", filters={"segment_id": segment_id})
    else:
        blasts = graph.query_by_label("BlastEvent")

    return sorted(blasts, key=lambda x: x.get("blast_number", 0))


def get_gas_trend(graph: MineKnowledgeGraph, gas_column: str = "MQ4_CH4_ppm",
                  max_points: int = 50) -> list[dict]:
    """
    Return a concentration timeline for a specific gas across all GasAnomaly nodes.
    Each returned dict has 'timestamp', 'value', and 'hazard_alert'.
    """
    all_gas = graph.query_by_label("GasAnomaly")
    trend = []
    for g in all_gas:
        readings = g.get("gas_readings", {})
        if isinstance(readings, str):
            import json
            try:
                readings = json.loads(readings.replace("'", '"'))
            except Exception:
                readings = {}
        value = readings.get(gas_column, None)
        if value is not None:
            trend.append({
                "timestamp": g.get("timestamp", 0.0),
                "timestamp_str": g.get("timestamp_str", ""),
                "value": float(value),
                "hazard_alert": g.get("hazard_alert", 0),
            })

    trend.sort(key=lambda x: x["timestamp"])
    return trend[:max_points]


def get_segment_risk_profile(graph: MineKnowledgeGraph, segment_id: str) -> dict:
    """
    Aggregate hazard counts and severity metrics for a tunnel segment.
    Returns a risk profile dict.
    """
    history = graph.get_segment_history(segment_id)

    # Count hazardous events
    blast_count = len(history.get("BlastEvent", []))
    vib_hazards = [v for v in history.get("VibrationEvent", []) if v.get("hazard_flag") == 1]
    gas_hazards = [g for g in history.get("GasAnomaly", []) if g.get("hazard_alert") == 1]
    env_anomalies = [e for e in history.get("EnvironmentalReading", []) if e.get("anomaly_flag") == 1]

    # Compute max PPV
    ppv_values = [v.get("ppv", 0.0) for v in history.get("VibrationEvent", [])]
    max_ppv = max(ppv_values) if ppv_values else 0.0

    # Compute risk score (simple weighted sum)
    risk_score = (
        len(vib_hazards) * 3.0 +
        len(gas_hazards) * 2.5 +
        len(env_anomalies) * 1.5 +
        blast_count * 0.5
    )

    # Classify risk level
    if risk_score > 50:
        risk_level = "CRITICAL"
    elif risk_score > 20:
        risk_level = "HIGH"
    elif risk_score > 5:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "segment_id": segment_id,
        "segment_name": graph.get_node(segment_id).get("name", "Unknown") if graph.get_node(segment_id) else "Unknown",
        "total_blasts": blast_count,
        "vibration_hazards": len(vib_hazards),
        "gas_hazards": len(gas_hazards),
        "env_anomalies": len(env_anomalies),
        "max_ppv_mm_s": max_ppv,
        "sensors_installed": len(history.get("SensorNode", [])),
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
    }


def correlate_events(graph: MineKnowledgeGraph, event_node_id: str) -> list[dict]:
    """
    Find all causally linked events for a given node.
    Traverses CAUSED_BY, CORRELATED_WITH, and PRECEDED_BY edges.
    """
    causal_rels = ["CAUSED_BY", "CORRELATED_WITH", "PRECEDED_BY"]
    related = []
    for rel in causal_rels:
        neighbors = graph.query_neighbors(event_node_id, rel_type=rel, depth=2)
        for n in neighbors:
            n["causal_relationship"] = rel
            related.append(n)

    return related


def get_equipment_status(graph: MineKnowledgeGraph, segment_id: str = None) -> list[dict]:
    """
    Return all equipment nodes, optionally filtered to a segment.
    """
    if segment_id:
        return graph.query_by_label("Equipment", filters={"location_segment_id": segment_id})
    return graph.query_by_label("Equipment")


def get_sensor_inventory(graph: MineKnowledgeGraph) -> list[dict]:
    """
    Return all registered sensor nodes with their locations.
    """
    return graph.query_by_label("SensorNode")
