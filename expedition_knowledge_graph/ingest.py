"""
Expedition Knowledge Graph (EKG) - Data Ingestion Pipelines
=============================================================
Populates the MineKnowledgeGraph from the workspace CSV datasets:
  1. Blast events + vibration readings   (vibration/data/vibration_features.csv)
  2. Gas anomaly events                  (gas_sensors/data/FIELDMIND_physics_dataset.csv)
  3. Environmental telemetry readings    (temperature_humidity/data/data_clean/iot_telemetry_clean.csv)
  4. Sensor node placements              (synthetic, based on modality types)
  5. Equipment entries                   (synthetic mine equipment)
"""

import os
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict

from schema import (
    TunnelSegment, SensorNode, BlastEvent, VibrationEvent,
    GasAnomaly, EnvironmentalReading, Equipment,
    LOCATED_IN, RECORDED_BY, OCCURRED_IN, CAUSED_BY,
    CORRELATED_WITH, OPERATES_IN,
)
from graph_store import MineKnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coord_to_segment_id(gx: float, gy: float, gelev: float, resolution: float = 500.0) -> str:
    """
    Cluster spatial coordinates into discrete tunnel segments
    by rounding to a grid. Returns a deterministic segment_id.
    """
    qx = int(round(gx / resolution))
    qy = int(round(gy / resolution))
    qe = int(round(gelev / 50.0))
    raw = f"seg_{qx}_{qy}_{qe}"
    return raw


def _ensure_segment(graph: MineKnowledgeGraph, seg_id: str,
                    gx: float = 0.0, gy: float = 0.0, gelev: float = 0.0):
    """Create the TunnelSegment node if it doesn't already exist."""
    if graph.get_node(seg_id) is None:
        seg = TunnelSegment(
            segment_id=seg_id,
            name=f"Tunnel Section {seg_id.replace('seg_', '').replace('_', '-')}",
            depth_m=abs(gelev),
            gx=gx, gy=gy, gelev=gelev,
        )
        graph.add_node(seg_id, "TunnelSegment", seg.to_dict())


# ---------------------------------------------------------------------------
# 1. Blast events + Vibration readings
# ---------------------------------------------------------------------------

def ingest_blast_events(graph: MineKnowledgeGraph, csv_path: str, max_blasts: int = 200):
    """
    Reads vibration_features.csv and creates:
      - BlastEvent nodes (one per unique blast number)
      - VibrationEvent nodes (one per row, sampled)
      - TunnelSegment nodes (auto-clustered from gx/gy/gelev)
      - CAUSED_BY edges (VibrationEvent → BlastEvent)
      - OCCURRED_IN edges (BlastEvent / VibrationEvent → TunnelSegment)
    """
    if not os.path.exists(csv_path):
        print(f"  [SKIP] Blast CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"  Ingesting blast data: {len(df)} rows from {os.path.basename(csv_path)}")

    # Group by blast number
    blast_groups = df.groupby("blast")
    blast_ids_map = {}   # blast_number -> node_id
    blast_count = 0
    vib_count = 0

    for blast_num, group in blast_groups:
        if blast_count >= max_blasts:
            break

        first_row = group.iloc[0]
        seg_id = _coord_to_segment_id(first_row["gx"], first_row["gy"], first_row["gelev"])
        _ensure_segment(graph, seg_id, first_row["gx"], first_row["gy"], first_row["gelev"])

        # Create BlastEvent node
        blast_node = BlastEvent(
            blast_number=int(blast_num),
            timestamp=float(blast_count),  # synthetic temporal ordering
            total_charge=float(first_row.get("total_charge", 0)),
            max_charge=float(first_row.get("max_charge", 0)),
            num_holes=int(first_row.get("num_holes", 0)),
            detonator_code=int(first_row.get("detonator_code", 0)),
            segment_id=seg_id,
        )
        blast_nid = f"blast_{blast_num}"
        blast_ids_map[blast_num] = blast_nid
        graph.add_node(blast_nid, "BlastEvent", blast_node.to_dict())
        graph.add_edge(blast_nid, seg_id, OCCURRED_IN)
        blast_count += 1

        # Sample up to 5 vibration rows per blast
        sample = group.head(5)
        for _, row in sample.iterrows():
            vib = VibrationEvent(
                timestamp=float(blast_count),
                ppv=float(row.get("ppv", 0)),
                scaled_distance_usbm=float(row.get("scaled_distance_usbm", 0)),
                scaled_distance_langefors=float(row.get("scaled_distance_langefors", 0)),
                elevation_diff=float(row.get("elevation_diff", 0)),
                hazard_flag=int(row.get("vibration_hazard", 0)),
                blast_id=blast_nid,
                segment_id=seg_id,
            )
            vib_nid = f"vib_{vib.event_id}"
            graph.add_node(vib_nid, "VibrationEvent", vib.to_dict())
            graph.add_edge(vib_nid, blast_nid, CAUSED_BY)
            graph.add_edge(vib_nid, seg_id, OCCURRED_IN)
            vib_count += 1

    print(f"    Created {blast_count} BlastEvent nodes, {vib_count} VibrationEvent nodes")


# ---------------------------------------------------------------------------
# 2. Gas anomaly events
# ---------------------------------------------------------------------------

def ingest_gas_anomalies(graph: MineKnowledgeGraph, csv_path: str, max_events: int = 500):
    """
    Reads FIELDMIND_physics_dataset.csv and creates:
      - GasAnomaly nodes (one per hazard row, sampled)
      - OCCURRED_IN edges to a default tunnel segment
      - CORRELATED_WITH edges to nearest blast (if Blast_Event == 1)
    """
    if not os.path.exists(csv_path):
        print(f"  [SKIP] Gas CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"  Ingesting gas data: {len(df)} rows from {os.path.basename(csv_path)}")

    # Default tunnel segment for gas sensors
    default_seg = "seg_gas_main"
    _ensure_segment(graph, default_seg, gx=324000.0, gy=7412000.0, gelev=-200.0)

    # Focus on hazard rows (Hazard_Alert == 1) and sample
    hazard_df = df[df["Hazard_Alert"] == 1]
    if len(hazard_df) > max_events:
        hazard_df = hazard_df.sample(n=max_events, random_state=42)

    # Also add some normal readings for context (every 1000th row)
    normal_sample = df[df["Hazard_Alert"] == 0].iloc[::1000]

    gas_count = 0
    for _, row in pd.concat([hazard_df, normal_sample]).iterrows():
        gas_readings = {}
        for col in ["MQ2_LPG_ppm", "MQ4_CH4_ppm", "MQ7_CO_ppm", "MQ135_NOx_ppm",
                     "MQ2_Smoke_ppm", "MQ3_Benzene_ppm", "MQ136_H2S_ppm",
                     "MG811_CO2_ppm", "PM25_Dust_ugm3"]:
            if col in row.index:
                gas_readings[col] = float(row[col])

        anomaly = GasAnomaly(
            timestamp=float(gas_count),
            timestamp_str=str(row.get("Timestamp", "")),
            gas_readings=gas_readings,
            hazard_alert=int(row.get("Hazard_Alert", 0)),
            blast_correlated=int(row.get("Blast_Event", 0)),
            segment_id=default_seg,
        )
        gas_nid = f"gas_{anomaly.event_id}"
        graph.add_node(gas_nid, "GasAnomaly", anomaly.to_dict())
        graph.add_edge(gas_nid, default_seg, OCCURRED_IN)

        # Link to blast events if correlated
        if anomaly.blast_correlated == 1:
            # Find the nearest blast node in the graph
            blast_nodes = graph.query_by_label("BlastEvent")
            if blast_nodes:
                nearest = blast_nodes[0]  # simplified nearest
                graph.add_edge(gas_nid, nearest["node_id"], CORRELATED_WITH,
                               {"time_delta_s": 0.0})

        gas_count += 1

    print(f"    Created {gas_count} GasAnomaly nodes ({len(hazard_df)} hazard + {len(normal_sample)} normal)")


# ---------------------------------------------------------------------------
# 3. Environmental telemetry readings
# ---------------------------------------------------------------------------

def ingest_environmental_readings(graph: MineKnowledgeGraph, csv_path: str,
                                   sample_every_n: int = 2000, max_readings: int = 200):
    """
    Reads iot_telemetry_clean.csv and creates:
      - EnvironmentalReading nodes (sampled every Nth row)
      - OCCURRED_IN edges to a default tunnel segment
    Flags anomalies using a simple threshold (temp > 28 or humidity > 80).
    """
    if not os.path.exists(csv_path):
        print(f"  [SKIP] Env CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"  Ingesting env data: {len(df)} rows from {os.path.basename(csv_path)}")

    # Default tunnel segment for environmental sensors
    env_seg = "seg_env_main"
    _ensure_segment(graph, env_seg, gx=324100.0, gy=7412100.0, gelev=-180.0)

    # Sample to keep the graph manageable
    sample_df = df.iloc[::sample_every_n].head(max_readings)

    env_count = 0
    for _, row in sample_df.iterrows():
        temp = float(row.get("temp", 0.0))
        humidity = float(row.get("humidity", 0.0))
        anomaly_flag = 1 if (temp > 28.0 or humidity > 80.0) else 0

        reading = EnvironmentalReading(
            timestamp=float(row.get("ts", 0.0)),
            temp=temp,
            humidity=humidity,
            anomaly_flag=anomaly_flag,
            segment_id=env_seg,
        )
        env_nid = f"env_{reading.reading_id}"
        graph.add_node(env_nid, "EnvironmentalReading", reading.to_dict())
        graph.add_edge(env_nid, env_seg, OCCURRED_IN)
        env_count += 1

    print(f"    Created {env_count} EnvironmentalReading nodes")


# ---------------------------------------------------------------------------
# 4. Sensor node placements
# ---------------------------------------------------------------------------

def ingest_sensor_nodes(graph: MineKnowledgeGraph):
    """
    Creates representative SensorNode entries for each modality
    and links them to tunnel segments via LOCATED_IN edges.
    """
    sensor_defs = [
        ("sensor_gas_mq4",     "gas",        "seg_gas_main",  "MQ4 Methane Sensor Array"),
        ("sensor_gas_smoke",   "gas",        "seg_gas_main",  "Smoke & Fire Sensor Array"),
        ("sensor_gas_lpg",     "gas",        "seg_gas_main",  "LPG/CNG Hazard Sensor"),
        ("sensor_gas_co",      "gas",        "seg_gas_main",  "CO/NOx Toxic Gas Sensor"),
        ("sensor_env_iot",     "env",        "seg_env_main",  "IoT Environmental Monitor"),
        ("sensor_env_occ",     "env",        "seg_env_main",  "Occupancy Detection Sensor"),
        ("sensor_vib_geo",     "vibration",  None,            "Geophone Vibration Array"),
        ("sensor_ultra_nav",   "ultrasonic", None,            "Ultrasonic Navigation Array (24-sensor)"),
    ]

    count = 0
    for sid, stype, seg_id, name in sensor_defs:
        node = SensorNode(
            sensor_id=sid,
            sensor_type=stype,
            location_segment_id=seg_id or "",
            install_date="2024-01-01",
        )
        props = node.to_dict()
        props["name"] = name
        graph.add_node(sid, "SensorNode", props)

        if seg_id and graph.get_node(seg_id):
            graph.add_edge(sid, seg_id, LOCATED_IN)
        count += 1

    # Link vibration sensor to the first available blast segment
    blast_segments = graph.query_by_label("TunnelSegment")
    vib_segments = [s for s in blast_segments if "gas" not in s["node_id"] and "env" not in s["node_id"]]
    if vib_segments:
        graph.add_edge("sensor_vib_geo", vib_segments[0]["node_id"], LOCATED_IN)
        graph.add_edge("sensor_ultra_nav", vib_segments[0]["node_id"], LOCATED_IN)

    print(f"    Created {count} SensorNode entries")


# ---------------------------------------------------------------------------
# 5. Equipment entries
# ---------------------------------------------------------------------------

def ingest_equipment(graph: MineKnowledgeGraph):
    """
    Creates representative equipment nodes and links them to tunnel segments.
    """
    equipment_defs = [
        ("equip_drill_001", "drill_rig",        "Atlas Copco Boomer S2",      "seg_gas_main"),
        ("equip_loader_001", "loader",           "CAT R1700 LHD",             "seg_env_main"),
        ("equip_conv_001",  "conveyor",          "Conveyor Belt Section A",    "seg_gas_main"),
        ("equip_vent_001",  "ventilation_fan",   "Main Ventilation Fan Unit",  "seg_env_main"),
    ]

    count = 0
    for eid, etype, name, seg_id in equipment_defs:
        equip = Equipment(
            equipment_id=eid,
            equipment_type=etype,
            name=name,
            location_segment_id=seg_id,
        )
        graph.add_node(eid, "Equipment", equip.to_dict())
        if graph.get_node(seg_id):
            graph.add_edge(eid, seg_id, OPERATES_IN)
        count += 1

    print(f"    Created {count} Equipment entries")


# ---------------------------------------------------------------------------
# Master ingestion function
# ---------------------------------------------------------------------------

def run_full_ingestion(graph: MineKnowledgeGraph, workspace_root: str):
    """
    Execute all ingestion pipelines to populate the EKG from workspace data.
    """
    print("=" * 80)
    print("EXPEDITION KNOWLEDGE GRAPH (EKG) - DATA INGESTION")
    print("=" * 80)

    # 1. Blast + Vibration
    blast_csv = os.path.join(workspace_root, "vibration", "data", "vibration_features.csv")
    ingest_blast_events(graph, blast_csv)

    # 2. Gas anomalies
    gas_csv = os.path.join(workspace_root, "gas_sensors", "data", "FIELDMIND_physics_dataset.csv")
    ingest_gas_anomalies(graph, gas_csv)

    # 3. Environmental readings
    env_csv = os.path.join(workspace_root, "temperature_humidity", "data", "data_clean", "iot_telemetry_clean.csv")
    ingest_environmental_readings(graph, env_csv)

    # 4. Sensor placements
    ingest_sensor_nodes(graph)

    # 5. Equipment
    ingest_equipment(graph)

    print("-" * 80)
    summary = graph.summary()
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Total edges: {summary['total_edges']}")
    print(f"  Nodes by label:")
    for label, count in sorted(summary["nodes_by_label"].items()):
        print(f"    {label:<25} {count:>6}")
    print(f"  Edges by type:")
    for rel, count in sorted(summary["edges_by_type"].items()):
        print(f"    {rel:<25} {count:>6}")
    print("=" * 80)

    return summary
