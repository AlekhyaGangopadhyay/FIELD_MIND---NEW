"""
Expedition Knowledge Graph (EKG) - End-to-End Demo
====================================================
1. Creates a fresh MineKnowledgeGraph
2. Runs all ingestion pipelines from workspace CSV data
3. Prints a graph summary
4. Executes sample queries (segment history, risk profiles, gas trends)
5. Saves the graph to JSON and reloads to verify persistence
"""

import os
import sys
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from graph_store import MineKnowledgeGraph
from ingest import run_full_ingestion
from query_api import (
    get_recent_anomalies,
    get_blast_history,
    get_gas_trend,
    get_segment_risk_profile,
    correlate_events,
    get_equipment_status,
    get_sensor_inventory,
    get_navigation_events,
)


def main():
    # ----------------------------------------------------------------
    # 1. Create graph and ingest data
    # ----------------------------------------------------------------
    graph = MineKnowledgeGraph()
    summary = run_full_ingestion(graph, workspace_root)

    # ----------------------------------------------------------------
    # 2. Sample Queries
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SAMPLE QUERIES")
    print("=" * 80)

    # 2a. List all tunnel segments
    segments = graph.query_by_label("TunnelSegment")
    print(f"\n--- Tunnel Segments ({len(segments)}) ---")
    for seg in segments[:10]:
        print(f"  {seg['node_id']:<25} | {seg.get('name', ''):<35} | Depth: {seg.get('depth_m', 0):.0f} m")

    # 2b. Risk profile for each segment
    print(f"\n--- Risk Profiles ---")
    for seg in segments:
        profile = get_segment_risk_profile(graph, seg["node_id"])
        if profile["total_blasts"] > 0 or profile["gas_hazards"] > 0 or profile["env_anomalies"] > 0:
            print(f"  Segment: {profile['segment_name']}")
            print(f"    Blasts: {profile['total_blasts']} | Vib Hazards: {profile['vibration_hazards']} | "
                  f"Gas Hazards: {profile['gas_hazards']} | Env Anomalies: {profile['env_anomalies']}")
            print(f"    Max PPV: {profile['max_ppv_mm_s']:.2f} mm/s | Sensors: {profile['sensors_installed']} | "
                  f"Risk Score: {profile['risk_score']} ({profile['risk_level']})")

    # 2c. Recent anomalies
    anomalies = get_recent_anomalies(graph, max_count=5)
    print(f"\n--- Recent Anomalies (top 5) ---")
    for a in anomalies:
        atype = a.get("anomaly_type", "?")
        if atype == "gas":
            print(f"  [{atype.upper():>5}] Hazard at {a.get('timestamp_str', '?')} | "
                  f"Segment: {a.get('segment_id', '?')}")
        else:
            print(f"  [{atype.upper():>5}] Temp: {a.get('temp', '?')}°C | Humidity: {a.get('humidity', '?')}% | "
                  f"Segment: {a.get('segment_id', '?')}")

    # 2d. Gas trend (Methane)
    trend = get_gas_trend(graph, gas_column="MQ4_CH4_ppm", max_points=10)
    print(f"\n--- Methane (CH4) Trend (first 10 readings) ---")
    for t in trend:
        marker = " ⚠️ HAZARD" if t["hazard_alert"] == 1 else ""
        print(f"  {t['timestamp_str']:<22} | CH₄: {t['value']:>8.2f} ppm{marker}")

    # 2e. Blast history (first segment with blasts)
    blast_segments = [s for s in segments if "gas" not in s["node_id"] and "env" not in s["node_id"]]
    if blast_segments:
        target_seg = blast_segments[0]["node_id"]
        blasts = get_blast_history(graph, segment_id=target_seg)
        print(f"\n--- Blast History for {target_seg} ({len(blasts)} events) ---")
        for b in blasts[:5]:
            print(f"  Blast #{b.get('blast_number', '?')} | Charge: {b.get('total_charge', 0):.1f} kg | "
                  f"Max Charge: {b.get('max_charge', 0):.1f} kg | Holes: {b.get('num_holes', 0)}")

    # 2f. Event correlation
    vib_events = graph.query_by_label("VibrationEvent", filters={"hazard_flag": 1})
    if vib_events:
        target_vib = vib_events[0]["node_id"]
        related = correlate_events(graph, target_vib)
        print(f"\n--- Event Correlation for {target_vib} ---")
        for r in related[:5]:
            print(f"  → {r.get('label', '?')} [{r.get('causal_relationship', '?')}] | "
                  f"Node: {r['node_id']}")

    # 2g. Equipment status
    equipment = get_equipment_status(graph)
    print(f"\n--- Equipment Inventory ({len(equipment)} units) ---")
    for e in equipment:
        print(f"  {e.get('name', '?'):<30} | Type: {e.get('equipment_type', '?'):<18} | "
              f"Status: {e.get('status', '?')} | Segment: {e.get('location_segment_id', '?')}")

    # 2h. Sensor inventory
    sensors = get_sensor_inventory(graph)
    print(f"\n--- Sensor Inventory ({len(sensors)} sensors) ---")
    for s in sensors:
        print(f"  {s.get('name', s['node_id']):<35} | Type: {s.get('sensor_type', '?'):<12} | "
              f"Segment: {s.get('location_segment_id', 'unassigned')}")

    # 2i. Navigation events (ultrasonic)
    nav_events = get_navigation_events(graph, collision_only=False, max_count=10)
    collision_events = get_navigation_events(graph, collision_only=True)
    print(f"\n--- Ultrasonic Navigation Events (top 10 of {len(graph.query_by_label('NavigationEvent'))}) ---")
    print(f"    Collision-risk events: {len(collision_events)}")
    for n in nav_events[:10]:
        risk_marker = " ** COLLISION RISK" if n.get('collision_risk') == 1 else ""
        print(f"  Command: {n.get('command', '?'):<20} | Min Distance: {n.get('min_distance', 0):.3f} m | "
              f"Segment: {n.get('segment_id', '?')}{risk_marker}")

    # ----------------------------------------------------------------
    # 3. Persistence: Save and Reload
    # ----------------------------------------------------------------
    save_path = os.path.join(script_dir, "data", "mine_graph.json")
    print(f"\n{'=' * 80}")
    print("PERSISTENCE TEST")
    print(f"{'=' * 80}")

    graph.save(save_path)

    # Reload into a fresh graph
    graph2 = MineKnowledgeGraph()
    graph2.load(save_path)

    # Verify counts match
    s1 = graph.summary()
    s2 = graph2.summary()
    assert s1["total_nodes"] == s2["total_nodes"], "Node count mismatch after reload!"
    assert s1["total_edges"] == s2["total_edges"], "Edge count mismatch after reload!"
    print(f"  ✅ Persistence verified: {s2['total_nodes']} nodes, {s2['total_edges']} edges match after reload.")

    # Quick query on reloaded graph
    segments_reloaded = graph2.query_by_label("TunnelSegment")
    print(f"  ✅ Query on reloaded graph: {len(segments_reloaded)} TunnelSegment nodes found.")

    print(f"\n{'=' * 80}")
    print("EKG Demo completed successfully!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
