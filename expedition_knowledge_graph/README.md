# Expedition Knowledge Graph (EKG) - FIELD-MIND Layer 2B

The **Expedition Knowledge Graph (EKG)** is FIELD-MIND's persistent mine memory. It maintains a continuously updated property-graph representation of the entire underground operation: tunnel geometry, sensor placements, blast records, gas anomalies, vibration events, environmental readings, **robot navigation events (ultrasonic)**, and equipment status. Every observation is linked to a spatial location, timestamp, and prior local history, enabling detection of slow-developing hazards, recurring failure patterns, and navigation collision risks.

---

## Architecture

```
              [ Workspace CSV Datasets ]          [ AI Agent ALERTs ]
                       |                                  |
          +------------+------------+--------+           |
          |            |            |        |           |
    vibration/   gas_sensors/  temp_hum/  ultrasonic/   |
          |            |            |        |           |
          v            v            v        v           v
    +-------------------------------------------------------------+
    |           Ingestion Pipelines (ingest.py)                   |
    |  blast_events | gas_anomalies | env_data | nav_events       |
    |                    + EKGAgent (sensor_agents/)              |
    +-------------------------------------------------------------+
              |
              v
    +------------------------------------------+
    |     MineKnowledgeGraph (graph_store.py)   |
    |  NetworkX DiGraph + Label Index           |
    |  JSON Persistence (mine_graph.json)       |
    +------------------------------------------+
              |
              v
    +------------------------------------------+
    |     Query API (query_api.py)              |
    |  risk_profile | gas_trend | correlate     |
    |  nav_events   | sensor_inventory          |
    +------------------------------------------+
              |
              v
    [ Layer 3: Scientific Reasoning Core / MineOrchestratorAgent ]
```

---

## Property Graph Schema

### Node Types

| Label | Key Properties | Source |
|-------|---------------|--------|
| `TunnelSegment` | segment_id, name, depth_m, gx, gy, gelev, status | Auto-clustered from spatial coordinates |
| `SensorNode` | sensor_id, sensor_type, location_segment_id | Synthetic (modality-based) |
| `BlastEvent` | blast_number, total_charge, max_charge, num_holes, detonator_code | vibration_features.csv |
| `VibrationEvent` | ppv, scaled_distance_usbm, hazard_flag, blast_id | vibration_features.csv |
| `GasAnomaly` | gas_readings (dict), hazard_alert, blast_correlated | FIELDMIND_physics_dataset.csv |
| `EnvironmentalReading` | temp, humidity, anomaly_flag | iot_telemetry_clean.csv |
| `NavigationEvent` | timestamp, steering_decision, collision_risk, min_distance, severity, segment_id | sensor_readings_24.csv / UltrasonicSensorAgent |
| `Equipment` | equipment_type, name, status | Synthetic mine equipment |

### Edge / Relationship Types

| Relationship | From -> To | Description |
|-------------|-----------|-------------|
| `LOCATED_IN` | SensorNode / Equipment -> TunnelSegment | Spatial placement |
| `OCCURRED_IN` | Event -> TunnelSegment | Event location |
| `CAUSED_BY` | VibrationEvent -> BlastEvent | Causal link |
| `CORRELATED_WITH` | GasAnomaly -> BlastEvent | Temporal correlation |
| `OPERATES_IN` | Equipment -> TunnelSegment | Equipment location |
| `NAVIGATION_IN` | NavigationEvent -> TunnelSegment | Robot path segment |
| `PRECEDED_BY` | NavigationEvent -> NavigationEvent | Sequential navigation chain |

---

## Graph Statistics (after ingestion)

| Metric | Count |
|--------|-------|
| Total Nodes | 1,389 |
| Total Edges | 1,444+ |
| TunnelSegments | 4 |
| BlastEvents | 62 |
| VibrationEvents | 310 |
| GasAnomalies | 547 |
| EnvironmentalReadings | 200 |
| **NavigationEvents** | **260** (written by `UltrasonicSensorAgent` / AI agent session) |
| SensorNodes | 8 |
| Equipment | 4 |

> NavigationEvent count grows each time the AI agent demo is run (`sensor_agents/demo_agents.py`).

---

## Running the Demo

```bash
python -X utf8 expedition_knowledge_graph/demo_ekg.py
```

This will:
1. Ingest data from all workspace CSV datasets (vibration, gas, env, **ultrasonic navigation**)
2. Print node/edge summary statistics
3. Execute sample queries (risk profiles, gas trends, blast history, event correlations, **navigation events**)
4. Save the graph to `expedition_knowledge_graph/data/mine_graph.json`
5. Reload and verify persistence integrity

To populate the graph with live ultrasonic `NavigationEvent` nodes from the AI agent system:
```bash
# Run the AI agent demo — EKGAgent writes all ALERT events (including ultrasonic) into the graph
py -X utf8 sensor_agents/demo_agents.py
```

---

## Query API

The `query_api.py` module provides high-level functions for the downstream reasoning agent:

| Function | Description |
|----------|-------------|
| `get_recent_anomalies(graph, max_count)` | Latest gas and environmental anomalies |
| `get_blast_history(graph, segment_id)` | Blast records for a tunnel section |
| `get_gas_trend(graph, gas_column, max_points)` | Concentration timeline for a specific gas |
| `get_segment_risk_profile(graph, segment_id)` | Aggregated risk score and hazard counts |
| `correlate_events(graph, event_node_id)` | Causally linked events (blast -> vibration -> gas) |
| `get_equipment_status(graph, segment_id)` | Equipment inventory and status |
| `get_sensor_inventory(graph)` | Registered sensor nodes and locations |
| `get_navigation_events(graph, collision_only, segment_id, max_count)` | Ultrasonic robot navigation events; filter to collision-risk-only events |

---

## Files

| File | Description |
|------|-------------|
| `schema.py` | Property graph schema (dataclasses for all node types, edge constants) |
| `graph_store.py` | NetworkX-backed graph engine with JSON persistence |
| `ingest.py` | Data ingestion pipelines from workspace CSV datasets |
| `query_api.py` | High-level query functions for the reasoning agent |
| `demo_ekg.py` | End-to-end demonstration and verification script |
| `data/mine_graph.json` | Serialised graph (created after running demo) |

---

## Design Decisions

- **NetworkX + JSON** instead of Neo4j: Zero-server, zero-dependency, runs on Jetson Nano's 4 GB RAM without a JVM. Same property-graph schema, easily migratable to Neo4j if needed.
- **Coordinate clustering**: Tunnel segments are auto-generated from blast geophone coordinates using a spatial grid (500 m resolution), avoiding manual segment definitions.
- **Sampling**: Environmental readings are sampled every 2000th row (from 405k) to keep the graph manageable while preserving anomaly distribution.
- **Ultrasonic NavigationEvents via EKGAgent**: Rather than ingesting `sensor_readings_24.csv` in bulk (which is a wall-following benchmark, not spatially located), ultrasonic navigation events are written live into the graph by the `EKGAgent` in `sensor_agents/`. This means every collision-risk event detected by `UltrasonicSensorAgent` is timestamped and persisted in real-time as a `NavigationEvent` node, linked to the active tunnel segment. This is architecturally cleaner because navigation decisions are inherently temporal stream data, not static batch data.
