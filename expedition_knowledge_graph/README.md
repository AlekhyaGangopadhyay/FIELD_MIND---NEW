# Expedition Knowledge Graph (EKG) - FIELD-MIND Layer 2B

The **Expedition Knowledge Graph (EKG)** is FIELD-MIND's persistent mine memory. It maintains a continuously updated property-graph representation of the entire underground operation: tunnel geometry, sensor placements, blast records, gas anomalies, vibration events, environmental readings, and equipment status. Every observation is linked to a spatial location, timestamp, and prior local history, enabling detection of slow-developing hazards and recurring failure patterns.

---

## Architecture

```
                  [ Workspace CSV Datasets ]
                           |
              +------------+------------+
              |            |            |
        vibration/    gas_sensors/   temperature_humidity/
              |            |            |
              v            v            v
    +------------------------------------------+
    |     Ingestion Pipelines (ingest.py)       |
    |  blast_events | gas_anomalies | env_data  |
    +------------------------------------------+
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
    +------------------------------------------+
              |
              v
    [ Layer 3: Scientific Reasoning Core ]
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
| `Equipment` | equipment_type, name, status | Synthetic mine equipment |

### Edge / Relationship Types

| Relationship | From -> To | Description |
|-------------|-----------|-------------|
| `LOCATED_IN` | SensorNode / Equipment -> TunnelSegment | Spatial placement |
| `OCCURRED_IN` | Event -> TunnelSegment | Event location |
| `CAUSED_BY` | VibrationEvent -> BlastEvent | Causal link |
| `CORRELATED_WITH` | GasAnomaly -> BlastEvent | Temporal correlation |
| `OPERATES_IN` | Equipment -> TunnelSegment | Equipment location |

---

## Graph Statistics (after ingestion)

| Metric | Count |
|--------|-------|
| Total Nodes | 1,135 |
| Total Edges | 1,444 |
| TunnelSegments | 4 |
| BlastEvents | 62 |
| VibrationEvents | 310 |
| GasAnomalies | 547 |
| EnvironmentalReadings | 200 |
| SensorNodes | 8 |
| Equipment | 4 |

---

## Running the Demo

```bash
python -X utf8 expedition_knowledge_graph/demo_ekg.py
```

This will:
1. Ingest data from all workspace CSV datasets
2. Print node/edge summary statistics
3. Execute sample queries (risk profiles, gas trends, blast history, event correlations)
4. Save the graph to `expedition_knowledge_graph/data/mine_graph.json`
5. Reload and verify persistence integrity

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
