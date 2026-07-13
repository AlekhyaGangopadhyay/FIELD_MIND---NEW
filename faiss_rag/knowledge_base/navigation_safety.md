# Robot Navigation Safety — Ultrasonic Collision Protocols

## Overview

FIELD-MIND deploys autonomous robotic platforms for inspection of underground mine workings. The `UltrasonicSensorAgent` uses a 24-sensor array and a trained Random Forest classifier to predict navigation decisions and detect collision risks in real-time.

---

## Navigation Decision Classes

The ultrasonic classifier predicts one of four steering commands:

| Class Label | Description | Safety Implication |
|-------------|-------------|-------------------|
| `Move-Forward` | Clear path ahead | Safe to proceed |
| `Slight-Right-Turn` | Minor obstacle detected | Low risk, gentle correction |
| `Slight-Left-Turn` | Minor obstacle on right | Low risk, gentle correction |
| `Sharp-Right-Turn` | Imminent collision risk | **HIGH RISK — Evasive action** |

**Collision Risk Trigger**: FIELD-MIND raises a `collision_risk = 1` flag whenever `Sharp-Right-Turn` is predicted by the navigation model OR when the minimum detected distance falls below **0.3 m** (30 cm proximity alert).

---

## Sensor Configuration

### 24-Sensor Array (Primary — `sensor_readings_24.csv`)

- Sensors US1–US24 placed circumferentially around the robot chassis
- Readings in metres (typical range: 0.02 m to 5.0 m)
- Scan rate: ~10 Hz
- Field of View: 360° coverage

### 4-Sensor Array (Compact — `sensor_readings_4.csv`)

- Sensors: SD_front, SD_left, SD_right, SD_back
- Suitable for narrow heading inspections

### 2-Sensor Array (Minimal — `sensor_readings_2.csv`)

- Sensors: SD_front, SD_left
- Used for forward-only inspection in tight spaces

---

## Collision Avoidance Protocols

### Level 1: Proximity Warning (distance < 1.0 m)

- Reduce robot speed by 50%
- Increase sensor scan rate
- Log position and reading to EKG as `NavigationEvent`

### Level 2: Proximity Alert (distance < 0.3 m)

- Halt robot immediately
- Activate audible alarm (if equipped)
- Alert control room via FIELD-MIND AgentBus ALERT message (Severity: HIGH)
- Write `NavigationEvent` with `collision_risk=1` to Expedition Knowledge Graph
- Wait for operator confirmation or autonomous recovery manoeuvre

### Level 3: Imminent Collision (Sharp-Right-Turn predicted)

- Execute evasive steering command immediately (no wait)
- Record full 24-sensor vector at time of evasion
- Trigger EMERGENCY state in MineOrchestratorAgent if 3 consecutive sharp turns detected
- Full system review if more than 5 collision events per tunnel segment per shift

---

## Underground Robot Operational Standards

### AS 4024.3302 (Australian Standard — Robots and Related Devices)

- Risk assessment mandatory before deploying autonomous platforms underground
- Emergency stop accessible within 0.5 seconds at any robot state
- Minimum 0.5 m clearance from all walls during normal operations
- Tethered communications required in areas with >3 corners (GPS unavailable underground)

### ISO 10218 (Industrial Robot Safety)

- Speed limits in collaborative zones: 250 mm/s maximum
- Force limits for human contact: < 150 N
- All autonomous systems must have defined safe states

### Underground-Specific Requirements

1. **Explosion-proofing**: Robot electrical systems must be ATEX/IECEx certified for Zone 1/2 hazardous areas (flammable gas present).
2. **Lighting**: Minimum 100 lux at robot head height in all traversed areas.
3. **Communication**: Leaky feeder cable or mesh radio for real-time telemetry.
4. **Fail-safe**: Robot must default to STOP state on communication loss > 5 seconds.
5. **Mapping**: SLAM-based localisation required; robot position logged to EKG every 30 seconds.

---

## Navigation Event Logging (EKG)

Every `NavigationEvent` node in the Expedition Knowledge Graph stores:

```json
{
  "timestamp"       : "<ISO timestamp>",
  "steering_decision": "Sharp-Right-Turn",
  "collision_risk"  : 1,
  "min_distance"    : 0.18,
  "severity"        : "HIGH",
  "segment_id"      : "TUNNEL_A3",
  "agent_source"    : "UltrasonicSensorAgent"
}
```

### Querying Navigation History

```python
from expedition_knowledge_graph.query_api import get_navigation_events

# All collision events in the last session
collisions = get_navigation_events(graph, collision_only=True)

# Navigation events in a specific tunnel segment
segment_nav = get_navigation_events(graph, segment_id="TUNNEL_A3")
```

---

## FIELD-MIND UltrasonicSensorAgent

### Model

| Model | Type | Input | Output |
|-------|------|-------|--------|
| `best_ultrasonic_24.joblib` | RF Classifier | US1–US24 (24 features) | Navigation class (4 classes) |

### Self-Learning

- Learns from `sensor_readings_24.csv` original dataset
- Label for learning: `collision_risk = 1` if class == `Sharp-Right-Turn`
- Experience replay buffer: 200 samples
- After 2 refits (300 ticks): in-sample accuracy = **0.975**
- Refitted model is atomically swapped into the live inference pipeline

### AgentBus Integration

- Publishes ALERT (Severity: HIGH/CRITICAL) on collision detection
- ALERT received by EKGAgent → writes `NavigationEvent` node
- ALERT received by MineOrchestratorAgent → contributes 0.15 weight to global hazard score
- Multiple navigation ALERTs combined with gas or vibration alerts → EMERGENCY state
