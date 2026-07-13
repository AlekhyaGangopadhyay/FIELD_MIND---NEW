"""
ekg_agent.py — Expedition Knowledge Graph Integration Agent
===========================================================
The EKG Agent acts as the system's "long-term memory".

Responsibilities:
  1. Subscribe to ALL AgentBus ALERT messages
  2. Write confirmed hazard events to the Expedition Knowledge Graph
  3. Answer QUERY messages from other agents with historical context
  4. Provide risk profile summaries on demand

This agent does NOT run a self-learning loop (it is not a predictive model).
Instead, it provides temporal context that helps other agents calibrate their
confidence (e.g., "this tunnel has had 12 gas hazards in the past hour").
"""

import sys
import os
import time
from typing import Any, Dict, List, Optional

from .agent_bus import AgentBus, AgentMessage, MessageType, Severity


class EKGAgent:
    """
    Knowledge Graph Integration Agent — subscribes to the bus and
    writes ALERT events into the Expedition Knowledge Graph.

    Parameters
    ----------
    workspace_root : str
    bus            : AgentBus
    verbose        : bool
    """

    def __init__(self, workspace_root: str, bus: AgentBus, verbose: bool = True):
        self.workspace_root = workspace_root
        self.bus            = bus
        self.verbose        = verbose
        self.agent_name     = "EKGAgent"

        # Try to import the EKG (graceful fallback if graph store not available)
        self._graph = None
        self._ekg_available = False
        self._try_load_ekg()

        # Subscribe to all alert messages from any source
        self.bus.subscribe("*", self._on_bus_message)

        # Metrics
        self._events_written = 0
        self._queries_answered = 0
        self._recent_alerts: List[AgentMessage] = []  # rolling buffer

        print(f"  [EKGAgent] Initialized. EKG available: {self._ekg_available}")

    # -----------------------------------------------------------------------
    # EKG Loading
    # -----------------------------------------------------------------------

    def _try_load_ekg(self) -> None:
        """Attempt to load the Expedition Knowledge Graph."""
        try:
            # Add EKG module to path
            ekg_dir = os.path.join(self.workspace_root, "expedition_knowledge_graph")
            if ekg_dir not in sys.path:
                sys.path.insert(0, ekg_dir)

            from graph_store import MineKnowledgeGraph
            graph_json = os.path.join(ekg_dir, "data", "mine_graph.json")

            self._graph = MineKnowledgeGraph()
            if os.path.exists(graph_json):
                self._graph.load(graph_json)
                print(f"  [EKGAgent] ✓ Knowledge graph loaded from {graph_json}")
            else:
                print(f"  [EKGAgent] ⚠ No existing graph file — starting with empty graph.")
            self._ekg_available = True
        except Exception as e:
            print(f"  [EKGAgent] EKG unavailable (will run in log-only mode): {e}")
            self._ekg_available = False

    # -----------------------------------------------------------------------
    # Bus Event Handler
    # -----------------------------------------------------------------------

    def _on_bus_message(self, msg: AgentMessage) -> None:
        """Receive and process all messages from AgentBus."""
        if msg.source == self.agent_name:
            return   # Don't process own messages

        if msg.msg_type == MessageType.ALERT:
            self._handle_alert(msg)

        elif msg.msg_type == MessageType.QUERY:
            self._handle_query(msg)

        elif msg.msg_type == MessageType.LEARNING_UPDATE:
            if self.verbose:
                print(f"  [EKGAgent] Learning update from {msg.source}: {msg.reason}")

    # -----------------------------------------------------------------------
    # Alert Handling → EKG Write
    # -----------------------------------------------------------------------

    def _handle_alert(self, msg: AgentMessage) -> None:
        """Write an ALERT event into the knowledge graph and local buffer."""
        self._recent_alerts.append(msg)
        if len(self._recent_alerts) > 200:
            self._recent_alerts.pop(0)

        if self.verbose:
            print(f"  [EKGAgent] ← ALERT from {msg.source} [{msg.severity.name}]: {msg.reason}")

        if not self._ekg_available or self._graph is None:
            self._events_written += 1   # count even in log-only mode
            return

        # Map agent source to EKG node type + properties
        try:
            node_id = f"agent_alert_{msg.source}_{int(msg.timestamp * 1000)}"
            payload = msg.payload or {}

            if "GasSensor" in msg.source:
                self._graph.add_node(node_id, label="GasAnomaly", properties={
                    "timestamp"    : msg.timestamp,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.timestamp)),
                    "hazard_alert" : 1,
                    "severity"     : msg.severity.name,
                    "gas_readings" : str(payload),
                    "agent_source" : msg.source,
                    "reason"       : msg.reason,
                })
            elif "EnvSensor" in msg.source:
                self._graph.add_node(node_id, label="EnvironmentalReading", properties={
                    "timestamp"    : msg.timestamp,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.timestamp)),
                    "anomaly_flag" : 1,
                    "severity"     : msg.severity.name,
                    "env_readings" : str(payload),
                    "agent_source" : msg.source,
                    "reason"       : msg.reason,
                })
            elif "Vibration" in msg.source:
                ppv = payload.get("predicted_ppv", 0.0)
                self._graph.add_node(node_id, label="VibrationEvent", properties={
                    "timestamp"    : msg.timestamp,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.timestamp)),
                    "hazard_flag"  : 1,
                    "ppv"          : float(ppv),
                    "severity"     : msg.severity.name,
                    "agent_source" : msg.source,
                    "reason"       : msg.reason,
                })
            elif "Ultrasonic" in msg.source:
                self._graph.add_node(node_id, label="NavigationEvent", properties={
                    "timestamp"      : msg.timestamp,
                    "timestamp_str"  : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.timestamp)),
                    "collision_risk" : 1,
                    "steering"       : payload.get("steering_decision", "unknown"),
                    "severity"       : msg.severity.name,
                    "agent_source"   : msg.source,
                    "reason"         : msg.reason,
                })

            self._events_written += 1

        except Exception as e:
            print(f"  [EKGAgent] Failed to write node to graph: {e}")

    # -----------------------------------------------------------------------
    # Query Handling
    # -----------------------------------------------------------------------

    def _handle_query(self, msg: AgentMessage) -> None:
        """Answer a QUERY message by publishing a RESPONSE."""
        query_type = msg.payload.get("query_type", "recent_alerts")
        self._queries_answered += 1

        if query_type == "recent_alerts":
            n = msg.payload.get("n", 10)
            alerts = self._recent_alerts[-n:]
            self.bus.publish(AgentMessage(
                source    = self.agent_name,
                msg_type  = MessageType.RESPONSE,
                severity  = Severity.LOW,
                payload   = {"alerts": [{"source": a.source, "reason": a.reason, "ts": a.timestamp} for a in alerts]},
                reason    = f"Last {len(alerts)} alerts from bus history",
                timestamp = time.time(),
                correlation_id = msg.correlation_id,
            ))

        elif query_type == "risk_profile" and self._ekg_available:
            segment_id = msg.payload.get("segment_id", "")
            try:
                sys.path.insert(0, os.path.join(self.workspace_root, "expedition_knowledge_graph"))
                from query_api import get_segment_risk_profile
                profile = get_segment_risk_profile(self._graph, segment_id)
                self.bus.publish(AgentMessage(
                    source    = self.agent_name,
                    msg_type  = MessageType.RESPONSE,
                    severity  = Severity.LOW,
                    payload   = profile,
                    reason    = f"Risk profile for segment {segment_id}",
                    timestamp = time.time(),
                    correlation_id = msg.correlation_id,
                ))
            except Exception as e:
                print(f"  [EKGAgent] Risk profile query failed: {e}")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def save_graph(self, path: Optional[str] = None) -> None:
        """Persist the knowledge graph to JSON."""
        if not self._ekg_available or self._graph is None:
            return
        if path is None:
            path = os.path.join(
                self.workspace_root, "expedition_knowledge_graph", "data", "mine_graph.json"
            )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._graph.save(path)
            print(f"  [EKGAgent] Graph saved to {path} ({self._events_written} new events)")
        except Exception as e:
            print(f"  [EKGAgent] Graph save failed: {e}")

    def get_recent_alerts(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last N alerts received by this agent."""
        return [
            {"source": a.source, "severity": a.severity.name, "reason": a.reason, "ts": a.timestamp}
            for a in self._recent_alerts[-n:]
        ]

    def status_report(self) -> str:
        lines = [
            f"╔══ EKGAgent Status Report ══╗",
            f"  EKG available      : {self._ekg_available}",
            f"  Events written     : {self._events_written}",
            f"  Queries answered   : {self._queries_answered}",
            f"  Recent alerts kept : {len(self._recent_alerts)}",
        ]
        return "\n".join(lines)
