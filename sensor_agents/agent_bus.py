"""
agent_bus.py — Publish/Subscribe Message Broker for Sensor Agents
=================================================================
Lightweight, in-process event bus that allows sensor agents to
communicate without direct dependencies on each other.

Message Flow:
  SensorAgent.act()  --publish()-->  AgentBus  --deliver()-->  Subscribers
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MessageType(Enum):
    ALERT    = "ALERT"            # Hazard detected — immediate action required
    INFO     = "INFO"             # Normal status update
    CLEAR    = "CLEAR"            # Previous hazard has been resolved
    LEARNING_UPDATE = "LEARNING_UPDATE"  # Agent updated its model weights
    QUERY    = "QUERY"            # Agent requesting context from EKG
    RESPONSE = "RESPONSE"         # EKG/agent replying to a QUERY


class Severity(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


# ---------------------------------------------------------------------------
# Message Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """
    Structured message exchanged between agents via the AgentBus.

    Attributes
    ----------
    source      : str           Name of the publishing agent (e.g. 'GasSensorAgent')
    msg_type    : MessageType   Category of the message
    severity    : Severity      Urgency level
    payload     : dict          Domain-specific data (inference results, embeddings, etc.)
    timestamp   : float         Unix timestamp of the event (default = now)
    reason      : str           Human-readable explanation string
    correlation_id : str        Optional ID to group related messages (chain trace)
    """
    source         : str
    msg_type       : MessageType
    severity       : Severity
    payload        : Dict[str, Any]
    reason         : str          = ""
    timestamp      : float        = field(default_factory=time.time)
    correlation_id : str          = ""

    def __repr__(self) -> str:
        ts = self.timestamp
        return (
            f"[{self.msg_type.value}|{self.severity.name}] "
            f"{self.source} @ t={ts:.2f} — {self.reason}"
        )


# ---------------------------------------------------------------------------
# AgentBus Singleton
# ---------------------------------------------------------------------------

class AgentBus:
    """
    In-process publish/subscribe event bus.

    Usage
    -----
    bus = AgentBus()

    # Subscribe
    def my_handler(msg: AgentMessage):
        print(msg)
    bus.subscribe("GasSensorAgent", my_handler)          # listen to specific source
    bus.subscribe("*", my_handler)                       # listen to all sources

    # Publish
    bus.publish(AgentMessage(source="GasSensorAgent", ...))

    # Retrieve recent messages (optional polling mode)
    recent = bus.get_messages(n=10)
    """

    def __init__(self, history_limit: int = 500):
        # source_name -> list of handler callables
        self._subscribers: Dict[str, List[Callable[[AgentMessage], None]]] = {}
        # Rolling history for inspection / debugging
        self._history: deque = deque(maxlen=history_limit)
        # Alert count per source
        self._alert_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, source_filter: str, handler: Callable[[AgentMessage], None]) -> None:
        """
        Register a handler for messages from a specific source.
        Use "*" to receive messages from ALL sources.

        Parameters
        ----------
        source_filter : str   Source agent name or "*" wildcard
        handler       : callable  Function that accepts an AgentMessage
        """
        if source_filter not in self._subscribers:
            self._subscribers[source_filter] = []
        self._subscribers[source_filter].append(handler)

    def unsubscribe(self, source_filter: str, handler: Callable[[AgentMessage], None]) -> None:
        """Remove a specific handler for a source filter."""
        if source_filter in self._subscribers:
            try:
                self._subscribers[source_filter].remove(handler)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, message: AgentMessage) -> None:
        """
        Broadcast a message to all subscribers matching the source or wildcard.

        Parameters
        ----------
        message : AgentMessage   The message to broadcast
        """
        self._history.append(message)

        # Count alerts per source
        if message.msg_type == MessageType.ALERT:
            self._alert_counts[message.source] = (
                self._alert_counts.get(message.source, 0) + 1
            )

        # Deliver to exact source subscribers
        for handler in self._subscribers.get(message.source, []):
            try:
                handler(message)
            except Exception as e:
                print(f"[AgentBus] Handler error for {message.source}: {e}")

        # Deliver to wildcard subscribers
        for handler in self._subscribers.get("*", []):
            try:
                handler(message)
            except Exception as e:
                print(f"[AgentBus] Wildcard handler error: {e}")

    # ------------------------------------------------------------------
    # History & inspection
    # ------------------------------------------------------------------

    def get_messages(
        self,
        n: int = 50,
        source_filter: Optional[str] = None,
        msg_type_filter: Optional[MessageType] = None,
    ) -> List[AgentMessage]:
        """Return the most recent N messages, optionally filtered."""
        msgs = list(self._history)
        if source_filter:
            msgs = [m for m in msgs if m.source == source_filter]
        if msg_type_filter:
            msgs = [m for m in msgs if m.msg_type == msg_type_filter]
        return msgs[-n:]

    def get_alert_counts(self) -> Dict[str, int]:
        """Return cumulative ALERT counts per agent source."""
        return dict(self._alert_counts)

    def clear_history(self) -> None:
        """Wipe the rolling message history (does not affect subscribers)."""
        self._history.clear()

    def summary(self) -> str:
        """Return a formatted summary of bus activity."""
        lines = ["=== AgentBus Summary ==="]
        lines.append(f"  Messages in history : {len(self._history)}")
        lines.append(f"  Active subscribers  : {len(self._subscribers)}")
        lines.append("  Alert counts per source:")
        for src, cnt in self._alert_counts.items():
            lines.append(f"    {src:<30} {cnt}")
        return "\n".join(lines)
