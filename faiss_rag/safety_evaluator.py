"""Protocol-aware comparison of sensor-model output with RAG safety rules.

This module is deliberately deterministic.  FAISS retrieves the supporting
OSHA/NIOSH/IS/ISEE/AS-4024 passages, while the small set of comparison rules
below makes the important safety decision explicit and measurable.  A language
model can then explain the result without having to infer a threshold from a
long text chunk.
"""

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProtocolCheck:
    domain: str
    metric: str
    reading: float
    unit: str
    protocol_limit: str
    source: str
    severity: str
    status: str
    model_signal: Any = None
    message: str = ""
    action: str = ""


@dataclass
class SafetyAssessment:
    """Structured output suitable for a CLI, chatbot prompt, or telemetry log."""

    overall_status: str
    checks: List[ProtocolCheck] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    rag_results: List[Dict[str, Any]] = field(default_factory=list)
    rag_context: str = ""
    retrieval_queries: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["checks"] = [asdict(check) for check in self.checks]
        return data

    def format_report(self) -> str:
        lines = [f"SAFETY ASSESSMENT: {self.overall_status}"]
        if self.checks:
            lines.append("MODEL VS PROTOCOL CHECKS:")
            for check in self.checks:
                signal = f"; model={check.model_signal}" if check.model_signal is not None else ""
                lines.append(
                    f"- [{check.severity}] {check.domain}/{check.metric}: "
                    f"{check.reading:g} {check.unit} — {check.status}{signal}. {check.message}"
                )
        if self.actions:
            lines.append("PRIORITIZED ACTIONS:")
            lines.extend(f"{i}. {action}" for i, action in enumerate(self.actions, 1))
        if self.rag_context:
            lines.append(self.rag_context)
        return "\n".join(lines)


class SafetyProtocolEvaluator:
    """Compare FIELD-MIND predictions with explicit safety protocol limits.

    ``readings`` should contain raw measurements (for example
    ``MQ4_CH4_ppm`` or ``predicted_ppv``); ``predictions`` should contain the
    model outputs (for example ``methane_hazard`` or ``vibration_hazard``).
    Both dictionaries are intentionally permissive so the evaluator can be
    used with Tier1Monitor, sensor agents, or a small chatbot form.
    """

    def __init__(self, retriever: Optional[Any] = None):
        self.retriever = retriever

    def assess(
        self,
        readings: Optional[Dict[str, Any]] = None,
        predictions: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        min_score: float = 0.25,
    ) -> SafetyAssessment:
        started = time.perf_counter()
        readings = readings or {}
        predictions = predictions or {}
        checks: List[ProtocolCheck] = []
        actions: List[str] = []
        queries: List[str] = []

        self._evaluate_gas(readings, predictions, checks, actions, queries)
        self._evaluate_environment(readings, predictions, checks, actions, queries)
        self._evaluate_vibration(readings, predictions, checks, actions, queries)
        self._evaluate_navigation(readings, predictions, checks, actions, queries)

        rag_results: List[Dict[str, Any]] = []
        rag_context = ""
        if self.retriever and queries:
            grouped = self.retriever.retrieve_many(queries, top_k=top_k, min_score=min_score)
            # Keep the strongest evidence once when domain queries overlap.
            unique: Dict[Tuple[str, int], Dict[str, Any]] = {}
            for result_set in grouped.values():
                for item in result_set:
                    unique[(item["source"], item["start_char"])] = item
            rag_results = sorted(unique.values(), key=lambda item: item["score"], reverse=True)
            for rank, item in enumerate(rag_results, 1):
                item["rank"] = rank
            rag_results = rag_results[: max(1, top_k * len(queries))]
            rag_context = self.retriever.format_context(rag_results)

        overall_status = self._overall_status(checks)
        if not checks:
            actions.append("No recognized sensor readings were supplied; continue baseline monitoring.")

        return SafetyAssessment(
            overall_status=overall_status,
            checks=checks,
            actions=list(dict.fromkeys(actions)),
            rag_results=rag_results,
            rag_context=rag_context,
            retrieval_queries=queries,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    @staticmethod
    def _value(values: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            if key in values and values[key] is not None:
                try:
                    return float(values[key])
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _signal(values: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in values:
                return values[key]
        return None

    @staticmethod
    def _is_alert(signal: Any) -> Optional[bool]:
        if signal is None:
            return None
        if isinstance(signal, str):
            return signal.strip().lower() not in {
                "0", "false", "no", "none", "move-forward", "safe", "normal", "nominal"
            }
        try:
            return bool(float(signal))
        except (TypeError, ValueError):
            return bool(signal)

    def _check(
        self,
        checks: List[ProtocolCheck],
        actions: List[str],
        *,
        domain: str,
        metric: str,
        reading: float,
        unit: str,
        limit: float,
        limit_text: str,
        critical: Optional[float],
        source: str,
        model_signal: Any,
        action: str,
        query: str,
        queries: List[str],
        warning_message: str,
        critical_message: str,
        expected_alert_at: Optional[float] = None,
        higher_is_worse: bool = True,
    ) -> None:
        queries.append(query)
        protocol_alert = reading > limit if higher_is_worse else reading < limit
        severity = "WARNING" if protocol_alert else "OK"
        message = warning_message if protocol_alert else f"Within {limit_text}."
        is_critical = (
            reading >= critical if higher_is_worse else reading <= critical
        ) if critical is not None else False
        if is_critical:
            severity = "CRITICAL"
            message = critical_message
        model_alert = self._is_alert(model_signal)
        comparison_limit = expected_alert_at if expected_alert_at is not None else limit
        expected_model_alert = (
            reading > comparison_limit if higher_is_worse else reading < comparison_limit
        )
        status = "ALERT" if protocol_alert else "WITHIN_LIMIT"
        if model_alert is not None and model_alert != expected_model_alert:
            status = "MODEL_MISS" if expected_model_alert else "MODEL_ALERT"
            message += (
                f" Model output disagrees with the configured {source} comparison threshold "
                f"({comparison_limit:g} {unit}); inspect calibration, labels, and sensor quality."
            )
            actions.append(
                f"Review {domain} model disagreement for {metric} against {source} threshold."
            )
        if protocol_alert:
            actions.append(action)
        checks.append(ProtocolCheck(
            domain=domain,
            metric=metric,
            reading=reading,
            unit=unit,
            protocol_limit=limit_text,
            source=source,
            severity=severity,
            status=status,
            model_signal=model_signal,
            message=message,
            action=action if protocol_alert else "",
        ))

    def _evaluate_gas(self, r, p, checks, actions, queries):
        ch4 = self._value(r, "MQ4_CH4_ppm", "ch4_ppm", "methane_ppm")
        if ch4 is not None:
            self._check(checks, actions, domain="gas", metric="methane (CH4)", reading=ch4,
                unit="ppm", limit=1000, limit_text="OSHA PEL 1,000 ppm; NIOSH IDLH 5,000 ppm",
                critical=5000, source="OSHA/NIOSH", model_signal=self._signal(p, "methane_hazard"),
                action="Increase ventilation and alert personnel; evacuate the affected section if methane reaches the emergency range.",
                query="OSHA NIOSH methane CH4 underground mine exposure evacuation ventilation protocol",
                queries=queries, warning_message="Exceeds the OSHA time-weighted exposure limit.",
                critical_message="At or above the NIOSH IDLH reference; treat as an emergency and evacuate.")
        co = self._value(r, "MQ7_CO_ppm", "co_ppm", "carbon_monoxide_ppm")
        if co is not None:
            self._check(checks, actions, domain="gas", metric="carbon monoxide (CO)", reading=co,
                unit="ppm", limit=25, limit_text="post-blast re-entry 25 ppm; OSHA PEL 50 ppm",
                critical=200, source="OSHA/NIOSH", model_signal=self._signal(p, "co_nox_hazard"),
                action="Investigate the source, increase ventilation, and prevent re-entry until CO is below 25 ppm.",
                query="OSHA NIOSH carbon monoxide CO post blast re-entry ventilation limit",
                queries=queries, warning_message="Above the configured post-blast re-entry limit.",
                critical_message="Danger range; evacuate personnel and initiate emergency ventilation.")
        lpg = self._value(r, "MQ2_LPG_ppm", "lpg_ppm")
        if lpg is not None:
            self._check(checks, actions, domain="gas", metric="LPG/CNG", reading=lpg,
                unit="ppm", limit=1000, limit_text="OSHA PEL 1,000 ppm",
                critical=21000, source="OSHA", model_signal=self._signal(p, "lpg_hazard"),
                action="Remove ignition sources and increase ventilation; investigate LPG/CNG leakage.",
                query="OSHA LPG CNG lower explosive limit underground mining ventilation protocol",
                queries=queries, warning_message="Exceeds the OSHA reference limit.",
                critical_message="At the LPG lower explosive limit reference; shut down ignition sources and evacuate.")
        nox = self._value(r, "MQ135_NOx_ppm", "nox_ppm", "no2_ppm")
        if nox is not None:
            self._check(checks, actions, domain="gas", metric="NOx/NO2", reading=nox,
                unit="ppm", limit=3, limit_text="NIOSH NO2 TWA 3 ppm",
                critical=5, source="NIOSH", model_signal=self._signal(p, "co_nox_hazard"),
                action="Increase ventilation and investigate diesel or post-blast fumes.",
                query="NIOSH NO2 NOx underground mine exposure limit blast fumes",
                queries=queries, warning_message="Exceeds the NIOSH NO2 time-weighted reference.",
                critical_message="At or above the NIOSH short-term reference; restrict exposure and ventilate.")

    def _evaluate_environment(self, r, p, checks, actions, queries):
        temp = self._value(r, "temp", "temperature_c", "temperature")
        if temp is not None:
            self._check(checks, actions, domain="environment", metric="temperature", reading=temp,
                unit="°C", limit=28, limit_text="OSHA/NIOSH warning reference 28 °C",
                critical=35, source="OSHA/NIOSH/MHSA", model_signal=self._signal(p, "anomaly_detected"),
                action="Increase cooling and enforce hydration/rest cycles for workers.",
                query="OSHA NIOSH MHSA underground mine temperature heat stress wet bulb protocol",
                queries=queries, warning_message="Heat-stress caution range; increase environmental controls.",
                critical_message="Danger range; restrict work and activate engineering heat controls.",
                expected_alert_at=28)
        humidity = self._value(r, "humidity", "humidity_pct", "relative_humidity")
        if humidity is not None:
            self._check(checks, actions, domain="environment", metric="relative humidity", reading=humidity,
                unit="%", limit=85, limit_text="operating range 15–85% RH",
                critical=90, source="FIELD-MIND environmental protocol", model_signal=self._signal(p, "anomaly_detected"),
                action="Control humidity and check condensation, slip, and heat-stress risk.",
                query="underground mine humidity heat stress OSHA environmental safety limit",
                queries=queries, warning_message="Outside the normal operating range.",
                critical_message="Extreme humidity; treat heat stress and equipment condensation as immediate risks.")

    def _evaluate_vibration(self, r, p, checks, actions, queries):
        ppv = self._value(r, "predicted_ppv", "ppv", "PPV")
        if ppv is not None:
            self._check(checks, actions, domain="vibration", metric="PPV", reading=ppv,
                unit="mm/s", limit=1, limit_text="FIELD-MIND early-warning threshold 1 mm/s",
                critical=25, source="IS 6922/ISEE", model_signal=self._signal(p, "vibration_hazard"),
                action="Pause further blasting and inspect the tunnel and nearby structures.",
                query="IS 6922 ISEE USBM blast vibration PPV underground mine safety limit protocol",
                queries=queries, warning_message="Above the conservative FIELD-MIND early-warning threshold.",
                critical_message="High damage-risk range under IS 6922; stop blasting and perform structural inspection.")

    def _evaluate_navigation(self, r, p, checks, actions, queries):
        distance = self._value(r, "min_distance", "minimum_distance_m", "distance_m")
        steering = self._signal(p, "steering_decision", "navigation_class")
        collision = self._signal(p, "sharp_turn_required", "collision_risk")
        if distance is not None:
            model_signal = collision if collision is not None else steering
            self._check(checks, actions, domain="navigation", metric="minimum clearance", reading=distance,
                unit="m", limit=1.0, limit_text="Level 1 warning at 1.0 m; Level 2 stop at 0.3 m",
                critical=0.3, source="AS 4024/ISO 10218", model_signal=model_signal,
                action="Reduce speed below 1 m; halt the robot immediately below 0.3 m and alert the control room.",
                query="AS 4024 ISO 10218 underground robot collision avoidance proximity stop protocol",
                queries=queries, warning_message="Proximity warning; slow the robot and increase scan rate.",
                critical_message="Collision-alert range; stop the robot and execute the fail-safe procedure.",
                expected_alert_at=0.3, higher_is_worse=False)
        elif steering is not None or collision is not None:
            queries.append("AS 4024 ISO 10218 underground robot collision avoidance proximity stop protocol")
            if self._is_alert(collision if collision is not None else steering):
                actions.append("Follow the model's evasive navigation command and verify the full sensor array before proceeding.")

    @staticmethod
    def _overall_status(checks: List[ProtocolCheck]) -> str:
        if any(check.severity == "CRITICAL" for check in checks):
            return "CRITICAL"
        if any(check.status in {"MODEL_MISS", "MODEL_ALERT"} for check in checks):
            return "REVIEW_MODEL_DISAGREEMENT"
        if any(check.severity == "WARNING" for check in checks):
            return "WARNING"
        return "SAFE"
