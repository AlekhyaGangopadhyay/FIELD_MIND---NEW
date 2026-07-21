"""
chat_assistant.py — Conversational Mine Safety Assistant
========================================================
Main conversational agent that communicates safety conditions to users.
Analyses active sensor data inputs, checks EKG segment memories, retrieves
FAISS safety guidelines, and generates a conversational response summarizing
hazards and recommended safety measures.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

from .llm_runner import OfflineLLMRunner
from .agent_loop import ScientificReasoningCore
from faiss_rag import SafetyProtocolEvaluator

# Add EKG path to import safely
EKG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "expedition_knowledge_graph")
if EKG_DIR not in sys.path:
    sys.path.insert(0, EKG_DIR)

try:
    from graph_store import MineKnowledgeGraph
    from query_api import get_segment_risk_profile, get_blast_history
    EKG_AVAILABLE = True
except ImportError:
    EKG_AVAILABLE = False


class MineSafetyChatAssistant:
    """
    Conversational AI Safety Assistant that communicates safety conditions to mine operators,
    analyzes data inputs, and references regulation safety measures.
    """

    def __init__(
        self,
        workspace_root: str,
        model_path: Optional[str] = None,
        faiss_dir: Optional[str] = None,
        ekg_json_path: Optional[str] = None
    ):
        self.workspace_root = workspace_root
        self.core = ScientificReasoningCore(
            workspace_root=workspace_root,
            model_path=model_path,
            faiss_dir=faiss_dir,
            ekg_json_path=ekg_json_path
        )
        self.llm_runner = self.core.llm_runner
        self.ekg_json_path = self.core.ekg_json_path
        self.rag_retriever = self.core.rag_retriever
        self.protocol_evaluator = SafetyProtocolEvaluator(self.rag_retriever)

    def chat(
        self,
        user_message: str,
        segment_id: str,
        active_anomalies: Dict[str, Any],
        model_predictions: Optional[Dict[str, Any]] = None,
        sensor_readings: Optional[Dict[str, Any]] = None,
        trend_context: Optional[str] = None,
    ) -> str:
        """
        Processes a user question, analyzes active readings, retrieves EKG and RAG,
        and generates a conversational response.
        """
        # 1. Fetch EKG context
        ekg_context = ""
        if EKG_AVAILABLE and os.path.exists(self.ekg_json_path):
            try:
                graph = MineKnowledgeGraph()
                graph.load(self.ekg_json_path)
                self.core._ensure_tunnel_segment(graph, segment_id)
                profile = get_segment_risk_profile(graph, segment_id)
                ekg_context = (
                    f"EKG context for segment {segment_id}: "
                    f"active hazards count={profile.get('hazard_count', 0)}, "
                    f"gas anomalies={profile.get('gas_anomalies_count', 0)}, "
                    f"vibration events={profile.get('vibration_events_count', 0)}."
                )
                blasts = get_blast_history(graph, segment_id)
                if blasts:
                    ekg_context += f" Recent blasting events ({len(blasts)}) registered in history."
            except Exception as e:
                ekg_context = f"EKG lookup context: {e}"
        else:
            ekg_context = f"EKG context for segment {segment_id} is unavailable."

        # 2. Compare raw readings with explicit OSHA/NIOSH/industry limits and
        # retrieve the supporting FAISS passages in one batched operation.
        model_predictions = model_predictions or {}
        try:
            assessment = self.protocol_evaluator.assess(
                readings=sensor_readings or active_anomalies,
                predictions=model_predictions,
                top_k=2,
            )
        except Exception as exc:
            # A missing/offline embedding model must not suppress deterministic
            # safety checks or the operator response.
            assessment = SafetyProtocolEvaluator().assess(
                readings=sensor_readings or active_anomalies,
                predictions=model_predictions,
                top_k=2,
            )
            assessment.rag_context = f"FAISS retrieval unavailable: {exc}"
        rag_context = assessment.rag_context or "FAISS safety guidelines index is unavailable."
        protocol_report = assessment.format_report()

        # 3. Format Prompt
        prompt = (
            "You are FIELD-MIND, an expert offline underground mining safety assistant.\n"
            "Your job is to communicate clearly with the operator, analyse data inputs, "
            "and suggest safety measures based on regulations.\n\n"
            f"LOCATION SEGMENT: {segment_id}\n"
            f"ACTIVE DATA INPUTS (anomalies/readings): {active_anomalies}\n"
            f"ALL SENSOR READINGS: {sensor_readings or active_anomalies}\n"
            f"MODEL PREDICTIONS: {model_predictions}\n"
            f"MODEL VS PROTOCOL ASSESSMENT:\n{protocol_report}\n"
            f"MULTI-NODE TIME-SERIES TREND:\n{trend_context or 'No historical trend supplied.'}\n"
            f"SAFETY MEASURES & REGULATIONS (RAG context):\n{rag_context}\n"
            f"EKG HISTORICAL GRAPH CONTEXT:\n{ekg_context}\n\n"
            f"USER QUERY: {user_message}\n\n"
            "Generate a helpful, conversational, safety-focused response to the user. "
            "Discuss and analyse the data inputs and explain corresponding safety measures."
        )

        # 4. Run through model generator or fallback conversational engine
        response = self.llm_runner.run_reasoning(
            prompt=prompt,
            anomalies=active_anomalies,
            ekg_history=ekg_context,
            rag_context=rag_context,
            task_type="chat"
        )

        if response == "Unknown task type.":
            # If the fallback doesn't support task_type="chat", build conversational response locally
            response = self._build_conversational_response(
                user_message,
                segment_id,
                active_anomalies,
                ekg_context,
                rag_context,
                assessment=assessment,
                sensor_readings=sensor_readings or active_anomalies,
                model_predictions=model_predictions,
                trend_context=trend_context,
            )

        return response

    def _build_conversational_response(
        self,
        user_message: str,
        segment_id: str,
        active_anomalies: Dict[str, Any],
        ekg_context: str,
        rag_context: str,
        assessment=None,
        sensor_readings: Optional[Dict[str, Any]] = None,
        model_predictions: Optional[Dict[str, Any]] = None,
        trend_context: Optional[str] = None,
    ) -> str:
        """
        Fallback conversational response builder analyzing data inputs and safety measures.
        """
        # The old rule-by-query response is retained for compatibility with
        # callers that invoke this private method directly. Normal chat turns
        # now use the structured assessment below.
        if assessment is not None:
            return self._build_assessment_response(
                user_message=user_message,
                segment_id=segment_id,
                assessment=assessment,
                ekg_context=ekg_context,
                sensor_readings=sensor_readings or active_anomalies,
                model_predictions=model_predictions or {},
                trend_context=trend_context,
            )

        # Parse query keywords
        q = user_message.lower()
        is_methane = "methane" in q or "ch4" in q or "gas" in q
        is_vib = "vibration" in q or "ppv" in q or "blast" in q
        is_env = "temp" in q or "humidity" in q or "heat" in q
        is_nav = "robot" in q or "navigation" in q or "collision" in q

        # Parse values
        max_methane = float(active_anomalies.get("MQ4_CH4_ppm", 0))
        max_co = float(active_anomalies.get("MQ7_CO_ppm", 0))
        max_ppv = float(active_anomalies.get("predicted_ppv", 0.0) or active_anomalies.get("ppv", 0.0))
        min_dist = float(active_anomalies.get("min_distance", 5.0))
        temp = float(active_anomalies.get("temp", 20.0))

        lines = [
            f"Hello. This is **FIELD-MIND**, your on-site safety assistant. Let me analyze the current safety parameters for **{segment_id}** and answer your request.",
            ""
        ]

        # ───────────────────────────────────────────────────────────────────
        # Analysis of Data Inputs
        # ───────────────────────────────────────────────────────────────────
        lines.append("### 📊 Active Data Inputs Analysis")
        if not active_anomalies:
            lines.append("• All monitored sensor parameters are currently within normal baseline ranges.")
        else:
            if max_methane > 0:
                status = "🚨 CRITICAL" if max_methane > 10000 else "⚠ ELEVATED"
                lines.append(f"• **Methane (CH4)**: {max_methane:.1f} ppm [{status}].")
            if max_co > 0:
                status = "⚠ ELEVATED" if max_co > 50 else "NOMINAL"
                lines.append(f"• **Carbon Monoxide (CO)**: {max_co:.1f} ppm [{status}].")
            if max_ppv > 0:
                status = "🚨 HIGH RISK" if max_ppv > 10.0 else "NOMINAL"
                lines.append(f"• **Ground Vibration (PPV)**: {max_ppv:.2f} mm/s [{status}].")
            if min_dist < 5.0:
                status = "🚨 IMMEDIATE COLLISION RISK" if min_dist < 0.3 else "NOMINAL"
                lines.append(f"• **Robot Proximity Distance**: {min_dist:.2f} m [{status}].")
            if temp != 20.0:
                lines.append(f"• **Ambient Temperature**: {temp:.1f}°C.")

        # EKG history integration
        if "recent blasting" in ekg_context.lower() or "hazards" in ekg_context.lower():
            lines.append(f"• **Memory Logs**: {ekg_context}")
        lines.append("")

        # ───────────────────────────────────────────────────────────────────
        # Analysis of Safety Measures
        # ───────────────────────────────────────────────────────────────────
        lines.append("### 🛡️ Regulation Safety Measures")
        if is_methane or (max_methane > 5000 or max_co > 25):
            lines.append("• **Gas Safety (OSHA/NIOSH)**: Methane concentrations exceeding 1.0% (10,000 ppm) require immediate evacuation. CO must remain below 25 ppm for safe post-blast entry.")
        if is_vib or max_ppv > 1.0:
            lines.append("• **Vibration Limits (IS 6922)**: Industrial structure limits are 5 mm/s, and residential structures are capped at 12.5 mm/s. PPV values exceeding 12.5 mm/s require a halt in blasting operations and a visual geological inspection.")
        if is_env or temp > 28.0:
            lines.append("• **Environmental Comfort (OSHA)**: Ambient work temperatures above 28°C require worker hydration cycles (15 minutes rest per hour) and secondary air conditioning controls.")
        if is_nav or min_dist < 0.5:
            lines.append("• **Navigation Safety (AS 4024)**: Robotic platforms must stop if an obstacle is within 30 cm (0.3 m) to prevent structural damage.")
        
        if not (is_methane or is_vib or is_env or is_nav):
            lines.append("• **General Safety Rules**: Standard operating rules mandate continuous secondary ventilation, visual inspection of tunnel headings, and log persistence on EKG.")
        lines.append("")

        # ───────────────────────────────────────────────────────────────────
        # Action Plan
        # ───────────────────────────────────────────────────────────────────
        lines.append("### 📢 Recommended Safety Actions")
        actions = []
        if max_methane > 10000:
            actions.append("**EVACUATE** segment immediately. Power down all non-intrinsically safe electrical grids.")
            actions.append("Override and speed up auxiliary exhaust ventilation fans.")
        elif max_co > 50:
            actions.append("Enforce a post-blast safety dilution period. Do not enter the area for 30 minutes.")
            
        if max_ppv > 12.5:
            actions.append("Initiate roof integrity scans to check for rock fall hazards.")
            
        if min_dist < 0.3:
            actions.append("Halt autonomous vehicle movement and command robot to backup.")

        if not actions:
            actions.append("All systems within safe parameters. Maintain continuous baseline surveillance.")

        for act in actions:
            lines.append(f"1. {act}")

        return "\n".join(lines)

    def _build_assessment_response(
        self,
        user_message: str,
        segment_id: str,
        assessment,
        ekg_context: str,
        sensor_readings: Dict[str, Any],
        model_predictions: Dict[str, Any],
        trend_context: Optional[str] = None,
    ) -> str:
        """Render one consistent, dynamic response from protocol checks."""
        lines = [
            f"FIELD-MIND safety assessment for {segment_id}",
            f"Overall status: **{assessment.overall_status}**",
            "",
        ]

        active_checks = [
            check for check in assessment.checks
            if check.severity != "OK" or check.status not in {"WITHIN_LIMIT", "ALERT"}
        ]
        normal_checks = [
            check for check in assessment.checks
            if check.severity == "OK" and check.status == "WITHIN_LIMIT"
        ]
        if active_checks:
            lines.append("### Findings")
            for check in active_checks:
                signal = ""
                if check.model_signal is not None:
                    signal = f" Model output: `{check.model_signal}`."
                lines.append(
                    f"- **{check.metric}**: {check.reading:g} {check.unit} - "
                    f"{check.severity} / {check.status}.{signal} {check.message}"
                )
            lines.append("")
        else:
            lines.extend([
                "### Findings",
                "All supplied numeric readings are within the configured protocol limits, and no model disagreement was detected.",
                "",
            ])

        if normal_checks:
            normal_summary = "; ".join(
                f"{check.metric} {check.reading:g} {check.unit}"
                for check in normal_checks
            )
            lines.extend([
                "### Normal checks",
                f"Within configured limits: {normal_summary}.",
                "",
            ])

        # Model outputs that are useful context but are not numeric protocol
        # checks should still be visible to the operator.
        model_context = []
        if "occupancy_state" in model_predictions:
            occupied = bool(model_predictions["occupancy_state"])
            model_context.append(f"Occupancy classifier: {'occupied' if occupied else 'unoccupied'}.")
        if "air_quality_score" in model_predictions:
            model_context.append(
                f"Air-quality model score: {float(model_predictions['air_quality_score']):.2f} "
                "(interpret against the model's documented scale; it is not a direct regulatory limit)."
            )
        if "steering_decision" in model_predictions:
            model_context.append(f"Navigation model decision: {model_predictions['steering_decision']}.")
        if model_context:
            lines.append("### Model context")
            lines.extend(f"- {item}" for item in model_context)
            lines.append("")

        if trend_context:
            lines.append("### Multi-node trend")
            lines.append(trend_context)
            lines.append("")

        lines.append("### Recommended actions")
        if assessment.actions:
            lines.extend(f"{index}. {action}" for index, action in enumerate(assessment.actions, 1))
        else:
            lines.append("1. Maintain baseline monitoring and normal mine operating procedures.")
        lines.append("")

        if ekg_context:
            lines.append(f"### Mine memory\n{ekg_context}\n")
        if assessment.rag_results:
            evidence = ", ".join(
                f"{item['source']} ({item['score']:.2f})"
                for item in assessment.rag_results[:4]
            )
            lines.append(f"### Grounding\nFAISS evidence consulted: {evidence}.")
        else:
            lines.append("### Grounding\nNo FAISS evidence was returned; treat this as baseline rule-engine output.")

        return "\n".join(lines)
