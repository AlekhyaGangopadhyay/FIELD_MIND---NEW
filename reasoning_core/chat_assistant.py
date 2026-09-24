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
    from query_api import get_segment_risk_profile, get_blast_history, get_self_learned_rules
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
        # Latency Optimization: Memory cache for instant exact query hits (~0ms)
        self._response_cache: Dict[str, str] = {}

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
        # Latency Optimization: Exact Match Cache Check (~0ms)
        cache_key = f"{user_message.strip().lower()}_{segment_id}_{hash(str(sensor_readings or active_anomalies))}"
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        # Fast path for common greetings / simple queries (~0ms latency)
        clean_q = user_message.strip().lower()
        if clean_q in {"hi", "hello", "hey", "status", "help"}:
            readings = sensor_readings or active_anomalies
            tbl = self._build_telemetry_markdown_table(readings)
            has_crit = any(float(readings.get(k, 0)) > limit for k, limit in [("MQ4_CH4_ppm", 5000), ("MQ7_CO_ppm", 50), ("vibration_pulses", 10)])
            status_summary = "🚨 CRITICAL HAZARD" if has_crit else "✔ SYSTEM NOMINAL"
            fast_reply = (
                f"### ❓ Asked Question\n> **{user_message}**\n\n"
                f"---\n"
                f"### 🛡️ FIELD-MIND Quick Status — Sector {segment_id}\n"
                f"Overall Status: **{status_summary}**\n\n"
                f"{tbl}\n\n"
                f"---\n"
                f"### 📢 Prioritized Safety Actions\n"
                f"1. {'Immediately inspect active alarms and prepare evacuation if gas levels rise.' if has_crit else 'Maintain baseline monitoring and normal mine operating procedures.'}\n"
                f"2. Inquire about any specific sensor, regulation limit (OSHA/NIOSH/IS), or emergency precaution."
            )
            self._response_cache[cache_key] = fast_reply
            return fast_reply

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

                learned_rules = get_self_learned_rules(graph, segment_id)
                if learned_rules:
                    ekg_context += f" EKG Self-Learned Rules ({len(learned_rules)}): " + " | ".join(r.get("rule_text", "") for r in learned_rules[:2])
            except Exception as e:
                ekg_context = f"EKG lookup context: {e}"
        else:
            ekg_context = f"EKG context for segment {segment_id} is unavailable."

        # 2. Compare raw readings with explicit OSHA/NIOSH/industry limits
        model_predictions = model_predictions or {}
        try:
            assessment = self.protocol_evaluator.assess(
                readings=sensor_readings or active_anomalies,
                predictions=model_predictions,
                top_k=2,
            )
        except Exception as exc:
            assessment = SafetyProtocolEvaluator().assess(
                readings=sensor_readings or active_anomalies,
                predictions=model_predictions,
                top_k=2,
            )
            assessment.rag_context = f"FAISS retrieval unavailable: {exc}"
        rag_context = assessment.rag_context or "FAISS safety guidelines index is unavailable."
        protocol_report = assessment.format_report()

        # 3. Format Structured ChatML Prompt
        system_instructions = (
            "You are FIELD-MIND, an expert on-device underground mining safety assistant deployed on NVIDIA Jetson.\n"
            "Your role is to protect miners and autonomous equipment by analyzing real-time sensor telemetry, "
            "evaluating ML model predictions against regulatory standards (OSHA, NIOSH, IS 6922, AS 4024), "
            "and providing clear, detailed, and prioritized safety recommendations.\n\n"
            "Formatting Rules:\n"
            "1. ALWAYS start with the header '### ❓ Asked Question' followed by the user question in a blockquote.\n"
            "2. Present all safety protocols, real-time sensor values, and regulatory checks in a Markdown Table.\n"
            "3. Conclude with at most 3 concise, numbered prioritized actions (1, 2, 3) with NO nested sub-bullets.\n"
            "4. Use clear colors and emojis (🚨 CRITICAL, ⚠ WARNING, ✔ SAFE) to segregate portions."
        )

        user_content = (
            f"LOCATION / SECTOR: {segment_id}\n\n"
            f"--- [ACTIVE SENSOR TELEMETRY] ---\n"
            f"Active Anomalies: {active_anomalies}\n"
            f"All Sensor Readings: {sensor_readings or active_anomalies}\n"
            f"Model Predictions: {model_predictions}\n\n"
            f"--- [SAFETY PROTOCOL & REGULATORY CHECKS] ---\n"
            f"{protocol_report}\n\n"
            f"--- [MULTI-NODE HISTORICAL TREND] ---\n"
            f"{trend_context or 'No historical multi-node trend recorded.'}\n\n"
            f"--- [RETRIEVED LITERATURE & GUIDELINES (FAISS RAG)] ---\n"
            f"{rag_context}\n\n"
            f"--- [EXPEDITION KNOWLEDGE GRAPH (EKG MEMORY)] ---\n"
            f"{ekg_context}\n\n"
            f"--- [OPERATOR QUESTION] ---\n"
            f"{user_message}\n\n"
            "Please provide a structured response answering the question, including the asked question at the top, a regulatory checks table, and max 3 prioritized actions."
        )

        prompt = self.llm_runner.format_chat_prompt(system_instructions, user_content)

        # 4. Run through model generator or fallback engine
        response = self.llm_runner.run_reasoning(
            prompt=prompt,
            anomalies=active_anomalies,
            ekg_history=ekg_context,
            rag_context=rag_context,
            task_type="chat",
            max_tokens=min(512, max(256, self.llm_runner.n_ctx // 2))
        )

        if not response or response in {"FIELD-MIND active. Safety regulations and model predictions evaluated.", "Unknown task type."}:
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

        # Guarantee Asked Question is present at top
        if "### ❓ Asked Question" not in response:
            response = f"### ❓ Asked Question\n> **{user_message}**\n\n---\n" + response

        self._response_cache[cache_key] = response
        return response

    def chat_stream(
        self,
        user_message: str,
        segment_id: str,
        active_anomalies: Dict[str, Any],
        model_predictions: Optional[Dict[str, Any]] = None,
        sensor_readings: Optional[Dict[str, Any]] = None,
        trend_context: Optional[str] = None,
    ):
        """
        Streams chat response tokens/blocks to achieve low latency (TTFT < 300ms).
        """
        cache_key = f"{user_message.strip().lower()}_{segment_id}_{hash(str(sensor_readings or active_anomalies))}"
        if cache_key in self._response_cache:
            yield self._response_cache[cache_key]
            return

        clean_q = user_message.strip().lower()
        if clean_q in {"hi", "hello", "hey", "status", "help"}:
            readings = sensor_readings or active_anomalies
            tbl = self._build_telemetry_markdown_table(readings)
            has_crit = any(float(readings.get(k, 0)) > limit for k, limit in [("MQ4_CH4_ppm", 5000), ("MQ7_CO_ppm", 50), ("vibration_pulses", 10)])
            status_summary = "🚨 CRITICAL HAZARD" if has_crit else "✔ SYSTEM NOMINAL"
            fast_reply = (
                f"### ❓ Asked Question\n> **{user_message}**\n\n"
                f"---\n"
                f"### 🛡️ FIELD-MIND Quick Status — Sector {segment_id}\n"
                f"Overall Status: **{status_summary}**\n\n"
                f"{tbl}\n\n"
                f"---\n"
                f"### 📢 Prioritized Safety Actions\n"
                f"1. {'Immediately inspect active alarms and prepare evacuation if gas levels rise.' if has_crit else 'Maintain baseline monitoring and normal mine operating procedures.'}\n"
                f"2. Inquire about any specific sensor, regulation limit (OSHA/NIOSH/IS), or emergency precaution."
            )
            self._response_cache[cache_key] = fast_reply
            yield fast_reply
            return

        # Fetch EKG and Protocol Assessment
        ekg_context = ""
        if EKG_AVAILABLE and os.path.exists(self.ekg_json_path):
            try:
                graph = MineKnowledgeGraph()
                graph.load(self.ekg_json_path)
                self.core._ensure_tunnel_segment(graph, segment_id)
                profile = get_segment_risk_profile(graph, segment_id)
                ekg_context = f"EKG context for segment {segment_id}: active hazards={profile.get('hazard_count', 0)}."
            except Exception:
                ekg_context = "EKG memory loaded."

        model_predictions = model_predictions or {}
        try:
            assessment = self.protocol_evaluator.assess(
                readings=sensor_readings or active_anomalies,
                predictions=model_predictions,
                top_k=2,
            )
        except Exception:
            assessment = SafetyProtocolEvaluator().assess(readings=sensor_readings or active_anomalies)

        # If LLM model is available, stream tokens
        if self.llm_runner.ensure_loaded():
            system_instructions = (
                "You are FIELD-MIND, an expert underground mining safety assistant.\n"
                "ALWAYS start with '### ❓ Asked Question\\n> **<question>**'.\n"
                "Include a Markdown Table for real-time sensor values & regulatory limits.\n"
                "End with max 3 numbered prioritized safety actions."
            )
            user_content = f"LOCATION: {segment_id}\nREADINGS: {sensor_readings or active_anomalies}\nQUESTION: {user_message}"
            prompt = self.llm_runner.format_chat_prompt(system_instructions, user_content)
            
            first_chunk = True
            for chunk in self.llm_runner.stream_reasoning(prompt, active_anomalies, ekg_context, assessment.rag_context):
                if first_chunk and "### ❓ Asked Question" not in chunk:
                    yield f"### ❓ Asked Question\n> **{user_message}**\n\n---\n"
                    first_chunk = False
                yield chunk
            return

        # Fallback structured response
        full_res = self._build_conversational_response(
            user_message=user_message,
            segment_id=segment_id,
            active_anomalies=active_anomalies,
            ekg_context=ekg_context,
            rag_context=assessment.rag_context,
            assessment=assessment,
            sensor_readings=sensor_readings or active_anomalies,
            model_predictions=model_predictions,
            trend_context=trend_context,
        )
        self._response_cache[cache_key] = full_res
        yield full_res

    @staticmethod
    def _build_telemetry_markdown_table(readings: Dict[str, Any], checks: Optional[List[Any]] = None) -> str:
        """Constructs a presentable custom TUI table for sensor telemetry and regulatory checks."""
        lines = [
            "### 📊 Real-Time Telemetry & Regulatory Protocol Checks",
            "| Metric / Sensor Stream | Real-Time Value | Regulatory Standard / Limit | Safety Check Status |",
            "| :--- | :---: | :---: | :---: |"
        ]
        if checks:
            for check in checks:
                status_badge = "🚨 CRITICAL" if check.severity == "CRITICAL" else ("⚠ WARNING" if check.severity == "WARNING" else "✔ SAFE")
                lines.append(f"| {check.metric.title()} ({check.domain.upper()}) | {check.reading:g} {check.unit} | {check.protocol_limit} | {status_badge} ({check.status}) |")
        else:
            ch4 = float(readings.get("MQ4_CH4_ppm", 0))
            ch4_stat = "🚨 CRITICAL (Evacuate)" if ch4 > 5000 else ("⚠ WARNING (Elevated)" if ch4 > 1000 else "✔ SAFE (Nominal)")
            lines.append(f"| Methane (MQ-4 CH4) | {ch4:.1f} ppm | < 1,000 ppm (OSHA PEL) | {ch4_stat} |")

            co = float(readings.get("MQ7_CO_ppm", 0))
            co_stat = "🚨 DANGER (Toxic)" if co > 50 else ("⚠ WARNING (Elevated)" if co > 25 else "✔ SAFE (Nominal)")
            lines.append(f"| Carbon Monoxide (MQ-7 CO) | {co:.1f} ppm | < 25 ppm (Post-blast entry) | {co_stat} |")

            temp = float(readings.get("temp", 22.0))
            t_stat = "⚠ CAUTION (Heat Stress)" if temp > 28 else "✔ SAFE (Nominal)"
            lines.append(f"| Ambient Temperature | {temp:.1f} °C | 18 – 28 °C (Safe Thermal Zone) | {t_stat} |")

            hum = float(readings.get("humidity", 55.0))
            h_stat = "⚠ HIGH RH (Condensation Risk)" if hum > 85 else "✔ SAFE (Nominal)"
            lines.append(f"| Relative Humidity | {hum:.1f} % | 15 – 85 % RH | {h_stat} |")

            vib = float(readings.get("vibration_pulses", 0.0))
            v_stat = "🚨 CRITICAL (Level 2 Shock)" if vib > 10 else "✔ SAFE (Baseline)"
            lines.append(f"| Shock Pulses (SW-420) | {vib:.0f} p/s | < 5 pulses/s | {v_stat} |")

            dist = float(readings.get("min_distance", 2.5))
            d_stat = "🚨 STOP ALARM (< 0.3m)" if dist < 0.3 else "✔ SAFE (Clear)"
            lines.append(f"| Obstacle Proximity | {dist:.2f} m | > 0.50 m (AS 4024 Clearance) | {d_stat} |")

        return "\n".join(lines)

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
        Fallback conversational response builder with structured TUI table, colors, and concise actions.
        """
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

        readings = sensor_readings or active_anomalies
        lines = [
            f"### ❓ Asked Question",
            f"> **{user_message}**",
            "",
            "---",
            f"### 🛡️ FIELD-MIND Safety Assessment — Sector {segment_id}",
            "",
            "### 📊 Real-Time Telemetry & Regulatory Protocol Checks",
            "| Metric / Sensor Stream | Real-Time Value | Regulatory Standard / Limit | Safety Status |",
            "| :--- | :---: | :---: | :---: |"
        ]

        ch4 = float(readings.get("MQ4_CH4_ppm", 0))
        ch4_stat = "🚨 CRITICAL (Evacuate)" if ch4 > 5000 else ("⚠ WARNING (Elevated)" if ch4 > 1000 else "✔ NOMINAL (Safe)")
        lines.append(f"| Methane (MQ-4 CH4) | {ch4:.1f} ppm | < 1,000 ppm (OSHA PEL) | {ch4_stat} |")

        co = float(readings.get("MQ7_CO_ppm", 0))
        co_stat = "🚨 DANGER (Toxic)" if co > 50 else ("⚠ WARNING (Elevated)" if co > 25 else "✔ NOMINAL (Safe)")
        lines.append(f"| Carbon Monoxide (MQ-7 CO) | {co:.1f} ppm | < 25 ppm (Post-blast entry) | {co_stat} |")

        temp = float(readings.get("temp", 22.0))
        t_stat = "⚠ CAUTION (Heat Stress)" if temp > 28 else "✔ NOMINAL (Safe)"
        lines.append(f"| Ambient Temperature | {temp:.1f} °C | 18 – 28 °C (Safe Thermal Zone) | {t_stat} |")

        hum = float(readings.get("humidity", 55.0))
        h_stat = "⚠ HIGH RH (Condensation Risk)" if hum > 85 else "✔ NOMINAL (Safe)"
        lines.append(f"| Relative Humidity | {hum:.1f} % | 15 – 85 % RH | {h_stat} |")

        vib = float(readings.get("vibration_pulses", 0.0))
        v_stat = "🚨 CRITICAL (Level 2 Shock)" if vib > 10 else "✔ NOMINAL (Baseline)"
        lines.append(f"| Shock Pulses (SW-420) | {vib:.0f} p/s | < 5 pulses/s | {v_stat} |")

        dist = float(readings.get("min_distance", 2.5))
        d_stat = "🚨 STOP ALARM (< 0.3m)" if dist < 0.3 else "✔ NOMINAL (Clear)"
        lines.append(f"| Obstacle Proximity | {dist:.2f} m | > 0.50 m (AS 4024 Clearance) | {d_stat} |")

        lines.extend(["", "---", "### 📢 Prioritized Safety Actions"])
        actions = []
        if ch4 > 5000:
            actions.append("EVACUATE tunnel heading immediately and power down non-explosion-proof electrical grids.")
            actions.append("Override and max out auxiliary exhaust ventilation fans to dilute combustible CH4.")
        elif co > 25:
            actions.append("Enforce 30-minute post-blast safety air dilution timer before allowing worker re-entry.")
        if dist < 0.3:
            actions.append("Halt autonomous vehicle movement immediately and execute emergency reverse clearance.")
        if not actions:
            actions.append("Maintain continuous automated baseline monitoring and standard ventilation protocols.")

        # Limit to max 3 concise main actions without subpoints
        for idx, act in enumerate(actions[:3], 1):
            lines.append(f"{idx}. {act}")

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
        """Render consistent, structured response with Q&A header, custom TUI table, colors, and max 3 concise actions."""
        status_color = "🚨 CRITICAL" if assessment.overall_status == "CRITICAL" else ("⚠ WARNING" if assessment.overall_status in {"WARNING", "REVIEW_MODEL_DISAGREEMENT"} else "✔ SAFE")
        
        lines = [
            f"### ❓ Asked Question",
            f"> **{user_message}**",
            "",
            "---",
            f"### 🛡️ FIELD-MIND Safety Assessment — Sector {segment_id}",
            f"Overall Status: **{status_color}** (`{assessment.overall_status}`)",
            "",
            "### 📊 Real-Time Telemetry & Regulatory Protocol Checks",
            "| Metric / Sensor Stream | Real-Time Value | Regulatory Standard / Limit | Safety Check Status |",
            "| :--- | :---: | :---: | :---: |"
        ]

        # Add protocol checks to the TUI table
        if assessment.checks:
            for check in assessment.checks:
                status_badge = "🚨 CRITICAL" if check.severity == "CRITICAL" else ("⚠ WARNING" if check.severity == "WARNING" else "✔ SAFE")
                lines.append(f"| {check.metric.title()} ({check.domain.upper()}) | {check.reading:g} {check.unit} | {check.protocol_limit} | {status_badge} ({check.status}) |")
        else:
            lines.append("| Baseline Sensors | Nominal | OSHA / NIOSH Baseline | ✔ SAFE |")

        # Supplement with key telemetry if missing from checks
        readings = sensor_readings or {}
        if "MQ4_CH4_ppm" in readings and not any("ch4" in c.metric.lower() for c in assessment.checks):
            ch4_val = float(readings["MQ4_CH4_ppm"])
            lines.append(f"| Methane (MQ-4 CH4) | {ch4_val:.1f} ppm | < 1,000 ppm (OSHA PEL) | {'🚨 CRITICAL' if ch4_val > 5000 else '✔ SAFE'} |")
        if "temp" in readings and not any("temp" in c.metric.lower() for c in assessment.checks):
            t_val = float(readings["temp"])
            lines.append(f"| Ambient Temperature | {t_val:.1f} °C | 18 – 28 °C | {'⚠ CAUTION' if t_val > 28 else '✔ SAFE'} |")

        lines.extend(["", "---"])

        if trend_context:
            lines.extend(["### 📈 Multi-Node Trend", trend_context, "", "---"])

        # Prioritized actions: Cap at top 3 main actions, deduplicated, no subpoints
        lines.append("### 📢 Prioritized Safety Actions")
        raw_actions = list(dict.fromkeys(assessment.actions)) if assessment.actions else []
        if not raw_actions:
            raw_actions.append("Maintain baseline monitoring and normal mine operating procedures.")

        top_actions = raw_actions[:3]
        for index, action in enumerate(top_actions, 1):
            lines.append(f"{index}. {action}")

        lines.extend(["", "---"])

        if ekg_context and "unavailable" not in ekg_context.lower():
            lines.extend([f"### 🧠 Expedition Knowledge Graph (EKG Memory)", ekg_context, ""])

        return "\n".join(lines)
