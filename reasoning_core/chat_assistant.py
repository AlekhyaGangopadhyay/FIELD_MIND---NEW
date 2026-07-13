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

    def chat(
        self,
        user_message: str,
        segment_id: str,
        active_anomalies: Dict[str, Any]
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

        # 2. Fetch FAISS RAG safety guidelines
        rag_context = ""
        if self.rag_retriever:
            try:
                # Retrieve using terms in the user query + active anomalies
                search_query = f"{user_message} " + " ".join(active_anomalies.keys())
                results = self.rag_retriever.retrieve(search_query, top_k=2)
                rag_context = self.rag_retriever.format_context(results)
            except Exception as e:
                rag_context = f"FAISS safety regulations retrieval error: {e}"
        else:
            rag_context = "FAISS safety guidelines index is unavailable."

        # 3. Format Prompt
        prompt = (
            "You are FIELD-MIND, an expert offline underground mining safety assistant.\n"
            "Your job is to communicate clearly with the operator, analyse data inputs, "
            "and suggest safety measures based on regulations.\n\n"
            f"LOCATION SEGMENT: {segment_id}\n"
            f"ACTIVE DATA INPUTS (anomalies/readings): {active_anomalies}\n"
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
                user_message, segment_id, active_anomalies, ekg_context, rag_context
            )

        return response

    def _build_conversational_response(
        self,
        user_message: str,
        segment_id: str,
        active_anomalies: Dict[str, Any],
        ekg_context: str,
        rag_context: str
    ) -> str:
        """
        Fallback conversational response builder analyzing data inputs and safety measures.
        """
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
