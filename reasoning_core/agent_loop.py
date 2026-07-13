"""
agent_loop.py — LangGraph Workflow Definition
==============================================
Builds the LangGraph state machine compiling the multi-step diagnostic reasoning loop:
Observe → EKG-Retrieve → RAG-Retrieve → Hypothesize → Suggest → Update-EKG
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from .state import AgentState
from .llm_runner import OfflineLLMRunner

# Safe imports for EKG and RAG
from faiss_rag.retriever import RAGRetriever

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


class ScientificReasoningCore:
    """
    Coordinates offline multimodal diagnostic reasoning via a LangGraph state machine.
    """

    def __init__(
        self,
        workspace_root: str,
        model_path: Optional[str] = None,
        faiss_dir: Optional[str] = None,
        ekg_json_path: Optional[str] = None
    ):
        self.workspace_root = workspace_root
        self.llm_runner = OfflineLLMRunner(model_path)
        self.ekg_json_path = ekg_json_path or os.path.join(workspace_root, "expedition_knowledge_graph", "data", "mine_graph.json")
        
        # Load FAISS Retriever
        faiss_save_dir = faiss_dir or os.path.join(workspace_root, "faiss_rag", "data")
        index_path = os.path.join(faiss_save_dir, "faiss_index.bin")
        meta_path = os.path.join(faiss_save_dir, "chunks_metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.rag_retriever = RAGRetriever(index_path, meta_path)
            print("  [ReasoningCore] ✓ FAISS RAG Retriever loaded.")
        else:
            self.rag_retriever = None
            print("  [ReasoningCore] ⚠ FAISS index files not found. RAG retriever runs in fallback mode.")

        # Build Graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("observe", self._node_observe)
        builder.add_node("ekg_retrieve", self._node_ekg_retrieve)
        builder.add_node("rag_retrieve", self._node_rag_retrieve)
        builder.add_node("hypothesize", self._node_hypothesize)
        builder.add_node("suggest", self._node_suggest)
        builder.add_node("update_ekg", self._node_update_ekg)

        # Set edges
        builder.set_entry_point("observe")
        builder.add_edge("observe", "ekg_retrieve")
        builder.add_edge("ekg_retrieve", "rag_retrieve")
        builder.add_edge("rag_retrieve", "hypothesize")
        builder.add_edge("hypothesize", "suggest")
        builder.add_edge("suggest", "update_ekg")
        builder.add_edge("update_ekg", END)

        return builder.compile()

    # -----------------------------------------------------------------------
    # Node Implementation
    # -----------------------------------------------------------------------

    def _node_observe(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append("Step 1: OBSERVE node started.")
        
        anomalies = state.get("anomalies", {})
        findings = []
        for key, val in anomalies.items():
            findings.append(f"{key}={val}")
            
        trace.append(f"Observed anomalies parameters: {', '.join(findings)}")
        return {
            "trace": trace,
            "status": "SUCCESS"
        }

    def _ensure_tunnel_segment(self, graph: MineKnowledgeGraph, segment_id: str) -> None:
        """Helper to ensure a segment node exists in EKG so query/updates don't raise key errors."""
        if graph.get_node(segment_id) is None:
            graph.add_node(segment_id, "TunnelSegment", {
                "segment_id": segment_id,
                "name": f"Tunnel Section {segment_id.replace('TUNNEL_', '')}",
                "depth_m": 150.0,
                "status": "active"
            })

    def _node_ekg_retrieve(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append("Step 2: EKG-RETRIEVE node started.")
        
        segment_id = state.get("segment_id", "TUNNEL_A1")
        ekg_history = ""

        if EKG_AVAILABLE and os.path.exists(self.ekg_json_path):
            try:
                graph = MineKnowledgeGraph()
                graph.load(self.ekg_json_path)
                
                # Ensure the segment node exists in graph to prevent lookup errors
                self._ensure_tunnel_segment(graph, segment_id)
                
                # Fetch segment risk profile
                profile = get_segment_risk_profile(graph, segment_id)
                history_lines = [
                    f"EKG segment {segment_id} risk profile: active hazards count={profile.get('hazard_count', 0)}",
                    f"Gas anomalies: {profile.get('gas_anomalies_count', 0)}, Vibration events: {profile.get('vibration_events_count', 0)}"
                ]
                
                # Check for recent blast event history
                blasts = get_blast_history(graph, segment_id)
                if blasts:
                    history_lines.append(f"Recent blast records: {len(blasts)} blasts recorded in this sector.")
                    
                ekg_history = "\n".join(history_lines)
                trace.append("Retrieved historical events from EKG successfully.")
            except Exception as e:
                ekg_history = f"Error reading EKG: {e}"
                trace.append(f"Warning: EKG lookup encountered an error: {e}")
        else:
            ekg_history = f"EKG file not found or EKG module unavailable. Location segment={segment_id}."
            trace.append("EKG file not found; running with localized EKG fallback context.")

        return {
            "ekg_history": ekg_history,
            "trace": trace
        }

    def _node_rag_retrieve(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append("Step 3: RAG-RETRIEVE node started.")
        
        anomalies = state.get("anomalies", {})
        queries = []
        if any(k for k in anomalies if "gas" in k.lower() or "ppm" in k.lower()):
            queries.append("methane carbon monoxide safety limit threshold")
        if any(k for k in anomalies if "vib" in k.lower() or "ppv" in k.lower()):
            queries.append("blast vibration PPV peak particle velocity safety limit")
        if any(k for k in anomalies if "temp" in k.lower() or "humid" in k.lower()):
            queries.append("OSHA wet bulb temperature comfort index heat stress")
        if any(k for k in anomalies if "us" in k.lower() or "steering" in k.lower() or "collision" in k.lower()):
            queries.append("robot navigation collision proximity alert steering")
            
        if not queries:
            queries.append("underground mining safety general rules")

        rag_context = ""
        if self.rag_retriever:
            try:
                combined_chunks = []
                for query in queries:
                    res = self.rag_retriever.retrieve(query, top_k=2)
                    combined_chunks.extend(res)
                # Format
                rag_context = self.rag_retriever.format_context(combined_chunks)
                trace.append("Successfully retrieved literature context from FAISS.")
            except Exception as e:
                rag_context = f"Error querying RAG: {e}"
                trace.append(f"Warning: RAG lookup failed: {e}")
        else:
            rag_context = "FAISS index not loaded. Falling back to rule thresholds."
            trace.append("FAISS index unavailable; running with baseline threshold limits.")

        return {
            "rag_context": rag_context,
            "trace": trace
        }

    def _node_hypothesize(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append("Step 4: HYPOTHESIZE node started.")
        
        anomalies = state.get("anomalies", {})
        ekg_history = state.get("ekg_history", "")
        rag_context = state.get("rag_context", "")

        # Format Prompt
        prompt = (
            "You are an on-device safety reasoning agent in an underground mine.\n"
            f"Active Anomalies: {anomalies}\n"
            f"EKG History:\n{ekg_history}\n"
            f"RAG Safety Regulations:\n{rag_context}\n\n"
            "Based on the inputs above, formulate a clear, concise hypothesis explaining "
            "the root cause of these anomalies."
        )

        hypothesis = self.llm_runner.run_reasoning(
            prompt=prompt,
            anomalies=anomalies,
            ekg_history=ekg_history,
            rag_context=rag_context,
            task_type="hypothesis"
        )
        
        trace.append(f"Hypothesis generated: {hypothesis}")
        return {
            "hypothesis": hypothesis,
            "trace": trace
        }

    def _node_suggest(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append("Step 5: SUGGEST node started.")
        
        anomalies = state.get("anomalies", {})
        ekg_history = state.get("ekg_history", "")
        rag_context = state.get("rag_context", "")
        hypothesis = state.get("hypothesis", "")

        # Format Prompt
        prompt = (
            "You are an on-device safety reasoning agent in an underground mine.\n"
            f"Active Anomalies: {anomalies}\n"
            f"Hypothesis: {hypothesis}\n"
            f"RAG Safety Regulations:\n{rag_context}\n\n"
            "Generate a list of prioritized, actionable safety suggestions to resolve this threat."
        )

        raw_suggestions = self.llm_runner.run_reasoning(
            prompt=prompt,
            anomalies=anomalies,
            ekg_history=ekg_history,
            rag_context=rag_context,
            task_type="suggestions"
        )

        # Parse suggestions into list
        suggestions = [s.strip("- *").strip() for s in raw_suggestions.split("\n") if s.strip()]
        
        trace.append(f"Generated {len(suggestions)} safety recommendations.")
        return {
            "suggestions": suggestions,
            "trace": trace
        }

    def _node_update_ekg(self, state: AgentState) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append("Step 6: UPDATE_EKG node started.")
        
        segment_id = state.get("segment_id", "TUNNEL_A1")
        hypothesis = state.get("hypothesis", "")
        suggestions = state.get("suggestions", [])

        if EKG_AVAILABLE and os.path.exists(self.ekg_json_path):
            try:
                graph = MineKnowledgeGraph()
                graph.load(self.ekg_json_path)
                
                # Ensure the segment node exists
                self._ensure_tunnel_segment(graph, segment_id)
                
                # Add reasoning event node to graph
                node_id = f"reasoning_event_{int(time.time() * 1000)}"
                graph.add_node(node_id, label="ReasoningResolution", properties={
                    "timestamp": time.time(),
                    "hypothesis": hypothesis,
                    "suggestions": "; ".join(suggestions),
                    "segment_id": segment_id
                })
                
                # Link to TunnelSegment in NetworkX DiGraph (graph.G)
                if segment_id in graph.G:
                    graph.add_edge(node_id, segment_id, rel_type="RESOLVED_FOR")
                
                graph.save(self.ekg_json_path)
                trace.append("Saved reasoning resolution node to Expedition Knowledge Graph successfully.")
            except Exception as e:
                trace.append(f"Warning: EKG update encountered an error: {e}")
        else:
            trace.append("Skipped EKG graph update (graph files not initialized or module missing).")

        trace.append("Workflow completed successfully.")
        return {
            "trace": trace,
            "status": "SUCCESS"
        }

    # -----------------------------------------------------------------------
    # Public Inference Hook
    # -----------------------------------------------------------------------

    def reason(self, anomalies: Dict[str, Any], segment_id: str) -> Dict[str, Any]:
        """
        Execute the complete LangGraph reasoning workflow.
        """
        initial_state: AgentState = {
            "anomalies": anomalies,
            "segment_id": segment_id,
            "ekg_history": "",
            "rag_context": "",
            "hypothesis": "",
            "suggestions": [],
            "trace": [],
            "status": "PENDING"
        }

        # Run StateGraph workflow
        return self.workflow.invoke(initial_state)
