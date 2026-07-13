"""
llm_runner.py — Offline Inference and Expert Rule Reasoner
===========================================================
Manages model inference. If llama-cpp-python is available and a valid GGUF 
model is provided, runs a local quantized LLM. Otherwise, falls back to a 
high-quality, domain-informed Expert Rule Engine that matches anomalies, 
EKG history, and RAG context to formulate reasoning hypotheses and safety advices.
"""

import os
import re
from typing import Any, Dict, List, Optional


class OfflineLLMRunner:
    """
    Runner that executes on-device reasoning using a local quantized LLM,
    with an embedded expert-system fallback.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._llm = None
        self._initialized = False

        if model_path and os.path.exists(model_path):
            try:
                from llama_cpp import Llama
                print(f"  [LLMRunner] Initializing Llama model from {model_path} ...")
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                self._initialized = True
                print("  [LLMRunner] ✓ Quantized Llama model loaded.")
            except Exception as e:
                print(f"  [LLMRunner] ⚠ Failed to load GGUF model: {e}. Fallback to Expert System enabled.")
        else:
            print("  [LLMRunner] No valid GGUF model path provided. Defaulting to Expert System.")

    def run_reasoning(
        self,
        prompt: str,
        anomalies: Dict[str, Any],
        ekg_history: str,
        rag_context: str,
        task_type: str = "hypothesis"
    ) -> str:
        """
        Executes reasoning using the loaded LLM or falls back to the Expert Rule Engine.
        """
        if self._initialized and self._llm:
            try:
                # LLM execution
                response = self._llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.2,
                    stop=["\n\n\n", "User:", "System:"],
                )
                text = response["choices"][0]["text"].strip()
                return text
            except Exception as e:
                print(f"  [LLMRunner] LLM inference failed: {e}. Falling back to Expert System.")

        # Expert System Fallback
        return self._expert_fallback(anomalies, ekg_history, rag_context, task_type)

    def _expert_fallback(
        self,
        anomalies: Dict[str, Any],
        ekg_history: str,
        rag_context: str,
        task_type: str
    ) -> str:
        """
        An offline expert system ruleset that generates mining hypotheses and safety suggestions.
        """
        # Parse active anomalies
        gas_alert = any(k for k, v in anomalies.items() if "gas" in k.lower() or "ppm" in k.lower())
        vibration_alert = any(k for k, v in anomalies.items() if "vib" in k.lower() or "ppv" in k.lower())
        env_alert = any(k for k, v in anomalies.items() if "temp" in k.lower() or "humid" in k.lower())
        nav_alert = any(k for k, v in anomalies.items() if "us" in k.lower() or "steering" in k.lower() or "collision" in k.lower())

        # Check thresholds
        max_methane = float(anomalies.get("MQ4_CH4_ppm", 0))
        max_co = float(anomalies.get("MQ7_CO_ppm", 0))
        max_ppv = float(anomalies.get("predicted_ppv", 0.0) or anomalies.get("ppv", 0.0))
        min_dist = float(anomalies.get("min_distance", 5.0))
        temp = float(anomalies.get("temp", 20.0))

        # Check blast correlation in EKG history
        blast_correlated = "blast" in ekg_history.lower() or "caused_by" in ekg_history.lower()

        if task_type == "hypothesis":
            hypotheses = []
            if gas_alert:
                if max_methane > 10000:
                    hypotheses.append(
                        f"CRITICAL Methane build-up detected ({max_methane:.1f} ppm). "
                        "Likely caused by ventilation pocket failure or sudden gas pocket release in the heading."
                    )
                elif max_co > 50:
                    hypotheses.append(
                        f"Elevated Carbon Monoxide ({max_co:.1f} ppm) detected. "
                        + ("This correlates with recent blasting activity reported in the segment history." if blast_correlated else "Likely due to diesel engine emissions or slow coal oxidation (spontaneous combustion).")
                    )
                else:
                    hypotheses.append("General gas concentration levels rising, indicating reduced ventilation flow.")

            if vibration_alert:
                if max_ppv > 5.0:
                    hypotheses.append(
                        f"High peak particle velocity ({max_ppv:.2f} mm/s) registered. "
                        + ("Directly triggered by recent detonator blasting." if blast_correlated else "Unexplained seismic vibration, indicating potential rockburst or structural shifts.")
                    )
                else:
                    hypotheses.append(f"Minor seismic vibration detected ({max_ppv:.2f} mm/s).")

            if env_alert:
                if temp > 35.0:
                    hypotheses.append(f"Critical ambient temperature anomaly ({temp:.1f}°C), suggesting a cooling system failure or localized fire.")
                else:
                    hypotheses.append("Environmental comfort indices deviated from normal baseline monitoring parameters.")

            if nav_alert:
                if min_dist < 0.3:
                    hypotheses.append(f"Robot proximity alert (US minimum distance={min_dist:.2f}m). A wall or physical obstacle is blocking the path.")

            if not hypotheses:
                return "Baseline nominal monitoring. Anomaly readings are near normal levels."

            return " | ".join(hypotheses)

        elif task_type == "suggestions":
            advices = []
            if gas_alert:
                if max_methane > 10000:
                    advices.append("IMMEDIATELY EVACUATE all personnel from the affected segment.")
                    advices.append("Activate secondary auxiliary ventilation fans to dilute combustible concentrations.")
                    advices.append("Power down all non-explosion-proof electrical equipment in the sector.")
                elif max_co > 50:
                    advices.append("Initiate post-blast dilution wait timer (minimum 30 minutes).")
                    advices.append("Ensure air velocity at the face is above 0.5 m/s before allowing re-entry.")

            if vibration_alert:
                if max_ppv > 12.5:
                    advices.append("Inspect tunnel roof structures for damage, new cracks, or rock spalling.")
                    advices.append("Halt all subsequent blasting operations in adjacent segments until geological inspection completes.")
                else:
                    advices.append("Ensure seismometer arrays are calibrated and logged to graph memory.")

            if env_alert:
                if temp > 30.0:
                    advices.append("Increase chilled air refrigeration flow to the working face.")
                    advices.append("Enforce worker hydration cycles (15-min rest per hour in thermal caution zones).")

            if nav_alert:
                if min_dist < 0.5:
                    advices.append("Command robotic platform to execute immediate reverse/backup maneuvers.")
                    advices.append("Recalibrate autonomous navigation steering coefficients.")

            if not advices:
                advices.append("Maintain continuous low-power monitoring and standard reporting.")

            return "\n".join([f"- {adv}" for adv in advices])

        return "Unknown task type."
