"""
llm_runner.py — Offline Inference and Expert Rule Reasoner
===========================================================
Manages model inference. If llama-cpp-python is available and a valid GGUF 
model is provided, runs a local quantized LLM. Otherwise, falls back to a 
high-quality, domain-informed Expert Rule Engine that matches anomalies, 
EKG history, and RAG context to formulate reasoning hypotheses and safety advices.
"""

import gc
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class OfflineLLMRunner:
    """
    Runner that executes on-device reasoning using a local quantized LLM
    (e.g., Qwen2.5-7B-Instruct-Q4_K_M.gguf on NVIDIA Jetson Orin Nano 8GB),
    with an embedded expert-system fallback and self-reflection engine.
    """

    def __init__(self, model_path: Optional[str] = None, workspace_root: Optional[str] = None, lazy_load: bool = True):
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path(__file__).resolve().parents[1]
        if not model_path:
            # Candidate GGUF model paths on Jetson Orin Nano 8GB
            candidates = [
                "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                "models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
                "models/Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break

        if model_path:
            candidate = Path(model_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.workspace_root / candidate
            model_path = str(candidate.resolve())
        else:
            candidates = [
                self.workspace_root / 'reasoning_core' / 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
                self.workspace_root / 'reasoning_core' / 'models' / 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
                self.workspace_root / 'gas_sensors' / 'models' / 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
                self.workspace_root / 'models' / 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
            ]
            model_path = next((str(c) for c in candidates if c.is_file()), None)
        self.model_path = model_path
        self._load_error = None
        self.n_ctx = max(512, int(os.getenv('FIELDMIND_LLM_CONTEXT', '2048')))
        self.n_threads = max(1, int(os.getenv('FIELDMIND_LLM_THREADS', '6')))
        self.n_gpu_layers = int(os.getenv('FIELDMIND_LLM_GPU_LAYERS', '-1'))
        self.lazy_load = lazy_load
        self._llm = None
        self._initialized = False

        if self.model_path and os.path.isfile(self.model_path) and not lazy_load:
            try:
                from llama_cpp import Llama
                print(f"  [LLMRunner] Initializing Qwen2.5-7B / 5B+ GGUF model on Jetson Orin Nano from {model_path} ...")
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )
                self._initialized = True
                print("  [LLMRunner] ✓ Quantized Qwen2.5-7B-Instruct model loaded with GPU acceleration.")
            except Exception as e:
                print(f"  [LLMRunner] ⚠ Failed to load GGUF model ({e}). Fallback to Expert System enabled.")
        elif not self.model_path:
            print("  [LLMRunner] No GGUF model found at 'reasoning_core/Qwen2.5-7B-Instruct-Q4_K_M.gguf'. Defaulting to Expert System & Reflection Engine.")

    def ensure_loaded(self) -> bool:
        if self._initialized and self._llm:
            return True
        if not self.model_path or not os.path.isfile(self.model_path):
            return False
        try:
            from llama_cpp import Llama
            print(f'  [LLMRunner] Loading local GGUF from {self.model_path} (ctx={self.n_ctx}, gpu_layers={self.n_gpu_layers}) ...')
            self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_threads=self.n_threads, n_gpu_layers=self.n_gpu_layers, n_batch=min(512, self.n_ctx), verbose=False)
            self._initialized = True
            self._load_error = None
        except Exception as exc:
            self._load_error = str(exc)
            self._llm = None
            self._initialized = False
            print(f'  [LLMRunner] GGUF load failed ({exc}); Expert System fallback remains active.')
        return self._initialized

    def unload(self) -> None:
        self._llm = None
        self._initialized = False
        gc.collect()

    def health(self) -> Dict[str, Any]:
        return {'model_path': self.model_path, 'available': bool(self.model_path and os.path.isfile(self.model_path)), 'loaded': self._initialized, 'load_error': self._load_error, 'n_ctx': self.n_ctx, 'n_gpu_layers': self.n_gpu_layers}

    def format_chat_prompt(self, system_msg: str, user_msg: str) -> str:
        """
        Formats system and user prompts using standard ChatML template (used by Qwen2.5)
        or fallback instruction tags.
        """
        return (
            f"<|im_start|>system\n{system_msg.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def run_reasoning(
        self,
        prompt: str,
        anomalies: Dict[str, Any],
        ekg_history: str,
        rag_context: str,
        task_type: str = "hypothesis",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Executes reasoning, self-reflection, or feasibility evaluation using the loaded LLM or falls back to the Expert System.
        """
        if self.ensure_loaded():
            try:
                # Dynamic token budget calculation: allow generous output length
                if max_tokens is None:
                    max_gen_tokens = min(1024, max(384, self.n_ctx // 2))
                else:
                    max_gen_tokens = max_tokens

                # ChatML & standard instruct stop sequences
                stop_sequences = [
                    "<|im_end|>",
                    "<|endoftext|>",
                    "<|im_start|>",
                    "### Human:",
                    "### User:",
                    "[User Query]",
                ]

                response = self._llm(
                    prompt,
                    max_tokens=max_gen_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    stop=stop_sequences,
                )
                text = response["choices"][0]["text"].strip()
                if text:
                    return text
            except Exception as e:
                print(f"  [LLMRunner] LLM inference failed: {e}. Falling back to Expert System.")

        # Expert System Fallback & Reflection Engine
        return self._expert_fallback(anomalies, ekg_history, rag_context, task_type)

    def _expert_fallback(
        self,
        anomalies: Dict[str, Any],
        ekg_history: str,
        rag_context: str,
        task_type: str
    ) -> str:
        """
        An offline expert system ruleset that generates mining hypotheses, safety suggestions,
        feasibility checks, and conversational safety answers.
        """
        # Parse active anomalies
        gas_alert = any(k for k, v in anomalies.items() if "gas" in k.lower() or "ppm" in k.lower())
        vibration_alert = any(k for k, v in anomalies.items() if "vib" in k.lower() or "ppv" in k.lower())
        env_alert = any(k for k, v in anomalies.items() if "temp" in k.lower() or "humid" in k.lower())
        nav_alert = any(k for k, v in anomalies.items() if "us" in k.lower() or "steering" in k.lower() or "collision" in k.lower())

        # Check thresholds
        max_methane = float(anomalies.get("MQ4_CH4_ppm", 0))
        max_co = float(anomalies.get("MQ7_CO_ppm", 0))
        humidity = float(anomalies.get("humidity", 50.0))
        max_ppv = float(anomalies.get("predicted_ppv", 0.0) or anomalies.get("ppv", 0.0))
        min_dist = float(anomalies.get("min_distance", 5.0))
        temp = float(anomalies.get("temp", 20.0))

        # Check blast correlation in EKG history
        blast_correlated = "blast" in ekg_history.lower() or "caused_by" in ekg_history.lower()

        if task_type == "feasibility":
            evaluations = []
            is_feasible = True
            
            # Check humidity condensation drift on gas sensors
            if max_co > 25.0 and humidity > 80.0 and not blast_correlated:
                evaluations.append(
                    f"UNFEASIBLE / DRIFT DISCREPANCY: Elevated CO reading ({max_co:.1f} ppm) detected at high humidity ({humidity:.1f}%). "
                    "In the absence of blasting in EKG history, this is physically unfeasible as a true combustion event and indicates moisture condensation drift on the MQ-7 sensor."
                )
                is_feasible = False
            elif max_co > 25.0 and blast_correlated:
                evaluations.append(
                    f"FEASIBLE: Elevated CO ({max_co:.1f} ppm) directly correlates with recorded blasting operations in EKG history."
                )
            
            if max_ppv > 5.0 and not blast_correlated:
                evaluations.append(
                    f"UNFEASIBLE / SEISMIC DISCREPANCY: High predicted PPV ({max_ppv:.2f} mm/s) registered without corresponding blast event in EKG history. "
                    "May indicate sensor calibration offset or local machinery vibration."
                )
                is_feasible = False
            elif max_ppv > 5.0 and blast_correlated:
                evaluations.append(
                    f"FEASIBLE: PPV vibration prediction ({max_ppv:.2f} mm/s) aligns with active detonator charges in this sector."
                )

            if min_dist < 0.5:
                evaluations.append(f"FEASIBLE: Ultrasonic wall clearance ({min_dist:.2f}m) indicates physical obstacle proximity.")

            if not evaluations:
                evaluations.append("FEASIBLE: Sensor inputs and model predictions match physical environmental baseline parameters.")

            status_prefix = "FEASIBILITY VERIFIED" if is_feasible else "PHYSICAL DISCREPANCY DETECTED"
            return f"[{status_prefix}]\n" + "\n".join(f"• {e}" for e in evaluations)

        elif task_type == "hypothesis":
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

            return "\n".join(f"✔ {a}" for a in advices)

        elif task_type == "reflection":
            explanation = str(anomalies.get("explanation", "Real-world ground truth differed from initial model prediction."))
            actual_situation = str(anomalies.get("actual_situation", "Verified physical situation."))
            rule = (
                f"When real-time telemetry indicates '{actual_situation}' despite conflicting model output, "
                f"apply corrective contextual rule: {explanation}. Override initial prediction and update experience replay memory."
            )
            return rule

        elif task_type == "chat":
            return "FIELD-MIND active. Safety regulations and model predictions evaluated."

        return "Standard operation."

