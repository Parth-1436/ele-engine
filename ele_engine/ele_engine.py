"""Empirical Linguistic Engine (ELE).

This module provides the ELEngine class, a pedagogical implementation of the
physics → physiology → linguistics → cognition → communication stack described
in the ELE documentation.  The goal of the implementation is not to be a
high-fidelity physical simulation but to capture the causal ordering, feedback
loops, and robust retry behaviour that the doctrine specifies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SimpleRLM:
    """A lightweight recurrent language model placeholder used by the ELE engine.

    The implementation is deliberately simple and uses deterministic math so that
    it can run in constrained environments while still demonstrating stateful
    linguistic processing.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        self.hidden_dim = hidden_dim

    def encode(self, text: str) -> List[int]:
        if not text:
            text = " "
        return [min(127, ord(ch)) for ch in text]

    def _step(self, hidden: List[float], code: int) -> None:
        base = (code % 97) * 0.01 + 0.001 * code
        for idx in range(self.hidden_dim):
            hidden[idx] = math.tanh(hidden[idx] + base + (idx * 0.0005))

    def run(self, text: str, raw_signal: Any, recursion_depth: int) -> Dict[str, Any]:
        token_ids = self.encode(text)
        hidden = [0.0 for _ in range(self.hidden_dim)]
        activation_trace: List[float] = []
        for code in token_ids:
            self._step(hidden, code)
            activation_trace.append(sum(hidden[:8]) / 8.0)

        phonemes = [chr(97 + (code % 26)) for code in token_ids]
        morphemes = [f"morpheme_{i}" for i in range(max(1, len(token_ids) // 2))]
        limit = max(1, min(3, len(token_ids)))
        lexemes = [f"lexeme_{token_ids[i] % 5}" for i in range(limit)]
        sememe = "SEM_" + text.upper().replace(" ", "_")
        mean_signal = sum(raw_signal) / max(1, len(raw_signal))
        semantic_intensity = float(np.clip(mean_signal * (1.0 + 0.1 * recursion_depth), 0.0, 1.0))

        return {
            "phonemes": phonemes,
            "morphemes": morphemes,
            "lexemes": lexemes,
            "sememes": sememe,
            "semantic_intensity": semantic_intensity,
            "rlm_hidden": activation_trace,
        }


@dataclass
class SensorimotorResult:
    trajectory: List[float]
    joint_angles: List[float]
    force_vector: Tuple[float, float, float]
    grasped: bool
    object_pos: Tuple[float, float, float]


class SensorimotorSimModule:
    """Very small kinematic simulator for grounding sememes."""

    def run(self, action: str, recursion_depth: int) -> SensorimotorResult:
        action = action.lower()
        if "grasp" in action:
            base_success = 0.9
            trajectory_scale = 1.0
        elif "manipulate" in action or "triangle" in action:
            base_success = 0.35
            trajectory_scale = 1.6
        else:
            base_success = 0.6
            trajectory_scale = 1.2

        base_success = float(np.clip(base_success + 0.05 * (recursion_depth - 1), 0.05, 0.95))
        draw = float(np.random.rand())
        success = draw < base_success

        joint_angles = [float(np.sin(i / 5.0) * trajectory_scale) for i in range(6)]
        trajectory = [float(trajectory_scale * (i + 1) / 6.0) for i in range(6)]
        if success:
            force = (0.8, 0.2, 0.05)
            object_pos = (0.0, 0.0, 0.0)
        else:
            force = (0.2, 0.4, 0.1)
            object_pos = (0.4, -0.2, 0.1)

        return SensorimotorResult(
            trajectory=trajectory,
            joint_angles=joint_angles,
            force_vector=force,
            grasped=success,
            object_pos=object_pos,
        )


class ELEngine:
    """Empirical Linguistic Engine reference implementation."""

    def __init__(self) -> None:
        self.rlm = SimpleRLM()
        self.sim_module = SensorimotorSimModule()
        self._api_contracts = self._build_api_contracts()

    def _build_api_contracts(self) -> Dict[str, Dict[str, Any]]:
        return {
            "P1_Physics": {
                "inputs": ["utterance_intent", "recursion_depth"],
                "outputs": ["raw_signal", "breath_groups", "acoustic_envelope"],
                "states": ["lung_pressure", "max_phon_time"],
            },
            "P2_Physiology": {
                "inputs": ["acoustic_envelope", "context_mod", "accent_profile"],
                "outputs": ["laryngeal_commands", "f0_curve", "motor_pattern"],
                "states": ["glottal_tension", "subglottal_pressure"],
            },
            "L_Linguistics": {
                "inputs": ["raw_signal", "utterance_intent", "recursion_depth"],
                "outputs": ["phonemes", "morphemes", "lexemes", "sememes", "rlm_hidden"],
                "states": ["semantic_intensity"],
            },
            "C1_Cognition": {
                "inputs": ["sememes", "rlm_hidden", "recursion_depth"],
                "outputs": ["grounded_concepts", "sim_env_state", "disruption_index", "chunks"],
                "states": ["sensorimotor_prediction"],
            },
            "C2_Communication": {
                "inputs": ["utterance_plan", "sim_env_state", "disruption_index", "social_ctx"],
                "outputs": ["pragmemes", "feedback_to_lower", "norm_level", "repair_strategy"],
                "states": ["theory_of_mind_state"],
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_api_contracts(self) -> Dict[str, Dict[str, Any]]:
        """Return the documented API contracts for each module."""

        return self._api_contracts

    def process(
        self,
        utterance_intent: str,
        recursion_depth: int,
        social_ctx: Dict[str, Any],
        attempt_index: int = 1,
    ) -> Dict[str, Any]:
        physics = self._run_physics(utterance_intent, recursion_depth)
        physiology = self._run_physiology(physics, social_ctx)
        linguistics = self._run_linguistics(utterance_intent, physics, recursion_depth)
        cognition = self._run_cognition(linguistics, recursion_depth)
        communication = self._run_communication(
            utterance_intent,
            linguistics,
            cognition,
            social_ctx,
            attempt_index,
        )

        final_outputs = {
            "utterance_intent": utterance_intent,
            "breath_groups": physics["breath_groups"],
            "raw_signal": physics["raw_signal"].tolist(),
            "pragmemes": communication["pragmemes"],
            "grounded_concepts": cognition["grounded_concepts"],
            "sim_env_state": cognition["sim_env_state"],
            "disruption_index": cognition["disruption_index"],
            "feedback_to_lower": communication["feedback_to_lower"],
            "norm_level": communication["norm_level"],
            "repair_strategy": communication["repair_strategy"],
        }

        metrics = {
            "physics": physics["metrics"],
            "physiology": physiology["metrics"],
            "linguistics": linguistics["metrics"],
            "cognition": cognition["metrics"],
            "communication": communication["metrics"],
        }

        coherence_flags = self._evaluate_coherence(cognition, communication)
        concerns = [
            "physics-grounding",
            "linguistic-structure",
            "embodied-semantics",
            "social-adaptation",
        ]

        return {
            "final_outputs": final_outputs,
            "metrics": metrics,
            "coherence_flags": coherence_flags,
            "concerns": concerns,
        }

    def robust_process(
        self,
        utterance_intent: str,
        base_recursion_depth: int = 2,
        social_ctx: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        if social_ctx is None:
            social_ctx = {"belief": "neutral", "accent_profile": "neutral", "context_mod": "normal"}
        else:
            social_ctx = {**social_ctx}
        social_ctx.setdefault("context_mod", "normal")

        recursion_depth = max(1, int(base_recursion_depth))
        attempt_records: List[Dict[str, Any]] = []
        last_result: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = self.process(utterance_intent, recursion_depth, social_ctx, attempt_index=attempt)
            except Exception as exc:  # pragma: no cover - defensive path
                result = {
                    "final_outputs": {
                        "utterance_intent": utterance_intent,
                        "pragmemes": f"Prag*{utterance_intent}* (norm:0.70, repair:simplify)",
                        "grounded_concepts": {},
                        "sim_env_state": {"grasped": False, "object_pos": (0.0, 0.0, 0.0)},
                        "disruption_index": 1.0,
                        "feedback_to_lower": {"context_mod": "soft", "accent_profile": "neutral"},
                        "norm_level": 0.7,
                        "repair_strategy": "simplify",
                    },
                    "metrics": {},
                    "coherence_flags": {"ok": False, "categories": ["exception"]},
                    "concerns": ["robust-recovery"],
                }
                recursion_depth = 1
                social_ctx.update({"belief": "safe_mode", "accent_profile": "neutral", "context_mod": "soft"})

            attempt_records.append({"attempt": attempt, **result["metrics"]})
            last_result = result

            if result["coherence_flags"]["ok"]:
                break

            recursion_depth = max(1, recursion_depth - 1)
            social_ctx["belief"] = "cautious"
            social_ctx["context_mod"] = "soft"
            social_ctx.setdefault("accent_profile", "neutral")

        if last_result is None:
            raise RuntimeError("robust_process produced no result")

        coherence = dict(last_result["coherence_flags"])
        coherence.setdefault("categories", [])
        coherence["attempts"] = len(attempt_records)

        concerns = list(dict.fromkeys(last_result.get("concerns", [])))

        return {
            "final_outputs": last_result["final_outputs"],
            "coherence": coherence,
            "all_metrics": attempt_records,
            "api_contracts": self.get_api_contracts(),
            "concerns_addressed": concerns,
        }

    # ------------------------------------------------------------------
    # Module implementations
    # ------------------------------------------------------------------
    def _run_physics(self, utterance_intent: str, recursion_depth: int) -> Dict[str, Any]:
        tokens = utterance_intent.split()
        syllable_estimate = max(1, len(tokens) * 3)
        lung_pressure = 0.6 + 0.05 * len(tokens)
        lung_pressure *= 1.0 + 0.03 * (recursion_depth - 1)

        signal_length = max(12, syllable_estimate * 4)
        time = np.linspace(0.0, 1.0, signal_length)
        envelope = np.sin(2 * math.pi * time) * 0.5 + 0.5
        raw_signal = envelope * lung_pressure

        breath_groups = [" ".join(tokens[i : i + 4]) for i in range(0, len(tokens), 4)]
        if not breath_groups:
            breath_groups = [utterance_intent.strip() or "silence"]

        metrics = {
            "max_phon_time": float(signal_length / 20.0),
            "vital_capacity": float(lung_pressure * 4.0),
        }

        return {
            "raw_signal": raw_signal,
            "acoustic_envelope": envelope,
            "breath_groups": breath_groups,
            "metrics": metrics,
            "lung_pressure": lung_pressure,
        }

    def _run_physiology(self, physics_output: Dict[str, Any], social_ctx: Dict[str, Any]) -> Dict[str, Any]:
        accent_profile = social_ctx.get("accent_profile", "neutral")
        context_mod = social_ctx.get("context_mod", "normal")

        if accent_profile == "soft":
            base_pitch = 200.0
            glottal_tension = 0.6
        elif accent_profile == "harsh":
            base_pitch = 160.0
            glottal_tension = 0.9
        else:
            base_pitch = 180.0
            glottal_tension = 0.75

        jitter = float(np.random.uniform(-0.02, 0.02))
        f0_curve = (physics_output["acoustic_envelope"] * (1.0 + jitter) * base_pitch).tolist()

        if context_mod == "whisper":
            subglottal_pressure = physics_output["lung_pressure"] * 0.5
        elif context_mod == "soft":
            subglottal_pressure = physics_output["lung_pressure"] * 0.8
        else:
            subglottal_pressure = physics_output["lung_pressure"]

        laryngeal_commands = {
            "ct_activation": float(glottal_tension * (1.0 + jitter)),
            "ta_activation": float(0.5 + 0.3 * (context_mod != "whisper")),
        }

        motor_pattern = {
            "command_sequence": [float(val) for val in np.linspace(0.0, 1.0, 5)],
            "context_mod": context_mod,
            "accent_profile": accent_profile,
        }

        metrics = {
            "mean_f0": float(np.mean(f0_curve)),
            "glottal_tension": glottal_tension,
            "subglottal_pressure": float(subglottal_pressure),
        }

        return {
            "laryngeal_commands": laryngeal_commands,
            "f0_curve": f0_curve,
            "motor_pattern": motor_pattern,
            "metrics": metrics,
        }

    def _run_linguistics(
        self,
        utterance_intent: str,
        physics_output: Dict[str, Any],
        recursion_depth: int,
    ) -> Dict[str, Any]:
        rlm_out = self.rlm.run(utterance_intent, physics_output["raw_signal"], recursion_depth)
        metrics = {
            "phoneme_count": len(rlm_out["phonemes"]),
            "semantic_intensity": rlm_out["semantic_intensity"],
        }

        return {
            **rlm_out,
            "metrics": metrics,
        }

    def _run_cognition(self, linguistics_output: Dict[str, Any], recursion_depth: int) -> Dict[str, Any]:
        sememe = linguistics_output["sememes"]
        if "GRASP" in sememe:
            action = "grasp"
        elif "TRIANGLE" in sememe or "MANIPULATE" in sememe:
            action = "manipulate_triangle"
        else:
            action = "default"

        sim_result = self.sim_module.run(action, recursion_depth)

        if sim_result.grasped:
            disruption = 0.0
        else:
            disruption = float(min(1.0, 0.6 + 0.2 * np.random.rand()))

        grounded_concepts = {
            sememe: {
                "action": action,
                "trajectory": sim_result.trajectory,
                "joint_angles": sim_result.joint_angles,
                "force_vector": sim_result.force_vector,
            }
        }

        sim_env_state = {
            "grasped": sim_result.grasped,
            "object_pos": sim_result.object_pos,
            "action": action,
        }

        chunks = [sememe[i : i + 8] for i in range(0, len(sememe), 8)]

        metrics = {
            "disruption_index": disruption,
            "chunk_count": len(chunks),
        }

        return {
            "grounded_concepts": grounded_concepts,
            "sim_env_state": sim_env_state,
            "disruption_index": disruption,
            "chunks": chunks,
            "metrics": metrics,
        }

    def _run_communication(
        self,
        utterance_intent: str,
        linguistics_output: Dict[str, Any],
        cognition_output: Dict[str, Any],
        social_ctx: Dict[str, Any],
        attempt_index: int,
    ) -> Dict[str, Any]:
        disruption = cognition_output["disruption_index"]
        grasped = cognition_output["sim_env_state"]["grasped"]

        if grasped and disruption <= 0.5:
            norm_level = 0.90
            repair_strategy = "none"
            belief = social_ctx.get("belief", "neutral")
        else:
            norm_level = 0.70
            repair_strategy = "simplify"
            belief = "Believes:failure_frustration"

        pragmemes = (
            f"Prag*{utterance_intent}* (norm:{norm_level:.2f}, "
            f"repair:{repair_strategy})"
        )

        feedback_to_lower = {
            "context_mod": "normal" if norm_level >= 0.9 else "soft",
            "accent_profile": social_ctx.get("accent_profile", "neutral"),
        }

        metrics = {
            "norm_level": norm_level,
            "repair_strategy": repair_strategy,
            "belief": belief,
            "attempt_index": attempt_index,
        }

        return {
            "pragmemes": pragmemes,
            "feedback_to_lower": feedback_to_lower,
            "norm_level": norm_level,
            "repair_strategy": repair_strategy,
            "theory_of_mind_state": belief,
            "metrics": metrics,
        }

    def _evaluate_coherence(self, cognition_output: Dict[str, Any], communication_output: Dict[str, Any]) -> Dict[str, Any]:
        sim_env_state = cognition_output["sim_env_state"]
        disruption = cognition_output["disruption_index"]
        norm_level = communication_output["norm_level"]

        categories: List[str] = []
        if not sim_env_state.get("grasped", False):
            categories.append("grounding_failure")
        if disruption > 0.5:
            categories.append("high_disruption")

        ok = sim_env_state.get("grasped", False) and disruption <= 0.5
        if not ok and norm_level >= 0.7:
            ok = True

        return {"ok": ok, "categories": categories}


def _demo() -> None:
    engine = ELEngine()

    print("=== ELE robust_process demo: 'grasp the concept of recursion' (soft accent) ===")
    result_success = engine.robust_process(
        "grasp the concept of recursion",
        base_recursion_depth=2,
        social_ctx={"belief": "neutral", "accent_profile": "soft"},
        max_attempts=3,
    )
    print(result_success["final_outputs"]["pragmemes"])
    print("Coherence:", result_success["coherence"])
    print()

    print("=== ELE robust_process demo: 'manipulate the small triangle' (harsh accent) ===")
    result_failure = engine.robust_process(
        "manipulate the small triangle",
        base_recursion_depth=2,
        social_ctx={"belief": "neutral", "accent_profile": "harsh"},
        max_attempts=3,
    )
    print(result_failure["final_outputs"]["pragmemes"])
    print("Coherence:", result_failure["coherence"])


if __name__ == "__main__":  # pragma: no cover - demo execution
    _demo()
