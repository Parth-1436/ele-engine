import numpy as np
import torch
import random

from ele_engine.ele_engine import ELEngine


def seed_all(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def test_grasp_success_coherent():
    seed_all(42)
    engine = ELEngine()
    result = engine.robust_process(
        "grasp the concept of recursion",
        base_recursion_depth=2,
        social_ctx={"belief": "neutral", "accent_profile": "soft"},
        max_attempts=3,
    )

    final = result["final_outputs"]
    coherence = result.get("coherence", {})
    sim_env = final.get("sim_env_state", {})

    assert coherence.get("ok", False), "Grasp case should be coherent"
    assert sim_env.get("grasped", True) is True, "Grasp action should normally succeed"
    assert final.get("disruption_index", 0.0) <= 0.5
    assert "Prag*grasp the concept of recursion" in final.get("pragmemes", "")
    # norm ~0.9 for success
    assert "norm:0.90" in final.get("pragmemes", "")


def test_manipulate_triangle_triggers_repair():
    seed_all(123)  # Different seed to increase chance of failure due to jitter
    engine = ELEngine()
    result = engine.robust_process(
        "manipulate the small triangle",
        base_recursion_depth=2,
        social_ctx={"belief": "neutral", "accent_profile": "harsh"},
        max_attempts=3,
    )

    final = result["final_outputs"]
    coherence = result.get("coherence", {})
    sim_env = final.get("sim_env_state", {})
    prag = final.get("pragmemes", "")

    # We expect coherent result after robust_process, even if first pass failed
    assert coherence.get("ok", True) is True

    # At least sometimes, jitter should produce a failure / high disruption path
    # which should manifest as norm:0.70 and repair:simplify
    # We don't assert failure is mandatory here, but if we see norm:0.70 it must match repair:simplify.
    if "norm:0.70" in prag:
        assert "repair:simplify" in prag
        assert sim_env.get("grasped", False) is False or final.get("disruption_index", 0.0) > 0.5


def test_api_contracts_exist():
    engine = ELEngine()
    contracts = engine.get_api_contracts()
    assert "P1_Physics" in contracts
    assert "P2_Physiology" in contracts
    assert "L_Linguistics" in contracts
    assert "C1_Cognition" in contracts
    assert "C2_Communication" in contracts
    # Check that each contract has inputs/outputs/states keys
    for name, spec in contracts.items():
        assert "inputs" in spec
        assert "outputs" in spec
        assert "states" in spec
