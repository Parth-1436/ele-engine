# ELE Usage Guide

This guide explains how to use ELE Engine in your own projects.

## Basic Usage

```python
from ele_engine.ele_engine import ELEngine

engine = ELEngine()

result = engine.robust_process(
    "grasp the concept of recursion",
    base_recursion_depth=2,
    social_ctx={"belief": "neutral", "accent_profile": "soft"},
    max_attempts=3,
)

print(result["final_outputs"]["pragmemes"])
print(result["coherence"])
```

## Interpreting `final_outputs`

Key keys in `final_outputs`:

* `pragmemes`: The final, context-aware utterance string.
* `grounded_concepts`: Fully grounded sememe → action mapping.
* `sim_env_state`: Kinematic outcome (`grasped=True/False`, `object_pos`).
* `disruption_index`: How disrupted the internal state is.
* `feedback_to_lower`: How C2 modulated P2 (`context_mod`, `accent_profile`).

## Coherence

The `coherence` field indicates whether the engine considers the run “coherent”:

```python
coherence = result.get("coherence", {})
print(coherence)
# => {"ok": True, "categories": [], "attempts": 1}
```

If `ok=False`, check `categories` (e.g., `["grounding_failure", "high_disruption"]`).

## Embedding in Larger Systems

ELE can be embedded as a “linguistic body” within:

* Multi-agent systems,
* Conversational agents,
* Cognitive simulations,
* Frameworks like Logos, Word Calculator, AMR controllers, etc.

High-level systems provide:

* `utterance_intent`,
* `social_ctx` (beliefs, accent_profile, etc.),

and read:

* `final_outputs`,
* `coherence`,
* `all_metrics` for diagnostics and adaptation.

For research usage, you can log:

* disruption_index over time,
* norm_level changes,
* repair_strategy selections.

This enables empirical study of coherence, failure, and repair in a grounded linguistic system.
