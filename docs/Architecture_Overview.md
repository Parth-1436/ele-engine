# ELE Architecture Overview

This document gives a high-level overview of the ELE Engine architecture.

## Module Stack

- **P1 – Physics**: Aerodynamics, VC, breath groups.
- **P2 – Physiology**: CT/TA activations, f0, accent profiles, context_mod.
- **L – Linguistics**: PyTorch GRU-based RLM (SimpleRLM), language units.
- **C1 – Cognition**: Grounded sememes via kinematic simulation, disruption_index.
- **C2 – Communication**: Pragmemes, ToM, norm_level, repair_strategy, feedback_to_lower.

## Data Flow

1. `utterance_intent` enters P1.
2. P1 → P2 with `acoustic_envelope`.
3. P2 → L with motor context (indirectly via f0, but in this implementation L uses raw_signal).
4. L → C1 with `sememes` + `rlm_hidden`.
5. C1 → C2 with `grounded_concepts`, `sim_env_state`, `disruption_index`.
6. C2 → P2 with `feedback_to_lower` (`context_mod`, `accent_profile`).
7. P2 → L → C1 → C2 re-run.

## Robustness Layer

- `ELEngine.robust_process`:
  - Runs one full cycle,
  - Checks coherence,
  - If needed, adjusts recursion_depth and social_ctx,
  - Retries until coherent or limit reached.

For detailed behavior, see `README.md` and `ELE_Doctrine_v1.0.md`.
