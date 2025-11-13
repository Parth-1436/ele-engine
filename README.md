# Empirical Linguistic Engine (ELE)

The **Empirical Linguistic Engine (ELE)** is a unified, interaction-dominant architecture that ties language, physics, cognition, and social behavior into a single coherent system.

It implements a full causal continuum:

> **P1 → P2 → L → C1 → C2**  
> Physics → Physiology → Linguistics → Cognition → Communication  

with mandatory feedback:

> **C2 → P2** (and rerun **L → C1 → C2**)

ELE V1.x is not just a language model wrapper. It is:

- **Physically grounded** (airflow, breath, vocal muscle activation),
- **Linguistically structured** (phoneme → morpheme → lexeme → sememe),
- **Embodied** (sememes drive kinematic simulations),
- **Socially adaptive** (pragmatics, ToM, accent handling),
- **Robust** (auto-repair loop that interprets disruption and retries until coherent or safely exhausted).

---

## Core Idea

A linguistic intention (e.g., `"grasp the concept of recursion"`) flows through:

1. **P1 – Physics**  
   - Computes Vital Capacity (VC), lung pressure, airflow.  
   - Decides how much content can fit into a breath group.  
   - Emits `raw_signal` and `breath_groups`.

2. **P2 – Physiology (Pneuma)**  
   - Takes `acoustic_envelope`, `context_mod` (normal/whisper/shout), and `accent_profile`.  
   - Produces laryngeal motor commands (CT/TA activations) and f₀ (pitch).  
   - Realizes speech as a motor pattern.

3. **L – Linguistics (RLM)**  
   - Uses a PyTorch GRU-based **Recursive Language Model (SimpleRLM)**.  
   - Converts `raw_signal` into:
     - phoneme/morpheme/lexeme descriptors,  
     - sememes (semantic string),  
     - RLM hidden state (`rlm_hidden`).

4. **C1 – Cognition (Embodied Grounding & Disruption)**  
   - Decodes sememes into actions (`grasp`, `manipulate_triangle`, `default`).  
   - Calls `SensorimotorSimModule` to simulate kinematics:
     - trajectory, joint_angles, force_vector, success/failure.  
   - Builds:
     - `grounded_concepts` (sememe → sim data),  
     - `chunks` (dynamic segmentation for working memory),  
     - `disruption_index` (how unstable / disrupted the attempt is).

5. **C2 – Communication (Pragmatics, ToM, Repair)**  
   - Receives:
     - `utterance_plan` (all upstream info),
     - `sim_env_state` (e.g., `grasped=True/False`),
     - `disruption_index`,
     - `social_ctx` (belief, accent_profile).  
   - Infers:
     - ToM-like belief (e.g., `Believes:failure_frustration`),  
     - `norm_level` (0.7–0.9),  
     - `repair_strategy` (`none`, `simplify`, etc.).  
   - Emits:
     - `pragmemes` (the situated utterance),
     - `feedback_to_lower` → `{"context_mod": "whisper"/"normal", "accent_profile": ...}`.

Then the engine can **re-run P2 → L → C1 → C2** with this feedback applied.

---

## Robust Loop: `robust_process()`

The main entrypoint is:

```python
from ele_engine.ele_engine import ELEngine

engine = ELEngine()

result = engine.robust_process(
    "manipulate the small triangle",
    base_recursion_depth=2,
    social_ctx={"belief": "neutral", "accent_profile": "harsh"},
    max_attempts=3,
)
```

`robust_process` will:

1. Run a full cycle (`process`) with feedback.
2. Check coherence:

   * If `sim_env_state.grasped` is False or `disruption_index > 0.5`, it treats this as disruption.
3. If incoherent:

   * Reduce recursion_depth (simplify),
   * Optionally nudge `social_ctx.belief` to "cautious",
   * Re-run up to `max_attempts`.
4. On exceptions:

   * Categorize (`rlm_error`, `sim_error`, etc.),
   * Fall back to safe settings (depth=1, neutral accent, “safe_mode” belief),
   * Re-run until success or exhaustion.

The returned `result` includes:

* `final_outputs` (pragmemes, grounded_concepts, sim_env_state, disruption_index, etc.),
* `coherence`: `{ "ok": True/False, "categories": [...], "attempts": N }`,
* `all_metrics` per module,
* `api_contracts` describing inputs/outputs per module,
* `concerns_addressed` summarizing doctrinal alignment.

---

## Installation / Requirements

This repo assumes:

* Python 3.9+
* NumPy
* PyTorch (CPU-only is fine)

Example:

```bash
pip install numpy torch
```

(You can add this to `pyproject.toml` or `requirements.txt`.)

---

## Repository Layout

```text
ele-engine/
├─ ele_engine/
│  ├─ __init__.py
│  └─ ele_engine.py
├─ tests/
│  └─ test_ele_engine.py
├─ docs/
│  ├─ ELE_Doctrine_v1.0.md
│  ├─ Architecture_Overview.md
│  └─ Usage_Guide.md
├─ README.md
├─ pyproject.toml  # optional
└─ LICENSE         # your choice (MIT/Apache-2.0/etc.)
```

---

## Running the Demo

From the `ele-engine` root:

```bash
python -m ele_engine.ele_engine
```

You should see output similar to:

```text
=== ELE robust_process demo: 'grasp the concept of recursion' (soft accent) ===
Prag*grasp the concept of recursion* (norm:0.90, repair:none)
Coherence: {'ok': True, 'categories': [], 'attempts': 1}

=== ELE robust_process demo: 'manipulate the small triangle' (harsh accent) ===
Prag*manipulate the small triangle* (norm:0.70, repair:simplify)
Coherence: {'ok': True, 'categories': ['grounding_failure', 'high_disruption'], 'attempts': 1 or 2}
```

---

## Tests

We use `pytest` for simple behavioral checks:

* P1 produces non-negative `max_phon_time`.
* A successful run with `"grasp..."` has:

  * `grasped=True`,
  * `disruption_index == 0.0`,
  * `norm_level ≈ 0.9`.
* A failure case (with jitter, `"manipulate the small triangle"`) eventually produces:

  * `grasped=False` at least sometimes over multiple runs,
  * `norm_level ≈ 0.7`,
  * `repair_strategy="simplify"`.

Run tests with:

```bash
pytest
```

---

## Docs

See:

* `docs/ELE_Doctrine_v1.0.md` – Philosophy and constraints (the Doctrine).
* `docs/Architecture_Overview.md` – Module breakdown, diagrams.
* `docs/Usage_Guide.md` – How to embed ELE in larger systems (LLMs, agents, etc.).

---

## License

Choose your license (MIT recommended if you want wide reuse).

---

## Status

ELE Engine v1.x is a **canonical reference** for:

* physically grounded language,
* recursive cognition,
* embodied semantics,
* and interaction-dominant communication.

All future work should either build *on* this engine or clearly specify where it diverges from this architecture.

