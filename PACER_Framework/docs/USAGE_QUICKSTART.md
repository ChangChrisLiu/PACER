# PACER quickstart for a new robot/VLA

This is the shortest path for someone who downloads the repo and wants to try
PACER on their own system.

## 1. Implement your hooks

Create a small file in your project, e.g. `my_pacer_hooks.py`, that implements:

- `RobotHooks.target_region(component, collection_config)`
- `RobotHooks.integrate_action_chunk(start_tcp, action_chunk, observation)`
- `RobotHooks.safety_flags(trace_or_prediction)`
- `VLAHooks.predict_action_chunk(model_id, observation, prompt, deterministic)`
- `VLAHooks.stop_emitted(prediction)`
- optionally `TrainerHooks.train_weighted_view(...)`

See `docs/ADAPTER_HOOKS.md` for examples for EEF-delta arms, joint-space arms,
absolute waypoint policies, bimanual robots, mobile manipulators, simulators,
OpenVLA/OpenPI/RT-style/diffusion/custom VLA models, and custom trainers.

## 2. Define targets for each task/config

For each task target and scene configuration, record:

```json
{
  "target_id": "scene_001:door_handle",
  "component": "door_handle",
  "point": [0.42, -0.10, 0.31],
  "radius": 0.015,
  "orientation_tolerance_rad": 0.25
}
```

Use your own tolerances. Do not infer the target from where the policy went.

## 3. Record three data sources

- rollouts from the current/base VLA policy;
- human corrections from failure/partial states;
- supplementary clean demos for cells the policy cannot reach.

Chunk them into standard PACER rows and validate them:

```python
from pacer_framework import validate_training_row, training_row_warnings

errors = [e for row in rows for e in validate_training_row(row)]
warnings = [w for row in rows for w in training_row_warnings(row)]
assert not errors
```

## 4. Build weighted views

```python
from pacer_framework import CANDIDATE_POOL, BASELINE_MODES, compile_weights, compile_baseline, export_training_view

for name, eta in CANDIDATE_POOL.items():
    weighted, manifest = compile_weights(rows, eta, eta_id=name)
    export_training_view(weighted, manifest, f"views/pacer_candidates/{name}")

for mode in BASELINE_MODES:
    if mode == "pacer_selected":
        continue
    view_rows, view_manifest = compile_baseline(rows, mode)
    export_training_view(view_rows, view_manifest, f"views/baselines/{mode}")
```

Each exported row has `loss_weight`. Your trainer multiplies the native action
loss by that value.

## 5. Train candidate VLA models externally

Use your own training stack. Keep the starting checkpoint, recipe, update
budget, data sampling, and normalization policy identical across candidates and
baselines.

## 6. Evaluate candidates offline

For each validation row and trained candidate:

1. call your `VLAHooks.predict_action_chunk`;
2. call your `RobotHooks.integrate_action_chunk` to get TCP/EEF positions;
3. compute PACER submetrics and blockers;
4. compute J_val and audits.

```python
from pacer_framework import (
    blocker_flags_from_open_loop,
    component_balanced_j_val,
    evaluate_candidate,
    open_loop_submetrics,
    select_candidate,
    validate_validation_row,
)

# Build validation_rows from your hook outputs. For a paper-faithful selector,
# validate strict rows and include blocker flags plus align/no_regression
# submetrics before scoring. Compute the reference policy component scores once.
strict_errors = [
    err
    for row in validation_rows
    for err in validate_validation_row(row, strict_paper=True)
]
assert not strict_errors
_, _, reference_manifest = component_balanced_j_val(reference_validation_rows)

eval_result = evaluate_candidate(
    validation_rows=validation_rows,
    weight_manifest=manifest,
    w_max=eta.w_max,
    reference_component_scores=reference_manifest["component_scores"],
    delta_reg=0.05,
    require_reference_component_scores=True,
)
selected = select_candidate({"candidate": eval_result["j_val"]}, {"candidate": eval_result["feasible"]})
```

Only deploy an audited selected candidate. If none passes, keep the reference
policy.
