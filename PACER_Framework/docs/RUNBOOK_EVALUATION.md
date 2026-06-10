# Runbook 3 — Offline evaluation and hardware candidate selection (pipeline steps 6–7)

Input: one trained model per candidate/baseline (Runbook 2) + the validation
split with its rollout/reference EEF traces.
Output: J_val per candidate, feasibility audits, the comparison tables, and
**the one model that goes to the actual hardware run**.

## 0. Adapter hooks for any robot/VLA stack

Before scoring candidates, implement the integration boundary in your own code
(interfaces live in `pacer_framework/hooks.py`):

- `VLAHooks.predict_action_chunk(...)`: call your candidate VLA on a logged
  observation. This can wrap OpenVLA, OpenPI, RT-style token decoders,
  diffusion/flow action heads, or any custom model.
- `RobotHooks.integrate_action_chunk(...)`: inverse-normalize and integrate the
  predicted action chunk from the logged start TCP pose into base-frame EEF/TCP
  positions. This is where joint-space vs EEF-delta actions, gripper channels,
  bimanual arms, mobile bases, and simulator-specific conventions live.
- `RobotHooks.target_region(...)` / `TargetRegion`: return the fixed target
  region captured before scoring.

PACER starts after these hooks: it consumes standard validation rows, target
regions, and TCP/EEF positions. It never imports your robot SDK or model code.

## 1. Offline validation scoring (one pass per candidate)

For every scoring validation row (roles clean / correction / auto_success /
partial; failure / excluded rows are audit-only):

1. Query the trained candidate ONCE on the logged observation with
   deterministic decoding (fix the flow/sampling noise for all rows and all
   candidates).
2. Inverse-normalize the predicted action chunk and integrate it **open-loop
   from the logged start TCP pose** with your robot's own action convention.
   This is robot-specific and stays in your code; the framework consumes the
   resulting base-frame TCP positions.
3. Compute submetrics and blocker flags:

```python
from pacer_framework import (
    RobotHooks, TargetRegion, TrainerHooks, VLAHooks,
    open_loop_submetrics, with_no_regression, blocker_flags_from_open_loop,
)

sub = open_loop_submetrics(
    predicted_positions,
    target_point=row_target_point, component=row_component,
    phase_ending=row_phase_ending, stop_emitted=predicted_stop_emitted,
    reference_positions=matched_reference_positions,   # omit if no match
)
sub = with_no_regression(sub, reference_policy_submetrics, delta_reg=0.05)
flags = blocker_flags_from_open_loop(
    predicted_positions, target_point=row_target_point, component=row_component,
    non_target_regions=[{"point": p, "radius": r} for p, r in other_targets],
    workspace_bounds={"min": [...], "max": [...]},
    orientation_ok=endpoint_orientation_within_tolerance,
    stop_source=predicted_stop_source,
)
validation_row = {"row_id": ..., "component": ..., "role": ..., "submetrics": sub, **flags}
```

## 2. How the EEF cosine / direction metrics work (eq:app_align)

Two distinct cosines, both on **TCP/EEF positions in the base frame** — never
on raw action vectors:

- **direction** submetric: cosine between the predicted chunk's net
  displacement (endpoint − start) and the start→target direction, clipped to
  [0, 1] (`alignment.target_direction_cosine`). Same form as the training
  feature e_dir, evaluated on the candidate's open-loop chunk.
- **align** submetric: when a matched reference chunk r(j) exists (a clean
  demonstration or correction covering the same state), the cosine between the
  generated chunk direction u_j and the reference chunk direction u_ref,
  clipped to [0, 1] (`alignment.reference_alignment`) — eq:app_align verbatim.

Both are omitted automatically when unavailable (the λ coefficients renormalize
per row). A diagnostic action-vector cosine, if you log one, must stay out of
the submetrics dict used for ranking.

## 3. J_val, audits, and selection

```python
from pacer_framework import (component_balanced_j_val, evaluate_candidate,
                             select_candidate, candidate_ranking_table)

# Reference policy scores once (for the no-regression audit):
_, _, ref = component_balanced_j_val(reference_validation_rows)

evaluations = {
    name: evaluate_candidate(
        validation_rows=rows_for[name],
        weight_manifest=weight_manifests[name],     # from compile_weights
        w_max=candidates[name].w_max,
        reference_component_scores=ref["component_scores"],
        delta_reg=0.05,                              # fix BEFORE ranking
    )
    for name in candidates
}
j_vals  = {n: e["j_val"]   for n, e in evaluations.items()}
audited = {n: e["feasible"] for n, e in evaluations.items()}
selected = select_candidate(j_vals, audited)         # None -> fall back to pi_theta0
ranking = candidate_ranking_table(j_vals, audited, selected)
```

Semantics, all paper-exact: v_j = B_j · renormalized weighted mean of the
applicable submetrics (eq:row_val); J_val = mean over components of
per-component mean row scores (eq:j_val); feasibility F(η) =
A_audit·A_wrong·A_safe·A_reg with zero tolerance on wrong-target and unsafe
leakage (eq:app_feasibility). J_val is a model-selection diagnostic — never a
training loss and never a substitute for deployment success.

## 4. Held-out comparison, exactly as reported

Evaluate ONLY the selected candidate and the baselines on held-out scene
configurations that appeared nowhere in the buffer, with the same reset,
timeout, label vocabulary, and operator rubric for every method, then:

```python
from pacer_framework import success_table, paired_component_bootstrap

results = {  # per method: component/task -> (successes, trials); fill from YOUR held-out run
    "pacer_selected": {
        "task_a": (successes_a, trials_a),
        "task_b": (successes_b, trials_b),
    },
    "vanilla_post_sft": {...},
    "uniform_all_eligible": {...},
}
table = success_table(results)                       # Wilson 95% CIs
diff = paired_component_bootstrap(results["pacer_selected"],
                                  results["vanilla_post_sft"])
```

`wilson_interval` gives per-method binomial confidence intervals; the paired
component cluster-bootstrap resamples your task/component groups with
replacement with both methods sharing each draw. With few components, report
bootstrap intervals as descriptive diagnostics, not as a replacement for the
actual held-out success rates.

## 5. Hand off to hardware (pipeline step 7)

The offline J_val + audits exist to answer one question: **which trained
candidate goes on the robot**. The decision rule, in full:

1. A candidate is eligible only if it passed every audit — zero forbidden
   positive-weight rows, healthy weight statistics, zero wrong-target leakage,
   zero unsafe leakage, and per-component no-regression against the reference
   policy.
2. Among audited candidates, deploy the one with the highest component-balanced
   J_val (`select_candidate`).
3. If NO candidate is audited, deploy the reference policy π_θ0 unchanged —
   PACER never forces a post-trained model onto hardware.
4. Freeze the selected checkpoint BEFORE the hardware evaluation, run the
   held-out protocol of §4 once, and report J_val only as the selection
   diagnostic it is — the hardware success rate is the empirical claim.
