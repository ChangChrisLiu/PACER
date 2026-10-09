# Offline evaluation and audited candidate selection

Input: trained candidates, a common validation split, declared task geometry and
a baseline policy. Output: comparable component-balanced scores, full feasibility
checks and the selected candidate or reference fallback.

## Integration boundary

Implement `VLAHooks.predict_action_chunk`, `RobotHooks.integrate_action_chunk`
and `RobotHooks.target_region` for your model and robot. Inverse-normalize and
integrate predicted chunks with the deployment action convention. PACER consumes
base-frame EEF/TCP positions and metadata; it does not simulate contact dynamics
or import model/robot SDKs in the portable framework.

A runnable synthetic, no-hardware workflow is available immediately:

```bash
python -m pacer_framework.demo --out-dir .local/demo
python -m pacer_framework.scoring --write-config .local/scoring.json
python -m pacer_framework.demo --scoring-config .local/scoring.json --out-dir .local/configured-demo
```

See [Scoring configuration](SCORING_CONFIGURATION.md) for the JSON/API contract.
Defaults are generic starting values; customize and freeze your own protocol.

## 1. Build matched validation inputs

Query policies deterministically on the same logged observations. Keep row
support, reference matching, valid horizons, target regions, coordinate frames
and blockers fixed across candidates. Row IDs identify observations, not the
model that produced a prediction.

```python
from pacer_framework import ScoringConfig, open_loop_submetrics, blocker_flags_from_open_loop

config = ScoringConfig.load("scoring.json")
submetrics = open_loop_submetrics(
    predicted_positions,
    target_point=row_target_point,
    component=row_component,
    phase_ending=row_phase_ending,
    stop_emitted=predicted_stop_emitted,
    reference_positions=matched_reference_positions,
    scoring_config=config,
)
flags = blocker_flags_from_open_loop(
    predicted_positions,
    target_point=row_target_point,
    component=row_component,
    non_target_regions=declared_non_target_regions,
    workspace_bounds=calibrated_workspace_bounds,
    orientation_ok=endpoint_orientation_within_tolerance,
    stop_source=predicted_stop_source,
)
validation_row = {
    "row_id": observation_id,
    "component": row_component,
    "role": row_role,
    "submetrics": submetrics,
    "reference_alignment_mode": config.reference_alignment_mode,
    **flags,
}
```

Build baseline-policy rows on the same inputs. The matched trajectory for align
is not automatically the baseline policy for no-regression. Do not fabricate
missing reference geometry or call every recorded partial trajectory an expert
success demonstration.

## 2. Direction and whole-chunk alignment

Direction compares net displacement with the start-to-target vector. Align
compares concatenated adjacent EEF/TCP displacements over the common valid
horizon. Endpoint alignment remains explicit. These are spatial EEF/TCP vectors,
not mixed-unit raw joint/gripper actions.

Trajectory mode requires a start and successor on both sides. A short matched
trajectory is an input error, not a reason to silently remove alignment. If no
matched reference exists under your declared applicability rule, omit it
consistently across candidates. Non-phase-ending rows omit stop; observed
zero-valued metrics remain in the score.

## 3. Score and audit under one configuration

```python
from pacer_framework import evaluate_candidate, select_candidate, candidate_ranking_table

evaluations = {
    name: evaluate_candidate(
        validation_rows=rows_for[name],
        reference_validation_rows=baseline_rows,
        weight_manifest=weight_manifests[name],
        w_max=candidates[name].w_max,
        scoring_config=config,
    )
    for name in candidates
}
j_vals = {name: result["j_val"] for name, result in evaluations.items()}
audited = {name: result["feasible"] for name, result in evaluations.items()}
selected = select_candidate(j_vals, audited)
ranking = candidate_ranking_table(j_vals, audited, selected)
```

Matched reference geometry recomputes no-regression under the same coefficients.
Reference component scores are rebuilt with that configuration, not copied from
a differently weighted run. Precomputed reference component scores remain an
explicit caller-owned same-protocol input.

The row score is the gated, normalized weighted mean of applicable terms.
Component-balanced aggregation averages positive rows within each component,
then components equally. Blocked positive rows remain zero in the denominator;
failure/excluded rows are audit-only. Manifests record the config, fingerprint,
counts and indicator provenance.

Full feasibility includes training-view/ESS checks, wrong-target and unsafe
leakage, scoring support, reference component coverage and component-level
no-regression. Missing references fail closed by default. An explicit reference
opt-out is a labelled diagnostic, not a complete selection audit.

`robust_profile_j_val` without a profile uses this same component-mean route.
Named lower-tail/factor-family diagnostics are explicit opt-ins, not the default.

## 4. Select, freeze and test separately

Choose the highest score among feasible candidates. If none passes, use the
reference fallback rather than forcing an unqualified candidate onto hardware.
Freeze the checkpoint and scoring protocol before protected testing. J_val is
not a training loss, task-success probability or safety certificate.

Keep actual held-out outcomes separate. `success_table` and
`paired_component_bootstrap` summarize your measured results. Match resampling
to the collection protocol; few task/component clusters support only limited
uncertainty claims. Synthetic demo counts are not experimental evidence.

For real hardware, follow the integration and safety runbooks, including
attended dry runs, calibrated targets and independent physical stop mechanisms.
