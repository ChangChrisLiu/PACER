# PACER algorithm notes

PACER learns from **process evidence**, not only final success/failure labels.
The goal is to turn rollout traces, corrections, demonstrations, target-region
checks, stop events, and provenance checks into controlled training weights and
validation views for VLA policy improvement.

## Core path

1. Convert rollout, correction, and clean-demo episodes into action chunks.
2. Attach process evidence to each chunk:
   - row role: clean demo, human correction, model success, partial, failure;
   - task progress toward the target;
   - proximity to the target region;
   - terminal/inside-region evidence;
   - direction/alignment evidence when available;
   - model stop/handoff context;
   - operator label/rank evidence;
   - provenance and safety validity.
3. Apply hard gates for invalid rows:
   - unsafe or quarantined trials;
   - invalid policy-loss masks;
   - missing or inconsistent provenance;
   - target mismatch for positive geometric credit;
   - validation/test rows that must not enter training.
4. Map surviving evidence to a raw PACER score with eta coefficients.
5. Normalize within strata using robust median/MAD statistics.
6. Clip/floor final weights according to role and candidate eta settings.
7. Expose weights as training/export fields such as `returns.loss_weight`.
8. Train candidate policies in the external VLA stack.
9. Score candidates on validation data. Audited candidate selection
   (`evaluate_candidate` / `select_candidate`) defaults to the endpoint-weighted
   strict `component_balanced_j_val` protocol. The current default
   trajectory-robust PACER calculation
   (`trajectory_robust_component_role_q25_A_plus`) is exposed by
   `robust_profile_j_val`: outcome and no-regression terms remain nonzero,
   whole-chunk EEF/TCP trajectory alignment is the primary process-fidelity
   family, blocked rows score zero, and the scalar score is the lower quartile
   over component×role cells. Report A+ as a process-fidelity diagnostic unless
   it was frozen before a held-out evaluation.
10. Optionally propose the next eta candidate with Bayesian optimization.

## Main code entry points

The Python package path currently remains `pacer` for backward compatibility
with existing scripts and saved data, but the method name used in user-facing
docs is PACER.

- `pacer.weighting.action_chunk_compiler`
  - loads episode sidecars and action traces;
  - generates action chunks;
  - assigns roles/evidence;
  - computes target-distance and process features.

- `pacer.weighting.pacer_bo_weights`
  - defines `PacerEta`;
  - provides the default eta pool;
  - computes raw PACER evidence scores;
  - performs robust stratum normalization;
  - writes clipped loss weights.

- `pacer.weighting.pacer_validation_scoring`
  - validates candidate/row scoring inputs;
  - computes controlled validation aggregates;
  - keeps evaluation definitions explicit and auditable.

- `pacer.weighting.pacer_eval_contract`
  - validates evaluation report schemas;
  - guards Bayesian-optimization inputs.

- `pacer.weighting.pacer_bo`
  - proposes eta candidates from validation scores using GP/EI-style search.

## Current validation calculation

The generic framework exposes the current PACER validation calculation through
`pacer_framework.validation.robust_profile_j_val`. Calling it without an
explicit profile uses:

```text
trajectory_robust_component_role_q25_A_plus
```

This default A+ profile uses three factor families and should receive
whole-chunk `align` values produced with
`open_loop_submetrics(..., reference_alignment_mode="trajectory")`:

```text
outcome    = mean(progress, proximity, terminal)
trajectory = 0.15 * direction + 0.85 * align
row_score  = 0.20 * outcome + 0.70 * trajectory + 0.10 * no_regression
aggregation = lower quartile over component × role cells
```

This is the framework-level scoring method. Actual experiment result tables
belong in generated evaluation outputs or project-specific artifacts, not in
this GitHub method definition.

## Evidence dimensions

PACER’s current evidence vector is intentionally explicit:

- `progress`: signed or normalized progress toward the task target;
- `proximity`: distance/inside-band evidence near the target;
- `terminal`: terminal target-region evidence;
- `direction`: local movement direction or target-alignment evidence when
  available;
- `stop`: correct model stop/handoff behavior near the target;
- `operator`: human label/rank evidence;
- `provenance`: manifest, mask, VLM-authority, and quarantine checks.

Target mismatch does not have to delete an entire row, but it must prevent
positive geometric credit from being assigned to the wrong target. This keeps
operator/provenance diagnostics visible while preventing leakage into positive
imitation evidence.

## Verification-control notes

- PACER eta search proposes candidate weights; it is not a guarantee of globally
  converged optimality.
- EEF/TCP cosine and related geometric scores are diagnostics, not final robot
  task-success guarantees.
- Validation scores guide candidate selection; held-out robot success remains
  separate deployment evidence.
- Fixed norm-stat discipline is required for fair comparisons between candidate
  policies.
- Public/reusable docs should stay robot/VLA-agnostic unless a section is
  explicitly marked as runtime-adapter specific.
