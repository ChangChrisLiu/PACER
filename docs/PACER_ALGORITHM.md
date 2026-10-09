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
9. Score candidates using one shared `ScoringConfig`: whole-chunk trajectory
   alignment, target-direction and other applicable metrics, gated direct row
   weighting and component-balanced mean. Candidate and baseline geometry use
   the same coefficients for no-regression. `evaluate_candidate` checks full
   feasibility; `select_candidate` takes the audited argmax. Named factor-family
   diagnostics remain explicit opt-ins and do not define the default score.
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

`pacer_framework.component_balanced_j_val` is the canonical scorer.
`robust_profile_j_val` without a profile and the runtime row-scoring facade use
that same calculation. `open_loop_submetrics` defaults to whole-chunk trajectory
alignment; endpoint mode must be explicit.

`ScoringConfig` exposes the shared coefficients, alignment representation,
no-regression margin and reference-audit requirement. Export an editable JSON
configuration through `python -m pacer_framework.scoring --write-config`, then
reuse it for every candidate and baseline. No task-specific parameter recipe is
part of this method description. See
[`SCORING_CONFIGURATION.md`](../PACER_Framework/docs/SCORING_CONFIGURATION.md).

A+ and other named lower-tail profiles remain explicit compatibility diagnostics.
Changing aggregation changes the score definition, not only its coefficients.
Root aggregate BO/smoke adapters retain their existing report contracts; they
are not substitutes for canonical row-level scoring or complete audits in a new
integration. Synthetic or proxy outputs are not physical validation evidence.

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
