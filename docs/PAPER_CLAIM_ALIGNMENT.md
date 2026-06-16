# PACER paper claim-to-code alignment

This document maps the publication-facing PACER claims to the reusable code in
this repository. It is intentionally generic: it records formulas, schemas, and
tests, not private datasets, raw logs, checkpoint paths, or paper result tables.

## Scope

The paper describes PACER as a data-side VLA post-training framework. PACER does
not change the VLA backbone, action representation, action decoder, or native
per-horizon supervised action objective. It changes only the row contribution to
that objective through fixed process evidence, eligibility gates, and
validation-selected eta configurations.

The implementation is split into two layers:

- `PACER_Framework/`: paper-faithful, robot/VLA-agnostic reference
  implementation for publication and reuse.
- `pacer/` and `scripts/`: runtime-adapter utilities and no-hardware smokes for
  labs that already have compatible robot, camera, and policy servers.

## Claim matrix

| Paper claim | Publication code | Tests / checks |
| --- | --- | --- |
| Typed rows have roles `clean`, `correction`, `auto_success`, `partial`, `failure`, and `excluded`. | `PACER_Framework/pacer_framework/roles.py` | `PACER_Framework/tests/test_roles.py` |
| Source/label mapping is deterministic and failure/excluded roles are audit-only. | `role_from_source_and_label`, `normalize_role` | `test_roles.py`, `test_gate.py` |
| Evidence vector is bounded over `{prog, prox, term, dir, stop, op, prov}`. | `PACER_Framework/pacer_framework/evidence.py`, `eta.EVIDENCE_KEYS` | `test_evidence.py`, `test_eta.py` |
| Geometry uses the declared target, not nearest-object reassignment. | `target_consistency`, `_geometric_evidence` | `test_evidence.py`, `test_alignment.py` |
| Progress/proximity/terminal/direction follow the appendix equations over the valid mask prefix. | `geometric_distances`, `_valid_position_prefix`, `_geometric_evidence` | `test_evidence.py` |
| Operator evidence follows the closed vocabulary; clean/correction rows carry constant operator evidence. | `PSI_OP`, `psi_op` | `test_evidence.py` |
| Provenance evidence checks manifest/timing/component/mask validity. | `PROVENANCE_FLAGS`, `provenance_check` | `test_evidence.py`, `test_gate.py` |
| Eligibility gate is `G_split * G_mask * G_prov * G_safe * G_role`. | `PACER_Framework/pacer_framework/gate.py` | `test_gate.py` |
| Failure and excluded rows are retained for audit but receive zero positive imitation weight. | `gate.py`, `weights.row_weight` | `test_gate.py`, `test_weights.py` |
| Eta has eight free fields and nine fixed fields with declared bounds. | `PACER_Framework/pacer_framework/eta.py` | `test_eta.py` |
| The process score is the raw weighted sum of evidence, not a learned reward or value model. | `eta.process_score` | `test_eta.py` |
| Robust centering uses training-only median and `1.4826 * MAD` with fallback from `(component, phase, role)` to `(component, phase)` to global. | `weights.robust_center_scale`, `weights.compile_weights` | `test_weights.py`, `test_end_to_end.py` |
| Runtime-adapter PACER eta weighting uses the same robust median/MAD scale floor and does not substitute a standard-deviation fallback. | `pacer/weighting/pacer_bo_weights.py` | `tests/weighting/test_pacer_bo_weights.py::test_runtime_robust_stats_use_mad_floor_without_standard_deviation_fallback` |
| Row weight is gate-dominated, clipped, and floor-aware: `g_i * clip(max(f_role, gamma_role * exp(...)), 0, w_max)`. | `weights.row_weight`, `weights.compile_weights` | `test_weights.py` |
| Runtime-adapter PACER eta weighting uses the same paper placement for role multipliers: raw score first, then `gamma_role * exp(...)` in the final weight. | `pacer/weighting/pacer_bo_weights.py` | `tests/weighting/test_pacer_bo_weights.py::test_pacer_weight_applies_role_multiplier_outside_centered_exponential` |
| Loss normalizer is the weighted masked-horizon mass; PACER exports weights but external VLA training computes the native action loss. | `weights.compile_weights`, `export.py` | `test_weights.py`, `test_export.py` |
| Effective weighted horizon count is computed for pre-training audit. | `weights.compile_weights` manifest field `n_eff_horizons` | `test_weights.py`, `test_end_to_end.py` |
| Validation score is component-balanced over positive-scoring roles; failure/excluded validation rows are audit-only. | `validation.component_balanced_j_val` | `test_validation.py` |
| Current default trajectory-robust PACER calculation is A+: outcome/non-regression remain nonzero, whole-chunk EEF/TCP trajectory alignment is primary, blocked rows score zero, and aggregation is lower quartile over component×role cells. | `validation.DEFAULT_ROBUST_SCORER_PROFILE`, `validation.robust_profile_j_val`, `docs/TRAJECTORY_ROBUST_SCORING.md` | `test_validation.py::test_trajectory_robust_default_profile_is_a_plus`, `test_validation.py::test_trajectory_robust_profile_uses_component_role_lower_tail_not_mean` |
| Blocker zeros wrong-target, non-target exclusion, invalid orientation, or unsafe validation rows; terminal non-entry is scored, not blocked. | `validation.blocker`, `validation.row_score` | `test_validation.py` |
| Candidate feasibility combines pre-training audit, wrong-target audit, safety audit, and no-regression audit. | `validation.candidate_feasibility`, `evaluate_candidate` | `test_validation.py`, `test_end_to_end.py` |
| Selection is `argmax J_val` among audited candidates, with fallback to the reference policy if none pass. | `validation.select_candidate` | `test_validation.py`, `test_end_to_end.py` |
| Baseline/export utilities are controlled views over the same rows rather than hidden model changes. | `baselines.py`, `export.py`, `reporting.py` | `test_baselines.py`, `test_export.py`, `test_reporting.py` |
| ZMQ collection is optional data movement, not a safety boundary. | `collect_zmq.py`, `docs/ZMQ_COLLECTION.md` | `test_collect_zmq.py` |
| Publication repo excludes private logs, checkpoints, local paths, internal result tables, and lab-specific scratch artifacts. | `.gitignore`, README, scope docs, hygiene test | `tests/test_publication_hygiene.py` |

## Runtime-adapter note

The root `pacer/` package contains compatibility utilities for action-chunk
compilation, runtime collection, and validation-score contracts. It is useful for
existing robot/VLA deployments, but the paper-faithful minimal implementation is
`PACER_Framework/`. Runtime adapters may carry legacy field names for saved-data
compatibility; user-facing documentation should describe the method as PACER and
use paper-facing names such as `J_val`, `fixed_geometry`, and `terminal_stop_heavy`.

## Privacy and result-boundary rule

Do not add raw experiment values, private checkpoint identifiers, local machine
paths, cluster job paths, lab IPs, or final paper result tables to this repo.
When a test needs an example checkpoint/config/string, use generic placeholders
such as `/tmp/example_ckpt`, `pacer_runtime_lora_10hz`, or `example_training_run`.
When a document needs results, link to the paper after publication rather than
embedding unpublished tables in the code repository.
