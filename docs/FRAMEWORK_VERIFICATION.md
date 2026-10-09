# PACER framework verification

This document maps PACER framework capabilities to the reusable code and tests in this repository. It is for users who want to inspect what the framework implements and how each behavior is verified.

It intentionally avoids experiment result tables, private datasets, checkpoint paths, lab logs, or project-specific history.

## Implementation matrix

| Framework capability | Code | Tests / checks |
| --- | --- | --- |
| Typed rows have roles `clean`, `correction`, `auto_success`, `partial`, `failure`, and `excluded`. | `PACER_Framework/pacer_framework/roles.py` | `PACER_Framework/tests/test_roles.py` |
| Source/label mapping is deterministic and failure/excluded roles are audit-only. | `role_from_source_and_label`, `normalize_role` | `test_roles.py`, `test_gate.py` |
| Evidence vector is bounded over `{prog, prox, term, dir, stop, op, prov}`. | `PACER_Framework/pacer_framework/evidence.py`, `eta.EVIDENCE_KEYS` | `test_evidence.py`, `test_eta.py` |
| Geometry uses the declared target, not nearest-object reassignment. | `target_consistency`, `_geometric_evidence` | `test_evidence.py`, `test_alignment.py` |
| Progress/proximity/terminal/direction are computed over the valid mask prefix. | `geometric_distances`, `_valid_position_prefix`, `_geometric_evidence` | `test_evidence.py` |
| Operator evidence follows a closed vocabulary; clean/correction rows carry constant operator evidence. | `PSI_OP`, `psi_op` | `test_evidence.py` |
| Provenance evidence checks manifest/timing/component/mask validity. | `PROVENANCE_FLAGS`, `provenance_check` | `test_evidence.py`, `test_gate.py` |
| Eligibility gate is `G_split * G_mask * G_prov * G_safe * G_role`. | `PACER_Framework/pacer_framework/gate.py` | `test_gate.py` |
| Failure and excluded rows are retained for audit but receive zero positive imitation weight. | `gate.py`, `weights.row_weight` | `test_gate.py`, `test_weights.py` |
| Eta has eight free fields and nine fixed fields with declared bounds. | `PACER_Framework/pacer_framework/eta.py` | `test_eta.py` |
| Process score is a raw weighted sum of evidence, not a learned reward or value model. | `eta.process_score` | `test_eta.py` |
| Robust centering uses training-only median and `1.4826 * MAD` with fallback from `(component, phase, role)` to `(component, phase)` to global. | `weights.robust_center_scale`, `weights.compile_weights` | `test_weights.py`, `test_end_to_end.py` |
| Runtime-adapter eta weighting uses the same robust median/MAD scale floor. | `pacer/weighting/pacer_bo_weights.py` | `tests/weighting/test_pacer_bo_weights.py` |
| Row weight is gate-dominated, clipped, and floor-aware. | `weights.row_weight`, `weights.compile_weights` | `test_weights.py` |
| Loss normalizer is the weighted masked-horizon mass; PACER exports weights but external VLA training computes the native action loss. | `weights.compile_weights`, `export.py` | `test_weights.py`, `test_export.py` |
| Effective weighted horizon count is computed for pre-training audit. | `weights.compile_weights` manifest field `n_eff_horizons` | `test_weights.py`, `test_end_to_end.py` |
| Validation score is component-balanced over positive-scoring roles; failure/excluded validation rows are audit-only. | `validation.component_balanced_j_val` | `test_validation.py` |
| Shared configurable direct component-mean default, whole-chunk alignment, reference-aware rescoring and configuration fingerprints | `ScoringConfig`, `component_balanced_j_val`, `robust_profile_j_val`, `SCORING_CONFIGURATION.md` | `test_scoring_config.py`, `test_scoring_cli.py`, explicit-profile replay tests |
| Blocker zeros wrong-target, non-target exclusion, invalid orientation, or unsafe validation rows; terminal non-entry is scored, not blocked. | `validation.blocker`, `validation.row_score` | `test_validation.py` |
| Candidate feasibility combines pre-training audit, wrong-target audit, safety audit, and no-regression audit. | `validation.candidate_feasibility`, `evaluate_candidate` | `test_validation.py`, `test_end_to_end.py` |
| Selection is argmax over audited candidates, with fallback to the reference policy if none pass. | `validation.select_candidate` | `test_validation.py`, `test_end_to_end.py` |
| Baseline/export utilities are controlled views over the same rows rather than hidden model changes. | `baselines.py`, `export.py`, `reporting.py` | `test_baselines.py`, `test_export.py`, `test_reporting.py` |
| ZMQ collection is optional data movement, not a safety boundary. | `collect_zmq.py`, `docs/ZMQ_COLLECTION.md` | `test_collect_zmq.py` |
| Public repo excludes private logs, checkpoints, local paths, internal result tables, and lab-specific scratch artifacts. | `.gitignore`, README, scope docs, hygiene tests | root test suite hygiene checks |

## Runtime-adapter note

The root `pacer/` package contains compatibility utilities for action-chunk compilation, runtime collection, and validation-score contracts. It is useful for existing robot/VLA deployments, but new integrations should start from `PACER_Framework/`.

## Repository boundary

Do not add raw experiment values, private checkpoint identifiers, local machine paths, cluster job paths, lab IPs, internal result tables, or unpublished evaluation outputs to this repository. Use generic placeholders such as `/tmp/example_ckpt`, `pacer_runtime_lora_10hz`, or `example_training_run` in examples and tests.
