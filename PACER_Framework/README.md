# PACER_Framework — process-aware VLA post-training for any robot, any VLA

**PACER** (Process-Aware Correction and Evidence Reweighting) turns the mixed
data a deployed robot policy actually produces — imperfect rollouts, human
corrections, a few clean demonstrations — into auditable training credit for
the **native** VLA action loss, without a reward model, a value function, or
any architecture change. This package is the generic, robot/VLA-agnostic
reference implementation: bring your own arm, your own VLA, your own targets,
and run the pipeline below. Pure Python stdlib, offline-only — no robot
drivers, no model code, no hardware imports.

Framework contracts are implemented directly and locked by tests (verification map below).

## The PACER pipeline

```
 (1) define targets ─> (2) record 3 data sources ─> (3) standard rows
        │                                                  │
        v                                                  v
 (7) pick best candidate <─ (6) offline open-loop  <─ (4) evidence/gates/weights
     for the hardware run       evaluation: EEF        -> 8 training views
     (J_val + audits)           cosine, v_j, J_val,        │
                                audits, tables         (5) train candidate
                                                           VLAs externally
```

1. **Define target regions and reference geometry.** For each task (component
   class) and scene configuration: capture the target point(s) with your
   teleop/jog interface BEFORE any scoring, pick position/orientation
   tolerances, and record the EEF/TCP frame convention. These declared targets
   are fixed and never reassigned from where the policy happened to go.
   → `docs/RUNBOOK_DATA_COLLECTION.md` §1
2. **Record three data sources** in the standard schema, all with your
   current/base VLA policy in the loop:
   - **rollouts** of the current policy, each given one closed-vocabulary
     outcome label;
   - **human corrections** recorded after/around failures and partial
     successes (recoveries from policy-induced states);
   - **supplementary clean demonstrations** for the cells the policy cannot
     reach. → `docs/RUNBOOK_DATA_COLLECTION.md` §2–4
3. **Convert records into PACER standard rows**: per-horizon masks, process
   roles, target metadata, safety/provenance flags, and either raw
   geometry/EEF traces or precomputed evidence — then validate
   (`schema.validate_training_row`) and split configuration-disjointly
   (`split.assign_case_group_split`). → `docs/DATA_CONTRACT.md`
4. **Compute evidence, gates, scores, and weights, and build the training
   datasets/views** (`compile_weights`, `compile_baseline`,
   `export_training_view`):
   - **pacer_selected** — the PACER-weighted dataset (per candidate η);
   - **vanilla_post_sft** — clean + correction at weight 1;
   - **uniform_all_eligible**; **outcome_only**; **clean_only**;
     **correction_only**; **fixed_geometry**;
   - **no_post_train** — the reference policy, no further training.
   → `docs/RUNBOOK_TRAINING_PREP.md`
5. **Train next-step VLA candidates externally** with those views: your
   trainer, your checkpointing, the native action loss multiplied by each
   row's exported `loss_weight` — same starting checkpoint, recipe, and update
   budget for every candidate and baseline.
6. **Evaluate the trained candidates offline / open-loop** using
   rollout/reference EEF traces: EEF cosine and reference-direction alignment
   (`alignment.reference_alignment`, eq:app_align), row scores v_j with
   blockers, component-balanced J_val, feasibility audits, and the comparison
   tables (`evaluate_candidate`, `success_table`,
   `paired_component_bootstrap`). PACER also exposes named trajectory-primary
   robust scorer profiles (`robust_profile_j_val`) for diagnostics where
   whole-chunk process fidelity is the declared validation objective. The
   current default robust profile is `trajectory_robust_component_role_q25_A_plus`:
   it keeps outcome/no-regression terms nonzero, scores blocked rows as zero,
   and aggregates by the lower quartile over component×role cells so controls
   are judged by the same robustness rule as PACER candidates. See
   `docs/TRAJECTORY_ROBUST_SCORING.md` for the exact default parameters and
   score-computation recipe.
   → `docs/RUNBOOK_EVALUATION.md`
7. **Pick the model for the hardware run**: the audited candidate with the
   highest J_val (`select_candidate`); when no candidate passes the audits,
   fall back to the reference policy. Only the selected model (plus baselines,
   for the comparison) goes to the protected hardware evaluation.

Try it immediately:

```bash
cd PACER_Framework
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pacer_framework.check_setup
python3 -m pytest tests -q            # framework tests
python3 examples/end_to_end_demo.py   # full pipeline on a synthetic robot
```

For full GitHub-style onboarding, including optional robot/VLA packages, read
`docs/SETUP.md` first. Then implement the hook boundary in
`docs/ADAPTER_HOOKS.md`. If your robot/simulator streams logs from another
process, use `docs/ZMQ_COLLECTION.md` and the optional
`python -m pacer_framework.collect_zmq` receiver. For the concrete terminal
layout and the PACER runtime-adapter collection/inference scripts, read
`docs/TELEOP_COLLECTION_INFERENCE.md`.

`examples/end_to_end_demo.py` walks steps 1–7 on an imaginary arm with two
targets; `tests/test_end_to_end.py` locks the same journey, and
`examples/*.jsonl` are contract-valid record templates to copy.

## Package layout

```
docs/      SETUP.md                   <- GitHub clone/install/verify guide
           USAGE_QUICKSTART.md        <- shortest path for new robot/VLA users
           DATA_CONTRACT.md           <- standard record schema (step 3)
           ADAPTER_HOOKS.md           <- connect your robot, VLA, and trainer
           TELEOP_COLLECTION_INFERENCE.md <- concrete teleop/ZMQ/data/inference workflow
           ZMQ_COLLECTION.md           <- optional live JSONL data collection over ZeroMQ
           RUNBOOK_DATA_COLLECTION.md <- steps 1-3
           RUNBOOK_TRAINING_PREP.md   <- steps 4-5
           RUNBOOK_EVALUATION.md      <- steps 6-7
           TRAJECTORY_ROBUST_SCORING.md <- current default trajectory-primary scorer
examples/  training_rows.jsonl, validation_rows.jsonl,
           candidate_terminal_stop_heavy.json, end_to_end_demo.py
pacer_framework/
           roles.py      <- process roles, label mapping, strict success
           eta.py        <- candidate configurations (8 free + 9 fixed fields)
           evidence.py   <- 7-field evidence vector incl. target consistency T_i
           gate.py       <- eligibility gate g_i (split/mask/provenance/safety/role)
           weights.py    <- stratum centering + the row-weight equation + manifests
           schema.py     <- executable data contract + templates
           split.py      <- configuration-disjoint case-group split
           baselines.py  <- the 8 training views of step 4
           export.py     <- trainer-ready rows.jsonl with loss_weight
           hooks.py      <- RobotHooks/VLAHooks/TrainerHooks integration boundary
           alignment.py  <- EEF cosine, reference alignment, open-loop submetrics, blockers
           validation.py <- v_j, J_val, feasibility audits, candidate selection
           reporting.py  <- Wilson CIs, paired component bootstrap, ranking tables
tests/     one file per module + test_end_to_end.py + test_examples.py
```

## Integration hooks for your robot and VLA

PACER does not know your action convention or model API. Keep those in your
own code and implement the small hook protocols in `pacer_framework/hooks.py`
(full guide: `docs/ADAPTER_HOOKS.md`):

- `RobotHooks`: declare fixed target regions, integrate a predicted action
  chunk into base-frame TCP/EEF positions, and expose safety flags. This is
  where robot type differences live: single-arm vs bimanual, joint-space vs
  EEF-delta actions, absolute waypoints, gripper channels, mobile-base motion,
  simulator vs real robot.
- `VLAHooks`: query a candidate model once on a logged observation and report
  whether it emitted the stop/handoff token. This works for OpenVLA-style
  policies, Diffusion/Flow action heads, RT-style action tokenizers, OpenPI
  policies, or custom PyTorch/JAX stacks — PACER only needs the resulting
  action chunk.
- `TrainerHooks`: launch your training stack on an exported view. The only
  required contract is that the native per-horizon action loss is multiplied by
  each row's `loss_weight`.

The framework consumes standard rows and base-frame TCP/EEF positions after
those hooks run; it never imports robot SDKs, model code, ROS, PyTorch, JAX,
OpenPI, or LeRobot.

## Framework contract → code → test map

| Framework contract | Implementation | Locked by test |
|---|---|---|
| eq:process_role, Table app_rolemap | `roles.role_from_source_and_label` | `test_roles.py::test_role_from_source_and_label_matches_table_app_rolemap` |
| eq:app_strict (strict success) | `roles.strict_success` | `test_roles.py::test_strict_success_eq_app_strict` |
| Eta free-field bounds | `eta.FREE_BOUNDS`, eta validation helpers | `test_eta.py` |
| Eta fixed protocol fields | eta configuration fields | `test_eta.py` |
| Built-in candidate pool | `eta.CANDIDATE_POOL` | `test_eta.py` |
| s_eta = Σ β_k e_k (raw coefficients) | `eta.process_score` | `test_eta.py::test_process_score_uses_raw_coefficients_without_renormalization` |
| eq:app_dist / eq:app_geom (incl. T_i) | `evidence.compute_evidence`, `target_consistency` | `test_evidence.py::test_target_inconsistency_zeroes_geometric_evidence_only` |
| eq:app_term + component tolerances | `evidence.r_target_for/d_ref_for` | `test_evidence.py::test_cpu_uses_tighter_tolerance` |
| ψ_op lookup; e_op = 1 for clean/correction | `evidence.psi_op` | `test_evidence.py::test_psi_op_lookup` |
| P_i provenance check | `evidence.provenance_check` | `test_evidence.py::test_provenance_check_requires_all_four_flags` |
| S_i stop evidence | `evidence.stop_evidence` | `test_evidence.py::test_stop_evidence_semantics` |
| eq:app_gate (5 factors, zero-mass mask) | `gate.eligibility_gate` | `test_gate.py` (one test per factor) |
| eq:app_center (median/MAD, σ_min) | `weights.robust_center_scale` | `test_weights.py::test_robust_center_scale_is_median_mad_with_sigma_floor` |
| eq:app_fallback (3-level) | `weights.compile_weights` | `test_weights.py::test_compile_weights_stratum_statistics_and_fallback_levels` |
| eq:weight (γ outside exp, clip [0, w_max], floors, gate dominance) | `weights.row_weight` | `test_weights.py::test_row_weight_matches_eq_weight_hand_computation` |
| eq:n_eff + eq:loss denominator | `weights.compile_weights` manifest | `test_weights.py::test_n_eff_and_loss_normalizer_over_weighted_masked_horizons` |
| Case-group split (app:split) | `split.assign_case_group_split` | `test_split.py::test_split_is_configuration_disjoint_and_whole_group` |
| Standard records (app:roles / app:valscore) | `schema.validate_*`, templates | `test_schema.py`, `test_examples.py` |
| eq:row_val + per-row λ renormalization | `validation.row_score` | `test_validation.py::test_row_score_renormalizes_over_applicable_submetrics` |
| eq:app_bj (blocker; terminal failure never blocked) | `validation.blocker`, `alignment.blocker_flags_from_open_loop` | `test_validation.py`, `test_alignment.py::test_blocker_flags_wrong_target_and_exclusion` |
| **eq:app_align (EEF reference-direction cosine)** | `alignment.reference_alignment` | `test_alignment.py::test_reference_alignment_eq_app_align` |
| Direction submetric (EEF target cosine) | `alignment.target_direction_cosine` | `test_alignment.py::test_target_direction_cosine_reads_chunk_displacement` |
| eq:app_vhat / eq:app_reg | `validation.geometric_row_score`, `no_regression`, `alignment.with_no_regression` | `test_validation.py`, `test_alignment.py` |
| eq:j_val (component-balanced) | `validation.component_balanced_j_val` | `test_validation.py::test_component_balanced_j_val_averages_components_not_rows` |
| Current default trajectory-primary robust diagnostic (framework completion for process-fidelity scoring) | `validation.DEFAULT_ROBUST_SCORER_PROFILE`, `validation.scorer_profile`, `validation.robust_profile_j_val`, `docs/TRAJECTORY_ROBUST_SCORING.md` | `test_validation.py::test_trajectory_robust_profile_contract_is_named_and_explicit`, `test_validation.py::test_trajectory_robust_default_profile_is_a_plus`, `test_validation.py::test_trajectory_robust_profile_uses_component_role_lower_tail_not_mean` |
| eq:app_feasibility (A_audit·A_wrong·A_safe·A_reg) | `validation.candidate_feasibility`, `evaluate_candidate` | `test_validation.py::test_candidate_feasibility_eq_app_feasibility` |
| eq:selection + reference fallback | `validation.select_candidate` | `test_validation.py::test_select_candidate_argmax_with_reference_fallback` |
| Table app_baselines (8 views) + eq:app_fixed_geometry | `baselines.compile_baseline` | `test_baselines.py` |
| Reporting utilities for held-out success CIs | `reporting.wilson_interval`, `success_table` | `test_reporting.py::test_wilson_interval_matches_independent_synthetic_cases` |
| Paired task/component bootstrap comparison | `reporting.paired_component_bootstrap` | `test_reporting.py::test_paired_component_bootstrap_reports_synthetic_point_differences` |
| Candidate ranking table | `reporting.candidate_ranking_table` | `test_reporting.py::test_candidate_ranking_table_shape_and_order` |
| eq:loss export channel | `export.export_training_view` | `test_export.py` |
| Whole pipeline from standard records | all of the above | `test_end_to_end.py::test_generic_user_end_to_end` |

## What you bring vs what PACER provides

| You (robot/VLA-specific) | PACER_Framework (generic) |
|---|---|
| Robot, teleop, target capture, prompts | data contract, validators, templates |
| Base policy π_θ0 + rollout/correction/demo recording | role taxonomy, labels, strict success |
| Action↔EEF conventions; open-loop integration of predicted chunks to base-frame TCP positions | evidence, gates, weights, splits, all 8 views, trainer-ready exports |
| The VLA training loop (consumes `loss_weight`) | EEF cosine/alignment, v_j, J_val, audits, selection, comparison tables |

Set tolerances per target class for your own platform. If a row omits explicit
`tolerances`, PACER falls back to the protocol defaults in `evidence.py`; for a
new robot, prefer recording `r_target` and `d_ref` in each target/geometry
configuration.

## Repository note

This folder is the generic PACER framework intended for new users. If this
repository also contains experiment-specific pipelines or historical artifacts,
treat them as provenance for that project only; build new integrations against
`PACER_Framework` and the hook/data-contract files above.
