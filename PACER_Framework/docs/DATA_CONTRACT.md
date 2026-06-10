# PACER standard data contract (robot/VLA-agnostic)

This is the prose companion to the executable contract in
`pacer_framework/schema.py` (`validate_training_row`, `validate_validation_row`,
`training_row_warnings`, and the `*_template()` builders). If the validators
accept your records, the whole pipeline — evidence, gates, weights, splits,
validation scoring, baselines — runs on them.

Units and frames: positions are `[x, y, z]` in meters in the **robot base
frame**. Roles may use either the paper vocabulary
(`clean / correction / auto_success / partial / failure / excluded`) or the
compiler vocabulary (`clean_demo / human_correction / model_success /
model_partial / model_failure`); everything normalizes internally.

## 1. Training row (one compiled action-chunk row)

One row = one action chunk of one trace (paper app:roles:
`x_i = (o_i, a_{i,1:H}, m_{i,1:H}, c_i, φ_i, t_i_act, q_i, z_i, ρ_i)`).
The observation/action tensors `o_i, a_{i,1:H}` stay in your training stack;
PACER consumes the typed metadata:

| Field | Required | Meaning |
|---|---|---|
| `row_id` | yes | stable unique id (`trace:chunk` recommended) |
| `component` | yes | task target class `c_i` (e.g. "door_handle", "part_slot") |
| `phase` | yes | approach-stage label `φ_i` (stratum dimension) |
| `split` | yes\* | `train` / `val` / `heldout` — \*written for you by `assign_case_group_split` |
| `collection_config` | for splitting | physical placement / scene configuration id of the trace |
| `role` or `sample_role` | yes | process role `ρ_i`, either vocabulary |
| `operator_label` | autonomous rows | one of the five closed-vocabulary labels (drives `e_op`) |
| `mask` | yes | per-horizon list of 0/1 (`m_{i,1:H}`, eq:app_mask) |
| `evidence` | one of these two | precomputed `{prog, prox, term, dir, stop, op, prov}` each in [0,1] |
| `geometry` | one of these two | raw inputs for evidence (next table) |
| `target_meta` | for T_i | `target_id`, `declared_target_id`, optional `observed_target_id`, `consistency`, `geometry_record_present` |
| `stop` | recommended | `{phase_ending, stop_event, inside_terminal}` (drives `e_stop`) |
| `provenance` | recommended | `{manifest_valid, timing_aligned, component_identity_valid, action_mask_valid}` — absent block = trusted |
| `safety` | recommended | `{unsafe, manual_safety_stop, quarantined}` — absent block = safe |
| `rank_weight` | fixed-geometry baseline only | frozen prior-pipeline geometric rank scalar `r_rank` |

`geometry` block, either form:

| Form | Fields |
|---|---|
| trajectory | `tcp_positions` (list of `[x,y,z]`, ideally start + one per horizon), `target_point` `[x,y,z]`, optional `target_id`, `orientation_ok`, `terminal_region_member`, `r_target`, `d_ref` |
| summary | `d0`, `d_end`, optional `d_min`, `direction_cosine` (when distances were computed upstream) |

Tolerances are **yours to set per target class**: put `r_target`/`d_ref` in the
geometry block of each row (or wrap the defaults once for your platform). When
omitted, the paper-protocol defaults apply — 0.010 m with a tighter 0.005 m
for the component class literally named "cpu", and `d_ref = 2·r_target`.

**T_i strictness (paper eq:app_geom):** the geometric evidence of a row is
zeroed unless the component id, phase id, target id, and a target-geometry
record are present and consistent. `training_row_warnings(row)` tells you
before compilation which rows will be affected and why.

## 2. Validation-scoring row

One row = one validation record scored against a trained candidate
(paper app:valscore). Your evaluator queries the candidate on the logged
observation, integrates the predicted chunk open-loop to TCP positions, and
either calls `pacer_framework.alignment` to obtain submetrics + blocker flags
or supplies them directly:

| Field | Required | Meaning |
|---|---|---|
| `row_id`, `component` | yes | identity |
| `role` / `sample_role` | yes | scoring set = {clean, correction, auto_success, partial}; failure/excluded rows are audit-only |
| `submetrics` | yes | bounded values among `progress, proximity, terminal, direction, stop, align, no_regression`; absent metrics are omitted and λ renormalizes |
| `wrong_target`, `non_target_exclusion`, `invalid_orientation`, `unsafe` | recommended | blocker flags `B_j` (eq:app_bj); absent = false |

## 3. Candidate configuration (η)

JSON dict of `PaperEta` fields — 8 free (`beta_prog, beta_prox, beta_term,
beta_stop, gamma_correction, gamma_partial, tau, w_max`, bounds in
`eta.FREE_BOUNDS`) + 9 fixed protocol fields. See
`examples/candidate_terminal_stop_heavy.json`. `PaperEta.from_dict` loads it;
`validate_bounds()` (run automatically by `compile_weights`) enforces the
declared search ranges.

## 4. Produced manifests

- weight manifest (`compile_weights`): counts by role/component, strata,
  `n_eff_horizons`/`n_eff_rows`, `loss_normalizer`, weight stats,
  `forbidden_positive_weight_rows` (must be 0) — the pre-training audit input.
- split manifest (`assign_case_group_split`): per-component
  train/val/heldout configuration plan + seed.
- training view (`export_training_view`): `rows.jsonl` with top-level
  `loss_weight` per row + `manifest.json` — the only thing your trainer needs.
- baseline manifest (`compile_baseline`): `train_rows`, `valid_horizons`,
  `nominal_start_rows` (the Table app_baselines columns).
