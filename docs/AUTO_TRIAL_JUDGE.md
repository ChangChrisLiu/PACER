# Auto trial judge for rollout-only evaluation

`--auto-judge` lets `scripts/robot_runtime/run_rollout_eval.py` judge, stop and
label approach trials without an operator picking labels. Every predicted
action chunk is scored with PACER's open-loop geometry before it is executed;
the score is saved per chunk and drives the stop decision. The operator is
asked only when the plan moves to a new board position (`config_id/component`).

## Scope

- **Approach only.** The judge evaluates whether the TCP reaches the handoff
  region of the declared target. It does not evaluate downstream skills.
- **A geometric proxy.** The terminal check is a declared-region and pose
  predicate, not a measurement of physical success. It replaces the operator's
  label menu, not the operator: stay at the cell with the physical E-stop.
- **No collision model.** Blockers use the TCP point only (non-target regions,
  workspace box, action bounds, floor below the target).
- **OpenPI absolute-joint chunks.** Predicted TCP poses come from UR5e forward
  kinematics of `openpi_absolute_joint` chunks; `--auto-judge` refuses
  `openvla` / `openvla_oft` models (their `eef_delta` chunks would be misread).

## Rules

Thresholds: `r_target` = 0.005 m for CPU and 0.010 m otherwise; `d_ref` =
2 `r_target`; rotation tolerance 0.35 rad; floor = target z − 0.005 m;
`r_near` is set per session with `--auto-judge-r-near-m` (no default).

A pose is **inside** the target when position error ≤ `r_target`, rotation
error ≤ 0.35 rad and z ≥ floor. A target entry may declare `non_targets`:
components of the same configuration whose target points become obstacle
regions (radius = their `r_target`). Nothing is an obstacle unless declared, so
a component already removed at that position (the CPU fan before the CPU) never
blocks the approach.

For each predicted chunk, in order:

| Condition | Stop source | Chunk executed? |
|---|---|---|
| The policy's own stop token | `model_stop_token` | no |
| Actual TCP inside, and every predicted pose inside the same target point | `auto_success_stop` | no |
| Blocker on the predicted chunk: non-target region, workspace box, non-finite or > 2π joint values, any pose below the floor, or endpoint rotation > 0.35 rad once the endpoint is within `d_ref` | `auto_blocker_stop` | no |
| Otherwise | continue | yes |

Rollouts also end on timeout (600 steps) or an operator `s` / `q`. Auto stops
use the same robot stop handling as `model_stop_token` (passive settle and
OpenPI servo release).

Final labels (closed PACER vocabulary):

| Case | Label |
|---|---|
| Blocker: non-target region, workspace or action bound | `totally_off_wrong_region_or_target` |
| Blocker: below floor or invalid orientation | `wrong_orientation_or_wrong_location` |
| Final pose below the floor | `wrong_orientation_or_wrong_location` |
| Final pose inside the target | `success` |
| Position ≤ `d_ref`, rotation ≤ 0.35 rad | `near_but_not_accurate` |
| Position ≤ `d_ref` with rotation > 0.35 rad, or position ≤ `r_near` | `wrong_orientation_or_wrong_location` |
| Position > `r_near` | `totally_off_wrong_region_or_target` |
| No final TCP reading | `operator_uncertain_exclude` |
| Operator `q` | `unsafe_abort` |

Final poses within ±0.001 m of `r_target`, `d_ref`, `r_near` or the floor keep
their label and get `needs_review:<threshold>` in `operator_note`, so they stay
in the success-rate denominator and can be spot-checked from the recordings.
Feedback rows carry `operator = auto_judge` and `auto_assigned = true`.

Each trial directory also gets `chunk_scores.jsonl`: one row per policy query
with the decision and reasons, actual and predicted TCP, position and rotation
errors, PACER submetrics (`progress`, `proximity`, `terminal`, `direction`),
`v_hat`, blocker flags, `v_row` (zero when a blocker fired) and `fk_residual_m`
(distance between the previous chunk's predicted endpoint and the measured TCP;
it also contains servo tracking lag and safety clamping). `auto_judge_context.json`
records the target points, obstacle regions, workspace and thresholds the trial
was judged against, so re-recording a target later does not lose provenance.

## Recording targets

Record a demo at each board position and use its endpoint as the target:

```bash
python -m pacer.rollout_eval.auto_judge_targets add --targets .local/auto_judge_targets.json \
  --config-id config_001 --component ram --demo-episode <demo episode dir>   # last frame
python -m pacer.rollout_eval.auto_judge_targets add --targets .local/auto_judge_targets.json \
  --config-id config_001 --component connector --tcp-pose 0.41 -0.12 0.05 3.14 0.0 0.0 \
  --non-target ram          # the RAM target is an obstacle while judging the connector
python -m pacer.rollout_eval.auto_judge_targets show --targets .local/auto_judge_targets.json
```

`--frame-index N` selects another demo frame; `--target-region-json` imports a
PACER `target_region.json` point set. An optional `"workspace": {"min": [x, y, z],
"max": [x, y, z]}` entry in the file enables the workspace blocker. Targets are
keyed by `config_id/component`, so reuse one only if the board returns to the
same placement; otherwise re-record it at the position prompt.

## Before the first live session

Check the forward kinematics against a recorded episode (frames with
`joint_positions` and `tcp_pose`):

```bash
python -m pacer.rollout_eval.ur5e_kinematics --check-episode <episode dir>
```

The check solves the tool offset once from frame 0 and compares the whole
episode, which is stricter than live use (the judge re-solves the offset at
every query and predicts only one chunk ahead). Residuals of a few millimetres
can come from nominal versus calibrated DH parameters; a centimetre-level
residual means the controller base frame differs from the DH base frame, and
the judge must not be used until that is resolved. During live runs, watch
`fk_residual_m`.

Then dry-run and run a short supervised session:

```bash
python scripts/robot_runtime/run_rollout_eval.py --run-id <id> --models-file <models.json> \
  --auto-judge --auto-judge-targets .local/auto_judge_targets.json --auto-judge-r-near-m <metres> --dry-run
```

The dry run records the judge settings in `eval_session_meta.json` and lists
plan cells that still lack a target.

## Reporting

- State that trial labels were assigned automatically by this rule set, and
  give the thresholds.
- Report agreement with operator labels on a labelled subset before relying on
  automatic labels.
- Keep every demo used to define a target out of the training data of every
  evaluated policy.
