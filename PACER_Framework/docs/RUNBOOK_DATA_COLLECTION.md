# Runbook 1 — Targets and the three data sources (pipeline steps 1–3)

Audience: anyone applying PACER to their own robot + VLA. "Component" below is
any task-target class on your platform (a connector, a door handle, a part
slot); "configuration" is any distinct physical placement/scene of it.

## 1. Define target regions and reference geometry (pipeline step 1)

Do this per component × configuration, BEFORE any rollout is scored:

1. **Capture the declared target**: jog/teleoperate the end effector to the
   target and record the point (set) in the robot base frame. This is the
   declared active target `t_act`. It is fixed by the task state and is never
   reassigned from wherever the policy later goes — that is what keeps
   wrong-target behavior identifiable.
2. **Pick tolerances**: a position tolerance `r_target` per component,
   a proximity normalizer `d_ref` (commonly `2·r_target`), and any orientation
   tolerance your downstream skill needs. These are your platform/task choices;
   record them once and put per-row/per-target overrides in the geometry block.
3. **Fix the EEF/TCP convention**: which frame your TCP positions are in
   (base frame, meters) and how predicted action chunks will later be
   integrated open-loop to TCP positions for evaluation. PACER consumes
   positions, never raw actions.
4. **Fix the prompt dictionary**: one canonical language instruction per
   component, reused for every recording and evaluation.
5. Decide the number of collection configurations per component (≥ 2; one
   becomes validation) and the split seed — now, not after looking at data.

Also fix the **success criterion**: a rollout is successful when the policy
reaches a region from which your fixed downstream skill/primitive could be
executed; PACER scores approach/handoff quality, not full task completion.

## 2. Data source A — rollouts of the current/base VLA policy

For each component × configuration:

1. Reset the scene; issue the component's canonical prompt; run the
   current/base policy π_θ0 as the action-chunk planner.
2. A rollout ends on the model stop token, a manual operator stop, a fixed
   timeout, or an unsafe abort.
3. Log per chunk: TCP positions (base frame), per-horizon action validity
   (→ `mask`), the stop source, manifest/timing health (→ `provenance`),
   safety events (→ `safety`).
4. The operator assigns exactly ONE closed-vocabulary label per rollout —
   `success`, `near_but_not_accurate`, `wrong_orientation_or_wrong_location`,
   `totally_off_wrong_region_or_target`, or `operator_uncertain_exclude` —
   from the final recorded state, with one rubric for every method, never
   relabeling after aggregate results exist. Strict success additionally
   requires the stop source ∉ {timeout, unsafe_abort, intervention}
   (`roles.strict_success`).
5. Roles follow deterministically (`roles.role_from_source_and_label`):
   success → auto_success, near → partial, the two wrong-* labels → failure,
   uncertain → excluded.

## 3. Data source B — human corrections (after/around failures)

Corrections are teleoperated **recoveries from policy-induced states** — they
cover exactly the failure neighborhoods that clean demonstrations never visit,
which is why PACER treats them as a distinct role with their own multiplier
and protocol floor.

1. When a rollout ends in a non-success state, the operator may teleoperate a
   recovery FROM THAT STATE to the target/handoff region.
2. Seed corrections only from states labeled near-miss / wrong-orientation /
   wrong-region. Never from unsafe aborts or operator-uncertain states.
3. Corrections are their own trace stream — not one-to-one with rollouts
   (successes seed none; one cell may seed several). Fix the correction budget
   per component-configuration in advance, before any candidate weighting
   exists, so collection cannot chase scores.
4. Tag each correction `role = correction` with the SAME `collection_config`
   as the rollout that seeded it, so the split keeps the pair together.

## 4. Data source C — supplementary clean demonstrations

1. BEFORE the rollout campaign, fix a coverage rule from the base policy's
   per-component success profile: record clean demos only for component-phase
   cells the policy does not reach.
2. Record them as ordinary teleoperated demonstrations under the same
   configurations and prompts; tag `role = clean` with the configuration id.
3. Keep them out of the original pre-training demonstration set: `clean` in
   PACER always means these supplementary demos in the post-training buffer.

## 5. Convert, validate, split (pipeline step 3)

1. Chunk every trace at your policy's action horizon H into standard rows —
   masks, roles, target metadata, safety/provenance, and geometry/EEF traces
   (or precomputed evidence). Field-by-field spec: `docs/DATA_CONTRACT.md`;
   shape reference: `schema.training_row_template()`; copyable records:
   `examples/training_rows.jsonl`.
2. Run `validate_training_row(row)` on every row and fix all errors. Run
   `training_row_warnings(row)` and review every "T_i = 0" warning — those
   rows will carry zero geometric evidence by design.
3. Split once with a fixed seed: `assign_case_group_split(rows, seed=...)`.
   Configurations are the split unit, so each rollout, its corrections, and
   its clean demos always land on the same side. Never reshuffle.
4. Reserve separate held-out **scene configurations** (physical setups, not
   logged rows) for the final hardware comparison; they never appear in the
   buffer.

Next: `docs/RUNBOOK_TRAINING_PREP.md` (steps 4–5).
