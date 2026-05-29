# PACER algorithm notes

PACER learns from process evidence rather than only final success/failure labels. The core path is:

1. Convert rollout/correction/demo episodes into action chunks.
2. Attach process evidence to each chunk: role, outcome, progress, hover/final target proximity, correction/demo source, stop-token context, etc.
3. Map evidence to a raw PACER score with eta coefficients.
4. Normalize scores within strata using robust median/MAD statistics.
5. Clip the final weights and expose them as `returns.loss_weight`.
6. Train candidate policies on materialized weighted views.
7. Validate candidates under fixed norm-stat discipline and feed validation scores to BO.

## Main code entry points

- `pacer.stage_b_rl.action_chunk_compiler`
  - episode/action trace loading
  - chunk generation
  - role/evidence assignment
  - target-distance features

- `pacer.stage_b_rl.pacer_bo_weights`
  - `PacerEta`
  - default eta pool
  - raw score computation
  - stratum normalization
  - clipped loss-weight generation

- `pacer.stage_b_rl.pacer_eval_contract`
  - eval report schema validation
  - guardrails for BO inputs

- `pacer.stage_b_rl.pacer_bo`
  - GP + Expected Improvement candidate proposal over eta settings

## Claim-control notes

- PACER BO proposes a next eta candidate; it is not a claim of globally converged optimum.
- EEF/TCP cosine can diagnose motion alignment but is not robot task success.
- Validation drives eta search; heldout remains final reporting only.
