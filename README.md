# PACER

PACER = Process-Aware Correction Evidence Reweighting.

This repository is an early, private, pre-publication GitHub package extracted from the SEA-VLA work. It intentionally starts from the point where a base robot policy already exists (for example a Pi0.5/OpenPI checkpoint). It does **not** include the initial SFT/training-data pipeline. The intended workflow is:

1. Serve or load an existing base policy.
2. Run rollout evaluation on real or replay tasks.
3. Collect correction and/or clean-demonstration evidence after failed or weak rollouts.
4. Compile the resulting evidence into process-aware action chunks.
5. Materialize PACER `returns.loss_weight` views.
6. Fine-tune/evaluate candidate eta settings with fixed norm-stat discipline.
7. Use validation scores to propose the next eta setting with BO.

This repo is deliberately private for pre-publication use. Do not treat it as a clean public release yet.

## Repository structure

```text
pacer/
  stage_b_rl/              # PACER core: chunk compiler, eta weights, eval contract, BO
  stage_b_pre/             # rollout/correction/demo schemas and writer utilities
  stage_b_rollout_eval/    # fixed-model rollout evaluation plan/writer utilities
scripts/sea_vla_integration/
  run_stage_b_rollout_eval.py     # SEA-VLA hardware integration reference
  run_stage_b_pre_collection.py   # SEA-VLA correction/demo collection reference
configs/
  pi05_rollout_then_correction.example.yaml
docs/
  QUICKSTART_PI05.md
  PACER_ALGORITHM.md
  DATA_CONTRACT.md
  PREPUBLICATION_SCOPE.md
  INTERNAL_HANDOFFS_NOT_INCLUDED.md
examples/
  eta_candidates.json
  minimal_scores.json
tests/
  stage_b_rl/              # lightweight PACER core tests
```

## Quick install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

## Important scope notes

- No raw robot videos, logs, checkpoints, token files, or lab-specific secrets are tracked.
- The SEA-VLA integration scripts are included as reference adapters; they expect the robot/camera/OpenPI runtime from the SEA-VLA stack.
- Validation scores may guide eta selection, but heldout/final test sets must remain untouched until final reporting.
- Method-specific norm stats must be injected at serving/evaluation time for fair fixed-norm comparisons.
- TCP/EEF cosine diagnostics are diagnostic only, not robot success metrics.

## Example workflow

See `docs/QUICKSTART_PI05.md` and `configs/pi05_rollout_then_correction.example.yaml`.
