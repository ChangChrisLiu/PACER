# Pi0.5-style quickstart: model -> rollout -> correction -> TRACE-VLA

TRACE-VLA starts after you already have a base robot policy, for example a Pi0.5/OpenPI checkpoint. This repository does not include the initial data-collection/SFT pipeline.

The commands below are intentionally explicit and use the actual integration-script flags. `configs/pi05_rollout_then_correction.example.yaml` is a planning template, not a consumed CLI config.

## 0. Prerequisites

```text
base_policy: existing Pi0.5/OpenPI checkpoint or LoRA-merged checkpoint
norm_stats: method/checkpoint-family-specific norm_stats available
runtime: SEA-VLA robot/camera/OpenPI adapters available when doing live hardware runs
```

Install the lightweight TRACE-VLA package for offline code/tests:

```bash
pip install -e '.[test]'
pytest -q
```

## 1. Roll out an existing model

Dry-run the evaluation plan first:

```bash
python scripts/sea_vla_integration/run_stage_b_rollout_eval.py \
  --run-id pi05_base_example \
  --model-id pi05_base \
  --model-type openpi \
  --checkpoint-path /path/to/pi05_or_openpi_checkpoint \
  --configs config_001 \
  --components cpu_fan,graphic_card,connector,ram,cpu \
  --trials-per-component 3 \
  --output-root outputs/rollout_eval \
  --dry-run
```

For live hardware, start the robot, camera, and policy server in separate terminals, verify method-specific norm-stat injection in the policy server, then rerun with `--confirm-hardware` instead of `--dry-run`.

## 2. Collect correction evidence

The correction/demo collector is a SEA-VLA integration reference. Its `--help` and `--dry-run` can be inspected from this repo, but live collection requires the full SEA-VLA robot runtime on `PYTHONPATH`.

Dry-run a corrector-only block:

```bash
python scripts/sea_vla_integration/run_stage_b_pre_collection.py \
  --config-id config_001 \
  --phase-block corrector_only \
  --components cpu_fan,graphic_card,connector,ram,cpu \
  --corrector-only-count 3 \
  --output-root outputs/correction/pi05_base \
  --operator operator \
  --dry-run
```

For live collection, run from the SEA-VLA hardware environment and add `--confirm-hardware`.

Contract reminders:

- `corrector_only` means one long corrector rollout per saved trial, not repeated hidden attempts.
- Model stop-token events should be classified by the operator when borderline.
- Human-corrected saves should not be silently converted into scripted success.

## 3. Compile TRACE-VLA chunks

Use `tracevla.stage_b_rl.action_chunk_compiler` on rollout/correction manifests to produce action chunks and process evidence.

## 4. Materialize TRACE-VLA weights

Use `tracevla.stage_b_rl.pacer_bo_weights` to map eta settings to clipped, stratum-normalized `returns.loss_weight` values.

## 5. Fine-tune and evaluate

Fine-tune candidate weighted views in your training stack. Evaluate each candidate with fixed method-specific norm stats at serving time. Write each validation result as a `scores.json` compatible with `tracevla.stage_b_rl.pacer_eval_contract`.

## 6. Propose the next eta

Use `tracevla.stage_b_rl.pacer_bo` with validation scores only. Do not use heldout/final-test scores for BO selection.
