# SEA-VLA integration scripts

The scripts under `scripts/sea_vla_integration/` are reference adapters extracted from the SEA-VLA robotics stack. They are useful for collaborators who already have the robot/camera/OpenPI runtime. They are not standalone hardware drivers.

Expected modes:

- `run_stage_b_rollout_eval.py --dry-run` works in this repo after `pip install -e .`. Live rollout requires the policy/robot/camera servers.
- `run_stage_b_pre_collection.py --dry-run` works for plan inspection. Live correction/demo collection requires the full SEA-VLA `src.*` runtime on `PYTHONPATH`.

Keep raw outputs under ignored paths such as `outputs/`, `data/`, `runs/`, or outside this repository.
