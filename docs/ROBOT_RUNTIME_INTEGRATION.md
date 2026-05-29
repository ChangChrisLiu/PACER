# robot-runtime integration scripts

The scripts under `scripts/robot_runtime/` are reference adapters extracted from the external robot runtime. They are useful for collaborators who already have the robot/camera/OpenPI runtime. They are not standalone hardware drivers.

Expected modes:

- `run_rollout_eval.py --dry-run` works in this repo after `pip install -e .`. Live rollout requires the policy/robot/camera servers.
- `run_correction_collection.py --dry-run` works for plan inspection. Live correction/demo collection requires a compatible robot runtime on `PYTHONPATH`.

Keep raw outputs under ignored paths such as `outputs/`, `data/`, `runs/`, or outside this repository.
