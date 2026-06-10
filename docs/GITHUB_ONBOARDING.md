# GitHub onboarding checklist

Use this as the review order for a new user landing on the PACER repository.
The repo has a generic framework layer plus a runtime-adapter layer. Both are
PACER; the runtime package keeps some legacy Python import paths and filenames
for compatibility with existing saved data.

## 1. Decide which path you need

Use `PACER_Framework/` if you want a reusable PACER pipeline for your own robot,
simulator, VLA model, or offline logs. This is the recommended first stop for
new users.

Use `pacer/` and `scripts/robot_runtime/` if you already have compatible
robot/camera/model servers and want to adapt the included PACER runtime adapter.
The package directory name is legacy; the method and repo should be described as
PACER.

## 2. Install and verify the generic framework with venv

```bash
cd PACER/PACER_Framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

## 3. Install and verify the generic framework with conda

```bash
cd PACER/PACER_Framework
conda create -n pacer-framework python=3.10 -y
conda activate pacer-framework
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

Use the conda path when you also need CUDA/PyTorch/JAX/OpenPI/OpenVLA/ROS/LeRobot
packages in the same environment. Install those large packages from their
upstream instructions after creating the conda environment.

Read next:

1. `PACER_Framework/docs/SETUP.md`
2. `PACER_Framework/docs/USAGE_QUICKSTART.md`
3. `PACER_Framework/docs/ADAPTER_HOOKS.md`
4. `PACER_Framework/docs/DATA_CONTRACT.md`
5. `PACER_Framework/docs/RUNBOOK_DATA_COLLECTION.md`
6. `PACER_Framework/docs/RUNBOOK_TRAINING_PREP.md`
7. `PACER_Framework/docs/RUNBOOK_EVALUATION.md`

## 4. If your robot streams data over ZMQ

Install optional ZMQ support:

```bash
cd PACER/PACER_Framework
python -m pip install -e '.[zmq]'
python -m pacer_framework.collect_zmq --help
```

Then read:

- `PACER_Framework/docs/ZMQ_COLLECTION.md`
- `PACER_Framework/docs/TELEOP_COLLECTION_INFERENCE.md`

## 5. If you are using the PACER runtime adapter

venv setup:

```bash
cd PACER
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest tests -q
```

conda setup:

```bash
cd PACER
conda create -n pacer-runtime python=3.10 -y
conda activate pacer-runtime
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest tests -q
```

Dry-run the operator-facing scripts before live hardware:

```bash
python scripts/hardware/run_policy_inference.py \
  --backend mock --mock-observation --prompt "move to the target"

python scripts/robot_runtime/run_correction_collection.py \
  --config-id config_999 --phase-block planner_only --components ram \
  --planner-only-count 1 --planner-skill-count 0 --corrector-only-count 0 \
  --output-root /tmp/pacer_collection_dry --dry-run

python scripts/robot_runtime/run_rollout_eval.py \
  --run-id dry_smoke --model-id mock_model --model-type openpi \
  --checkpoint-path /tmp/mock_ckpt --configs config_001 --components ram \
  --trials-per-component 1 --output-root /tmp/pacer_eval_dry --dry-run
```

Read next:

- `docs/ROBOT_RUNTIME_INTEGRATION.md`
- `docs/HARDWARE_SOFTWARE_SETUP.md`
- `docs/ZMQ_AGENT_CONTRACT.md`
- `docs/PACER_ALGORITHM.md`

## 6. Live robot checklist

Before running any command with `--confirm-hardware`:

- robot safety monitor and physical E-stop are active;
- robot server, cameras, and model server are each in their own terminal;
- dry-run plan matches the intended config/components/counts;
- target-region/teleop instructions are understood by the operator;
- output root points to an ignored data directory or external disk;
- no private tokens/checkpoints/logs will be committed;
- `--restart` is absent unless you intentionally want to discard progress.

## 7. What to commit vs. keep local

Commit:

- framework/source code;
- schemas and docs;
- small synthetic examples;
- tests.

Do not commit:

- raw frames, videos, robot logs, or `.pkl` files;
- checkpoints or norm-stat serving copies;
- private configs, `.env` files, tokens, robot IPs if sensitive;
- generated `data/`, `outputs/`, `runs/`, `logs/` artifacts.
