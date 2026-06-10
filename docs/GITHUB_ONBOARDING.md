# GitHub onboarding checklist

Use this as the review order for a new user landing on the PACER repository.

## 1. Decide which layer you need

- Use `PACER_Framework/` if you want a reusable PACER pipeline for your own
  robot, simulator, VLA model, or offline logs.
- Use `tracevla/` and `scripts/robot_runtime/` if you are adapting the
  TRACE-VLA robot-runtime reference code and already have compatible
  robot/camera/model servers.

## 2. Install and verify the generic framework

```bash
cd PACER/PACER_Framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

Read next:

1. `PACER_Framework/docs/SETUP.md`
2. `PACER_Framework/docs/USAGE_QUICKSTART.md`
3. `PACER_Framework/docs/ADAPTER_HOOKS.md`
4. `PACER_Framework/docs/DATA_CONTRACT.md`
5. `PACER_Framework/docs/RUNBOOK_DATA_COLLECTION.md`
6. `PACER_Framework/docs/RUNBOOK_TRAINING_PREP.md`
7. `PACER_Framework/docs/RUNBOOK_EVALUATION.md`

## 3. If your robot streams data over ZMQ

Install optional ZMQ support:

```bash
cd PACER/PACER_Framework
python -m pip install -e '.[zmq]'
python -m pacer_framework.collect_zmq --help
```

Then read:

- `PACER_Framework/docs/ZMQ_COLLECTION.md`
- `PACER_Framework/docs/TELEOP_COLLECTION_INFERENCE.md`

## 4. If you are using the TRACE-VLA reference integration

```bash
cd PACER
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest tests/weighting -q
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
- `docs/TRACEVLA_ALGORITHM.md`

## 5. Live robot checklist

Before running any command with `--confirm-hardware`:

- robot safety monitor and physical E-stop are active;
- robot server, cameras, and model server are each in their own terminal;
- dry-run plan matches the intended config/components/counts;
- target-region/teleop instructions are understood by the operator;
- output root points to an ignored data directory or external disk;
- no private tokens/checkpoints/logs will be committed;
- `--restart` is absent unless you intentionally want to discard progress.

## 6. What to commit vs. keep local

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
