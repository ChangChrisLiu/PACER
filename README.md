# PACER / TRACE-VLA

Process-Aware Correction and Evidence Reweighting (PACER) for improving
vision-language-action (VLA) robot policies from rollout traces, human
corrections, and clean demonstrations.

> **Status: private pre-publication research package** (see `LICENSE`). The
> code and docs are usable by collaborators today, but interfaces may change
> before public release, and no public result claims are made here.

This repository has two layers:

1. `PACER_Framework/` — a clean, robot/VLA-agnostic reference framework for new
   users. Start here if you are applying PACER to your own robot, simulator, or
   VLA model.
2. `tracevla/` + `scripts/` — the TRACE-VLA integration/reference package:
   weighting code, rollout/correction schemas, rollout-eval utilities, and
   robot-runtime adapter scripts for labs that already have compatible
   robot/camera/model servers.

The core framework is offline and lightweight. Live robot use requires your own
robot runtime, safety system, cameras, model server, and operator supervision.

New to the repo? `docs/GITHUB_ONBOARDING.md` is a step-by-step checklist that
walks both layers in order, including the live-robot and commit-hygiene checks.

## Quick start: generic PACER framework

```bash
git clone git@github.com:ChangChrisLiu/PACER.git
cd PACER

cd PACER_Framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

The demo is synthetic. Replace its toy records with your own robot/simulator
logs before drawing conclusions.

Key framework docs:

- `PACER_Framework/docs/SETUP.md` — install, optional dependencies, verification.
- `PACER_Framework/docs/USAGE_QUICKSTART.md` — shortest end-to-end user path.
- `PACER_Framework/docs/ADAPTER_HOOKS.md` — implement `RobotHooks`, `VLAHooks`,
  and `TrainerHooks` for your robot/model/training stack.
- `PACER_Framework/docs/DATA_CONTRACT.md` — standard PACER row schema.
- `PACER_Framework/docs/ZMQ_COLLECTION.md` — optional ZeroMQ JSONL collection
  from robot/simulator processes.
- `PACER_Framework/docs/TELEOP_COLLECTION_INFERENCE.md` — concrete terminal
  layout, teleop capture, ZMQ ports, collection modes, saved artifacts, and
  inference commands.
- `PACER_Framework/docs/RUNBOOK_DATA_COLLECTION.md` — target regions, rollouts,
  corrections, and clean demos.
- `PACER_Framework/docs/RUNBOOK_TRAINING_PREP.md` — PACER/baseline view export
  for external VLA fine-tuning.
- `PACER_Framework/docs/RUNBOOK_EVALUATION.md` — offline candidate scoring,
  EEF cosine/reference alignment, audits, and hardware candidate selection.

## Quick start: TRACE-VLA integration package

Use this if you are working with the existing `tracevla` package or adapting the
robot-runtime reference scripts.

```bash
cd PACER
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest tests/weighting -q
```

No-hardware smoke commands:

```bash
# Single policy inference smoke with a mock observation.
python scripts/hardware/run_policy_inference.py \
  --backend mock \
  --mock-observation \
  --prompt "move to the target" \
  --output /tmp/pacer_mock_infer.json

# Collection plan dry-run; writes metadata but touches no hardware.
python scripts/robot_runtime/run_correction_collection.py \
  --config-id config_999 \
  --phase-block planner_only \
  --components ram \
  --planner-only-count 1 \
  --planner-skill-count 0 \
  --corrector-only-count 0 \
  --output-root /tmp/pacer_collection_dry \
  --dry-run

# Rollout-only candidate-eval dry-run.
python scripts/robot_runtime/run_rollout_eval.py \
  --run-id dry_smoke \
  --model-id mock_model \
  --model-type openpi \
  --checkpoint-path /tmp/mock_ckpt \
  --configs config_001 \
  --components ram \
  --trials-per-component 1 \
  --output-root /tmp/pacer_eval_dry \
  --dry-run
```

TRACE-VLA docs:

- `docs/QUICKSTART_PI05.md` — Pi0.5/OpenPI-oriented package quickstart.
- `docs/HARDWARE_SOFTWARE_SETUP.md` — hardware/software bring-up notes.
- `docs/ROBOT_RUNTIME_INTEGRATION.md` — scope of robot-runtime adapter scripts.
- `docs/ZMQ_AGENT_CONTRACT.md` — ZMQ probe/agent contract.
- `docs/DATA_CONTRACT.md` — TRACE-VLA action-chunk/data contract.
- `docs/TRACEVLA_ALGORITHM.md` — weighting, eta, and validation algorithm notes.
- `docs/PREPUBLICATION_SCOPE.md` — what is intentionally excluded.
- `docs/INTERNAL_HANDOFFS_NOT_INCLUDED.md` — internal pipelines that do not ship
  in this package.

## Live robot architecture

A live robot run is normally split into four terminals/processes:

```text
T1 robot server     : robot command/state endpoint, typically ZMQ
T2 camera servers   : wrist/base camera endpoints
T3 model server     : one OpenPI/OpenVLA/OpenVLA-OFT policy checkpoint
T4 collector/eval   : run_correction_collection.py or run_rollout_eval.py
```

Default integration ports used by the reference scripts:

```text
robot command/state: 6000
observation client : 6002
wrist camera       : 5000
base camera        : 5001
planner model      : 8000
corrector model    : 8001
```

Live collection requires `--confirm-hardware`. Always run the matching dry-run
first and confirm your robot safety stack is active.

## What gets collected

PACER/TRACE-style data comes from three sources:

1. base-policy rollouts;
2. human corrections from weak/failure states;
3. clean demonstrations under the same target/config conventions.

The TRACE-VLA writer saves per-trial artifacts such as:

```text
<output_root>/<config_id>/<block>/<component>/<trial_id>/
  frame_0000.pkl
  episode_meta.json
  target_region.json
  tracevla_trial_feedback.json
  tracevla_auto_score.json
  vla_action_trace.jsonl
```

For new robots that do not use the TRACE-VLA runtime, stream or write standard
PACER rows following `PACER_Framework/docs/DATA_CONTRACT.md`.

## Safety and repository scope

- This repo does not include raw robot videos/logs, private checkpoints, token
  files, or lab secrets.
- Live robot scripts are reference adapters, not universal hardware drivers.
- ZMQ logging is for data collection, not hard real-time safety control.
- Keep robot safety interlocks in the robot/controller process.
- Validation scores help select candidate models, but held-out robot success is
  the deployment claim.
- Do not use `--restart` on live collections unless you intentionally want to
  discard existing progress for that config/run.

## Development checks

Before pushing changes:

```bash
cd PACER
python -m pytest tests/weighting -q

cd PACER_Framework
python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

## License

See `LICENSE`.
