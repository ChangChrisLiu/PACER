# PACER

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
[![Status: pre-publication](https://img.shields.io/badge/Status-pre--publication-orange.svg)](docs/PREPUBLICATION_SCOPE.md)

PACER stands for **Process-Aware Correction and Evidence Reweighting**. It is a
research codebase for improving vision-language-action (VLA) robot policies
from more than final success/failure labels. PACER keeps process evidence from
rollouts, human corrections, clean demonstrations, target geometry, stop events,
operator labels, and provenance checks, then turns that evidence into auditable
training and evaluation views for policy improvement.

This repository is organized for GitHub readers who want to understand,
install, adapt, and verify the method. It contains source code, schemas, small
synthetic examples, tests, setup notes, and runtime-adapter references. It does
not contain private robot logs, videos, checkpoints, credentials, raw lab data,
or final result tables.

> **Status:** pre-publication research code released under the
> [MIT License](LICENSE). Interfaces and schemas may still change before the
> accompanying paper is published. See
> [docs/PREPUBLICATION_SCOPE.md](docs/PREPUBLICATION_SCOPE.md) for exactly what
> is included and excluded.

New to the repo? [docs/GITHUB_ONBOARDING.md](docs/GITHUB_ONBOARDING.md) is the
fastest install-and-verify checklist; the sections below explain each component
and document in more depth.

## Repository at a glance

There are two implementation layers:

- [`PACER_Framework/`](PACER_Framework/) is the generic, robot/VLA-agnostic
  reference implementation. Start here if you want to apply PACER to your own
  simulator, robot, VLA model, or offline logs. It contains the standard row
  schema, evidence functions, eligibility gates, eta-weighted training views,
  baseline views, open-loop validation metrics, reporting utilities, adapter
  hook protocols, synthetic examples, and a framework test suite.
- [`pacer/`](pacer/) plus [`scripts/`](scripts/) is the runtime-adapter layer.
  It contains collection schemas, writer utilities, rollout/correction helpers,
  action-chunk compilation, PACER weighting utilities, validation/BO helpers,
  hardware probe scripts, and dry-run/live entry points for compatible robot,
  camera, and policy-server environments.

Use the framework layer to build a new integration. Use the runtime-adapter
layer when you already have compatible robot/camera/model services and want to
adapt the included reference scripts.

## What PACER does

A complete PACER loop usually looks like this:

1. Start from an existing base VLA policy checkpoint.
2. Record base-policy rollouts with observations, action chunks, target
   metadata, stop events, and outcome/operator labels.
3. Collect human corrections from weak or failed states, plus clean
   demonstrations for cells the policy cannot reliably reach.
4. Validate provenance, target identity, safety flags, and action masks.
5. Compute explicit process evidence for progress, proximity, terminal target
   membership, direction/alignment, stop/handoff behavior, operator evidence,
   and provenance.
6. Export PACER-weighted rows and controlled baseline views.
7. Train candidate VLA policies externally, multiplying the native action loss
   by each exported `loss_weight`.
8. Evaluate candidates with fixed validation rows, geometry diagnostics,
   component-balanced scores, and audit gates.
9. Select an audited candidate for any later protected hardware comparison.

The algorithm and runtime notes are summarized in
[`docs/PACER_ALGORITHM.md`](docs/PACER_ALGORITHM.md). The framework implementation
has a more detailed method-to-code map in
[`PACER_Framework/README.md`](PACER_Framework/README.md).
For the publication-wide claim matrix across both layers, see
[`docs/PAPER_CLAIM_ALIGNMENT.md`](docs/PAPER_CLAIM_ALIGNMENT.md).

## Install the generic framework with venv

Use this path for a lightweight local environment and for first-time framework
verification.

```bash
git clone https://github.com/ChangChrisLiu/PACER.git
cd PACER/PACER_Framework

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

## Install the generic framework with conda

Use conda when your simulator, VLA stack, CUDA stack, OpenPI/OpenVLA
dependencies, ROS environment, or robot SDK already needs conda isolation.

```bash
git clone https://github.com/ChangChrisLiu/PACER.git
cd PACER/PACER_Framework

conda create -n pacer-framework python=3.10 -y
conda activate pacer-framework
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

Optional framework extras:

```bash
# ZMQ collection from a robot, simulator, or bridge process.
python -m pip install -e '.[zmq]'

# Wheel/sdist build utilities and test dependency.
python -m pip install -e '.[dev]'

# Light model-side hints; full CUDA/robot stacks still come from upstream docs.
python -m pip install -e '.[vla-transformers]'   # transformers + accelerate
python -m pip install -e '.[lerobot]'            # LeRobot dataset workflows
```

## Install the runtime-adapter package with venv

Use this path when working with the root `pacer` package and the runtime
reference scripts.

```bash
git clone https://github.com/ChangChrisLiu/PACER.git
cd PACER

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'

python -m pytest tests -q
```

## Install the runtime-adapter package with conda

```bash
git clone https://github.com/ChangChrisLiu/PACER.git
cd PACER

conda create -n pacer-runtime python=3.10 -y
conda activate pacer-runtime
python -m pip install --upgrade pip
python -m pip install -e '.[test]'

python -m pytest tests -q
```

The root package also has a `robot` extra (`opencv-python`, `pyzmq`) used by
the hardware probe scripts:

```bash
python -m pip install -e '.[test,robot]'
```

If your live stack needs PyTorch, JAX, OpenPI, OpenVLA, LeRobot, ROS 2, CUDA
wheels, camera SDKs, or vendor robot SDKs, install those from their upstream
instructions inside the same environment. PACER keeps those dependencies out of
the default install because each lab uses different hardware and model stacks.

## Fast no-hardware checks

These commands should run without a robot. They are the quickest way to check
that the repository is installed correctly.

```bash
# Root runtime-adapter package tests.
cd PACER
python -m pytest tests -q
python -m pytest tests/weighting -q

# Generic framework tests and demo.
cd PACER_Framework
python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py

# Optional ZMQ collector CLI check.
python -m pacer_framework.collect_zmq --help
```

No-hardware runtime script smokes:

```bash
cd PACER

# Single policy inference smoke with a mock observation.
python scripts/hardware/run_policy_inference.py \
  --backend mock \
  --mock-observation \
  --prompt "move to the target" \
  --output /tmp/pacer_mock_infer.json

# Collection plan dry-run; writes only dry-run metadata and touches no hardware.
python scripts/robot_runtime/run_correction_collection.py \
  --config-id config_999 \
  --phase-block planner_only \
  --components ram \
  --planner-only-count 1 \
  --planner-skill-count 0 \
  --corrector-only-count 0 \
  --output-root /tmp/pacer_collection_dry \
  --dry-run

# Rollout-only candidate-evaluation dry-run.
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

## Generic framework contents

[`PACER_Framework/`](PACER_Framework/) is a self-contained Python package for
new PACER integrations. It deliberately avoids robot SDKs, ROS, PyTorch, JAX,
OpenPI, LeRobot, and hardware imports. Your integration converts robot/model
objects into plain Python rows; the framework then computes evidence, weights,
splits, validation scores, and exports.

Important framework files and docs:

- [`PACER_Framework/README.md`](PACER_Framework/README.md) explains the generic
  PACER pipeline, package layout, hook boundaries, and claim-to-test map.
- [`PACER_Framework/docs/SETUP.md`](PACER_Framework/docs/SETUP.md) covers clone,
  install, verification, and optional integration dependency patterns.
- [`PACER_Framework/docs/USAGE_QUICKSTART.md`](PACER_Framework/docs/USAGE_QUICKSTART.md)
  gives the shortest path for applying PACER to a new robot/VLA system.
- [`PACER_Framework/docs/DATA_CONTRACT.md`](PACER_Framework/docs/DATA_CONTRACT.md)
  defines the standard training rows, validation rows, eta configs, and produced
  manifests consumed by the framework.
- [`PACER_Framework/docs/ADAPTER_HOOKS.md`](PACER_Framework/docs/ADAPTER_HOOKS.md)
  shows how to connect robot geometry, VLA inference, and training without
  importing heavy runtime stacks into PACER.
- [`PACER_Framework/docs/RUNBOOK_DATA_COLLECTION.md`](PACER_Framework/docs/RUNBOOK_DATA_COLLECTION.md),
  [`RUNBOOK_TRAINING_PREP.md`](PACER_Framework/docs/RUNBOOK_TRAINING_PREP.md),
  and [`RUNBOOK_EVALUATION.md`](PACER_Framework/docs/RUNBOOK_EVALUATION.md)
  expand the data, training-view, and evaluation stages.
- [`PACER_Framework/docs/ZMQ_COLLECTION.md`](PACER_Framework/docs/ZMQ_COLLECTION.md)
  and [`PACER_Framework/docs/TELEOP_COLLECTION_INFERENCE.md`](PACER_Framework/docs/TELEOP_COLLECTION_INFERENCE.md)
  describe optional networked collection and a concrete terminal layout.
- [`PACER_Framework/examples/`](PACER_Framework/examples/) contains synthetic
  rows, candidate configs, and an end-to-end demo. These examples are templates
  for integration testing, not paper results.
- [`PACER_Framework/tests/`](PACER_Framework/tests/) locks the row schema,
  evidence math, weighting, baselines, validation, reporting, and the synthetic
  end-to-end path.

## Runtime-adapter package contents

[`pacer/`](pacer/) is the root runtime package. It is useful when adapting the
included reference scripts to an existing robot/VLA runtime.

- [`pacer/collection/`](pacer/collection/) contains collection configuration,
  feedback schemas, compatibility checks, target/reference caches, trial
  planning, teleop capture helpers, scoring utilities, inference wrappers, and
  trial writers.
- [`pacer/rollout_eval/`](pacer/rollout_eval/) contains rollout-evaluation plan,
  feedback, runner, and writer utilities.
- [`pacer/weighting/`](pacer/weighting/) contains action-chunk compilation,
  PACER eta-weight materialization, validation scoring, evaluation-contract
  checks, local validation helpers, training-manifest utilities, and
  Bayesian-optimization helpers over eta candidates.

The runtime package is not a complete robot controller. It expects the robot,
camera, policy server, and safety layers to be supplied by the deployment
environment.

## Scripts and live integration

[`scripts/`](scripts/) provides command-line entry points for no-hardware
smokes, dry-runs, and compatible live environments:

- [`scripts/hardware/run_policy_inference.py`](scripts/hardware/run_policy_inference.py)
  probes a policy backend or mock backend with a single observation.
- [`scripts/hardware/zmq_agent_probe.py`](scripts/hardware/zmq_agent_probe.py)
  sends safe probe requests such as `ping` and `status` to local ZMQ agents.
- [`scripts/robot_runtime/run_correction_collection.py`](scripts/robot_runtime/run_correction_collection.py)
  plans or runs rollout/correction/demo collection blocks.
- [`scripts/robot_runtime/run_rollout_eval.py`](scripts/robot_runtime/run_rollout_eval.py)
  plans or runs rollout-only candidate evaluation.

Live robot commands require explicit hardware confirmation in the scripts that
support motion. Always run the matching `--dry-run` first and confirm the output
root, config IDs, components, counts, safety monitor, and physical E-stop.

[`docs/ROBOT_RUNTIME_INTEGRATION.md`](docs/ROBOT_RUNTIME_INTEGRATION.md)
describes the runtime adapter scope.
[`docs/HARDWARE_SOFTWARE_SETUP.md`](docs/HARDWARE_SOFTWARE_SETUP.md) lists the
expected four-process topology: robot bridge, camera/observation service,
policy server, and PACER runner. [`docs/ZMQ_AGENT_CONTRACT.md`](docs/ZMQ_AGENT_CONTRACT.md)
documents the small JSON contract used by hardware probes and local bridges.

## Configs, examples, and tests

- [`configs/pacer_hardware.example.yaml`](configs/pacer_hardware.example.yaml)
  is a placeholder hardware/runtime config. Copy it locally and fill in private
  addresses, checkpoint paths, and secrets outside git.
- [`configs/pi05_rollout_then_correction.example.yaml`](configs/pi05_rollout_then_correction.example.yaml)
  documents the shape of a Pi0.5/OpenPI-style rollout-then-correction plan.
- [`examples/eta_candidates.json`](examples/eta_candidates.json) and
  [`examples/minimal_scores.json`](examples/minimal_scores.json) are small
  runtime-layer examples for weighting and validation helpers.
- [`tests/`](tests/) covers the root runtime package, hardware-script dry-run
  behavior, PACER tolerances, weighting, validation scoring, BO helpers, and
  manifest validation.

## Root documentation

The root [`docs/`](docs/) folder explains the publication-facing runtime and
repo-boundary topics:

- [`docs/GITHUB_ONBOARDING.md`](docs/GITHUB_ONBOARDING.md) is a compact
  checklist for installing, verifying, and deciding which layer to use.
- [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) gives a short runtime-layer
  data-contract summary for rollout, correction, clean-demo, and weighted-view
  artifacts.
- [`docs/PACER_ALGORITHM.md`](docs/PACER_ALGORITHM.md) summarizes process
  evidence, hard gates, eta scoring, weighting, validation, and claim-control
  rules.
- [`docs/PAPER_CLAIM_ALIGNMENT.md`](docs/PAPER_CLAIM_ALIGNMENT.md) maps paper
  claims to publication-safe code, docs, and tests.
- [`docs/QUICKSTART_PI05.md`](docs/QUICKSTART_PI05.md) shows a Pi0.5/OpenPI
  style model-to-rollout-to-correction-to-PACER workflow.
- [`docs/HARDWARE_SOFTWARE_SETUP.md`](docs/HARDWARE_SOFTWARE_SETUP.md) explains
  live stack prerequisites, process topology, dry-runs, and live hardware gates.
- [`docs/ROBOT_RUNTIME_INTEGRATION.md`](docs/ROBOT_RUNTIME_INTEGRATION.md)
  clarifies that the runtime scripts are adapters, not standalone hardware
  drivers.
- [`docs/ZMQ_AGENT_CONTRACT.md`](docs/ZMQ_AGENT_CONTRACT.md) defines safe ZMQ
  probe messages and bridge expectations.
- [`docs/PREPUBLICATION_SCOPE.md`](docs/PREPUBLICATION_SCOPE.md) records what
  is included and intentionally excluded from this research extraction.
- [`docs/archive/INTERNAL_HANDOFFS_NOT_INCLUDED.md`](docs/archive/INTERNAL_HANDOFFS_NOT_INCLUDED.md)
  archives the old internal-handoff boundary note: local run handoffs, private
  job paths, raw logs, and scratch reviews are not part of the repository.

## Data collection modes

PACER supports three common data sources:

- **Base-policy rollouts:** the current policy attempts the task. These runs
  provide success, near-miss, failure, stop, and target-proximity evidence.
- **Human corrections:** an operator corrects from weak or failed states. These
  runs provide recovery evidence and can receive higher credit when they are
  safe, aligned, and target-consistent.
- **Clean demonstrations:** an operator performs the task directly. These runs
  provide high-quality imitation data and calibration references.

For the runtime adapter scripts, saved trial folders follow this shape:

```text
<output_root>/<config_id>/<block>/<component>/<trial_id>/
  frame_0000.pkl
  episode_meta.json
  target_region.json
  pacer_trial_feedback.json
  pacer_auto_score.json
  vla_action_trace.jsonl
```

New integrations should follow the framework row schema in
[`PACER_Framework/docs/DATA_CONTRACT.md`](PACER_Framework/docs/DATA_CONTRACT.md).

## Live robot architecture

A live robot run is normally split into four terminals/processes:

```text
T1 robot server     : robot command/state endpoint, typically ZMQ
T2 camera servers   : wrist/base camera endpoints
T3 model server     : one OpenPI/OpenVLA/OpenVLA-OFT policy checkpoint
T4 collector/eval   : run_correction_collection.py or run_rollout_eval.py
```

Default adapter ports:

```text
robot command/state: 6000
observation client : 6002
wrist camera       : 5000
base camera        : 5001
planner model      : 8000
corrector model    : 8001
```

Live commands require explicit hardware confirmation where supported. Always run
the matching `--dry-run` first, confirm that the output root is correct, and
verify that the robot safety stack and physical E-stop are active.

## What PACER does not provide

PACER is not a complete robot stack. This repository does not provide:

- robot drivers;
- camera drivers;
- a hard real-time controller;
- private policy checkpoints;
- raw robot videos/logs;
- token files or API keys;
- a universal OpenPI/OpenVLA/LeRobot training recipe.

Instead, PACER provides the evidence schema, weighting logic, validation logic,
export utilities, reference adapters, examples, and tests that you connect to
your own robot/model infrastructure.

## Safety and repository hygiene

- Keep robot safety interlocks in the robot/controller process, not in the data
  logger.
- Treat ZMQ collection as data movement, not as a safety boundary.
- Do not commit raw frames, `.pkl` episode dumps, checkpoints, `.env` files,
  tokens, private configs, or lab-specific secrets.
- Keep generated data under ignored paths such as `data/`, `outputs/`, `runs/`,
  or `logs/`.
- Do not use `--restart` on live collections unless you intentionally want to
  discard existing progress for that config/run.

## Development checks before commit

```bash
cd PACER
python -m pytest tests -q
python -m pytest tests/weighting -q

cd PACER_Framework
python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

Also run `git status --short` and confirm that no generated robot data,
checkpoints, logs, pycache folders, or local scratch files are staged.

## License

This repository is released under the [MIT License](LICENSE)
(copyright © 2026 Chris Liu). The license covers the code, schemas, examples,
and documentation in this repository. Private datasets, checkpoints, and lab
infrastructure are not part of the repository and are not covered by this
grant.
