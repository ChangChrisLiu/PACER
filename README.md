# PACER

PACER stands for **Process-Aware Correction and Evidence Reweighting**. It is a
research package for improving vision-language-action (VLA) robot policies using
more than final success/failure labels: PACER keeps the process evidence from
rollouts, human corrections, clean demonstrations, target geometry, stop events,
and provenance checks, then converts that evidence into training/evaluation
views for policy improvement.

> **Status: private pre-publication research package** (see `LICENSE`). The code
> and docs are usable by collaborators today, but interfaces may change before a
> public release. The repository intentionally contains code, schemas, small
> synthetic examples, and tests only — not private robot logs, videos,
> checkpoints, tokens, or lab secrets.

## What is in this repository?

The repository has two practical layers. Both are PACER; they differ only in how
close they are to a particular robot runtime.

1. `PACER_Framework/`
   - Robot/VLA-agnostic PACER reference implementation.
   - Best starting point for new users.
   - Includes a clean row schema, evidence functions, gates, eta weights,
     validation scoring, export utilities, adapter hooks, synthetic examples,
     and a full test suite.
   - Use this if you want to plug PACER into your own simulator, robot, VLA
     policy, or offline logs.

2. `pacer/` and `scripts/`
   - Existing internal Python package path for the robot-runtime reference
     integration. The public-facing method name is still PACER.
   - Includes rollout/correction collection schemas, writer utilities, action
     chunk compilation, PACER weighting, Bayesian-optimization helper code,
     policy-inference smoke scripts, and no-hardware dry-runs.
   - Use this if you already have compatible robot, camera, and model servers
     and want to adapt the included runtime scripts.

New to the repo? Start with `docs/GITHUB_ONBOARDING.md`; it gives a concise
reading order and checklist.

## Installation option A: Python venv

Use this when you want a lightweight local Python environment.

### Generic PACER framework

```bash
git clone git@github.com:ChangChrisLiu/PACER.git
cd PACER/PACER_Framework

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

### Robot-runtime reference integration

```bash
cd PACER

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'

python -m pytest tests -q
```

## Installation option B: conda

Use conda if your VLA stack, CUDA stack, OpenPI/OpenVLA dependencies, or robot
runtime already uses conda. PACER itself is lightweight; conda is mainly useful
for keeping the larger ML/robot dependencies isolated.

### Generic PACER framework with conda

```bash
git clone git@github.com:ChangChrisLiu/PACER.git
cd PACER/PACER_Framework

conda create -n pacer-framework python=3.10 -y
conda activate pacer-framework
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

Optional packages:

```bash
# ZMQ collection from a robot/simulator process
python -m pip install -e '.[zmq]'

# Wheel/sdist build utilities
python -m pip install -e '.[dev]'
```

### Robot-runtime reference integration with conda

```bash
cd PACER

conda create -n pacer-runtime python=3.10 -y
conda activate pacer-runtime
python -m pip install --upgrade pip
python -m pip install -e '.[test]'

python -m pytest tests -q
```

If your robot/VLA stack requires PyTorch, JAX, OpenPI, OpenVLA, LeRobot, ROS 2,
or CUDA-specific wheels, install those following the upstream project
instructions inside the same conda environment. PACER does not pin those heavy
stacks because different labs use different robots and accelerator setups.

## Fast no-hardware verification

These commands should run without a robot. They are the quickest way to check
that the repository is installed correctly.

```bash
# Root package tests.
cd PACER
python -m pytest tests -q
python -m pytest tests/weighting -q

# Framework tests and demo.
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

## Recommended reading order

For most new users:

1. `docs/GITHUB_ONBOARDING.md` — repo-level checklist.
2. `PACER_Framework/docs/SETUP.md` — installation and dependency notes.
3. `PACER_Framework/docs/USAGE_QUICKSTART.md` — shortest framework workflow.
4. `PACER_Framework/docs/DATA_CONTRACT.md` — required row schema.
5. `PACER_Framework/docs/ADAPTER_HOOKS.md` — where to connect your robot/model.
6. `PACER_Framework/docs/RUNBOOK_DATA_COLLECTION.md` — how to collect rollouts,
   corrections, and clean demos.
7. `PACER_Framework/docs/RUNBOOK_TRAINING_PREP.md` — exporting PACER and
   baseline views for training.
8. `PACER_Framework/docs/RUNBOOK_EVALUATION.md` — offline validation,
   candidate selection, and audit checks.

For live robot or ZMQ integration:

- `PACER_Framework/docs/ZMQ_COLLECTION.md` — generic ZMQ JSONL ingestion.
- `PACER_Framework/docs/TELEOP_COLLECTION_INFERENCE.md` — terminal layout,
  teleop capture, default ports, saved artifacts, and inference workflow.
- `docs/HARDWARE_SOFTWARE_SETUP.md` — robot/software bring-up notes.
- `docs/ROBOT_RUNTIME_INTEGRATION.md` — scope of the runtime adapter scripts.
- `docs/ZMQ_AGENT_CONTRACT.md` — policy/agent ZMQ request-response contract.
- `docs/PACER_ALGORITHM.md` — PACER weighting, eta, validation, and BO notes.
- `docs/PREPUBLICATION_SCOPE.md` — what is intentionally not included.
- `docs/INTERNAL_HANDOFFS_NOT_INCLUDED.md` — internal pipelines excluded from
  this package.

## PACER workflow in one page

A complete PACER loop usually looks like this:

1. **Start from a base VLA policy.** Serve or load a policy checkpoint through
   your existing model stack.
2. **Run rollouts.** Record observations, model actions, target regions, stop
   events, and operator labels.
3. **Collect process evidence.** Add human corrections or clean demonstrations
   for weak or failed states, using the same target/config conventions.
4. **Validate provenance and safety.** Reject rows with invalid masks,
   mismatched target identity, unsafe stops, missing manifests, or quarantined
   trials.
5. **Compute PACER evidence.** Extract progress, proximity, terminal success,
   direction, stop/handoff behavior, operator evidence, and provenance evidence.
6. **Materialize training views.** Export PACER-weighted rows plus baseline
   views so comparisons are controlled.
7. **Train candidate policies externally.** PACER prepares data and weights; the
   actual VLA training loop stays in your model-specific stack.
8. **Evaluate candidates.** Use offline validation and, for selected candidates,
   controlled robot rollouts under fixed norm-stat and safety discipline.
9. **Update eta settings.** Validation scores can guide eta selection/BO, while
   held-out robot success remains the final deployment claim.

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
  pacer_trial_feedback.json      # legacy filename in the current writer
  pacer_auto_score.json          # legacy filename in the current writer
  vla_action_trace.jsonl
```

New integrations should follow the PACER row schema in
`PACER_Framework/docs/DATA_CONTRACT.md`.

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

Live commands require `--confirm-hardware`. Always run the matching `--dry-run`
first, confirm that the output root is correct, and verify that your robot safety
stack and physical E-stop are active.

## What PACER does not provide

PACER is not a complete robot stack. This repo does not provide:

- robot drivers;
- camera drivers;
- a hard real-time controller;
- private policy checkpoints;
- raw robot videos/logs;
- token files or API keys;
- a universal OpenPI/OpenVLA/LeRobot training recipe.

Instead, PACER provides the evidence schema, weighting logic, validation logic,
export utilities, reference adapters, and tests that you connect to your own
robot/model infrastructure.

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

See `LICENSE`.
