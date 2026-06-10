# Setup PACER_Framework from GitHub

PACER_Framework is deliberately lightweight: the core package is pure Python and
has **no required robot, model, ROS, PyTorch, JAX, OpenPI, or LeRobot imports**.
A new user installs the core first, verifies the demo, then connects their own
robot/VLA/trainer stack through the hooks in `pacer_framework/hooks.py`.

## 0. What you need before cloning

Minimum for the core framework:

- Linux/macOS/WSL or any Python-capable system;
- Python 3.10+;
- `git`;
- optionally a virtual environment tool (`python -m venv`, conda, uv, etc.).

For a real robot/VLA run, you additionally need your own stack:

- robot communication: ROS 2, vendor SDK, simulator API, or custom controller;
- model inference: OpenVLA/OpenPI/RT-style/diffusion/custom VLA runtime;
- training: PyTorch/JAX/OpenPI/LeRobot/custom trainer that can multiply its
  native action loss by PACER's exported `loss_weight`;
- logging: a way to save observations, action chunks, TCP/EEF traces, target
  metadata, safety/provenance flags, and operator labels. For a concrete
  networked option, install `pyzmq` with `python -m pip install -e '.[zmq]'`
  and use `docs/ZMQ_COLLECTION.md` / `python -m pacer_framework.collect_zmq`.
  For the concrete T1/T2/T3/T4 teleop, ZMQ, collection, and inference workflow,
  see `docs/TELEOP_COLLECTION_INFERENCE.md`.

PACER does not install those for you because every lab's robot/model stack is
different. Instead, it gives you a stable data contract and adapter hooks.

## 1. Clone and install the core

```bash
git clone git@github.com:ChangChrisLiu/PACER.git   # or your fork / mirror URL
cd PACER/PACER_Framework

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e '.[dev]'
```

If you do not want an editable install, use:

```bash
python -m pip install .
```

## 2. Verify the install

```bash
python -m pacer_framework.check_setup
python -m pytest tests -q
python examples/end_to_end_demo.py
```

Expected:

- setup check reports Python version, import path, and optional dependency
  availability;
- tests pass;
- demo prints `synthetic demo only` and walks the full pipeline on toy records.

The demo numbers are synthetic placeholders, not paper or lab dataset results.

## 3. Choose your integration path

### A. Simulator-only, ZMQ streams, or offline logs

Use this first if you are new to PACER.

1. If your simulator/robot can write files directly, export logged trajectories
   into the standard rows in `docs/DATA_CONTRACT.md`.
2. If your simulator/robot runs in another process or machine, stream JSON over
   ZMQ using `docs/ZMQ_COLLECTION.md`; the receiver appends raw events/rows to
   JSONL for later validation.
3. If you already have TCP/EEF positions, you do not need robot middleware.
4. Run `compile_weights`, `compile_baseline`, and `export_training_view`.
5. Train candidates in your own trainer.
6. Evaluate candidates by producing predicted TCP/EEF traces and calling the
   alignment/validation utilities.

### B. ROS 2 robot

Install ROS 2 from the official distro instructions for your OS. Typical Python
packages live in your ROS environment, not in PACER's `pyproject.toml`.

Your hook file should:

- subscribe/read observations from your logger;
- call your FK/action integration code to produce base-frame TCP/EEF positions;
- call PACER only after logs are converted to plain dictionaries.

Sketch:

```python
from pacer_framework import RobotHooks, TargetRegion

class RosRobotHooks:
    def target_region(self, component, collection_config):
        return TargetRegion(component=component, target_id=..., point=(...), radius=...)

    def integrate_action_chunk(self, *, start_tcp, action_chunk, observation):
        # Use your ROS/FK utilities here, return [(x,y,z), ...] in base frame.
        return positions

    def safety_flags(self, trace_or_prediction):
        return {"unsafe": False, "manual_safety_stop": False, "quarantined": False}
```

### C. Vendor SDK / direct controller

Install the vendor SDK separately. Keep SDK imports in your hook package, not in
PACER itself. The only required output is base-frame TCP/EEF positions and
safety flags.

### D. OpenVLA / transformer-style VLA

Install the model's official dependencies in your training/inference
environment. Your `VLAHooks.predict_action_chunk` should return the action chunk
in the representation your `RobotHooks.integrate_action_chunk` expects.

```python
class MyOpenVLAHooks:
    def predict_action_chunk(self, *, model_id, observation, prompt=None, deterministic=True):
        # preprocess observation, run model, decode action tokens
        return action_chunk

    def stop_emitted(self, prediction):
        return decoded_stop_token
```

### E. OpenPI / JAX policy

Install OpenPI/JAX separately following that project's instructions. PACER only
needs the exported rows for training and predicted chunks for validation.

### F. Diffusion/flow VLA

Fix the validation sampling seed/noise schedule before candidate ranking. PACER
expects deterministic offline evaluation across candidates.

## 4. Implement the three PACER hooks

Read: `docs/ADAPTER_HOOKS.md`.

Required boundaries:

- `RobotHooks`: target regions, action-chunk integration to TCP/EEF positions,
  safety flags;
- `VLAHooks`: model inference and stop-token extraction;
- `TrainerHooks` or equivalent script: train from exported `rows.jsonl` and
  `manifest.json`.

PACER starts after your hooks have turned robot/model-specific objects into
plain Python dictionaries and `[x, y, z]` TCP/EEF positions.

## 5. Prepare data

Follow `docs/RUNBOOK_DATA_COLLECTION.md`:

1. define target regions and tolerances per task/config;
2. record rollouts of the current/base VLA;
3. record human corrections after/around failures or partial successes;
4. record supplementary clean demos;
5. chunk traces into standard rows;
6. validate rows and split configuration-disjointly.

## 6. Build training views

Follow `docs/RUNBOOK_TRAINING_PREP.md`:

```python
from pacer_framework import CANDIDATE_POOL, BASELINE_MODES, compile_weights, compile_baseline, export_training_view

for name, eta in CANDIDATE_POOL.items():
    rows_w, manifest = compile_weights(rows, eta, eta_id=name)
    export_training_view(rows_w, manifest, f"views/pacer_candidates/{name}")

for mode in BASELINE_MODES:
    if mode == "pacer_selected":
        continue
    rows_b, manifest_b = compile_baseline(rows, mode)
    export_training_view(rows_b, manifest_b, f"views/baselines/{mode}")
```

Each exported row contains `loss_weight`. Your trainer applies it to the native
per-horizon action loss.

## 7. Train and evaluate candidates

Train one model per PACER candidate and baseline in your external stack. Then
follow `docs/RUNBOOK_EVALUATION.md`:

1. run each candidate on validation observations;
2. integrate predicted chunks to TCP/EEF positions;
3. compute EEF cosine/reference alignment and blockers;
4. compute `v_j`, component-balanced `J_val`, and audits;
5. select the highest-J_val audited candidate;
6. only then run the selected candidate on hardware.

## 8. Troubleshooting checklist

- `ModuleNotFoundError: pacer_framework`: run `python -m pip install -e .` from
  `PACER_Framework` or use `PYTHONPATH=.`.
- Tests fail after installing robot/model packages: create a clean venv for the
  PACER core and keep robot/model dependencies in your integration project.
- J_val is zero: inspect blocker flags and `training_row_warnings`; wrong target
  metadata or invalid masks often zero evidence/gates.
- No candidate is selected: audits failed; keep the reference policy rather than
  forcing a risky post-trained model onto hardware.
- Robot frames look wrong: confirm all TCP/EEF positions and target points are
  in the same base/world frame and in meters.
