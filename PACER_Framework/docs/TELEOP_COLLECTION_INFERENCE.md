# Teleop, ZMQ collection, and inference workflow

This document explains how PACER-style data collection is meant to work
when the GitHub repo is used on a new robot computer.

There are two layers:

1. `PACER_Framework/` — generic, robot/VLA-agnostic PACER package. It validates
   rows, computes evidence/weights, exports training views, and evaluates
   candidates. It also includes an optional generic ZMQ JSON collector:
   `python -m pacer_framework.collect_zmq`.
2. Robot runtime integration — your lab-specific robot/camera/model process.
   In the included reference runtime adapter, this is represented by
   `PACER/scripts/robot_runtime/run_correction_collection.py` and
   `PACER/scripts/robot_runtime/run_rollout_eval.py`. These scripts expect
   external robot-runtime imports such as `src.comms.robot_node.ZMQClientRobot`,
   `src.comms.camera_node.ZMQClientCamera`, `src.agents.vla_agent.VLAAgent`,
   joystick/teleop code, safety monitor, and CSV skill executor.

The generic framework does not command the robot directly. It receives logs or
standard rows produced by the robot runtime.

## 1. Environment pieces

Core PACER environment:

```bash
cd PACER_Framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pacer_framework.check_setup
```

Optional ZMQ collection support:

```bash
python -m pip install -e '.[zmq]'
python -m pacer_framework.check_setup   # should report zmq found
```

Live robot runtime environment, installed separately:

- robot ZMQ server/client implementation;
- camera ZMQ server/client implementation;
- joystick/teleop agent;
- safety monitor;
- VLA agent/model adapter;
- OpenPI/OpenVLA/OpenVLA-OFT client/server packages as needed;
- any vendor SDK / ROS 2 / simulator bindings required by the robot.

In one compatible runtime integration, the live collection script imports:

```text
src.agents.joystick_agent.JoystickAgent, load_home_pose
src.agents.safety.SafetyMonitor
src.agents.vla_agent.VLAAgent
src.comms.camera_node.ZMQClientCamera
src.comms.robot_node.ZMQClientRobot
src.data.episode_buffer.rebuild_phase_segments
src.skills.csv_skill_executor.CSVSkillExecutor
```

If those imports are absent, dry-runs still work, but live robot collection will
fail closed.

## 2. Terminal layout for live robot collection

Use separate terminals/processes so failures are isolated.

```text
T1 robot server     : launches robot controller / ZMQ robot endpoint
T2 camera servers   : launches wrist + base camera ZMQ endpoints
T3 model server     : launches one VLA policy server/checkpoint
T4 collector/eval   : runs PACER collection or rollout-eval script
```

Default ports from the current integration scripts:

```text
robot command/state: 6000
observation client : 6002
wrist camera       : 5000
base camera        : 5001
planner model      : 8000
corrector model    : 8001
```

For a new robot, keep the same conceptual layout but replace the actual server
commands with your runtime's launch commands.

## 3. Teleoperation environment

Teleop is used for three things:

1. target-region capture: operator jogs the end effector to the intended target
   / handoff pose and records the target geometry;
2. human recovery correction: after a failed/partial model rollout, operator
   teleops from the failure pose to a good recovery pose;
3. clean full demonstration: operator records a clean from-home demonstration
   under the same target/config conventions.

The reference teleop wrapper is:

```text
PACER/pacer/collection/teleop_capture.py
```

It wraps the runtime `JoystickAgent` and `ZMQClientRobot`:

- `TeleopCaptureController.tick()` calls `joystick_agent.act({})`;
- velocity actions are sent through `robot_client.command_cartesian_velocity(...)`;
- `snapshot_from_observation(...)` extracts:
  - `tcp_pose`
  - `joint_positions`
  - `gripper_obs`
  - timestamp
- snapshots become target/failure/reference points.

For a new repo user, implement the same shape in `RobotHooks` if not using the
included runtime adapter directly.

## 4. Data-collection modes

The current collection entrypoint is:

```text
PACER/scripts/robot_runtime/run_correction_collection.py
```

Main modes:

- `planner_only`: run base/planner VLA to target/handoff pose; operator labels
  result; failures can trigger recovery correction and clean demo recording.
- `planner_skill`: run planner then a fixed component skill; operator labels
  downstream success/failure; optional human recovery after failure.
- `corrector_only`: start from cached/captured failure poses and run corrector
  model; gripper/physical verifier plus operator borderline menu decide label.
- `all` / `planner_side`: convenience phase subsets.

Dry-run smoke, no hardware:

```bash
cd PACER
python3 scripts/robot_runtime/run_correction_collection.py \
  --config-id config_999 \
  --phase-block planner_only \
  --components ram \
  --planner-only-count 1 \
  --planner-skill-count 0 \
  --corrector-only-count 0 \
  --output-root /tmp/pacer_collection_dry \
  --dry-run
```

Live collection requires external robot runtime imports and explicit hardware
confirmation:

```bash
python3 scripts/robot_runtime/run_correction_collection.py \
  --config-id config_001 \
  --phase-block planner_side \
  --components ram,connector,cpu_fan,graphic_card,cpu \
  --output-root data/pacer_correction_rollouts \
  --robot-host 127.0.0.1 --robot-port 6000 --obs-port 6002 \
  --camera-host 127.0.0.1 --wrist-camera-port 5000 --base-camera-port 5001 \
  --server-host 127.0.0.1 \
  --planner-server-port 8000 \
  --corrector-server-port 8001 \
  --model-type openpi \
  --fps 10 --open-loop-horizon 10 \
  --confirm-hardware
```

Do not use `--restart` unless intentionally discarding existing progress for
that config.

## 5. What gets saved

The collection writer saves one directory per trial:

```text
<output_root>/<config_id>/<block>/<component>/<trial_id>/
  frame_0000.pkl
  frame_0001.pkl
  ...
  episode_meta.json
  target_region.json                 # when target capture exists
  pacer_trial_feedback.json       # operator label
  pacer_auto_score.json           # automatic geometric/physical score
  vla_action_trace.jsonl             # policy queries + executed actions

<output_root>/<config_id>/
  session_meta.json
  progress.json
  pacer_pre_manifest.jsonl
```

Important saved signals:

- frame-level observations/images/state from robot runtime;
- TCP/EEF and joint state in frame pickles;
- target region / declared target id;
- action chunks from the VLA query;
- executed/post-safety-clamped actions;
- stop source: `model_stop_token`, `manual_stop`, `timeout`, `unsafe_abort`;
- operator label;
- safety/provenance metadata.

These raw artifacts are later converted into PACER standard rows matching
`PACER_Framework/docs/DATA_CONTRACT.md`.

## 6. Generic ZMQ JSON collection path

If a new robot stack does not want to use the included runtime collection script, use the
framework's generic JSONL collector:

```bash
cd PACER_Framework
python -m pip install -e '.[zmq]'
mkdir -p data/raw
python -m pacer_framework.collect_zmq \
  --bind tcp://*:5557 \
  --socket PULL \
  --out data/raw/pacer_stream.jsonl
```

Robot/simulator process sends JSON:

```python
import zmq, time
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PUSH)
sock.connect('tcp://collector-host:5557')

sock.send_json({
    'message_type': 'standard_row',
    'row_id': 'scene_001:rollout_0001:chunk_000',
    'component': 'door_handle',
    'phase': 'approach',
    'collection_config': 'scene_001',
    'role': 'partial',
    'operator_label': 'near_but_not_accurate',
    'mask': [1, 1],
    'geometry': {
        'tcp_positions': [[0.30, -0.10, 0.31], [0.31, -0.10, 0.31]],
        'target_point': [0.42, -0.10, 0.31],
        'target_id': 'scene_001:door_handle',
        'orientation_ok': True,
    },
    'target_meta': {
        'target_id': 'scene_001:door_handle',
        'declared_target_id': 'scene_001:door_handle',
    },
    'stop': {'phase_ending': False, 'stop_event': 'timeout', 'inside_terminal': False},
    'safety': {'unsafe': False, 'manual_safety_stop': False, 'quarantined': False},
    'provenance': {'manifest_present': True, 'timing_ok': True, 'schema_ok': True},
    'sent_unix_time': time.time(),
})
```

See `PACER_Framework/docs/ZMQ_COLLECTION.md` for target-region messages, event
messages, and raw-event-to-row conversion notes.

## 7. Inference flow

Single-step smoke runner:

```text
PACER/scripts/hardware/run_policy_inference.py
```

Mock smoke, no model server:

```bash
cd PACER
python3 scripts/hardware/run_policy_inference.py \
  --backend mock \
  --mock-observation \
  --prompt 'move to the target' \
  --output /tmp/pacer_mock_infer.json
```

OpenPI dry-run, no server contact:

```bash
python3 scripts/hardware/run_policy_inference.py \
  --backend openpi \
  --dry-run \
  --mock-observation \
  --host 127.0.0.1 --port 8000 \
  --prompt 'move to the target'
```

OpenPI live inference expects `openpi_client.websocket_client_policy` and a
running OpenPI websocket server:

```bash
python3 scripts/hardware/run_policy_inference.py \
  --backend openpi \
  --observation-npz /path/to/observation.npz \
  --host 127.0.0.1 --port 8000 \
  --prompt 'move to the target' \
  --output /tmp/openpi_infer.json
```

Inside full collection/eval, inference is handled by:

```text
PACER/pacer/collection/inference_runner.py::TracedChunkRunner
```

It runs at 10 Hz by default:

1. read fresh observation;
2. build a frame;
3. if action queue is empty, query the VLA for an action chunk;
4. write `policy_query` to `vla_action_trace.jsonl`;
5. detect model stop token on the raw chunk;
6. optionally apply execution-only gripper cap before sending to robot;
7. send one action per control tick;
8. write `executed_action` trace rows;
9. stop on model stop, manual `s`, unsafe `q`, or timeout.

## 8. Rollout-only candidate evaluation

For PACER model selection / hardware comparison, use rollout eval, not the
training-data collector:

```text
PACER/scripts/robot_runtime/run_rollout_eval.py
```

Dry-run smoke:

```bash
cd PACER
python3 scripts/robot_runtime/run_rollout_eval.py \
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

Live eval uses the same T1/T2/T3/T4 structure, but it does **not** record target
regions, corrections, clean demos, or CSV skills. It writes rollout-only eval
artifacts and operator labels for candidate comparison.

## 9. Verified on this checkout

The following no-hardware checks were run successfully in this repo:

```bash
cd PACER
python3 scripts/hardware/run_policy_inference.py --backend mock --mock-observation --prompt 'move to the target'

python3 scripts/robot_runtime/run_correction_collection.py \
  --config-id config_999 --phase-block planner_only --components ram \
  --planner-only-count 1 --planner-skill-count 0 --corrector-only-count 0 \
  --output-root /tmp/pacer_collection_dry --dry-run

python3 scripts/robot_runtime/run_rollout_eval.py \
  --run-id dry_smoke --model-id mock_model --model-type openpi \
  --checkpoint-path /tmp/mock_ckpt --configs config_001 --components ram \
  --trials-per-component 1 --output-root /tmp/pacer_eval_dry --dry-run
```

Framework tests and legacy weighting tests should also pass before live use:

```bash
cd PACER_Framework && python3 -m pytest tests -q
cd ../PACER && python3 -m pytest tests/weighting -q
```
