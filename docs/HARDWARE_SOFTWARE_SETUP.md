# TRACE-VLA hardware and software setup

This is the complete bring-up checklist for running TRACE-VLA from an existing
VLA checkpoint. It is intentionally written with placeholders instead of lab
addresses or secrets; copy the example config and fill those values locally.

## 0. Scope

TRACE-VLA starts after a base policy already exists. A full live run needs four
pieces running at the same time:

1. Robot command/telemetry stack: UR/RTDE or equivalent robot bridge.
2. Camera/observation stack: base camera, wrist camera, robot state, gripper.
3. Policy server: Pi0.5/OpenPI or another VLA backend that returns action chunks.
4. TRACE-VLA runner: rollout evaluation or correction/demo collection scripts.

The repository includes offline tests and dry-runs, but it does not ship robot
firmware, vendor drivers, private checkpoints, raw videos, or norm-stat files.

## 1. Machine prerequisites

Recommended lab host:

```bash
python --version          # 3.10+
git --version
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,robot]'
python -m pytest -q
```

Robot-side/vendor dependencies usually live outside this repo:

- UR RTDE libraries, if controlling a UR5e/UR arm.
- Gripper driver, e.g. Robotiq Python package or local bridge.
- Camera SDKs or OpenCV capture service.
- OpenPI/OpenVLA runtime and checkpoint files.
- Method-specific norm statistics for the checkpoint being evaluated.

Do not commit local IPs, checkpoint paths, tokens, `.env` files, raw data, or
videos. Put local values in an untracked config copied from
`configs/tracevla_hardware.example.yaml`.

## 2. Network and process topology

A common lab layout is:

```text
policy host        : serves VLA action chunks over websocket/REST
robot bridge host  : receives command messages and publishes telemetry
camera host        : publishes RGB/depth or serves observations
operator terminal  : runs TRACE-VLA collection/evaluation
```

Default placeholders in the example config:

```text
OpenPI policy server : ws://127.0.0.1:8000
Robot command ZMQ    : tcp://ROBOT_IP_OR_HOSTNAME:5555
Robot telemetry ZMQ  : tcp://ROBOT_IP_OR_HOSTNAME:5556
Camera/obs ZMQ       : tcp://CAMERA_HOSTNAME:5560
```

Use whatever ports your deployed bridge actually uses; the important part is to
record them in the local config and verify each service before enabling motion.

## 3. Start the policy server

For OpenPI/Pi0.5, start the server in the OpenPI environment, not necessarily in
this repo's virtualenv. Example shape:

```bash
cd /path/to/openpi
uv run scripts/serve_policy.py \
  --port 8000 \
  --checkpoint /path/to/checkpoint \
  --norm-stats-key METHOD_SPECIFIC_NORM_STATS_KEY
```

Required invariant: the norm stats must match the method/checkpoint under test.
For fair fixed-norm comparison, do not silently reuse baseline norm stats for a
newly adapted checkpoint unless that is the declared condition.

Probe from this repo:

```bash
python scripts/hardware/run_policy_inference.py \
  --backend openpi \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt "test prompt" \
  --mock-observation \
  --dry-run
```

Remove `--dry-run` only after the OpenPI client package is installed and the
policy server is reachable.

## 4. Start and probe ZMQ robot/camera agents

TRACE-VLA treats ZMQ services as JSON REQ/REP endpoints for probing and as a
reference contract for local bridges. The minimal messages are documented in
`docs/ZMQ_AGENT_CONTRACT.md`.

Probe a command endpoint:

```bash
python scripts/hardware/zmq_agent_probe.py --endpoint tcp://ROBOT_IP_OR_HOSTNAME:5555 --request ping
python scripts/hardware/zmq_agent_probe.py --endpoint tcp://ROBOT_IP_OR_HOSTNAME:5555 --request status
```

If the bridge uses a different schema, keep an adapter in the local TRACE-VLA
runtime. Do not weaken safety gates in TRACE-VLA just to match a bridge.

## 5. Dry-run TRACE-VLA scripts

Rollout evaluation plan inspection:

```bash
python scripts/robot_runtime/run_rollout_eval.py \
  --config configs/pi05_rollout_then_correction.example.yaml \
  --dry-run
```

Correction/demo collection plan inspection:

```bash
python scripts/robot_runtime/run_correction_collection.py \
  --config configs/pi05_rollout_then_correction.example.yaml \
  --dry-run
```

Single policy inference smoke:

```bash
python scripts/hardware/run_policy_inference.py \
  --backend mock \
  --mock-observation \
  --prompt "insert the RAM stick" \
  --output outputs/inference_smoke.json
```

## 6. Live hardware gate

Before any live motion:

1. E-stop reachable and tested.
2. Robot is in remote-control mode, brakes released, speed slider low.
3. Workspace is clear; cameras and gripper visible in observation stream.
4. Home pose is correct for the current table and task.
5. Policy server returns action chunks with the expected action convention.
6. Safety clamps are enabled.
7. Operator confirms hardware explicitly with `--confirm-hardware`.

Never run live mode from a stale SSH session or after a server crash without
re-probing robot state, camera stream, and policy response.

## 7. Expected run order

Terminal A: robot bridge / vendor driver.
Terminal B: camera or observation server.
Terminal C: VLA policy server.
Terminal D: TRACE-VLA dry-run, then live runner with explicit confirmation.

Record every live run under an untracked output directory. Only synthetic small
examples and source code belong in git.
