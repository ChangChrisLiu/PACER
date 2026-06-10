# ZMQ data collection for PACER

Use this when your robot/simulator/policy process is separate from the PACER
Python environment. The robot process streams JSON messages over ZeroMQ; PACER's
receiver appends them to JSONL. Later you validate/chunk/export those messages
into standard PACER rows.

ZMQ is optional. The PACER core does not require it.

## 1. Install optional ZMQ support

```bash
cd PACER_Framework
python -m pip install -e '.[zmq]'
python -m pacer_framework.check_setup
```

`check_setup` should report `zmq found`.

## 2. Choose a socket pattern

Recommended patterns:

| Pattern | Use when | Receiver command |
|---|---|---|
| PUSH → PULL | one or more robot/logger clients push events to one collector | `--bind tcp://*:5557 --socket PULL` |
| PUB → SUB | robot broadcasts to several consumers; PACER is one subscriber | `--connect tcp://robot-host:5557 --socket SUB` |

For most labs, start with PUSH → PULL. The collector binds; robot/simulator
clients connect and push JSON.

## 3. Start the PACER collector

```bash
mkdir -p data/raw
python -m pacer_framework.collect_zmq \
  --bind tcp://*:5557 \
  --socket PULL \
  --out data/raw/pacer_stream.jsonl
```

For a short smoke test:

```bash
python -m pacer_framework.collect_zmq \
  --bind tcp://*:5557 \
  --socket PULL \
  --out data/raw/smoke.jsonl \
  --max-messages 3
```

## 4. Send messages from the robot/simulator side

Minimal PUSH client:

```python
import json
import time
import zmq

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://collector-host:5557")

def send(msg):
    msg.setdefault("sent_unix_time", time.time())
    sock.send_json(msg)

send({
    "message_type": "target_region",
    "collection_config": "scene_001",
    "component": "door_handle",
    "target_id": "scene_001:door_handle",
    "target_point": [0.42, -0.10, 0.31],
    "r_target": 0.015,
    "d_ref": 0.030,
    "orientation_tolerance_rad": 0.25
})

send({
    "message_type": "chunk",
    "trace_id": "rollout_scene_001_0001",
    "row_id": "rollout_scene_001_0001:chunk_000",
    "source": "rollout",
    "role": "partial",
    "operator_label": "near_but_not_accurate",
    "component": "door_handle",
    "phase": "approach",
    "collection_config": "scene_001",
    "prompt": "move to the door handle handoff pose",
    "observation": {"rgb_path": "logs/rollout_scene_001_0001/rgb_000.png"},
    "action_chunk": [[0.01, 0.00, 0.00], [0.01, 0.00, 0.00]],
    "mask": [1, 1],
    "geometry": {
        "tcp_positions": [[0.30, -0.10, 0.31], [0.31, -0.10, 0.31], [0.32, -0.10, 0.31]],
        "target_point": [0.42, -0.10, 0.31],
        "target_id": "scene_001:door_handle",
        "orientation_ok": True
    },
    "target_meta": {
        "target_id": "scene_001:door_handle",
        "declared_target_id": "scene_001:door_handle"
    },
    "stop": {
        "phase_ending": False,
        "stop_event": "timeout",
        "inside_terminal": False
    },
    "safety": {
        "unsafe": False,
        "manual_safety_stop": False,
        "quarantined": False
    },
    "provenance": {
        "manifest_present": True,
        "timing_ok": True,
        "schema_ok": True
    }
})
```

## 5. What to stream

You may stream either complete standard rows or lower-level events. Complete
rows are simplest and can be validated directly. Lower-level events are useful
if your robot process emits `episode_start`, `observation`, `action`,
`tcp_pose`, `operator_label`, and `episode_end` separately; in that case add an
offline converter that groups events into row dictionaries matching
`docs/DATA_CONTRACT.md`.

Recommended message types:

| message_type | Purpose |
|---|---|
| `target_region` | fixed target/tolerance declaration for one component/config |
| `episode_start` | rollout/correction/clean-demo trace metadata |
| `chunk` | one model/teleop action chunk with observation, action, TCP positions, mask |
| `operator_label` | final closed-vocabulary label for a rollout |
| `episode_end` | stop source, safety, timing/provenance summary |
| `standard_row` | already-converted PACER row; validate directly |

## 6. Validate collected rows

If each ZMQ line is already a standard PACER row:

```python
import json
from pacer_framework import validate_training_row, training_row_warnings

rows = [json.loads(line) for line in open("data/raw/pacer_stream.jsonl")]
rows = [r for r in rows if r.get("row_id")]
errors = [e for row in rows for e in validate_training_row(row)]
warnings = [w for row in rows for w in training_row_warnings(row)]
print("errors", errors)
print("warnings", warnings)
```

If you stream events, first convert them into standard rows, then run the same
validator. Keep raw JSONL unchanged so you can audit/rebuild rows later.

## 7. Collection workflow for PACER

1. Start the ZMQ collector.
2. Define target regions and stream/save `target_region` messages.
3. Run base-policy rollouts; stream `chunk` / `episode_end` / labels.
4. For failed/partial states, teleoperate corrections and stream correction
   chunks with `source="correction"` and `role="correction"`.
5. Record clean demos under the same target/config conventions with
   `source="clean_demo"` and `role="clean"`.
6. Convert/validate rows.
7. Run PACER weighting/export/training/evaluation.

## 8. Safety and reliability notes

- ZMQ logging is for data collection, not hard real-time robot control.
- Keep robot safety interlocks in the robot/controller process.
- Use monotonic timestamps and trace IDs so interrupted episodes can be audited.
- Save images/large arrays as files and send paths in JSON; do not push huge
  binary blobs unless you design a separate binary transport.
- Log raw actions and integrated TCP/EEF positions. PACER scoring uses the
  TCP/EEF positions, but raw actions are useful for debugging adapters.
- Use one closed operator-label vocabulary across all methods before looking at
  aggregate results.
