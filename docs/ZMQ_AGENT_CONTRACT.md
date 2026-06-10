# ZMQ agent contract

This document defines the small JSON contract used by PACER hardware probes
and by local robot/camera bridge adapters. It is deliberately minimal so labs can
wrap existing bridges without changing the core rollout/correction code.

## Transport

- Pattern: REQ/REP for command/status probes.
- Serialization: UTF-8 JSON object per request and response.
- Timeout: clients should fail fast, normally 1-3 seconds for probes.
- Endpoints: configured locally, never hard-coded in committed files.

## Common request fields

```json
{
  "type": "ping",
  "client": "pacer",
  "request_id": "optional-uuid"
}
```

Responses should include:

```json
{
  "ok": true,
  "type": "pong",
  "server_time": 0.0,
  "message": "optional"
}
```

## Request types

### ping

Liveness only. Must not move hardware.

```json
{"type": "ping"}
```

### status

Returns robot/camera/bridge readiness. Must not move hardware.

```json
{
  "type": "status"
}
```

Suggested response fields:

```json
{
  "ok": true,
  "mode": "idle",
  "e_stop": false,
  "protective_stop": false,
  "robot_connected": true,
  "camera_connected": true,
  "last_observation_age_s": 0.03
}
```

### get_observation

Returns the current observation metadata. Large images may be transported by the
local bridge separately; this contract only requires metadata and small arrays.

```json
{"type": "get_observation", "include_images": false}
```

Expected semantic keys for PACER adapters:

- `joint_positions`: six arm joints plus optional gripper.
- `gripper_position`: normalized or raw, with convention documented.
- `tcp_pose`: x, y, z, rx, ry, rz.
- `base_rgb` and `wrist_rgb`: arrays or references, depending on the bridge.

### move_joint / move_tcp / set_gripper

Live motion commands. These should be guarded by the local bridge as well as by
PACER's caller-side confirmation gate. Probes in this repo do not issue them
unless you explicitly pass a custom JSON file.

## Safety rules

- `ping`, `status`, and `get_observation` must never move hardware.
- Motion commands must reject requests unless the bridge is armed.
- All commands should return the post-clamp command actually sent to hardware.
- Stop-token gripper values must be intercepted at the execution boundary, not
  rewritten in raw VLA traces.
- Manual stop should stop the current loop without sending an extra RTDE stop
  after the operator has already stopped/held the robot.
