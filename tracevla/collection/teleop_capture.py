"""Teleoperation capture helpers for TRACE-VLA collection.

The hardware-facing class is intentionally small: it reuses JoystickAgent
without changing button_mapping.json. Unit tests cover the pure conversion
helpers; hardware smoke tests exercise the live loop.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .schemas import TargetPoint


@dataclass
class PoseSnapshot:
    tcp_pose: list[float]
    joint_positions: list[float]
    gripper_obs: int
    timestamp: float

    def to_target_point(self, tag: str = "target") -> TargetPoint:
        return TargetPoint(
            tcp_pose=self.tcp_pose,
            joint_positions=self.joint_positions,
            gripper_obs=self.gripper_obs,
            timestamp=self.timestamp,
            tag=tag,
        )


def snapshot_from_observation(obs: dict[str, Any], timestamp: float | None = None) -> PoseSnapshot:
    joints = obs.get("joint_positions", [])
    tcp = obs.get("tcp_pose")
    if tcp is None and "ee_pos_quat" in obs:
        # Caller should normally provide tcp_pose; this fallback keeps tests simple.
        ee = obs["ee_pos_quat"]
        tcp = list(ee[:3]) + [0.0, 0.0, 0.0]
    if tcp is None:
        raise ValueError("observation must contain tcp_pose or ee_pos_quat")
    grip = obs.get("gripper_pos", obs.get("gripper_position", 0))
    if not isinstance(grip, (int, float)):
        try:
            grip = float(grip[0]) * 255.0
        except Exception:
            grip = 0
    return PoseSnapshot(
        tcp_pose=[float(x) for x in list(tcp)[:6]],
        joint_positions=[float(x) for x in list(joints)[:6]],
        gripper_obs=int(round(float(grip))),
        timestamp=float(timestamp if timestamp is not None else obs.get("timestamp", time.time())),
    )


class TeleopCaptureController:
    """Thin hardware loop wrapper for target/failure/human-correction capture.

    The implementation is conservative. The new TRACE-VLA collection script can use this
    class for live sessions; dry-run/tests use the pure helpers above.
    """

    def __init__(self, joystick_agent, robot_client, rate_hz: float = 100.0):
        self.joystick_agent = joystick_agent
        self.robot_client = robot_client
        self.rate_hz = rate_hz
        self.active = False

    def set_active(self, active: bool) -> None:
        self.active = bool(active)
        if not active:
            try:
                self.robot_client.speed_stop()
            except Exception:
                pass

    def tick(self) -> None:
        if not self.active:
            return
        action = self.joystick_agent.act({})
        if isinstance(action, dict) and action.get("type") == "velocity":
            self.robot_client.command_cartesian_velocity(
                velocity=action["velocity"],
                acceleration=action["acceleration"],
                time_running=action["time"],
                gripper_vel=action.get("gripper_vel", 0.0),
            )


class TeleopThread(threading.Thread):
    """Background thread that drives a controller's tick() at a fixed cadence.

    Exits cleanly on first controller exception, storing it in last_error so
    callers can surface a real failure instead of silently leaving teleop dead.
    """

    def __init__(self, controller: Any, tick_sleep: float = 0.01) -> None:
        super().__init__(daemon=True)
        self.controller = controller
        self.tick_sleep = float(tick_sleep)
        self._running = True
        self.last_error: BaseException | None = None

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            try:
                self.controller.tick()
            except Exception as exc:
                self.last_error = exc
                self._running = False
                break
            time.sleep(self.tick_sleep)
