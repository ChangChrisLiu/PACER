"""Planner/corrector scoring helpers for TRACE-VLA collection collection."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .schemas import PlannerScore, TargetRegion


def _rot_error(a_rotvec: Sequence[float], b_rotvec: Sequence[float]) -> float:
    ra = Rotation.from_rotvec(np.asarray(a_rotvec, dtype=float))
    rb = Rotation.from_rotvec(np.asarray(b_rotvec, dtype=float))
    return float((ra.inv() * rb).magnitude())


def score_planner_final_pose(
    final_tcp_pose: Sequence[float] | None,
    target_region: TargetRegion | None,
    stop_source: str,
    steps_to_stop: int,
) -> PlannerScore:
    if final_tcp_pose is None or target_region is None or not target_region.points:
        return PlannerScore(
            pos_error_min_m=None,
            rot_error_min_rad=None,
            inside_region=None,
            closest_target_idx=None,
            stop_source=stop_source,
            steps_to_stop=steps_to_stop,
        )

    final = np.asarray(final_tcp_pose, dtype=float)
    if final.shape[0] < 6:
        raise ValueError("final_tcp_pose must contain at least 6 values [x,y,z,rx,ry,rz]")

    best_idx = None
    best_pos = math.inf
    best_rot = math.inf
    for idx, point in enumerate(target_region.points):
        target = np.asarray(point.tcp_pose, dtype=float)
        pos_err = float(np.linalg.norm(final[:3] - target[:3]))
        rot_err = _rot_error(final[3:6].tolist(), target[3:6].tolist())
        if (pos_err, rot_err) < (best_pos, best_rot):
            best_idx = idx
            best_pos = pos_err
            best_rot = rot_err

    inside = (
        best_pos <= target_region.position_tolerance_m
        and best_rot <= target_region.rotation_tolerance_rad
    )
    return PlannerScore(
        pos_error_min_m=best_pos,
        rot_error_min_rad=best_rot,
        inside_region=inside,
        closest_target_idx=best_idx,
        stop_source=stop_source,
        steps_to_stop=steps_to_stop,
    )


def _as_float_vector(value: Any, limit: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if limit is not None:
        arr = arr[:limit]
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def score_model_rollout_performance(
    frames: Sequence[dict[str, Any]],
    action_trace: Sequence[dict[str, Any]],
    *,
    arm_motion_threshold_rad: float = 1e-3,
    tcp_motion_threshold_m: float = 1e-4,
) -> dict[str, Any]:
    """Summarize whether commanded model rollout actually moved the robot.

    This is intentionally a diagnostic score, not a task outcome metric. It is
    designed to separate pipeline failures (commands were produced but the arm
    did not move / did not track them) from genuine policy failures (the arm
    moved, but to a bad pose or without a stop token).
    """
    joint_rows = [_as_float_vector(f.get("joint_positions"), limit=6) for f in frames]
    joint_rows = [j for j in joint_rows if j is not None and j.size >= 6]
    tcp_rows = [_as_float_vector(f.get("tcp_pose"), limit=6) for f in frames]
    tcp_rows = [t for t in tcp_rows if t is not None and t.size >= 6]

    joint_motion_total_norm = None
    joint_motion_max_abs = None
    if len(joint_rows) >= 2:
        joint_delta = joint_rows[-1][:6] - joint_rows[0][:6]
        joint_motion_total_norm = float(np.linalg.norm(joint_delta))
        joint_motion_max_abs = float(np.max(np.abs(joint_delta)))

    tcp_motion_total_m = None
    tcp_z_delta_m = None
    if len(tcp_rows) >= 2:
        tcp_delta = tcp_rows[-1][:3] - tcp_rows[0][:3]
        tcp_motion_total_m = float(np.linalg.norm(tcp_delta))
        tcp_z_delta_m = float(tcp_delta[2])

    executed = [r for r in action_trace if r.get("type") == "executed_action"]
    first_frame_joints = joint_rows[0][:6] if joint_rows else None
    commanded_delta_max_abs = None
    tracking_error_final_norm = None
    tracking_error_max_norm = None
    n_tracking = 0
    tracking_errors: list[float] = []
    commanded_deltas: list[float] = []
    for row in executed:
        target = _as_float_vector(row.get("post_safety_clamp_action"), limit=6)
        extra_raw = row.get("extra")
        extra = extra_raw if isinstance(extra_raw, dict) else {}
        before = _as_float_vector(extra.get("pre_action_obs_joints"), limit=6)
        if before is None:
            before = first_frame_joints
        after = _as_float_vector(extra.get("joints_after_cmd"), limit=6)
        if target is not None and before is not None and target.size >= 6 and before.size >= 6:
            commanded_deltas.append(float(np.max(np.abs(target[:6] - before[:6]))))
        if target is not None and after is not None and target.size >= 6 and after.size >= 6:
            err = float(np.linalg.norm(target[:6] - after[:6]))
            tracking_errors.append(err)
            n_tracking += 1
    if commanded_deltas:
        commanded_delta_max_abs = float(max(commanded_deltas))
    if tracking_errors:
        tracking_error_final_norm = float(tracking_errors[-1])
        tracking_error_max_norm = float(max(tracking_errors))

    arm_motion_detected = bool(
        (joint_motion_total_norm is not None and joint_motion_total_norm > arm_motion_threshold_rad)
        or (tcp_motion_total_m is not None and tcp_motion_total_m > tcp_motion_threshold_m)
    )
    commands_nontrivial = bool(commanded_delta_max_abs is not None and commanded_delta_max_abs > arm_motion_threshold_rad)
    pipeline_suspect = bool(commands_nontrivial and not arm_motion_detected)

    return {
        "num_frames": len(frames),
        "num_executed_actions": len(executed),
        "arm_motion_detected": arm_motion_detected,
        "joint_motion_total_norm": _round_or_none(joint_motion_total_norm),
        "joint_motion_max_abs": _round_or_none(joint_motion_max_abs),
        "tcp_motion_total_m": _round_or_none(tcp_motion_total_m),
        "tcp_z_delta_m": _round_or_none(tcp_z_delta_m),
        "commanded_joint_delta_max_abs": _round_or_none(commanded_delta_max_abs),
        "tracking_error_final_norm": _round_or_none(tracking_error_final_norm),
        "tracking_error_max_norm": _round_or_none(tracking_error_max_norm),
        "num_tracking_error_samples": n_tracking,
        "pipeline_suspect_no_motion_with_commands": pipeline_suspect,
    }
