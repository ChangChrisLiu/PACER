"""UR5e forward kinematics for scoring predicted joint-space action chunks.

The OpenPI adapter emits chunks of absolute joint targets
(``openpi_absolute_joint``: six joints plus an absolute gripper command). The
auto trial judge maps those targets to TCP poses with the standard UR5e
Denavit-Hartenberg parameters and a tool offset solved from one measured
``(joint_positions, tcp_pose)`` pair, so no TCP configuration is needed.

Assumption: the controller's base frame (the frame of ``tcp_pose``) equals the
DH base frame. ``fk_check_frames`` and the per-chunk ``fk_residual_m`` written by
the auto judge expose this assumption on recorded data; check one existing
episode with ``python -m pacer.rollout_eval.ur5e_kinematics --check-episode DIR``
before the first live use.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

# Standard DH parameters of the UR5e (metres, radians).
UR5E_D = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
UR5E_A = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
UR5E_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)


def _finite_vector(values: Sequence[float], length: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < length:
        raise ValueError(f"{name} needs at least {length} values, got {arr.size}")
    arr = arr[:length]
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def flange_transform(q: Sequence[float]) -> np.ndarray:
    """Base-to-flange 4x4 transform for six joint angles."""
    joints = np.asarray(q, dtype=float).reshape(-1)
    if joints.size != 6:
        raise ValueError(f"flange_transform expects 6 joint angles, got {joints.size}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("flange_transform received non-finite joint angles")
    T = np.eye(4)
    for theta, d, a, alpha in zip(joints, UR5E_D, UR5E_A, UR5E_ALPHA):
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        T = T @ np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return T


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """``[x, y, z, rx, ry, rz]`` (rotation vector) to a 4x4 transform."""
    p = _finite_vector(pose, 6, "pose")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[3:6]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def matrix_to_pose(T: np.ndarray) -> list[float]:
    """4x4 transform to ``[x, y, z, rx, ry, rz]`` (rotation vector)."""
    M = np.asarray(T, dtype=float)
    rotvec = Rotation.from_matrix(M[:3, :3]).as_rotvec()
    return [float(v) for v in (*M[:3, 3], *rotvec)]


def solve_tool_offset(q: Sequence[float], tcp_pose: Sequence[float]) -> np.ndarray:
    """Flange-to-TCP transform implied by one measured joint/TCP pair."""
    return np.linalg.inv(flange_transform(_finite_vector(q, 6, "joint_positions"))) @ pose_to_matrix(tcp_pose)


def predict_tcp_poses(joint_targets: Sequence[Sequence[float]], tool_offset: np.ndarray) -> list[list[float]]:
    """TCP poses for each joint-target row (only the first six entries are used)."""
    offset = np.asarray(tool_offset, dtype=float)
    return [
        matrix_to_pose(flange_transform(_finite_vector(row, 6, "joint target")) @ offset)
        for row in joint_targets
    ]


def _rotation_error(a: Sequence[float], b: Sequence[float]) -> float:
    ra = Rotation.from_rotvec(np.asarray(a, dtype=float)[3:6])
    rb = Rotation.from_rotvec(np.asarray(b, dtype=float)[3:6])
    return float((ra.inv() * rb).magnitude())


def fk_check_frames(frames: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Compare FK-predicted TCP poses with recorded ``tcp_pose`` over frames.

    The tool offset is solved from frame 0, then every frame's joints are mapped
    through FK and compared with its recorded pose.
    """
    if not frames:
        raise ValueError("fk_check_frames requires at least one frame")
    try:
        joints = [_finite_vector(f["joint_positions"], 6, "joint_positions") for f in frames]
        tcps = [_finite_vector(f["tcp_pose"], 6, "tcp_pose") for f in frames]
    except KeyError as exc:
        raise ValueError(f"frame is missing {exc.args[0]!r}") from exc
    offset = solve_tool_offset(joints[0], tcps[0])
    predicted = predict_tcp_poses(joints, offset)
    pos_res = [float(np.linalg.norm(np.asarray(p[:3]) - t[:3])) for p, t in zip(predicted, tcps)]
    rot_res = [_rotation_error(p, t) for p, t in zip(predicted, tcps)]
    return {
        "n_frames": len(frames),
        "max_position_residual_m": max(pos_res),
        "mean_position_residual_m": float(np.mean(pos_res)),
        "max_rotation_residual_rad": max(rot_res),
    }


def _load_episode_frames(episode_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(episode_dir.glob("frame_*.pkl"))
    if not paths:
        raise SystemExit(f"no frame_*.pkl files in {episode_dir}")
    frames = []
    for path in paths:
        with path.open("rb") as f:
            frames.append(pickle.load(f))
    return frames


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check UR5e FK against a recorded episode")
    ap.add_argument("--check-episode", type=Path, required=True,
                    help="Episode directory containing frame_*.pkl with joint_positions and tcp_pose")
    args = ap.parse_args(argv)
    print(json.dumps(fk_check_frames(_load_episode_frames(args.check_episode)), indent=2))


if __name__ == "__main__":
    main()
