from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from pacer.rollout_eval.ur5e_kinematics import (
    fk_check_frames,
    flange_transform,
    main,
    matrix_to_pose,
    pose_to_matrix,
    predict_tcp_poses,
    solve_tool_offset,
)

Q0 = [0.1, -1.2, 1.4, -1.6, -1.5, 0.3]
Q1 = [0.2, -1.1, 1.3, -1.7, -1.5, 0.4]


def _tool():
    return pose_to_matrix([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])


def _frames(perturb_index: int | None = None) -> list[dict]:
    frames = []
    for k in range(5):
        q = [v + 0.05 * k for v in Q0]
        tcp = matrix_to_pose(flange_transform(q) @ _tool())
        if perturb_index == k:
            tcp[0] += 0.05
        frames.append({"joint_positions": q + [0.0], "tcp_pose": tcp})
    return frames


def test_flange_zero_pose_matches_ur5e_reference():
    pose = matrix_to_pose(flange_transform([0.0] * 6))
    assert np.allclose(pose[:3], [-0.8172, -0.2329, 0.0628], atol=1e-4)


def test_pose_matrix_roundtrip():
    p = [0.1, -0.2, 0.3, 0.1, 0.2, -0.3]
    assert np.allclose(matrix_to_pose(pose_to_matrix(p)), p, atol=1e-9)


def test_tool_offset_reproduces_measured_and_new_pose():
    O = _tool()
    off = solve_tool_offset(Q0, matrix_to_pose(flange_transform(Q0) @ O))
    pred = predict_tcp_poses([Q0, Q1 + [0.5]], off)  # 7th entry (gripper) ignored
    assert np.allclose(pred[0], matrix_to_pose(flange_transform(Q0) @ O), atol=1e-9)
    assert np.allclose(pred[1], matrix_to_pose(flange_transform(Q1) @ O), atol=1e-9)


def test_fk_check_frames_small_for_consistent_large_for_perturbed():
    good = fk_check_frames(_frames())
    bad = fk_check_frames(_frames(perturb_index=3))
    assert good["n_frames"] == 5
    assert good["max_position_residual_m"] < 1e-9
    assert good["max_rotation_residual_rad"] < 1e-9
    assert bad["max_position_residual_m"] >= 0.049


def test_flange_rejects_bad_input():
    with pytest.raises(ValueError):
        flange_transform([0.0] * 5)
    with pytest.raises(ValueError):
        flange_transform([0.0] * 5 + [float("nan")])


def test_cli_check_episode_prints_json(tmp_path, capsys):
    for i, frame in enumerate(_frames()):
        with (tmp_path / f"frame_{i:04d}.pkl").open("wb") as f:
            pickle.dump(frame, f)
    main(["--check-episode", str(tmp_path)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_frames"] == 5
    assert payload["max_position_residual_m"] < 1e-9
