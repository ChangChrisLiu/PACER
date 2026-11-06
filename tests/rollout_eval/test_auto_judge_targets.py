from __future__ import annotations

import json
import pickle

import pytest

from pacer.rollout_eval.auto_judge_targets import (
    TARGETS_SCHEMA,
    JudgeTarget,
    load_targets,
    main,
    target_from_demo_episode,
    target_from_region_json,
    upsert_target,
)

POSE_A = (0.40, -0.10, 0.05, 3.14, 0.0, 0.0)
POSE_B = (0.30, -0.20, 0.06, 3.14, 0.0, 0.0)


def test_upsert_then_load_roundtrip(tmp_path):
    path = tmp_path / "targets.json"
    upsert_target(path, JudgeTarget("1", "ram", (POSE_A,), "tcp_pose"))
    book = load_targets(path)
    target = book.get("config_001", "ram")
    assert target is not None
    assert target.config_id == "config_001"
    assert target.points == (POSE_A,)
    assert target.source == "tcp_pose"
    assert json.loads(path.read_text())["schema"] == TARGETS_SCHEMA


def test_upsert_preserves_other_entries_and_workspace(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({
        "schema": TARGETS_SCHEMA,
        "targets": {"config_001/cpu": {"points": [list(POSE_B)], "source": "demo_endpoint"}},
        "workspace": {"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 1.0]},
    }))
    upsert_target(path, JudgeTarget("config_001", "ram", (POSE_A,), "tcp_pose"))
    book = load_targets(path)
    assert book.get("config_001", "cpu").points == (POSE_B,)
    assert book.get("config_001", "ram").points == (POSE_A,)
    assert book.workspace_min == (-1.0, -1.0, 0.0)
    assert book.workspace_max == (1.0, 1.0, 1.0)


def test_load_missing_file_is_empty(tmp_path):
    book = load_targets(tmp_path / "absent.json")
    assert book.targets == {}
    assert book.get("config_001", "ram") is None
    assert book.workspace_min is None and book.workspace_max is None


def test_bad_point_length_raises(tmp_path):
    with pytest.raises(ValueError):
        JudgeTarget("config_001", "ram", ((0.1, 0.2, 0.3),), "tcp_pose")
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({
        "schema": TARGETS_SCHEMA,
        "targets": {"config_001/ram": {"points": [[0.1, 0.2]], "source": "tcp_pose"}},
    }))
    with pytest.raises(ValueError):
        load_targets(path)


def test_wrong_schema_raises(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"schema": "something_else", "targets": {}}))
    with pytest.raises(ValueError):
        load_targets(path)


def test_unknown_component_raises():
    with pytest.raises(ValueError):
        JudgeTarget("config_001", "ssd", (POSE_A,), "tcp_pose")


def test_demo_episode_defaults_to_last_frame_and_honors_index(tmp_path):
    for i in range(3):
        with (tmp_path / f"frame_{i:04d}.pkl").open("wb") as f:
            pickle.dump({"tcp_pose": [0.1 * i, 0.0, 0.05, 3.14, 0.0, 0.0], "joint_positions": [0.0] * 7}, f)
    last = target_from_demo_episode(tmp_path, config_id="config_002", component="cpu")
    first = target_from_demo_episode(tmp_path, config_id="config_002", component="cpu", frame_index=0)
    assert last.points[0][0] == pytest.approx(0.2)
    assert first.points[0][0] == pytest.approx(0.0)
    assert last.source == "demo_endpoint"


def test_region_json_uses_all_points(tmp_path):
    path = tmp_path / "target_region.json"
    path.write_text(json.dumps({
        "config_id": "config_001", "component": "ram",
        "points": [{"tcp_pose": list(POSE_A)}, {"tcp_pose": list(POSE_B)}],
    }))
    target = target_from_region_json(path, config_id="config_001", component="ram")
    assert target.points == (POSE_A, POSE_B)
    assert target.source == "target_region"


def test_others_in_config_excludes_self_and_other_configs(tmp_path):
    path = tmp_path / "targets.json"
    upsert_target(path, JudgeTarget("config_001", "ram", (POSE_A,), "tcp_pose"))
    upsert_target(path, JudgeTarget("config_001", "cpu_fan", (POSE_B,), "tcp_pose"))
    upsert_target(path, JudgeTarget("config_002", "cpu", (POSE_B,), "tcp_pose"))
    others = load_targets(path).others_in_config("config_001", "ram")
    assert [(t.config_id, t.component) for t in others] == [("config_001", "cpu_fan")]


def test_cli_add_tcp_pose_writes_file(tmp_path):
    path = tmp_path / "targets.json"
    main(["add", "--targets", str(path), "--config-id", "config_003", "--component", "graphic_card",
          "--tcp-pose", *[str(v) for v in POSE_A]])
    target = load_targets(path).get("config_003", "graphic_card")
    assert target is not None
    assert target.points == (POSE_A,)
    assert target.source == "tcp_pose"
