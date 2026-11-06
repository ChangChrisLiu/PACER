from __future__ import annotations

import math

import numpy as np
import pytest

from pacer.rollout_eval.auto_judge import (
    AutoJudge,
    AutoJudgeConfig,
    TrialJudge,
    classify_final_pose,
)
from pacer.rollout_eval.auto_judge_targets import JudgeTarget, upsert_target
from pacer.rollout_eval.ur5e_kinematics import flange_transform, matrix_to_pose, pose_to_matrix

TARGET = (0.40, -0.10, 0.05, math.pi, 0.0, 0.0)
FAN_POINT = (0.30, -0.10, 0.10, math.pi, 0.0, 0.0)
CHUNK = [[0.0] * 7 for _ in range(10)]
ROW_KEYS = {
    "query_index", "decision", "decision_reasons", "actual_tcp", "actual_pos_err_m",
    "actual_rot_err_rad", "predicted_tcp_end", "fk_residual_m", "submetrics", "v_hat",
    "blocker", "v_row",
}


def at(dx=0.0, dy=0.0, dz=0.0, rot_err=0.0):
    """Pose offset from TARGET; rot_err rotates about x away from the target orientation."""
    return [TARGET[0] + dx, TARGET[1] + dy, TARGET[2] + dz, math.pi - rot_err, 0.0, 0.0]


def obs_at(pose):
    return {"joint_positions": [0.0] * 7, "tcp_pose": list(pose)}


def make_judge(predicted, *, model_stop=False, non_targets=(), workspace=None):
    script = list(predicted)

    def predict_fn(chunk, obs):
        return [list(p) for p in script]

    return TrialJudge(
        target=JudgeTarget("config_001", "ram", (TARGET,), "tcp_pose"),
        non_targets=list(non_targets),
        workspace=workspace,
        config=AutoJudgeConfig(r_near_m=0.05),
        model_stop_check=(lambda chunk, obs: model_stop),
        predict_fn=predict_fn,
    )


def test_success_stop_when_actual_and_predicted_inside():
    judge = make_judge([at(dz=0.002)] * 10)
    assert judge.check_chunk(CHUNK, obs_at(at(dz=0.003))) == "auto_success_stop"
    assert judge.chunk_scores[-1]["decision"] == "auto_success_stop"


def test_no_success_when_predicted_leaves_region():
    judge = make_judge([at(dz=0.002)] * 5 + [at(dz=0.03)] * 5)
    assert judge.check_chunk(CHUNK, obs_at(at(dz=0.003))) is False
    assert judge.chunk_scores[-1]["decision"] == "continue"


def test_no_success_when_actual_outside():
    judge = make_judge([at(dz=0.002)] * 10)
    assert judge.check_chunk(CHUNK, obs_at(at(dz=0.03))) is False


def test_floor_blocker():
    judge = make_judge([at(dz=0.02)] * 9 + [at(dz=-0.006)])
    assert judge.check_chunk(CHUNK, obs_at(at(dz=0.03))) == "auto_blocker_stop"
    assert "below_floor" in judge.chunk_scores[-1]["decision_reasons"]


def test_orientation_blocker_only_near_target():
    far = make_judge([at(dx=0.10, rot_err=0.5)] * 10)
    assert far.check_chunk(CHUNK, obs_at(at(dx=0.12))) is False
    near = make_judge([at(dx=0.015, rot_err=0.5)] * 10)
    assert near.check_chunk(CHUNK, obs_at(at(dx=0.05))) == "auto_blocker_stop"
    assert "invalid_orientation" in near.chunk_scores[-1]["decision_reasons"]


def test_non_target_blocker():
    fan = JudgeTarget("config_001", "cpu_fan", (FAN_POINT,), "tcp_pose")
    path = [at(dx=-0.05, dz=0.05)] * 4 + [list(FAN_POINT[:2]) + [FAN_POINT[2] + 0.005, math.pi, 0.0, 0.0]] + [at(dz=0.03)] * 5
    judge = make_judge(path, non_targets=[fan])
    assert judge.check_chunk(CHUNK, obs_at(at(dz=0.06))) == "auto_blocker_stop"
    assert "non_target_exclusion" in judge.chunk_scores[-1]["decision_reasons"]


def test_workspace_blocker():
    judge = make_judge([at(dz=0.5)] * 10, workspace=((-1.0, -1.0, 0.0), (1.0, 1.0, 0.3)))
    assert judge.check_chunk(CHUNK, obs_at(at(dz=0.06))) == "auto_blocker_stop"
    assert "workspace" in judge.chunk_scores[-1]["decision_reasons"]


def test_action_bound_blocker_without_prediction():
    judge = make_judge([at(dz=0.03)] * 10)
    bad = [[0.0] * 6 + [0.0] for _ in range(9)] + [[float("nan")] * 7]
    assert judge.check_chunk(bad, obs_at(at(dz=0.06))) == "auto_blocker_stop"
    row = judge.chunk_scores[-1]
    assert row["decision_reasons"] == ["action_bound"]
    assert row["predicted_tcp_end"] is None


def test_model_stop_token_takes_precedence():
    judge = make_judge([at(dz=-0.02)] * 10, model_stop=True)
    assert judge.check_chunk(CHUNK, obs_at(at(dz=0.03))) is True
    assert judge.chunk_scores[-1]["decision"] == "model_stop_token"


def test_chunk_scores_row_keys_and_values():
    judge = make_judge([at(dz=0.02)] * 10)
    judge.check_chunk(CHUNK, obs_at(at(dz=0.04)))
    judge.check_chunk(CHUNK, obs_at(at(dz=0.025)))
    first, second = judge.chunk_scores
    assert set(first) == ROW_KEYS and set(second) == ROW_KEYS
    assert first["query_index"] == 0 and second["query_index"] == 1
    assert first["fk_residual_m"] is None
    assert second["fk_residual_m"] == pytest.approx(0.005, abs=1e-9)
    assert 0.0 < first["v_hat"] <= 1.0 and first["v_row"] == first["v_hat"]
    assert set(first["submetrics"]) == {"progress", "proximity", "terminal", "direction"}

    blocked = make_judge([at(dz=-0.01)] * 10)
    blocked.check_chunk(CHUNK, obs_at(at(dz=0.03)))
    assert blocked.chunk_scores[-1]["v_row"] == 0.0


def test_missing_obs_keys_raise():
    judge = make_judge([at()] * 10)
    with pytest.raises(ValueError, match="tcp_pose"):
        judge.check_chunk(CHUNK, {"joint_positions": [0.0] * 6})


def test_default_fk_predict_hold_position():
    q0 = [0.1, -1.2, 1.4, -1.6, -1.5, 0.3]
    tcp = matrix_to_pose(flange_transform(q0) @ pose_to_matrix([0, 0, 0.2, 0, 0, 0]))
    target = JudgeTarget("config_001", "ram", (tuple(tcp),), "tcp_pose")
    judge = TrialJudge(target=target, non_targets=[], workspace=None, config=AutoJudgeConfig(r_near_m=0.05))
    decision = judge.check_chunk([q0 + [0.0]] * 10, {"joint_positions": q0 + [0.0], "tcp_pose": tcp})
    assert np.allclose(judge.chunk_scores[-1]["predicted_tcp_end"], tcp, atol=1e-9)
    assert decision == "auto_success_stop"


@pytest.mark.parametrize("pos,rot,zm,label,review", [
    (0.003, 0.1, 0.010, "success", False), (0.008, 0.1, 0.010, "near_but_not_accurate", False),
    (0.008, 0.5, 0.010, "wrong_orientation_or_wrong_location", False),
    (0.030, 0.1, 0.010, "wrong_orientation_or_wrong_location", False),
    (0.080, 0.1, 0.010, "totally_off_wrong_region_or_target", False),
    (0.0055, 0.1, 0.010, "near_but_not_accurate", True),
    (0.0045, 0.1, 0.010, "success", True),
    (0.003, 0.1, -0.002, "wrong_orientation_or_wrong_location", False),
    (0.003, 0.1, 0.0005, "success", True)])
def test_classify_final_pose_cpu(pos, rot, zm, label, review):
    got = classify_final_pose(pos_err_m=pos, rot_err_rad=rot, z_margin_m=zm,
                              r_target_m=0.005, config=AutoJudgeConfig(r_near_m=0.05))
    assert got[0] == label and got[2] is review
    assert ("needs_review" in got[1]) is review


def test_final_feedback_mapping():
    judge = make_judge([at(dz=0.02)] * 9 + [at(dz=-0.006)])
    judge.check_chunk(CHUNK, obs_at(at(dz=0.03)))
    fb = judge.final_feedback(stop_source="auto_blocker_stop", final_tcp=at(dz=0.03))
    assert fb.label == "wrong_orientation_or_wrong_location"
    assert fb.note.startswith("auto_blocker_stop;") and "below_floor" in fb.note
    assert fb.operator == "auto_judge" and fb.auto_assigned is True

    bound = make_judge([at()] * 10)
    bound.check_chunk([[float("nan")] * 7] * 10, obs_at(at(dz=0.03)))
    fb = bound.final_feedback(stop_source="auto_blocker_stop", final_tcp=at(dz=0.03))
    assert fb.label == "totally_off_wrong_region_or_target"
    assert "needs_review:action_bound" in fb.note

    fresh = make_judge([at()] * 10)
    assert fresh.final_feedback(stop_source="unsafe_abort", final_tcp=None).label == "unsafe_abort"
    none_fb = fresh.final_feedback(stop_source="timeout", final_tcp=None)
    assert none_fb.label == "operator_uncertain_exclude" and "final_tcp_unavailable" in none_fb.note
    border = fresh.final_feedback(stop_source="timeout", final_tcp=at(dx=0.0105))
    assert border.label == "near_but_not_accurate" and "needs_review" in border.note
    ok = fresh.final_feedback(stop_source="auto_success_stop", final_tcp=at(dz=0.002))
    assert ok.label == "success" and ok.note.startswith("auto_success_stop")


def test_auto_judge_missing_target_raises_keyerror_and_reload_picks_up_new_target(tmp_path):
    path = tmp_path / "targets.json"
    aj = AutoJudge(targets_path=path, config=AutoJudgeConfig(r_near_m=0.05))
    assert aj.has_target("config_001", "ram") is False
    with pytest.raises(KeyError):
        aj.trial_judge("config_001", "ram")
    upsert_target(path, JudgeTarget("config_001", "ram", (TARGET,), "tcp_pose"))
    upsert_target(path, JudgeTarget("config_001", "cpu_fan", (FAN_POINT,), "tcp_pose"))
    aj.reload()
    assert aj.has_target("config_001", "ram") is True
    judge = aj.trial_judge("config_001", "ram")
    assert [t.component for t in judge.non_targets] == ["cpu_fan"]
    meta = aj.to_dict()
    assert meta["config"]["r_near_m"] == 0.05
    assert meta["targets_path"] == str(path)


def test_config_rejects_r_near_not_above_dref():
    with pytest.raises(ValueError):
        AutoJudgeConfig(r_near_m=0.02)
