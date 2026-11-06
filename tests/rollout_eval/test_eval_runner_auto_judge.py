from __future__ import annotations

import json

from pacer.collection.inference_runner import RolloutResult
from pacer.rollout_eval.auto_judge import AutoJudge, AutoJudgeConfig
from pacer.rollout_eval.auto_judge_targets import JudgeTarget, upsert_target
from pacer.rollout_eval.eval_feedback import EvalFeedback
from pacer.rollout_eval.eval_plan import EvalProgressTracker, ModelSpec, build_eval_plan
from pacer.rollout_eval.eval_runner import HardwareHandles, RolloutEvalConfig, RolloutEvalRunner
from pacer.rollout_eval.eval_writer import RolloutEvalWriter
from pacer.rollout_eval.ur5e_kinematics import flange_transform, matrix_to_pose, pose_to_matrix

Q0 = [0.1, -1.2, 1.4, -1.6, -1.5, 0.3]
TCP = matrix_to_pose(flange_transform(Q0) @ pose_to_matrix([0, 0, 0.2, 0, 0, 0]))
OBS = {"joint_positions": Q0 + [0.0], "tcp_pose": TCP}
MODEL = ModelSpec(model_id="m1", model_type="openpi", checkpoint_path="ck")


class FakeRobot:
    def get_tcp_pose_raw(self):
        return list(TCP)


class FakeAdapter:
    def infer(self, obs, prompt):
        return [Q0 + [0.0] for _ in range(10)]

    def apply_action(self, action, obs, robot, safety):
        return None

    def get_current_state(self, obs):
        return None


class FakeAgent:
    def _check_chunk_stop(self, chunk, state):
        return False


class FakeChunkRunner:
    """Calls stop_check once on the adapter chunk, like the first query of a rollout."""

    def run(self, *, get_obs, infer, stop_check, prompt, **_ignored):
        obs = get_obs()
        result = stop_check(infer(obs, prompt), obs)
        stop_source = result if isinstance(result, str) else ("model_stop_token" if result else "timeout")
        return RolloutResult(frames=[{}], action_trace=[{"type": "rollout_stop", "stop_source": stop_source}],
                             stop_source=stop_source, steps=1, final_obs=obs)


def _plan(trials=1):
    return build_eval_plan(run_id="run1", models=[MODEL], configs=["config_001"], components=["ram"],
                           trials_per_component=trials)


def _runner(tmp_path, *, auto_judge, calls, obs=OBS, position_gate=None):
    writer = RolloutEvalWriter(tmp_path / "out", "run1")

    def record(name, value=None):
        def fn(*args, **kwargs):
            calls.setdefault(name, []).append(args or kwargs)
            return value
        return fn

    kwargs = {"position_gate": position_gate or record("position_gate")}
    return RolloutEvalRunner(
        config=RolloutEvalConfig(),
        hardware=HardwareHandles(robot=FakeRobot(), obs_client=None, cameras={}, safety_monitor=None,
                                 home_joints=None, home_gripper=0.0),
        writer=writer,
        progress=EvalProgressTracker(writer.progress_path()),
        build_frame=lambda obs, cams, size: {},
        create_adapter=lambda args: FakeAdapter(),
        get_obs=lambda *a, **k: dict(obs),
        vla_agent_factory=lambda **kw: FakeAgent(),
        stop_after_rollout=lambda src, model: calls.setdefault("stop_after", []).append(src),
        move_home=lambda: None,
        prompt_label=record("prompt_label", EvalFeedback(label="success")),
        readiness_gate=record("readiness_gate"),
        pre_rollout_gate=record("pre_rollout_gate"),
        chunk_runner_factory=lambda config, model: FakeChunkRunner(),
        manual_stop_reader=lambda: None,
        auto_judge=auto_judge,
        print_fn=lambda *_: None,
        **kwargs,
    ), writer


def _targets(tmp_path, *, with_target=True):
    path = tmp_path / "targets.json"
    if with_target:
        upsert_target(path, JudgeTarget("config_001", "ram", (tuple(TCP),), "demo_endpoint"))
    return path


def _summary(writer):
    return [json.loads(line) for line in writer.summary_jsonl_path().read_text().splitlines()]


def test_auto_mode_skips_pre_rollout_gate_and_labels_success(tmp_path):
    calls = {}
    aj = AutoJudge(targets_path=_targets(tmp_path), config=AutoJudgeConfig(r_near_m=0.05))
    runner, writer = _runner(tmp_path, auto_judge=aj, calls=calls)
    runner.run(plan=_plan(), models_by_id={"m1": MODEL})
    assert "pre_rollout_gate" not in calls and "prompt_label" not in calls
    (row,) = _summary(writer)
    assert row["operator_label"] == "success"
    assert row["stop_source"] == "auto_success_stop"
    assert row["operator_auto_assigned"] is True
    scores = (writer.trial_dir(_plan()[0]) / "chunk_scores.jsonl").read_text().splitlines()
    assert len(scores) == 1 and json.loads(scores[0])["decision"] == "auto_success_stop"
    ctx = json.loads((writer.trial_dir(_plan()[0]) / "auto_judge_context.json").read_text())
    assert ctx["target"]["points"] == [list(TCP)]
    assert ctx["config"]["r_near_m"] == 0.05


def test_auto_mode_rejects_non_openpi_model(tmp_path):
    calls = {}
    aj = AutoJudge(targets_path=_targets(tmp_path), config=AutoJudgeConfig(r_near_m=0.05))
    model = ModelSpec(model_id="m1", model_type="openvla", checkpoint_path="ck")
    runner, writer = _runner(tmp_path, auto_judge=aj, calls=calls)
    runner.run(plan=_plan(), models_by_id={"m1": model})
    progress = json.loads(writer.progress_path().read_text())
    assert "openpi" in next(iter(progress["failed"].values()))
    assert not writer.summary_jsonl_path().exists()


def test_auto_stop_uses_model_stop_token_hardware_semantics(tmp_path):
    calls = {}
    aj = AutoJudge(targets_path=_targets(tmp_path), config=AutoJudgeConfig(r_near_m=0.05))
    runner, _ = _runner(tmp_path, auto_judge=aj, calls=calls)
    runner.run(plan=_plan(), models_by_id={"m1": MODEL})
    assert calls["stop_after"] == ["model_stop_token"]


def test_position_gate_waits_for_target_then_runs(tmp_path):
    calls = {}
    path = _targets(tmp_path, with_target=False)
    gate_calls = []

    def gate(*, prompt, print_fn):
        gate_calls.append(prompt)
        if len(gate_calls) == 2:
            upsert_target(path, JudgeTarget("config_001", "ram", (tuple(TCP),), "demo_endpoint"))

    aj = AutoJudge(targets_path=path, config=AutoJudgeConfig(r_near_m=0.05))
    runner, writer = _runner(tmp_path, auto_judge=aj, calls=calls, position_gate=gate)
    runner.run(plan=_plan(trials=2), models_by_id={"m1": MODEL})
    assert len(gate_calls) == 2
    assert all("config_001/ram" in prompt for prompt in gate_calls)
    assert [r["operator_label"] for r in _summary(writer)] == ["success", "success"]


def test_judge_error_marks_trial_failed(tmp_path):
    calls = {}
    aj = AutoJudge(targets_path=_targets(tmp_path), config=AutoJudgeConfig(r_near_m=0.05))
    bad_obs = {"joint_positions": Q0 + [0.0]}
    runner, writer = _runner(tmp_path, auto_judge=aj, calls=calls, obs=bad_obs)
    runner.run(plan=_plan(), models_by_id={"m1": MODEL})
    progress = json.loads(writer.progress_path().read_text())
    assert list(progress["failed"]) == [_plan()[0].eval_trial_id]
    assert "tcp_pose" in next(iter(progress["failed"].values()))
    assert len(calls["readiness_gate"]) == 1
    # The robot still gets the settle + servo release before the operator gate.
    assert calls["stop_after"] == ["model_stop_token"]


def test_manual_mode_unchanged(tmp_path):
    calls = {}
    runner, writer = _runner(tmp_path, auto_judge=None, calls=calls)
    runner.run(plan=_plan(), models_by_id={"m1": MODEL})
    assert len(calls["prompt_label"]) == 1 and len(calls["pre_rollout_gate"]) == 1
    (row,) = _summary(writer)
    assert row["stop_source"] == "timeout" and row["operator_label"] == "success"
    assert not (writer.trial_dir(_plan()[0]) / "chunk_scores.jsonl").exists()


def test_save_trial_overwrites_chunk_scores(tmp_path):
    writer = RolloutEvalWriter(tmp_path / "out", "run1")
    trial = _plan()[0]
    trial_dir = writer.trial_dir(trial)
    trial_dir.mkdir(parents=True)
    (trial_dir / "chunk_scores.jsonl").write_text("{}\n{}\n{}\n")
    writer.save_trial(
        trial=trial, model=MODEL, frames=[{}], action_trace=None,
        feedback=EvalFeedback(label="success"), stop_source="auto_success_stop", steps=1,
        started_at_ts=0.0, ended_at_ts=1.0, wall_seconds=1.0, final_tcp=list(TCP),
        runner_args={}, chunk_scores=[{"query_index": 0}],
    )
    assert (trial_dir / "chunk_scores.jsonl").read_text().splitlines() == ['{"query_index": 0}']
