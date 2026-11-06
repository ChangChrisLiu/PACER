from __future__ import annotations

from pacer.collection.inference_runner import TracedChunkRunner

CHUNK = [[0.0] * 7 for _ in range(10)]


def _run(stop_result_for_query):
    calls = {"n": 0}

    def stop_check(chunk, obs):
        calls["n"] += 1
        return stop_result_for_query(calls["n"])

    runner = TracedChunkRunner(max_steps=25, sleep=lambda s: None)
    return runner.run(
        component="ram",
        prompt="p",
        phase="teleop",
        get_obs=lambda: {"tcp_pose": [0.0] * 6},
        infer=lambda obs, prompt: CHUNK,
        apply_action=lambda action, obs: None,
        build_frame=lambda obs: {},
        stop_check=stop_check,
    )


def _queries(result):
    return [row for row in result.action_trace if row["type"] == "policy_query"]


def test_bool_true_gives_model_stop_token():
    result = _run(lambda n: n == 2)
    assert result.stop_source == "model_stop_token"
    assert _queries(result)[-1]["stop_check_result"] is True
    assert _queries(result)[-1]["extra"]["stop_check_source"] == "model_stop_token"


def test_string_result_becomes_stop_source():
    result = _run(lambda n: "auto_success_stop" if n == 2 else False)
    assert result.stop_source == "auto_success_stop"
    assert result.action_trace[-1]["type"] == "rollout_stop"
    assert result.action_trace[-1]["stop_source"] == "auto_success_stop"
    stopping = _queries(result)[-1]
    assert stopping["stop_check_result"] is True
    assert stopping["extra"]["stop_check_source"] == "auto_success_stop"
    after = result.action_trace[result.action_trace.index(stopping) + 1:]
    assert [row["type"] for row in after] == ["rollout_stop"]


def test_false_runs_to_timeout():
    result = _run(lambda n: False)
    assert result.stop_source == "timeout"
    assert result.steps == 25
    assert all(q["extra"]["stop_check_source"] is None for q in _queries(result))
