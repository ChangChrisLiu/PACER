"""10 Hz inference runner with VLA action tracing for Stage-B-pre.

This module is written to be testable without robot hardware. Hardware callers
pass real observation/model/action callbacks; tests pass fakes.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schemas import VLAExecutedActionTrace, VLAQueryTrace

ObsFn = Callable[[], dict[str, Any]]
InferFn = Callable[[dict[str, Any], str], list[Any]]
ApplyActionFn = Callable[[Any, dict[str, Any]], Any]
FrameBuilderFn = Callable[[dict[str, Any]], dict[str, Any]]
StopCheckFn = Callable[[list[Any], dict[str, Any]], bool]
ManualStopFn = Callable[[], str | None]
QueryDiagFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class RolloutResult:
    frames: list[dict[str, Any]]
    action_trace: list[dict[str, Any]]
    stop_source: str
    steps: int
    final_obs: dict[str, Any] | None = None


@dataclass(frozen=True)
class GripperCapConfig:
    """Execution-only OpenPI planner gripper cap.

    The cap is intentionally applied after raw chunk stop detection and before
    `apply_action`, so raw stop-token semantics remain unchanged while the
    robot receives a bounded non-CPU planner gripper target.
    """

    enabled: bool = False
    value_norm: float = 180.0 / 255.0
    # Values at/above this are treated as stop-token semantics, not as physical
    # gripper close commands. Stop detection runs on the raw chunk before this
    # helper; this execution boundary is a second guard so permissive callers
    # never send 240-255/255 stop-token values to the robot gripper.
    stop_token_lower_norm: float = 240.0 / 255.0
    initial_held_gripper_norm: float = 0.0
    components: tuple[str, ...] = ("cpu_fan", "ram", "connector", "graphic_card")
    planner_phases: tuple[str, ...] = ("teleop",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "value_norm": float(self.value_norm),
            "stop_token_lower_norm": float(self.stop_token_lower_norm),
            "initial_held_gripper_norm": float(self.initial_held_gripper_norm),
            "components": list(self.components),
            "planner_phases": list(self.planner_phases),
        }


@dataclass
class TracedChunkRunner:
    fps: int = 10
    max_steps: int = 600
    open_loop_horizon: int = 10
    model_type: str = "openpi"
    checkpoint_name: str = ""
    adapter_action_format: str = "unknown"
    gripper_cap: GripperCapConfig = field(default_factory=GripperCapConfig)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time
    _query_index: int = field(default=0, init=False)

    def run(
        self,
        *,
        component: str,
        prompt: str,
        phase: str,
        get_obs: ObsFn,
        infer: InferFn,
        apply_action: ApplyActionFn,
        build_frame: FrameBuilderFn,
        stop_check: StopCheckFn | None = None,
        manual_stop: ManualStopFn | None = None,
        query_diag_fn: QueryDiagFn | None = None,
    ) -> RolloutResult:
        dt = 1.0 / float(self.fps)
        frames: list[dict[str, Any]] = []
        action_trace: list[dict[str, Any]] = []
        action_queue: list[Any] = []
        queue_query_index = -1
        action_idx = 0
        stop_source = "timeout"
        final_obs = None
        last_valid_gripper = float(self.gripper_cap.initial_held_gripper_norm)

        for step in range(self.max_steps):
            t0 = self.clock()
            manual = manual_stop() if manual_stop else None
            if manual:
                stop_source = str(manual)
                break

            obs = get_obs()
            final_obs = obs
            frame = build_frame(obs)
            frame["phase"] = phase
            frame["control_source"] = f"model_{phase}"
            frame["stage_b_step"] = step
            frames.append(frame)
            frame_idx = len(frames) - 1

            if not action_queue:
                # V0.7: time the infer() call so the per-trial startup
                # diagnostics can attribute "robot didn't move for N seconds"
                # to either model warmup vs. script gating.
                _infer_t0 = self.clock()
                chunk = list(infer(obs, prompt))
                _infer_latency_s = max(0.0, self.clock() - _infer_t0)
                n = min(self.open_loop_horizon, len(chunk))
                chunk = chunk[:n]
                stop_now = bool(stop_check(chunk, obs)) if stop_check else False
                query_extra = _safe_query_diag(query_diag_fn, obs)
                query_extra["query_inference_latency_s"] = float(_infer_latency_s)
                query = VLAQueryTrace(
                    query_index=self._query_index,
                    query_timestamp=self.clock(),
                    observation_frame_index=frame_idx,
                    prompt=prompt,
                    component=component,
                    model_type=self.model_type,
                    checkpoint_name=self.checkpoint_name,
                    raw_action_chunk=[_action_to_list(a) for a in chunk],
                    chunk_length=len(chunk),
                    open_loop_horizon=self.open_loop_horizon,
                    adapter_action_format=self.adapter_action_format,
                    stop_check_result=stop_now,
                    extra=query_extra,
                )
                action_trace.append({"type": "policy_query", **query.to_dict()})
                if stop_now:
                    stop_source = "model_stop_token"
                    break
                action_queue.extend(chunk)
                queue_query_index = self._query_index
                action_idx = 0
                self._query_index += 1

            action = action_queue.pop(0)
            action_to_execute, gripper_cap_extra, last_valid_gripper = _apply_gripper_cap(
                action,
                component=component,
                phase=phase,
                model_type=self.model_type,
                config=self.gripper_cap,
                last_valid_gripper=last_valid_gripper,
            )
            applied = apply_action(action_to_execute, obs)
            clamped_list, extra = _unpack_applied(applied)
            extra = {**gripper_cap_extra, **extra}
            executed = VLAExecutedActionTrace(
                frame_index=frame_idx,
                query_index=queue_query_index,
                action_index_within_chunk=action_idx,
                executed_action=_action_to_list(action_to_execute),
                post_safety_clamp_action=clamped_list,
                timestamp=self.clock(),
                extra=extra,
            )
            action_trace.append({"type": "executed_action", **executed.to_dict()})
            action_idx += 1

            elapsed = self.clock() - t0
            if elapsed < dt:
                self.sleep(dt - elapsed)
        else:
            stop_source = "timeout"

        action_trace.append(
            {
                "type": "rollout_stop",
                "stop_source": stop_source,
                "steps": len(frames),
                "timestamp": self.clock(),
            }
        )
        return RolloutResult(
            frames=frames,
            action_trace=action_trace,
            stop_source=stop_source,
            steps=len(frames),
            final_obs=final_obs,
        )


def _action_to_list(action: Any) -> list[float]:
    arr = np.asarray(action, dtype=float).reshape(-1)
    return [float(x) for x in arr]


def _apply_gripper_cap(
    action: Any,
    *,
    component: str,
    phase: str,
    model_type: str,
    config: GripperCapConfig,
    last_valid_gripper: float,
) -> tuple[Any, dict[str, Any], float]:
    """Return an execution action plus trace metadata for gripper policy.

    This helper never mutates the queued raw action. Raw chunks remain in the
    `policy_query.raw_action_chunk` row; only `executed_action` and the target
    passed to `apply_action` may differ when the cap/intercept is applicable.
    """
    raw_arr = np.asarray(action, dtype=float).reshape(-1)
    raw_gripper = float(raw_arr[6]) if raw_arr.size >= 7 else None
    cap_components = tuple(config.components)
    last_valid = float(last_valid_gripper)
    if not config.enabled:
        return action, {}, last_valid
    base_extra: dict[str, Any] = {
        "gripper_cap_enabled": True,
        "gripper_cap_applied": False,
        "gripper_cap_value_norm": float(config.value_norm),
        "gripper_stop_token_lower_norm": float(config.stop_token_lower_norm),
        "gripper_stop_token_intercepted": False,
        "gripper_stop_token_raw_value": None,
        "gripper_stop_token_held_value": last_valid,
        "gripper_cap_components": list(cap_components),
        "gripper_cap_raw_value": raw_gripper,
        "gripper_cap_executed_value": raw_gripper,
        "gripper_cap_phase": phase,
        "gripper_cap_component": component,
        "gripper_cap_reason": "not_applicable",
    }
    if model_type != "openpi":
        base_extra["gripper_cap_reason"] = f"model_type_exempt:{model_type}"
        return action, base_extra, last_valid
    if phase not in config.planner_phases:
        base_extra["gripper_cap_reason"] = f"phase_exempt:{phase}"
        return action, base_extra, last_valid
    if raw_arr.size < 7 or raw_gripper is None:
        base_extra["gripper_cap_reason"] = "invalid_action_shape"
        return action, base_extra, last_valid
    if raw_gripper >= float(config.stop_token_lower_norm):
        held = raw_arr.copy()
        held[6] = last_valid
        base_extra.update(
            {
                "gripper_stop_token_intercepted": True,
                "gripper_stop_token_raw_value": raw_gripper,
                "gripper_stop_token_held_value": last_valid,
                "gripper_cap_executed_value": last_valid,
                "gripper_cap_reason": "stop_token_region_held",
            }
        )
        return held, base_extra, last_valid
    if component == "cpu":
        # CPU is exempt from the 180/255 approach cap, but not from the
        # stop-token-band interception above.
        base_extra["gripper_cap_reason"] = "component_exempt:cpu"
        return action, base_extra, raw_gripper
    if component not in cap_components:
        base_extra["gripper_cap_reason"] = f"component_exempt:{component}"
        return action, base_extra, raw_gripper
    if raw_gripper <= float(config.value_norm):
        base_extra["gripper_cap_reason"] = "raw_below_cap"
        return action, base_extra, raw_gripper

    capped = raw_arr.copy()
    capped[6] = float(config.value_norm)
    base_extra.update(
        {
            "gripper_cap_applied": True,
            "gripper_cap_executed_value": float(capped[6]),
            "gripper_cap_reason": "planner_non_cpu_approach_band_cap_180_to_stop_token",
        }
    )
    return capped, base_extra, float(capped[6])


_ALLOWED_APPLIED_DICT_KEYS = frozenset({"post_safety_clamp_action", "extra"})


def _unpack_applied(applied: Any) -> tuple[list[float] | None, dict[str, Any]]:
    """Normalize the return value of apply_action.

    Strict contract — only three shapes are accepted:
      1. None                                            -> (None, {})
      2. flat action (np.ndarray / list)                 -> (list, {})
      3. dict {"post_safety_clamp_action": list|None,
               "extra": dict (optional)}                  -> (list|None, extra)

    Any deviation from #3 (unknown keys, missing required key, non-dict extra)
    is treated as a contract violation: the row's post_safety_clamp_action is
    forced to None and extra carries an apply_action_contract_error string so
    the failure is visible in the trace instead of silently corrupting state.
    """
    if applied is None:
        return None, {}
    if isinstance(applied, dict):
        errors: list[str] = []
        unknown = set(applied.keys()) - _ALLOWED_APPLIED_DICT_KEYS
        if unknown:
            errors.append(f"unknown_keys={sorted(unknown)}")
        if "post_safety_clamp_action" not in applied:
            errors.append("missing_key=post_safety_clamp_action")
        extra_field = applied.get("extra", {})
        if "extra" in applied and not isinstance(extra_field, dict):
            errors.append(f"non_dict_extra={type(extra_field).__name__}")
        if errors:
            return None, {"apply_action_contract_error": "; ".join(errors)}
        clamp = applied.get("post_safety_clamp_action")
        clamp_list = _action_to_list(clamp) if clamp is not None else None
        extra = extra_field if isinstance(extra_field, dict) else {}
        return clamp_list, dict(extra)
    return _action_to_list(applied), {}


def _safe_query_diag(fn: QueryDiagFn | None, obs: dict[str, Any]) -> dict[str, Any]:
    if fn is None:
        return {}
    try:
        out = fn(obs)
    except Exception as exc:
        return {"query_diag_error": repr(exc)}
    if not isinstance(out, dict):
        return {"query_diag_error": f"non_dict_return: {type(out).__name__}"}
    return out
