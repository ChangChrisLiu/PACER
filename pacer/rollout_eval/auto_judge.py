"""PACER-scored auto trial judge for approach-only rollout evaluation.

For every predicted action chunk (before execution) the judge maps the chunk
to TCP poses, scores it with the PACER framework's open-loop submetrics and
blockers, records one ``chunk_scores`` row and decides whether to stop:

1. the policy's own stop token -> stop (``model_stop_token``);
2. the actual TCP is inside the target region and every predicted pose stays
   inside it -> stop (``auto_success_stop``);
3. a blocker on the predicted chunk -> stop before executing it
   (``auto_blocker_stop``);
4. otherwise continue.

The final label uses the closed PACER vocabulary. It is a geometric proxy for
the operator's handoff judgement, not a measurement of physical success.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from pacer.collection.config import COMPONENT_SEQUENCE
from pacer.collection.schemas import default_position_tolerance_m
from pacer_framework import blocker_flags_from_open_loop, geometric_row_score, open_loop_submetrics

from .auto_judge_targets import TARGETS_SCHEMA, JudgeTarget, TargetBook, load_targets
from .eval_feedback import (
    NEAR_LABEL,
    SUCCESS_LABEL,
    TOTALLY_OFF_LABEL,
    UNCERTAIN_LABEL,
    UNSAFE_ABORT_AUTO_LABEL,
    WRONG_OR_LOC_LABEL,
    EvalFeedback,
)
from .ur5e_kinematics import predict_tcp_poses, solve_tool_offset

AUTO_JUDGE_OPERATOR = "auto_judge"
AUTO_SUCCESS_STOP = "auto_success_stop"
AUTO_BLOCKER_STOP = "auto_blocker_stop"

PredictFn = Callable[[Sequence[Sequence[float]], Mapping[str, Any]], list[list[float]]]
ModelStopCheck = Callable[[list, dict], bool]

_TOTALLY_OFF_REASONS = ("wrong_target", "non_target_exclusion", "workspace", "action_bound")
_WRONG_REASONS = ("below_floor", "invalid_orientation")


@dataclass(frozen=True)
class AutoJudgeConfig:
    r_near_m: float
    rotation_tolerance_rad: float = 0.35
    floor_margin_m: float = 0.005
    borderline_m: float = 0.001

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
        max_d_ref = max(2.0 * default_position_tolerance_m(c) for c in COMPONENT_SEQUENCE)
        if self.r_near_m <= max_d_ref:
            raise ValueError(f"r_near_m must exceed the largest d_ref ({max_d_ref} m), got {self.r_near_m}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite(values: Any, length: int, key: str) -> list[float]:
    try:
        arr = [float(v) for v in list(values)[:length]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"observation {key!r} must be numeric") from exc
    if len(arr) < length or not all(math.isfinite(v) for v in arr):
        raise ValueError(f"observation {key!r} needs {length} finite values, got {values!r}")
    return arr


def obs_joint_and_tcp(obs: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Six joint angles and the measured TCP pose from a robot observation."""
    for key in ("joint_positions", "tcp_pose"):
        if obs.get(key) is None:
            raise ValueError(f"observation is missing {key!r}; the auto judge needs joint_positions and tcp_pose")
    return _finite(obs["joint_positions"], 6, "joint_positions"), _finite(obs["tcp_pose"], 6, "tcp_pose")


def fk_predict(chunk: Sequence[Sequence[float]], obs: Mapping[str, Any]) -> list[list[float]]:
    """Predicted TCP poses for an absolute-joint chunk, tool offset from ``obs``."""
    q0, tcp0 = obs_joint_and_tcp(obs)
    return predict_tcp_poses(chunk, solve_tool_offset(q0, tcp0))


def _rot_err(a: Sequence[float], b: Sequence[float]) -> float:
    ra = Rotation.from_rotvec(np.asarray(a, dtype=float)[3:6])
    rb = Rotation.from_rotvec(np.asarray(b, dtype=float)[3:6])
    return float((ra.inv() * rb).magnitude())


def _pos_err(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)[:3] - np.asarray(b, dtype=float)[:3]))


def _closest(pose: Sequence[float], points: Sequence[Sequence[float]]) -> tuple[Sequence[float], float, float]:
    """Closest target point by position error, ties broken by rotation error."""
    best = min(points, key=lambda p: (_pos_err(pose, p), _rot_err(pose, p)))
    return best, _pos_err(pose, best), _rot_err(pose, best)


def classify_final_pose(
    *, pos_err_m: float, rot_err_rad: float, z_margin_m: float, r_target_m: float, config: AutoJudgeConfig,
) -> tuple[str, str, bool]:
    """Closed-vocabulary label for a final pose, plus a needs-review flag."""
    d_ref = 2.0 * r_target_m
    tol = config.rotation_tolerance_rad
    if z_margin_m < 0:
        label = WRONG_OR_LOC_LABEL
    elif pos_err_m <= r_target_m and rot_err_rad <= tol:
        label = SUCCESS_LABEL
    elif pos_err_m <= d_ref and rot_err_rad <= tol:
        label = NEAR_LABEL
    elif pos_err_m <= d_ref or pos_err_m <= config.r_near_m:
        label = WRONG_OR_LOC_LABEL
    else:
        label = TOTALLY_OFF_LABEL
    band = config.borderline_m
    names = [name for name, t in (("r_target", r_target_m), ("d_ref", d_ref), ("r_near", config.r_near_m))
             if abs(pos_err_m - t) <= band]
    if abs(z_margin_m) <= band:
        names.append("floor")
    note = f"needs_review:{','.join(names)}" if names else ""
    return label, note, bool(names)


class TrialJudge:
    """Per-trial judge; ``check_chunk`` is a ``TracedChunkRunner`` stop_check."""

    def __init__(
        self,
        *,
        target: JudgeTarget,
        non_targets: Sequence[JudgeTarget],
        workspace: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
        config: AutoJudgeConfig,
        model_stop_check: ModelStopCheck | None = None,
        predict_fn: PredictFn = fk_predict,
    ) -> None:
        self.target = target
        self.non_targets = list(non_targets)
        self.workspace = workspace
        self.config = config
        self.model_stop_check = model_stop_check
        self.predict_fn = predict_fn
        self.r_target = default_position_tolerance_m(target.component)
        self.d_ref = 2.0 * self.r_target
        self.chunk_scores: list[dict[str, Any]] = []
        self._non_target_regions = [
            {"point": list(p[:3]), "radius": default_position_tolerance_m(o.component)}
            for o in self.non_targets for p in o.points
        ]

    # -- helpers -----------------------------------------------------------
    def _inside(self, pose: Sequence[float], point: Sequence[float]) -> bool:
        return (
            _pos_err(pose, point) <= self.r_target
            and _rot_err(pose, point) <= self.config.rotation_tolerance_rad
            and float(pose[2]) >= float(point[2]) - self.config.floor_margin_m
        )

    def _outside_workspace(self, poses: Sequence[Sequence[float]]) -> bool:
        if self.workspace is None:
            return False
        lo, hi = self.workspace
        return any(float(p[i]) < lo[i] or float(p[i]) > hi[i] for p in poses for i in range(3))

    @staticmethod
    def _action_bounds_ok(chunk: Sequence[Sequence[float]]) -> bool:
        try:
            arr = [[float(v) for v in row] for row in chunk]
        except (TypeError, ValueError):
            return False
        if not arr or any(len(row) < 6 for row in arr):
            return False
        return all(math.isfinite(v) for row in arr for v in row) and all(
            abs(v) <= 2.0 * math.pi for row in arr for v in row[:6]
        )

    def _row(self, *, decision: str, reasons: list[str], actual: list[float], actual_point: Sequence[float],
             predicted_end: list[float] | None, submetrics: dict[str, float], v_hat: float | None,
             blocker: dict[str, bool]) -> dict[str, Any]:
        previous = self.chunk_scores[-1]["predicted_tcp_end"] if self.chunk_scores else None
        fk_residual = _pos_err(previous, actual) if previous is not None else None
        v_row = None if v_hat is None else (0.0 if reasons else v_hat)
        row = {
            "query_index": len(self.chunk_scores),
            "decision": decision,
            "decision_reasons": list(reasons),
            "actual_tcp": list(actual),
            "actual_pos_err_m": _pos_err(actual, actual_point),
            "actual_rot_err_rad": _rot_err(actual, actual_point),
            "predicted_tcp_end": predicted_end,
            "fk_residual_m": fk_residual,
            "submetrics": submetrics,
            "v_hat": v_hat,
            "blocker": blocker,
            "v_row": v_row,
        }
        self.chunk_scores.append(row)
        return row

    # -- stop_check ----------------------------------------------------------
    def check_chunk(self, chunk: Sequence[Sequence[float]], obs: Mapping[str, Any]) -> bool | str:
        _q0, actual = obs_joint_and_tcp(obs)
        actual_point, _, _ = _closest(actual, self.target.points)
        common = {"actual": actual, "actual_point": actual_point}

        if self.model_stop_check is not None and self.model_stop_check(list(chunk), dict(obs)):
            self._row(decision="model_stop_token", reasons=[], predicted_end=None, submetrics={},
                      v_hat=None, blocker={}, **common)
            return True

        if not self._action_bounds_ok(chunk):
            self._row(decision=AUTO_BLOCKER_STOP, reasons=["action_bound"], predicted_end=None, submetrics={},
                      v_hat=0.0, blocker={"action_bound": True}, **common)
            return AUTO_BLOCKER_STOP

        predicted = [list(map(float, p[:6])) for p in self.predict_fn(chunk, obs)]
        end = predicted[-1]
        end_point, end_pos, end_rot = _closest(end, self.target.points)
        tol = self.config.rotation_tolerance_rad
        orientation_ok = (end_rot <= tol) if end_pos <= self.d_ref else None
        xyz = [p[:3] for p in predicted]
        flags = blocker_flags_from_open_loop(
            xyz,
            target_point=list(end_point[:3]),
            component=self.target.component,
            r_target=self.r_target,
            non_target_regions=self._non_target_regions,
            workspace_bounds=None,
            orientation_ok=orientation_ok,
        )
        blocker = {
            "wrong_target": flags["wrong_target"],
            "non_target_exclusion": flags["non_target_exclusion"],
            "invalid_orientation": flags["invalid_orientation"],
            "workspace": self._outside_workspace(predicted),
            "action_bound": False,
            "below_floor": any(p[2] < float(end_point[2]) - self.config.floor_margin_m for p in predicted),
        }
        reasons = [name for name, hit in blocker.items() if hit]
        submetrics = open_loop_submetrics(
            [actual[:3]] + xyz,
            target_point=list(end_point[:3]),
            component=self.target.component,
            r_target=self.r_target,
            terminal_region_member=self._inside(end, end_point),
        )
        v_hat = float(geometric_row_score(submetrics))

        if self._inside(actual, actual_point) and all(self._inside(p, actual_point) for p in predicted):
            decision, result = AUTO_SUCCESS_STOP, AUTO_SUCCESS_STOP
        elif reasons:
            decision, result = AUTO_BLOCKER_STOP, AUTO_BLOCKER_STOP
        else:
            decision, result = "continue", False
        self._row(decision=decision, reasons=reasons if decision != AUTO_SUCCESS_STOP else [],
                  predicted_end=end, submetrics=submetrics, v_hat=v_hat, blocker=blocker, **common)
        return result

    # -- final label -----------------------------------------------------------
    def final_feedback(self, *, stop_source: str, final_tcp: Sequence[float] | None) -> EvalFeedback:
        def feedback(label: str, *tokens: str) -> EvalFeedback:
            note = ";".join([stop_source, *[t for t in tokens if t]])
            return EvalFeedback(label=label, note=note, operator=AUTO_JUDGE_OPERATOR, auto_assigned=True)

        if stop_source == "unsafe_abort":
            return feedback(UNSAFE_ABORT_AUTO_LABEL)
        if stop_source == AUTO_BLOCKER_STOP and self.chunk_scores:
            reasons = list(self.chunk_scores[-1]["decision_reasons"])
            review = ["needs_review:action_bound"] if "action_bound" in reasons else []
            if any(r in _TOTALLY_OFF_REASONS for r in reasons):
                return feedback(TOTALLY_OFF_LABEL, *reasons, *review)
            if any(r in _WRONG_REASONS for r in reasons):
                return feedback(WRONG_OR_LOC_LABEL, *reasons, *review)
        if final_tcp is None:
            return feedback(UNCERTAIN_LABEL, "final_tcp_unavailable")
        pose = [float(v) for v in list(final_tcp)[:6]]
        point, pos_err, rot_err = _closest(pose, self.target.points)
        z_margin = pose[2] - (float(point[2]) - self.config.floor_margin_m)
        label, note, _review = classify_final_pose(
            pos_err_m=pos_err, rot_err_rad=rot_err, z_margin_m=z_margin,
            r_target_m=self.r_target, config=self.config,
        )
        return feedback(label, f"pos_err_m={pos_err:.4f}", f"rot_err_rad={rot_err:.3f}", note)


class AutoJudge:
    """Holds the target book and builds one ``TrialJudge`` per trial."""

    def __init__(self, *, targets_path: Path, config: AutoJudgeConfig) -> None:
        self.targets_path = Path(targets_path)
        self.config = config
        self.book: TargetBook = load_targets(self.targets_path)

    def reload(self) -> None:
        self.book = load_targets(self.targets_path)

    def has_target(self, config_id: str, component: str) -> bool:
        return self.book.get(config_id, component) is not None

    def trial_judge(self, config_id: str, component: str, *, model_stop_check: ModelStopCheck | None = None) -> TrialJudge:
        target = self.book.get(config_id, component)
        if target is None:
            raise KeyError(f"no auto-judge target for {config_id}/{component} in {self.targets_path}")
        workspace = None
        if self.book.workspace_min is not None and self.book.workspace_max is not None:
            workspace = (self.book.workspace_min, self.book.workspace_max)
        return TrialJudge(
            target=target,
            non_targets=self.book.others_in_config(config_id, component),
            workspace=workspace,
            config=self.config,
            model_stop_check=model_stop_check,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "targets_path": str(self.targets_path),
            "targets_schema": TARGETS_SCHEMA,
        }
