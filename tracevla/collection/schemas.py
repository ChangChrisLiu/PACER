"""Typed schemas for TRACE-VLA collection rollout collection sidecars."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BlockName = Literal["planner_skill", "planner_only", "corrector_only"]
StopSource = Literal[
    "model_stop_token",
    "manual_stop",
    "timeout",
    "unsafe_abort",
    "not_run",
]


def default_position_tolerance_m(component: str) -> float:
    """Paper protocol target tolerance: CPU 5 mm, other PACER components 10 mm."""
    normalized = component.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized == "cpu":
        return 0.005
    return 0.010


@dataclass(frozen=True)
class TargetPoint:
    tcp_pose: list[float]
    joint_positions: list[float]
    gripper_obs: int
    timestamp: float
    source_event: str = "L25/start_recording"
    tag: str = "target"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetRegion:
    config_id: str
    component: str
    points: list[TargetPoint]
    position_tolerance_m: float | None = None
    rotation_tolerance_rad: float = 0.35
    source: str = "operator_joystick_points"

    def __post_init__(self) -> None:
        if self.position_tolerance_m is None:
            object.__setattr__(self, "position_tolerance_m", default_position_tolerance_m(self.component))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["points"] = [p.to_dict() for p in self.points]
        return data


@dataclass(frozen=True)
class TrialSpec:
    config_id: str
    block: BlockName
    component: str
    component_index: int
    trial_index: int
    global_index: int

    @property
    def trial_id(self) -> str:
        return f"{self.config_id}__{self.block}__{self.component}__{self.trial_index:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "trial_id": self.trial_id,
        }


@dataclass
class ControlSegment:
    phase: str
    control_source: str
    start: int
    end: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerScore:
    pos_error_min_m: float | None
    rot_error_min_rad: float | None
    inside_region: bool | None
    closest_target_idx: int | None
    stop_source: str
    steps_to_stop: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Feedback:
    label: str
    note: str = ""
    operator: str = "operator"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VLAQueryTrace:
    query_index: int
    query_timestamp: float
    observation_frame_index: int | None
    prompt: str
    component: str
    model_type: str
    checkpoint_name: str
    raw_action_chunk: list[list[float]]
    chunk_length: int
    open_loop_horizon: int
    adapter_action_format: str
    stop_check_result: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VLAExecutedActionTrace:
    frame_index: int | None
    query_index: int
    action_index_within_chunk: int
    executed_action: list[float]
    post_safety_clamp_action: list[float] | None = None
    stop_source: str | None = None
    timestamp: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
