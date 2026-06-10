"""Adapter hook interfaces for plugging PACER into any robot and any VLA.

PACER intentionally does not import robot SDKs or model frameworks.  Users keep
those dependencies in their own code and implement the small hooks below:

- ``RobotHooks`` converts robot-specific action chunks into base-frame TCP/EEF
  positions and declares target regions / workspace checks.
- ``VLAHooks`` queries a trained VLA candidate on a logged observation and
  inverse-normalizes its action chunk.
- ``TrainerHooks`` consumes PACER's exported ``rows.jsonl`` / ``manifest.json``
  views and launches whatever training recipe the user's stack requires.

The protocols are runtime-checkable documentation plus light validation. They
let examples and downstream projects type-check the integration boundary without
PACER depending on PyTorch, JAX, ROS, OpenPI, LeRobot, or any robot driver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

Vector3 = Sequence[float]
ActionChunk = Sequence[Any]
Observation = Mapping[str, Any]


@dataclass(frozen=True)
class TargetRegion:
    """Robot-agnostic target-region declaration.

    ``point`` is the target center in the robot base frame. ``radius`` is the
    terminal-position tolerance. ``orientation_tolerance_rad`` is optional
    because not every task/component has an orientation constraint.
    """

    target_id: str
    component: str
    point: tuple[float, float, float]
    radius: float
    orientation_tolerance_rad: float | None = None

    def as_geometry(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_point": list(self.point),
            "r_target": float(self.radius),
            "d_ref": float(2.0 * self.radius),
            **(
                {"orientation_tolerance_rad": float(self.orientation_tolerance_rad)}
                if self.orientation_tolerance_rad is not None
                else {}
            ),
        }


@runtime_checkable
class RobotHooks(Protocol):
    """Robot-side hooks PACER expects users to implement.

    Typical implementations wrap robot-specific FK, action integration, target
    capture, and safety checks. PACER only consumes the returned dictionaries /
    base-frame positions.
    """

    def target_region(self, component: str, collection_config: str) -> TargetRegion:
        """Return the fixed declared target region for a component/config."""
        ...

    def integrate_action_chunk(
        self,
        *,
        start_tcp: Vector3,
        action_chunk: ActionChunk,
        observation: Observation,
    ) -> list[tuple[float, float, float]]:
        """Convert a VLA action chunk into open-loop TCP/EEF positions.

        This is where robot-specific action conventions live: joint deltas,
        EEF deltas, absolute waypoints, gripper channels, action scaling, and
        inverse normalization. Return start + one TCP position per horizon in
        the robot base frame.
        """
        ...

    def safety_flags(self, trace_or_prediction: Mapping[str, Any]) -> dict[str, bool]:
        """Return PACER safety flags, e.g. unsafe/manual_safety_stop/quarantined."""
        ...


@runtime_checkable
class VLAHooks(Protocol):
    """Model-side hooks PACER expects users to implement."""

    def predict_action_chunk(
        self,
        *,
        model_id: str,
        observation: Observation,
        prompt: str | None = None,
        deterministic: bool = True,
    ) -> ActionChunk:
        """Query a trained VLA candidate once on a logged observation."""
        ...

    def stop_emitted(self, prediction: ActionChunk | Mapping[str, Any]) -> bool:
        """Return whether the candidate emitted the task/handoff stop token."""
        ...


@runtime_checkable
class TrainerHooks(Protocol):
    """Training-stack hook for consuming PACER exported views."""

    def train_weighted_view(
        self,
        *,
        view_dir: str,
        starting_checkpoint: str,
        output_model_id: str,
        recipe: Mapping[str, Any],
    ) -> str:
        """Train one candidate/baseline from a PACER view and return model id/path.

        The trainer must multiply its native per-horizon supervised action loss
        by each row's exported top-level ``loss_weight`` and use the manifest's
        loss-normalizer semantics. All candidates/baselines should share the
        same starting checkpoint, recipe, and update budget.
        """
        ...


def validate_target_region(region: TargetRegion) -> list[str]:
    """Small sanity check for target declarations before data collection."""

    errors: list[str] = []
    if not region.target_id:
        errors.append("target_id is required")
    if not region.component:
        errors.append("component is required")
    if len(region.point) != 3 or any(not isinstance(v, (int, float)) for v in region.point):
        errors.append("point must be three numeric base-frame coordinates")
    if region.radius <= 0:
        errors.append("radius must be positive")
    if region.orientation_tolerance_rad is not None and region.orientation_tolerance_rad <= 0:
        errors.append("orientation_tolerance_rad must be positive when supplied")
    return errors
