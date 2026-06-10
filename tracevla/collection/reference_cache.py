"""TRACE-VLA V0.1 (2026-05-20) reference caches.

Two caches live under `<output_root>/<config_id>/`:

1. `planner_reference_cache.json` — per-component planner target groups
   captured during planner_only and inherited by planner_skill. Implemented
   by ``PlannerReferenceCache``.
2. `corrector_start_pose_cache.json` — per-component failure_start pose
   captured during corrector_only and reused across later corrector trials.
   Implemented by ``CorrectorStartPoseCache``.

Both caches are JSON files persisted with the atomic-write pattern from
``trial_plan.atomic_write_json``. Schemas match the agreed-upon implementation
designs from internal TRACE-VLA protocol reviews.

Target-group schedule resolvers (``resolve_target_group``,
``resolve_planner_skill_target_group``) are pure functions so the planner
schedule is decidable in dry-run without touching disk.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import COMPONENT_SEQUENCE
from .schemas import TargetPoint, TargetRegion, default_position_tolerance_m

PLANNER_REFERENCE_CACHE_SCHEMA = "tracevla_planner_reference_cache.v0.1"
PLANNER_REFERENCE_BINDING_SCHEMA = "tracevla_planner_reference_binding.v0.1"
CORRECTOR_START_POSE_CACHE_SCHEMA = "tracevla_corrector_start_pose_cache.v0.1"

PLANNER_REFERENCE_CACHE_FILENAME = "planner_reference_cache.json"
CORRECTOR_START_POSE_CACHE_FILENAME = "corrector_start_pose_cache.json"

# Per-component planner_only target-group schedule.
# Fixed components capture once and reuse for every planner_only trial.
# RAM/connector capture at each group boundary.
PLANNER_TARGET_GROUP_SCHEDULES: dict[str, list[int]] = {
    "cpu_fan": [10],
    "graphic_card": [10],
    "cpu": [10],
    "ram": [5, 5],
    "connector": [2, 3, 2, 3],
}

FIXED_SCHEDULE_COMPONENTS: frozenset[str] = frozenset({"cpu_fan", "graphic_card", "cpu"})


def _schedule_for(component: str) -> list[int]:
    if component not in PLANNER_TARGET_GROUP_SCHEDULES:
        raise ValueError(
            f"No planner target-group schedule for component {component!r}; "
            f"expected one of {sorted(PLANNER_TARGET_GROUP_SCHEDULES)}"
        )
    return list(PLANNER_TARGET_GROUP_SCHEDULES[component])


def _group_id_for(component: str, group_index: int) -> str:
    """Deterministic group identifier used by both planner_only and planner_skill."""
    if component in FIXED_SCHEDULE_COMPONENTS:
        return f"{component}_fixed_{group_index:03d}"
    return f"{component}_target_{group_index:03d}"


def resolve_target_group(component: str, trial_index: int) -> str:
    """Map a planner_only trial index to its target_group_id."""
    schedule = _schedule_for(component)
    cursor = 0
    for group_index, length in enumerate(schedule):
        if trial_index < cursor + length:
            return _group_id_for(component, group_index)
        cursor += length
    raise ValueError(
        f"planner_only trial_index {trial_index} out of range for component "
        f"{component!r}; schedule {schedule} covers {cursor} trials."
    )


def planner_only_capture_index_for(component: str, trial_index: int) -> bool:
    """True if `trial_index` is the first trial in its group (operator must capture)."""
    schedule = _schedule_for(component)
    cursor = 0
    for length in schedule:
        if trial_index == cursor:
            return True
        if trial_index < cursor + length:
            return False
        cursor += length
    raise ValueError(
        f"planner_only trial_index {trial_index} out of range for component "
        f"{component!r}; schedule {schedule} covers {cursor} trials."
    )


def resolve_planner_skill_target_group(
    component: str,
    planner_skill_trial_index: int,
    *,
    policy: str = "representative_groups",
) -> str:
    """Map a planner_skill trial index to the inherited target_group_id.

    `representative_groups` (TRACE-VLA V0.1 default): map planner_skill trial
    index `i` to group `i mod num_groups`. With `--planner-skill-count 2`
    this gives connector groups 0 and 1 (representative, intentionally
    incomplete — caller should annotate the metadata).
    """
    if policy != "representative_groups":
        raise ValueError(f"Unsupported planner_skill target policy {policy!r}")
    schedule = _schedule_for(component)
    num_groups = len(schedule)
    group_index = planner_skill_trial_index % num_groups
    return _group_id_for(component, group_index)


def planner_skill_full_coverage_requires(component: str) -> int:
    """Number of distinct target groups for a component (informational)."""
    return len(_schedule_for(component))


def build_planner_skill_coverage_metadata(
    *,
    component: str,
    planner_skill_count: int,
    policy: str = "representative_groups",
) -> dict:
    """Build the per-trial coverage annotation for planner_skill.

    TRACE-VLA V0.1 Fix 1: planner_skill metadata must record the policy used
    and whether the chosen ``planner_skill_count`` actually covers every
    target group for the component. For connector with count=2 over 4 groups,
    ``full_target_group_coverage`` is False — downstream consumers can treat
    that as a flag for incomplete planner_skill evaluation.
    """
    num_groups = len(_schedule_for(component))
    if policy == "representative_groups":
        covered = min(int(planner_skill_count), num_groups)
    else:
        covered = int(planner_skill_count)
    return {
        "planner_skill_target_policy": policy,
        "num_target_groups": int(num_groups),
        "covered_target_group_count": int(covered),
        "full_target_group_coverage": covered >= num_groups,
    }


@dataclass(frozen=True)
class CachedPlannerReference:
    component: str
    group_id: str
    target_region: TargetRegion
    captured_in_trial_id: str
    updated_at: float


@dataclass(frozen=True)
class CachedCorrectorStartPose:
    component: str
    pose: TargetPoint
    revision: int
    captured_in_trial_id: str
    last_confirmed_trial_id: str | None
    updated_at: float


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def _schedule_kind(component: str) -> str:
    if component in FIXED_SCHEDULE_COMPONENTS:
        return "fixed"
    return "group_lengths"


class PlannerReferenceCache:
    """Read/write the per-config planner reference cache JSON.

    Layout (`planner_reference_cache.json`):

    .. code-block:: json

        {
          "schema_version": "tracevla_planner_reference_cache.v0.1",
          "config_id": "config_001",
          "updated_at": 1715000000.0,
          "components": {
            "cpu_fan": {
              "schedule_kind": "fixed",
              "group_lengths": [10],
              "groups": {
                "cpu_fan_fixed_000": {
                  "group_index": 0,
                  "planner_only_trial_indices": [0,1,2,3,4,5,6,7,8,9],
                  "captured_in_trial_id": "config_001__planner_only__cpu_fan__000",
                  "updated_at": 1715000000.0,
                  "target_region": { ... TargetRegion.to_dict() ... }
                }
              }
            }
          }
        }
    """

    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / PLANNER_REFERENCE_CACHE_FILENAME
        self._payload: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "schema_version": PLANNER_REFERENCE_CACHE_SCHEMA,
            "config_id": self.config_dir.name,
            "updated_at": time.time(),
            "components": {},
        }

    def _save(self) -> None:
        self._payload["updated_at"] = time.time()
        _atomic_write_json(self.path, self._payload)

    def get(self, component: str, group_id: str) -> CachedPlannerReference | None:
        comp = self._payload.get("components", {}).get(component)
        if not comp:
            return None
        grp = comp.get("groups", {}).get(group_id)
        if not grp:
            return None
        region_payload = grp.get("target_region")
        if region_payload is None:
            return None
        points = [
            TargetPoint(
                tcp_pose=list(p.get("tcp_pose", [])),
                joint_positions=list(p.get("joint_positions", [])),
                gripper_obs=int(p.get("gripper_obs", 0)),
                timestamp=float(p.get("timestamp", 0.0)),
                source_event=str(p.get("source_event", "L25/start_recording")),
                tag=str(p.get("tag", "target")),
            )
            for p in region_payload.get("points", [])
        ]
        region = TargetRegion(
            config_id=str(region_payload.get("config_id", self._payload.get("config_id", ""))),
            component=str(region_payload.get("component", component)),
            points=points,
            position_tolerance_m=float(
                region_payload.get("position_tolerance_m", default_position_tolerance_m(component))
            ),
            rotation_tolerance_rad=float(region_payload.get("rotation_tolerance_rad", 0.35)),
            source=str(region_payload.get("source", "operator_joystick_points")),
        )
        return CachedPlannerReference(
            component=component,
            group_id=group_id,
            target_region=region,
            captured_in_trial_id=str(grp.get("captured_in_trial_id", "")),
            updated_at=float(grp.get("updated_at", 0.0)),
        )

    def put(
        self,
        *,
        component: str,
        group_id: str,
        target_region: TargetRegion,
        captured_in_trial_id: str,
    ) -> CachedPlannerReference:
        components = self._payload.setdefault("components", {})
        comp = components.setdefault(
            component,
            {
                "schedule_kind": _schedule_kind(component),
                "group_lengths": _schedule_for(component),
                "groups": {},
            },
        )
        # Compute group_index from id suffix for back-compat audit.
        try:
            group_index = int(group_id.rsplit("_", 1)[-1])
        except ValueError:
            group_index = 0
        comp.setdefault("groups", {})[group_id] = {
            "group_index": group_index,
            "planner_only_trial_indices": _planner_only_trial_indices_for_group(
                component, group_index
            ),
            "captured_in_trial_id": captured_in_trial_id,
            "updated_at": time.time(),
            "target_region": target_region.to_dict(),
        }
        self._save()
        cached = self.get(component, group_id)
        assert cached is not None  # we just wrote it
        return cached

    def has(self, component: str, group_id: str) -> bool:
        return self.get(component, group_id) is not None


def _planner_only_trial_indices_for_group(component: str, group_index: int) -> list[int]:
    schedule = _schedule_for(component)
    cursor = 0
    for idx, length in enumerate(schedule):
        if idx == group_index:
            return list(range(cursor, cursor + length))
        cursor += length
    return []


def build_planner_reference_binding(
    cache: PlannerReferenceCache,
    *,
    component: str,
    group_id: str,
    reference_mode: str,
    target_region_source: str,
) -> dict:
    """Per-trial planner_reference dict for ``episode_meta.json``.

    ``reference_mode`` should be ``captured_now`` for the first planner_only
    trial that captured the reference, ``reused_from_planner_only_cache``
    for later planner_only trials in the same group, and
    ``inherited_from_planner_only`` for planner_skill trials.
    """
    cached = cache.get(component, group_id)
    captured_in_trial_id = cached.captured_in_trial_id if cached else ""
    return {
        "schema_version": PLANNER_REFERENCE_BINDING_SCHEMA,
        "target_group_id": group_id,
        "cache_path": str(cache.path.as_posix()),
        "reference_mode": reference_mode,
        "captured_in_trial_id": captured_in_trial_id,
        "target_region_source": target_region_source,
    }


class MissingPlannerReferenceError(RuntimeError):
    """Raised when planner_skill cannot find its inherited target_region.

    planner_skill must fail BEFORE robot motion when the planner_only cache
    has no entry for the component/group. The error message instructs the
    operator to run planner_only first.
    """


class CorrectorStartPoseCache:
    """Read/write the per-config corrector failure_start cache JSON."""

    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / CORRECTOR_START_POSE_CACHE_FILENAME
        self._payload: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "schema_version": CORRECTOR_START_POSE_CACHE_SCHEMA,
            "config_id": self.config_dir.name,
            "updated_at": time.time(),
            "components": {},
        }

    def _save(self) -> None:
        self._payload["updated_at"] = time.time()
        _atomic_write_json(self.path, self._payload)

    def get(self, component: str) -> CachedCorrectorStartPose | None:
        comp = self._payload.get("components", {}).get(component)
        if not comp:
            return None
        pose_payload = comp.get("pose")
        if pose_payload is None:
            return None
        pose = TargetPoint(
            tcp_pose=list(pose_payload.get("tcp_pose", [])),
            joint_positions=list(pose_payload.get("joint_positions", [])),
            gripper_obs=int(pose_payload.get("gripper_obs", 0)),
            timestamp=float(pose_payload.get("timestamp", 0.0)),
            source_event=str(pose_payload.get("source_event", "L25/start_recording")),
            tag=str(pose_payload.get("tag", "failure_start")),
        )
        return CachedCorrectorStartPose(
            component=component,
            pose=pose,
            revision=int(comp.get("revision", 1)),
            captured_in_trial_id=str(comp.get("captured_in_trial_id", "")),
            last_confirmed_trial_id=comp.get("last_confirmed_trial_id"),
            updated_at=float(comp.get("updated_at", 0.0)),
        )

    def put(
        self,
        *,
        component: str,
        pose: TargetPoint,
        captured_in_trial_id: str,
        is_adjustment: bool = False,
    ) -> CachedCorrectorStartPose:
        components = self._payload.setdefault("components", {})
        existing = components.get(component)
        revision = (int(existing.get("revision", 0)) + 1) if existing else 1
        components[component] = {
            "revision": revision,
            "captured_in_trial_id": captured_in_trial_id,
            "is_adjustment": bool(is_adjustment),
            "updated_at": time.time(),
            "pose": pose.to_dict(),
            "last_confirmed_trial_id": (
                existing.get("last_confirmed_trial_id") if existing else None
            ),
        }
        self._save()
        cached = self.get(component)
        assert cached is not None
        return cached

    def confirm_for_trial(self, component: str, trial_id: str) -> None:
        comp = self._payload.get("components", {}).get(component)
        if not comp:
            return
        comp["last_confirmed_trial_id"] = trial_id
        comp["updated_at"] = time.time()
        self._save()


def decide_corrector_start_action(
    cache: CorrectorStartPoseCache,
    component: str,
    *,
    input_fn: Callable[[str], str] | None = None,
) -> str:
    """Prompt the operator with the V0.1 corrector start cache options.

    Returns one of: ``capture`` (no cache exists, must capture), ``accept``
    (operator confirmed reuse), ``adjust``, ``recapture``, ``exclude``.
    """
    if cache.get(component) is None:
        return "capture"
    fn = input_fn if input_fn is not None else input
    print(
        f"\n[CORRECTOR START CACHE] cached failure_start exists for {component}.\n"
        "  Enter = accept and start rollout\n"
        "  j     = adjust with joystick and update cache\n"
        "  r     = recapture from current pose and update cache\n"
        "  x     = exclude/skip this trial"
    )
    try:
        raw = fn("Choice [enter=accept | j | r | x]: ").strip().lower()
    except EOFError:
        return "exclude"
    if raw in ("", "y", "yes", "accept", "a"):
        return "accept"
    if raw in ("j", "adjust"):
        return "adjust"
    if raw in ("r", "recapture"):
        return "recapture"
    if raw in ("x", "exclude", "skip"):
        return "exclude"
    # Unknown input -> conservative skip.
    print(f"[CORRECTOR START CACHE] unrecognized input {raw!r}; treating as exclude.")
    return "exclude"


__all__ = [
    "CORRECTOR_START_POSE_CACHE_FILENAME",
    "CORRECTOR_START_POSE_CACHE_SCHEMA",
    "CachedCorrectorStartPose",
    "CachedPlannerReference",
    "CorrectorStartPoseCache",
    "FIXED_SCHEDULE_COMPONENTS",
    "MissingPlannerReferenceError",
    "PLANNER_REFERENCE_BINDING_SCHEMA",
    "PLANNER_REFERENCE_CACHE_FILENAME",
    "PLANNER_REFERENCE_CACHE_SCHEMA",
    "PLANNER_TARGET_GROUP_SCHEDULES",
    "PlannerReferenceCache",
    "build_planner_reference_binding",
    "build_planner_skill_coverage_metadata",
    "decide_corrector_start_action",
    "planner_only_capture_index_for",
    "planner_skill_full_coverage_requires",
    "resolve_planner_skill_target_group",
    "resolve_target_group",
]
