"""Target poses for the auto trial judge, one entry per board position.

A target is the TCP pose the approach should reach, usually the endpoint of a
demo recorded at that position. Entries are keyed by ``config_id/component``.
An entry may declare ``non_targets``: components of the same configuration
whose target points are obstacles while judging this one. Nothing is an
obstacle unless declared, because a component removed earlier (the CPU fan
before the CPU) must not block the approach. The demo that defines a target
must not be used to train any policy that is evaluated against it.

File format::

    {"schema": "pacer_auto_judge_targets.v1",
     "targets": {"config_001/ram": {"points": [[x, y, z, rx, ry, rz]],
                                     "source": "demo_endpoint",
                                     "non_targets": ["connector"]}},   # optional
     "workspace": {"min": [x, y, z], "max": [x, y, z]}}   # optional
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pacer.collection.config import COMPONENT_SEQUENCE
from pacer.collection.trial_plan import normalize_config_id

TARGETS_SCHEMA = "pacer_auto_judge_targets.v1"


def _pose6(values: Sequence[Any], what: str) -> tuple[float, ...]:
    try:
        pose = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be six numbers") from exc
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise ValueError(f"{what} must be six finite numbers [x, y, z, rx, ry, rz], got {values!r}")
    return pose


def _xyz(values: Sequence[Any], what: str) -> tuple[float, float, float]:
    try:
        xyz = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be three numbers") from exc
    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
        raise ValueError(f"{what} must be three finite numbers, got {values!r}")
    return xyz  # type: ignore[return-value]


@dataclass(frozen=True)
class JudgeTarget:
    config_id: str
    component: str
    points: tuple[tuple[float, ...], ...]
    source: str
    non_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_id", normalize_config_id(self.config_id))
        if self.component not in COMPONENT_SEQUENCE:
            raise ValueError(f"unknown component {self.component!r}; expected one of {COMPONENT_SEQUENCE}")
        points = tuple(_pose6(p, f"target point for {self.key}") for p in self.points)
        if not points:
            raise ValueError(f"target {self.key} needs at least one point")
        object.__setattr__(self, "points", points)
        non_targets: list[str] = []
        for name in self.non_targets:
            if name not in COMPONENT_SEQUENCE or name == self.component:
                raise ValueError(f"non_targets of {self.key} must name other components, got {name!r}")
            if name not in non_targets:
                non_targets.append(name)
        object.__setattr__(self, "non_targets", tuple(non_targets))

    @property
    def key(self) -> str:
        return f"{self.config_id}/{self.component}"


@dataclass(frozen=True)
class TargetBook:
    targets: dict[tuple[str, str], JudgeTarget] = field(default_factory=dict)
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None

    def get(self, config_id: str, component: str) -> JudgeTarget | None:
        return self.targets.get((normalize_config_id(config_id), component))

    def non_targets_for(self, config_id: str, component: str) -> list[JudgeTarget]:
        """Targets declared as obstacles by this cell; a declared one must exist."""
        target = self.get(config_id, component)
        if target is None:
            return []
        out: list[JudgeTarget] = []
        for name in target.non_targets:
            other = self.get(config_id, name)
            if other is None:
                raise ValueError(f"{target.key} declares non-target {name!r}, which has no target in this file")
            out.append(other)
        return out


def _read_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": TARGETS_SCHEMA, "targets": {}}
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping) or raw.get("schema") != TARGETS_SCHEMA:
        raise ValueError(f"{path} is not a {TARGETS_SCHEMA} file")
    if not isinstance(raw.get("targets", {}), Mapping):
        raise ValueError(f"{path}: 'targets' must be an object")
    return dict(raw)


def load_targets(path: Path) -> TargetBook:
    raw = _read_raw(Path(path))
    targets: dict[tuple[str, str], JudgeTarget] = {}
    for key, entry in raw.get("targets", {}).items():
        config_id, sep, component = str(key).partition("/")
        if not sep or not isinstance(entry, Mapping):
            raise ValueError(f"malformed target entry {key!r}")
        target = JudgeTarget(
            config_id, component, tuple(entry.get("points") or ()), str(entry.get("source", "")),
            non_targets=tuple(entry.get("non_targets") or ()),
        )
        targets[(target.config_id, target.component)] = target
    workspace = raw.get("workspace")
    if workspace is None:
        return TargetBook(targets=targets)
    if not isinstance(workspace, Mapping):
        raise ValueError("'workspace' must be an object with 'min' and 'max'")
    return TargetBook(
        targets=targets,
        workspace_min=_xyz(workspace.get("min") or (), "workspace.min"),
        workspace_max=_xyz(workspace.get("max") or (), "workspace.max"),
    )


def upsert_target(path: Path, target: JudgeTarget) -> None:
    path = Path(path)
    raw = _read_raw(path)
    targets = dict(raw.get("targets", {}))
    entry: dict[str, Any] = {"points": [list(p) for p in target.points], "source": target.source}
    if target.non_targets:
        entry["non_targets"] = list(target.non_targets)
    targets[target.key] = entry
    raw["targets"] = targets
    raw["schema"] = TARGETS_SCHEMA
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(raw, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def target_from_demo_episode(
    episode_dir: Path, *, config_id: str, component: str, frame_index: int | None = None,
) -> JudgeTarget:
    frames = sorted(Path(episode_dir).glob("frame_*.pkl"))
    if not frames:
        raise ValueError(f"no frame_*.pkl files in {episode_dir}")
    path = frames[-1] if frame_index is None else Path(episode_dir) / f"frame_{int(frame_index):04d}.pkl"
    with path.open("rb") as f:
        frame = pickle.load(f)
    if not isinstance(frame, Mapping) or frame.get("tcp_pose") is None:
        raise ValueError(f"{path} has no tcp_pose")
    # No slicing: a 7-value pose (e.g. a quaternion) must fail, not be misread.
    return JudgeTarget(config_id, component, (tuple(frame["tcp_pose"]),), "demo_endpoint")


def target_from_region_json(path: Path, *, config_id: str, component: str) -> JudgeTarget:
    raw = json.loads(Path(path).read_text())
    points = tuple(tuple(p["tcp_pose"]) for p in raw.get("points") or ())
    return JudgeTarget(config_id, component, points, "target_region")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Record targets for the PACER auto trial judge")
    sub = ap.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="Add or replace the target of one config/component")
    add.add_argument("--targets", type=Path, required=True)
    add.add_argument("--config-id", required=True)
    add.add_argument("--component", required=True, choices=COMPONENT_SEQUENCE)
    src = add.add_mutually_exclusive_group(required=True)
    src.add_argument("--tcp-pose", type=float, nargs=6, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    src.add_argument("--demo-episode", type=Path, help="Episode dir with frame_*.pkl; uses the last frame")
    src.add_argument("--target-region-json", type=Path, help="PACER target_region.json")
    add.add_argument("--frame-index", type=int, default=None, help="With --demo-episode: frame to use")
    add.add_argument("--non-target", action="append", default=[], choices=COMPONENT_SEQUENCE,
                     help="Component of the same config whose target is an obstacle (repeatable). "
                          "Do not list components already removed at this position.")
    show = sub.add_parser("show", help="Print the targets file")
    show.add_argument("--targets", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.command == "show":
        book = load_targets(args.targets)
        for target in book.targets.values():
            extra = f", non_targets={list(target.non_targets)}" if target.non_targets else ""
            print(f"{target.key}: {len(target.points)} point(s), source={target.source}{extra}")
        if book.workspace_min is not None:
            print(f"workspace: min={book.workspace_min} max={book.workspace_max}")
        return

    if args.tcp_pose is not None:
        target = JudgeTarget(args.config_id, args.component, (tuple(args.tcp_pose),), "tcp_pose")
    elif args.demo_episode is not None:
        target = target_from_demo_episode(
            args.demo_episode, config_id=args.config_id, component=args.component,
            frame_index=args.frame_index,
        )
    else:
        target = target_from_region_json(args.target_region_json, config_id=args.config_id, component=args.component)
    if args.non_target:
        target = JudgeTarget(target.config_id, target.component, target.points, target.source,
                             non_targets=tuple(args.non_target))
    upsert_target(args.targets, target)
    print(f"saved {target.key} ({len(target.points)} point(s), source={target.source}) to {args.targets}")


if __name__ == "__main__":
    main()
