"""Read-only TRACE-VLA action-chunk compiler for offline RW-FMA.

This module is deliberately conservative.  It compiles *candidate* policy
training rows from saved TRACE-VLA rollout episodes, but it does not train, merge,
or command hardware.  The first supported target is Pi0.5/OpenPI RW-FMA:
nonnegative return/rank-weighted native flow-matching over aligned 10-step,
external 7-D action chunks.

Authority rules:
- VLM fields are sidecars only (`vlm_authority_used=false`).
- Numeric reward/weight fields are derived from robot trace, target-region
  score, operator labels, and human segment provenance, not VLM text.
- Unsafe/clamped/stale/auto-motion chunks are quarantined from policy loss.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracevla.collection.schemas import default_position_tolerance_m

try:  # scipy is already used by TRACE-VLA scoring; keep fallback for tiny tests.
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover - fallback path only for minimal envs
    Rotation = None

ACTION_CHUNK_SCHEMA_VERSION = "tracevla_action_chunks.v0.2"
TRIAL_SCHEMA_VERSION = "tracevla_trials.v0.1"
SEGMENT_SCHEMA_VERSION = "tracevla_segments.v0.1"
EXPECTED_HORIZON = 10
EXPECTED_ACTION_DIM = 7
PLANNER_BLOCKS = {"planner_only", "planner_skill"}
CORRECTOR_BLOCKS = {"corrector_only"}
TRAINABLE_PHASES = {"teleop", "recovery_correction", "clean_full_demo", "corrector_attempt"}
EXCLUDED_CONTROL_SOURCES = {"scripted_move_home", "auto_motion", "homing"}

PLANNER_LABEL_RANK = {
    "success_skill_completed": 5,
    "success_stop_token": 4,
    "success_manual_stop_near_target": 4,
    "success": 4,
    "stop_token_should_emit_here": 4,
    "near_miss_stop_token": 3,
    "near_miss_manual_stop": 3,
    "near_miss_timeout": 3,
    "planner_near_miss_position": 3,
    "near_but_not_accurate": 3,
    "planner_bad_position": 1,
    "planner_bad_orientation": 1,
    "wrong_orientation_or_wrong_location": 1,
    "wrong_target": 1,
    "bad_orientation": 1,
    "no_stop_timeout": 1,
    "manual_stop_bad": 0,
    "totally_off_wrong_region_or_target": 0,
    "totally_off": 0,
    "operator_uncertain_exclude": -1,
    "unsafe_abort": -1,
}

CORRECTOR_LABEL_RANK = {
    "corrector_success": 4,
    "human_corrected_then_save": 3,
    "near_miss": 2,
    "wrong_recovery_pose": 1,
    "retry_corrector": 1,
    "manual_drop_or_abort": 0,
    "operator_uncertain_exclude": -1,
    "unsafe_abort": -1,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _episode_id(ep: Path) -> str:
    return ep.name


def _infer_source(ep: Path) -> dict[str, Any]:
    parts = ep.parts
    # .../tracevla_correction_rollouts/config_XXX/block/component/trial
    config_id = block = component = None
    for i, part in enumerate(parts):
        if part.startswith("config_") and i + 3 < len(parts):
            config_id = part
            block = parts[i + 1]
            component = parts[i + 2]
    return {"config_id": config_id, "block": block, "component": component, "trial_id": ep.name}


def _label_rank(block: str | None, label: str | None) -> int | None:
    if not label:
        return None
    if block in CORRECTOR_BLOCKS:
        return CORRECTOR_LABEL_RANK.get(label)
    return PLANNER_LABEL_RANK.get(label)


def _role_for_segment(block: str | None, segment: Mapping[str, Any]) -> str:
    phase = segment.get("phase")
    control_source = segment.get("control_source")
    if block in CORRECTOR_BLOCKS or phase == "corrector_attempt":
        return "corrector"
    if phase in {"recovery_correction", "clean_full_demo"}:
        return "planner"
    if block in PLANNER_BLOCKS or control_source == "model_teleop":
        return "planner"
    return "unknown"


def _sample_role(segment: Mapping[str, Any], label: str | None, rank: int | None) -> str:
    phase = segment.get("phase")
    control_source = segment.get("control_source")
    if phase == "clean_full_demo" or control_source == "human_demo_from_home":
        return "clean_demo"
    if phase == "recovery_correction" or control_source == "human_correction_from_failure_pose":
        return "human_correction"
    if control_source == "model_teleop":
        if rank is not None and rank >= 4:
            return "model_success"
        if rank is not None and rank >= 2:
            return "model_partial"
        if rank is not None and rank >= 0:
            return "model_failure"
    return "excluded"


def _numeric_vec(x: Any, length: int) -> bool:
    return isinstance(x, list) and len(x) == length and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in x)


def _finite_float(x: Any) -> float | None:
    if not isinstance(x, (int, float)):
        return None
    value = float(x)
    return value if math.isfinite(value) else None


def _positive_float(x: Any) -> float | None:
    value = _finite_float(x)
    if value is None or value <= 0:
        return None
    return value


def _is_chunk(x: Any) -> bool:
    return isinstance(x, list) and len(x) == EXPECTED_HORIZON and all(_numeric_vec(a, EXPECTED_ACTION_DIM) for a in x)


def _actions_equal(a: Sequence[float], b: Sequence[float], tol: float = 1e-9) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def _load_frame(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _target_points(target_region: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = target_region.get("points")
    if not isinstance(points, list):
        return []
    out: list[dict[str, Any]] = []
    for p in points:
        if isinstance(p, dict) and _numeric_vec(p.get("tcp_pose"), 6):
            out.append(p)
    return out


_EMPTY_HYBRID_FIELDS = {
    "pos_error_norm_by_tol": None,
    "rot_error_norm_by_tol": None,
    "hybrid_pose_error_tol_units": None,
    "rotation_as_position_m": None,
    "combined_pose_distance_m": None,
}


def _hybrid_pose_distance_fields(
    pos_err: Any,
    rot_err: Any,
    pos_tol: Any,
    rot_tol: Any,
) -> dict[str, Any]:
    """Derive normalized + combined pose-distance fields for a single pose block.

    Returns five entries: per-tolerance normalized pos/rot errors, the L2 of
    those normalized errors in tolerance units, the rotation error expressed as
    an equivalent positional distance via the tolerance ratio, and the L2 of
    the original positional error with that rotation-equivalent distance.

    All entries become ``None`` if any input is missing/non-finite or if either
    tolerance is non-positive, so downstream consumers can safely skip rows.
    """
    pos_f = _finite_float(pos_err)
    rot_f = _finite_float(rot_err)
    pos_tol_f = _positive_float(pos_tol)
    rot_tol_f = _positive_float(rot_tol)
    if pos_f is None or rot_f is None or pos_tol_f is None or rot_tol_f is None:
        return dict(_EMPTY_HYBRID_FIELDS)
    pos_norm = pos_f / pos_tol_f
    rot_norm = rot_f / rot_tol_f
    hybrid = math.sqrt(pos_norm * pos_norm + rot_norm * rot_norm)
    rotation_as_position = rot_f * (pos_tol_f / rot_tol_f)
    combined = math.sqrt(pos_f * pos_f + rotation_as_position * rotation_as_position)
    return {
        "pos_error_norm_by_tol": round(pos_norm, 9),
        "rot_error_norm_by_tol": round(rot_norm, 9),
        "hybrid_pose_error_tol_units": round(hybrid, 9),
        "rotation_as_position_m": round(rotation_as_position, 9),
        "combined_pose_distance_m": round(combined, 9),
    }


def _empty_pose_error_block() -> dict[str, Any]:
    return {
        "pos_error_min_m": None,
        "rot_error_min_rad": None,
        "inside_region": None,
        "closest_target_idx": None,
        **dict(_EMPTY_HYBRID_FIELDS),
    }


def _pose_error_to_region(tcp_pose: Any, target_region: Mapping[str, Any]) -> dict[str, Any]:
    if not (_numeric_vec(tcp_pose, 6) and isinstance(tcp_pose, list)):
        return _empty_pose_error_block()
    points = _target_points(target_region)
    if not points:
        return _empty_pose_error_block()
    raw_pos_tol = target_region.get("position_tolerance_m")
    raw_rot_tol = target_region.get("rotation_tolerance_rad")
    pos_tol = _positive_float(raw_pos_tol)
    rot_tol = _positive_float(raw_rot_tol)
    if pos_tol is None and "position_tolerance_m" not in target_region:
        pos_tol = default_position_tolerance_m(str(target_region.get("component", "")))
    if rot_tol is None and "rotation_tolerance_rad" not in target_region:
        rot_tol = 0.35
    pose = [float(v) for v in tcp_pose[:6]]
    best_idx = None
    best_pos = float("inf")
    best_rot = float("inf")
    for idx, point in enumerate(points):
        target = [float(v) for v in point["tcp_pose"][:6]]
        pos_err = math.sqrt(sum((pose[i] - target[i]) ** 2 for i in range(3)))
        if Rotation is not None:
            try:
                rot_err = float((Rotation.from_rotvec(pose[3:6]).inv() * Rotation.from_rotvec(target[3:6])).magnitude())
            except Exception:
                rot_err = math.sqrt(sum((pose[i] - target[i]) ** 2 for i in range(3, 6)))
        else:
            rot_err = math.sqrt(sum((pose[i] - target[i]) ** 2 for i in range(3, 6)))
        if (pos_err, rot_err) < (best_pos, best_rot):
            best_idx, best_pos, best_rot = idx, pos_err, rot_err
    inside = None
    if pos_tol is not None and rot_tol is not None:
        inside = bool(best_pos <= pos_tol and best_rot <= rot_tol)
    return {
        "pos_error_min_m": round(best_pos, 9),
        "rot_error_min_rad": round(best_rot, 9),
        "inside_region": inside,
        "closest_target_idx": best_idx,
        **_hybrid_pose_distance_fields(best_pos, best_rot, raw_pos_tol, raw_rot_tol),
    }


def _load_segment_frames(ep: Path, start: int | None, end: int | None) -> list[dict[str, Any]]:
    return [frame for _, frame in _load_segment_frame_pairs(ep, start, end)]


def _load_segment_frame_pairs(ep: Path, start: int | None, end: int | None) -> list[tuple[int, dict[str, Any]]]:
    """Load segment frames with their original global frame indices."""
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        return []
    pairs: list[tuple[int, dict[str, Any]]] = []
    for idx in range(start, end + 1):
        frame = _load_frame(ep / f"frame_{idx:04d}.pkl")
        if frame is not None:
            pairs.append((idx, frame))
    return pairs


def _contiguous_non_auto_frame_runs(pairs: Sequence[tuple[int, dict[str, Any]]]) -> list[list[tuple[int, dict[str, Any]]]]:
    """Split frame pairs into physically contiguous non-auto-motion runs."""
    runs: list[list[tuple[int, dict[str, Any]]]] = []
    current: list[tuple[int, dict[str, Any]]] = []
    prev_idx: int | None = None
    for idx, frame in pairs:
        if frame.get("is_auto_motion"):
            if current:
                runs.append(current)
                current = []
            prev_idx = None
            continue
        if prev_idx is not None and idx != prev_idx + 1:
            if current:
                runs.append(current)
            current = []
        current.append((idx, frame))
        prev_idx = idx
    if current:
        runs.append(current)
    return runs


def _segment_distance_score(ep: Path, segment: Mapping[str, Any], target_region: Mapping[str, Any]) -> dict[str, Any]:
    frames = _load_segment_frames(ep, segment.get("start"), segment.get("end"))
    tcp_frames = [f for f in frames if _numeric_vec(f.get("tcp_pose"), 6)]
    if not tcp_frames:
        return {
            "present": False,
            "reason": "no_segment_tcp_pose_frames",
            "start": None,
            "end": None,
            "min": None,
            "min_pos_error_m": None,
            "delta_pos_error_m": None,
            "tcp_motion_total_m": None,
            "tcp_z_delta_m": None,
        }
    start_pose = tcp_frames[0].get("tcp_pose")
    end_pose = tcp_frames[-1].get("tcp_pose")
    start_score = _pose_error_to_region(start_pose, target_region)
    end_score = _pose_error_to_region(end_pose, target_region)
    all_scores = [_pose_error_to_region(f.get("tcp_pose"), target_region) for f in tcp_frames]
    indexed_pos = [
        (i, s["pos_error_min_m"])
        for i, s in enumerate(all_scores)
        if isinstance(s.get("pos_error_min_m"), (int, float))
    ]
    pos_vals = [v for _, v in indexed_pos]
    min_score = all_scores[min(indexed_pos, key=lambda kv: kv[1])[0]] if indexed_pos else _empty_pose_error_block()
    tcp_motion = None
    tcp_z_delta = None
    if _numeric_vec(start_pose, 6) and _numeric_vec(end_pose, 6) and isinstance(start_pose, list) and isinstance(end_pose, list):
        tcp_motion = math.sqrt(sum((float(end_pose[i]) - float(start_pose[i])) ** 2 for i in range(3)))
        tcp_z_delta = float(end_pose[2]) - float(start_pose[2])
    start_pos = start_score.get("pos_error_min_m")
    end_pos = end_score.get("pos_error_min_m")
    delta_pos = None
    if isinstance(start_pos, (int, float)) and isinstance(end_pos, (int, float)):
        delta_pos = round(float(end_pos) - float(start_pos), 9)
    return {
        "present": True,
        "reason": None,
        "start": start_score,
        "end": end_score,
        "min": min_score,
        "min_pos_error_m": min(pos_vals) if pos_vals else None,
        "any_inside_region": any(s.get("inside_region") is True for s in all_scores),
        "delta_pos_error_m": delta_pos,
        "tcp_motion_total_m": None if tcp_motion is None else round(tcp_motion, 9),
        "tcp_z_delta_m": None if tcp_z_delta is None else round(tcp_z_delta, 9),
    }


def _offset_target_region_z(target_region: Mapping[str, Any], clearance_m: float) -> dict[str, Any]:
    """Return a target-region-like dict with each tcp target lifted in z."""
    lifted = dict(target_region)
    points = []
    for point in _target_points(target_region):
        q = dict(point)
        pose = [float(v) for v in point["tcp_pose"][:6]]
        pose[2] += float(clearance_m)
        q["tcp_pose"] = pose
        q["tag"] = f"{point.get('tag', 'target')}_hover_{int(clearance_m * 100):02d}cm"
        points.append(q)
    lifted["points"] = points
    return lifted


def _hover_distance_score(ep: Path, segment: Mapping[str, Any], target_region: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for clearance in (0.05, 0.10):
        key = f"hover_{int(clearance * 100):02d}cm"
        score = _segment_distance_score(ep, segment, _offset_target_region_z(target_region, clearance))
        end = score.get("end") or {}
        out[key] = {
            **score,
            "clearance_m": clearance,
            "hover_reached": bool(score.get("any_inside_region") is True),
        }
    return out


def _empty_chunk_local_block(active_distance_target: str) -> dict[str, Any]:
    target_family = "hover_05cm" if active_distance_target == "hover_approach" else "final_target"
    return {
        "present": False,
        "active_distance_target": active_distance_target,
        "target_family": target_family,
        "start_pos_error_m": None,
        "end_pos_error_m": None,
        "min_pos_error_m": None,
        "delta_pos_error_m": None,
        "start_combined_pose_distance_m": None,
        "end_combined_pose_distance_m": None,
        "min_combined_pose_distance_m": None,
        "delta_combined_pose_distance_m": None,
        "inside_region_first_t": None,
        "inside_region_at_end": False,
        "hover_05cm_first_t": None,
        "hover_05cm_at_end": False,
        "hover_10cm_first_t": None,
        "per_step_pos_error_m": [],
        "per_step_combined_pose_distance_m": [],
    }


def _chunk_local_distance_features(
    ep: Path,
    chunk_frame_indices: Sequence[int | None],
    target_region: Mapping[str, Any],
    active_distance_target: str,
) -> dict[str, Any]:
    """Distance features computed strictly over the chunk's actual frame window.

    Frame indices come from the rollout (model: ``executed_action.frame_index``;
    human: the actual run indices for the action targets). If any frame in the
    window cannot be loaded or lacks a valid 6-D TCP pose, the block returns
    ``present=false`` rather than fabricating values.
    """
    empty = _empty_chunk_local_block(active_distance_target)
    if not chunk_frame_indices:
        return empty
    frames: list[dict[str, Any] | None] = []
    for idx in chunk_frame_indices:
        if not isinstance(idx, int):
            return empty
        f = _load_frame(ep / f"frame_{idx:04d}.pkl")
        if not isinstance(f, dict):
            return empty
        frames.append(f)
    tcp_poses = [f.get("tcp_pose") for f in frames]
    if not all(_numeric_vec(p, 6) for p in tcp_poses):
        return empty
    target_family = "hover_05cm" if active_distance_target == "hover_approach" else "final_target"
    if target_family == "hover_05cm":
        active_region = _offset_target_region_z(target_region, 0.05)
    else:
        active_region = target_region
    scores = [_pose_error_to_region(p, active_region) for p in tcp_poses]
    indexed = [(i, s.get("pos_error_min_m")) for i, s in enumerate(scores) if isinstance(s.get("pos_error_min_m"), (int, float))]
    if not indexed:
        return empty
    min_idx = min(indexed, key=lambda kv: kv[1])[0]
    start_score = scores[0]
    end_score = scores[-1]
    min_score = scores[min_idx]

    def _combined(s: Mapping[str, Any]) -> float | None:
        v = s.get("combined_pose_distance_m")
        return v if isinstance(v, (int, float)) else None

    start_pos = start_score.get("pos_error_min_m")
    end_pos = end_score.get("pos_error_min_m")
    min_pos = min_score.get("pos_error_min_m")
    delta_pos = None
    if isinstance(start_pos, (int, float)) and isinstance(end_pos, (int, float)):
        delta_pos = round(float(start_pos) - float(end_pos), 9)

    start_cd = _combined(start_score)
    end_cd = _combined(end_score)
    min_cd = _combined(min_score)
    delta_cd = None
    if isinstance(start_cd, (int, float)) and isinstance(end_cd, (int, float)):
        delta_cd = round(float(start_cd) - float(end_cd), 9)

    def _first_inside(region_scores: Sequence[Mapping[str, Any]]) -> int | None:
        for i, s in enumerate(region_scores):
            if s.get("inside_region") is True:
                return i
        return None

    h05_scores = [_pose_error_to_region(p, _offset_target_region_z(target_region, 0.05)) for p in tcp_poses]
    h10_scores = [_pose_error_to_region(p, _offset_target_region_z(target_region, 0.10)) for p in tcp_poses]

    return {
        "present": True,
        "active_distance_target": active_distance_target,
        "target_family": target_family,
        "start_pos_error_m": start_pos,
        "end_pos_error_m": end_pos,
        "min_pos_error_m": min_pos,
        "delta_pos_error_m": delta_pos,
        "start_combined_pose_distance_m": start_cd,
        "end_combined_pose_distance_m": end_cd,
        "min_combined_pose_distance_m": min_cd,
        "delta_combined_pose_distance_m": delta_cd,
        "inside_region_first_t": _first_inside(scores),
        "inside_region_at_end": bool(scores[-1].get("inside_region") is True),
        "hover_05cm_first_t": _first_inside(h05_scores),
        "hover_05cm_at_end": bool(h05_scores[-1].get("inside_region") is True),
        "hover_10cm_first_t": _first_inside(h10_scores),
        "per_step_pos_error_m": [s.get("pos_error_min_m") if isinstance(s.get("pos_error_min_m"), (int, float)) else None for s in scores],
        "per_step_combined_pose_distance_m": [_combined(s) for s in scores],
    }


def _trial_final_block(auto_score: Mapping[str, Any], target_region: Mapping[str, Any]) -> dict[str, Any]:
    """Trial-level final pose distance block, including hybrid fields.

    Used for both model rows and human rows when ``tracevla_auto_score.json``
    carries finite geometry; reverts to nulls otherwise.
    """
    if not isinstance(auto_score, Mapping):
        auto_score = {}
    if not isinstance(target_region, Mapping):
        target_region = {}
    pos = auto_score.get("pos_error_min_m")
    rot = auto_score.get("rot_error_min_rad")
    inside = auto_score.get("inside_region")
    closest = auto_score.get("closest_target_idx")
    pos_f = _finite_float(pos)
    rot_f = _finite_float(rot)
    if pos_f is None or rot_f is None:
        return {
            "pos_error_min_m": None,
            "rot_error_min_rad": None,
            "inside_region": None,
            "closest_target_idx": None,
            **dict(_EMPTY_HYBRID_FIELDS),
        }
    pos_tol = target_region.get("position_tolerance_m", 0.02)
    rot_tol = target_region.get("rotation_tolerance_rad", 0.35)
    hybrid = _hybrid_pose_distance_fields(pos_f, rot_f, pos_tol, rot_tol)
    return {
        "pos_error_min_m": pos_f,
        "rot_error_min_rad": rot_f,
        "inside_region": inside if isinstance(inside, bool) else None,
        "closest_target_idx": closest,
        **hybrid,
    }


def _phase_gated_progress(final_score: Mapping[str, Any], hover_scores: Mapping[str, Any]) -> dict[str, Any]:
    """Privileged phase-gated planner progress scaffold.

    Early planner credit should prefer reaching a hover/approach manifold before
    descending to the final target region. This returns categorical/numeric
    audit features; it is not a calibrated advantage.
    """
    h05 = hover_scores.get("hover_05cm") or {}
    h10 = hover_scores.get("hover_10cm") or {}
    h05_reached = bool(h05.get("hover_reached"))
    h10_reached = bool(h10.get("hover_reached"))
    final_terminal_reached = bool((final_score.get("end") or {}).get("inside_region") is True)
    # Terminal final-target success must override hover-approach semantics. Otherwise
    # use the final target after the segment has ever crossed a hover manifold.
    active = "final_target" if (final_terminal_reached or h05_reached or h10_reached) else "hover_approach"
    hover_delta = h05.get("delta_pos_error_m")
    final_delta = final_score.get("delta_pos_error_m")

    def progress_class(delta: Any, reached: bool = False) -> str:
        if reached:
            return "reached"
        if not isinstance(delta, (int, float)):
            return "unknown"
        if delta <= -0.03:
            return "strong_improvement"
        if delta < -0.005:
            return "weak_improvement"
        if delta >= 0.02:
            return "regressed"
        return "no_clear_progress"

    hover_progress = progress_class(hover_delta, h05_reached or h10_reached)
    final_progress = progress_class(final_delta, bool((final_score.get("end") or {}).get("inside_region") is True))
    if active == "hover_approach":
        previous_chunk_quality = {
            "strong_improvement": "improved",
            "weak_improvement": "weak_improvement",
            "no_clear_progress": "no_progress",
            "regressed": "regressed",
            "unknown": "not_judgeable",
            "reached": "improved",
        }[hover_progress]
        suggested = "descend" if hover_progress == "reached" else "approach_higher"
    else:
        previous_chunk_quality = {
            "strong_improvement": "improved",
            "weak_improvement": "weak_improvement",
            "no_clear_progress": "no_progress",
            "regressed": "regressed",
            "unknown": "not_judgeable",
            "reached": "improved",
        }[final_progress]
        suggested = "hold_orientation" if final_progress == "reached" else "descend"
    return {
        "active_distance_target": active,
        "hover_reached_05cm": h05_reached,
        "hover_reached_10cm": h10_reached,
        "hover_progress_class": hover_progress,
        "final_progress_class": final_progress,
        "previous_chunk_quality_label": previous_chunk_quality,
        "suggested_next_planner_constraint_label": suggested,
    }


def _vlmb_categorical_sidecar(sample_role: str, phase_gate: Mapping[str, Any], quarantined: bool, quarantine_reasons: Sequence[str]) -> dict[str, Any]:
    if quarantined and any("safety" in r or "clamped" in r or "operator_excluded" in r for r in quarantine_reasons):
        rl_use = "exclude_safety"
    elif quarantined:
        rl_use = "shadow_only"
    elif sample_role in {"clean_demo", "human_correction"}:
        rl_use = "planner_positive_candidate"
    elif sample_role == "model_success":
        rl_use = "planner_positive_candidate"
    elif sample_role == "model_partial":
        rl_use = "planner_retry_signal"
    elif sample_role == "model_failure":
        rl_use = "value_negative"
    else:
        rl_use = "shadow_only"
    quality = str(phase_gate.get("previous_chunk_quality_label") or "not_judgeable")
    return {
        "vlm_authority_used": False,
        "vlmb_feedback_schema": "tracevla_geometry_feedback.v0.1",
        "progress_class": phase_gate.get("hover_progress_class") if phase_gate.get("active_distance_target") == "hover_approach" else phase_gate.get("final_progress_class"),
        "previous_chunk_quality": quality,
        "suggested_next_planner_constraint": phase_gate.get("suggested_next_planner_constraint_label"),
        "rl_feedback_use": rl_use,
        "note": "categorical labels derived from privileged geometry for offline VLM-B supervision/audit only",
    }


def _frame_action(frame: Mapping[str, Any]) -> list[float] | None:
    joints = frame.get("joint_positions")
    grip = frame.get("gripper_pos")
    if not (_numeric_vec(joints, 6) and isinstance(joints, list) and isinstance(grip, (int, float))):
        return None
    return [float(v) for v in joints[:6]] + [max(0.0, min(1.0, float(grip) / 255.0))]


def _human_segment_chunk_rows(
    *,
    ep: Path,
    trial_row: Mapping[str, Any],
    segment: Mapping[str, Any],
    segment_distance: Mapping[str, Any],
    hover_scores: Mapping[str, Any],
    label: str | None,
    rank: int | None,
    target_region: Mapping[str, Any],
    rel_ep: str,
) -> list[dict[str, Any]]:
    phase = segment.get("phase")
    control_source = segment.get("control_source")
    if phase not in {"recovery_correction", "clean_full_demo", "human_corrector_demo"}:
        return []
    if control_source in EXCLUDED_CONTROL_SOURCES:
        return []
    frame_pairs = _load_segment_frame_pairs(ep, segment.get("start"), segment.get("end"))
    frame_runs = _contiguous_non_auto_frame_runs(frame_pairs)
    rows: list[dict[str, Any]] = []
    sample_role = "clean_demo" if phase in {"clean_full_demo", "human_corrector_demo"} else "human_correction"
    role = _role_for_segment(trial_row.get("block"), segment)
    # Non-overlapping 1-second windows inside each contiguous non-auto-motion run:
    # observation at frame i, targets i+1..i+10. Never stitch across auto/homing gaps.
    auto_score = _read_json(ep / "tracevla_auto_score.json")
    for run in frame_runs:
        for local_start in range(0, max(0, len(run) - EXPECTED_HORIZON), EXPECTED_HORIZON):
            obs_global_idx, obs_frame = run[local_start]
            target_pairs = run[local_start + 1 : local_start + 1 + EXPECTED_HORIZON]
            target_frames = [f for _, f in target_pairs]
            target_frame_indices = [gi for gi, _ in target_pairs]
            if len(target_frames) != EXPECTED_HORIZON:
                continue
            action_chunk = [_frame_action(f) for f in target_frames]
            if not all(a is not None for a in action_chunk):
                continue
            phase_gate = _phase_gated_progress(segment_distance, hover_scores)
            chunk_local = _chunk_local_distance_features(
                ep,
                target_frame_indices,
                target_region,
                str(phase_gate.get("active_distance_target") or "hover_approach"),
            )
            quarantine_reasons: list[str] = []
            if label in {"operator_uncertain_exclude", "unsafe_abort"}:
                quarantine_reasons.append(f"operator_excluded_label:{label}")
            quarantined = bool(quarantine_reasons)
            reward = _score_reward_components({}, label, rank, sample_role)
            reward["phase_gated_components"] = phase_gate
            route_components = _route_reward_components(chunk_local, sample_role)
            reward["route_reward_components"] = route_components
            weight = _rw_weight(sample_role, reward, quarantined, rank, route_components)
            mask = _policy_loss_mask(sample_role, weight, chunk_local)
            aw_info = _aw_fma_reward_components(chunk_local, sample_role, quarantined, quarantine_reasons, rank, label)
            stratum_key = _aw_fma_stratum_key(trial_row.get("config_id"), trial_row.get("block"), trial_row.get("component"))
            aw_returns = _aw_fma_returns_block(aw_info, sample_role, quarantined, quarantine_reasons, stratum_key)
            rows.append({
            "schema": ACTION_CHUNK_SCHEMA_VERSION,
            "row_kind": "tracevla_rl_action_chunk",
            "row_id": f"chunk:{trial_row.get('trial_id')}:seg{segment.get('segment_id')}:h{obs_global_idx:04d}",
            "trial_id": trial_row.get("trial_id"),
            "episode_dir": rel_ep,
            "config_id": trial_row.get("config_id"),
            "block": trial_row.get("block"),
            "component": trial_row.get("component"),
            "policy_role": role,
            "sample_role": sample_role,
            "segment_id": segment.get("segment_id"),
            "phase": phase,
            "control_source": control_source,
            "query_index": None,
            "observation": {
                **_frame_refs(ep, obs_global_idx),
                "prompt": trial_row.get("prompt") or trial_row.get("language_instruction"),
                "query_obs_joints": obs_frame.get("joint_positions"),
                "state_7d": _frame_action(obs_frame),
                "query_ee_pos_quat": None,
                "tcp_pose": obs_frame.get("tcp_pose"),
                "source": "frame_trajectory_human_segment",
            },
            "action_convention": {
                "model_type": "human_frame_trajectory",
                "adapter_action_format": "openpi_absolute_joint_from_frames",
                "external_action_dim": EXPECTED_ACTION_DIM,
                "horizon": EXPECTED_HORIZON,
                "internal_openpi_action_dim": 32,
                "delta_actions_first_six_joints": True,
                "gripper_absolute": True,
            },
            "actions": {
                "raw_action_chunk": action_chunk,
                "executed_action_chunk": action_chunk,
                "post_safety_clamp_action_chunk": action_chunk,
            },
            "masks": {
                "action_mask": [1] * EXPECTED_HORIZON,
                "policy_loss_mask": mask,
                "reward_pad_mask": [1] * EXPECTED_HORIZON,
            },
            "reward": reward,
            "distance_features": {
                "trial_final_to_target_region": _trial_final_block(auto_score, target_region),
                "segment_to_target_region": segment_distance,
                "hover_above_target_region": hover_scores,
                "phase_gated_progress": phase_gate,
                "chunk_local_to_active_target": chunk_local,
            },
            "returns": {
                "return_unvalidated": reward["scalar_reward_unvalidated"],
                "rank_weight_unvalidated": weight,
                "w_base_unvalidated": _w_base(sample_role),
                "R_route_unvalidated": route_components["R_route_unvalidated"],
                "advantage_valid": False,
                "advantage": None,
                **aw_returns,
            },
            "eligibility": {
                "usable_for_rw_fma_dry_run": weight > 0,
                "quarantined": quarantined,
                "quarantine_reasons": quarantine_reasons,
            },
            "semantic_sidecar": _vlmb_categorical_sidecar(sample_role, phase_gate, quarantined, quarantine_reasons),
            "provenance": {
                "source": "human_segment_frame_trajectory",
                "episode_meta_sha256": _sha256_file(ep / "episode_meta.json"),
                "target_region_sha256": _sha256_file(ep / "target_region.json"),
                "tracevla_trial_feedback_sha256": _sha256_file(ep / "tracevla_trial_feedback.json"),
            },
        })
    return rows


def _frame_refs(ep: Path, frame_index: int | None) -> dict[str, Any]:
    if frame_index is None:
        return {"frame_index": None, "frame_pkl": None, "frame_sha256": None}
    p = ep / f"frame_{int(frame_index):04d}.pkl"
    return {
        "frame_index": frame_index,
        "frame_pkl": p.as_posix() if p.exists() else None,
        "frame_sha256": _sha256_file(p),
    }


def _segment_for_frame(segments: Sequence[Mapping[str, Any]], frame_index: int | None) -> Mapping[str, Any] | None:
    if frame_index is None:
        return None
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        if isinstance(start, int) and isinstance(end, int) and start <= frame_index <= end:
            return seg
    return None


def _score_reward_components(auto_score: Mapping[str, Any], label: str | None, rank: int | None, sample_role: str) -> dict[str, Any]:
    inside = auto_score.get("inside_region")
    pos = auto_score.get("pos_error_min_m")
    rot = auto_score.get("rot_error_min_rad")
    score_present = bool(auto_score)
    # Conservative scalar: use label rank for coarse outcome and target-region score
    # only when present. This is a scaffold, not a calibrated advantage.
    label_component = 0.0
    if rank is not None:
        if rank >= 4:
            label_component = 1.0
        elif rank >= 2:
            label_component = 0.45
        elif rank >= 0:
            label_component = 0.05
        else:
            label_component = 0.0
    physical_component = None
    if inside is True:
        physical_component = 1.0
    elif inside is False and isinstance(pos, (int, float)):
        # 5 cm roughly zeroes out. Keeps positive weights nonnegative but low.
        physical_component = max(0.0, 1.0 - min(float(pos), 0.05) / 0.05)
    scalar = label_component if physical_component is None else 0.5 * label_component + 0.5 * physical_component
    if sample_role in {"clean_demo", "human_correction"}:
        scalar = max(scalar, 0.8 if sample_role == "clean_demo" else 0.65)
    return {
        "score_present": score_present,
        "label_component": round(label_component, 6),
        "physical_component": None if physical_component is None else round(physical_component, 6),
        "scalar_reward_unvalidated": round(float(scalar), 6),
        "pos_error_min_m": pos,
        "rot_error_min_rad": rot,
        "inside_region": inside,
        "stop_source": auto_score.get("stop_source"),
        "steps_to_stop": auto_score.get("steps_to_stop"),
    }


def _route_reward_components(chunk_local: Mapping[str, Any], sample_role: str) -> dict[str, Any]:
    """Compute the v3 privileged route/progress reward components.

    This is an offline compiler-side scalar used only for nonnegative RW-FMA
    weighting/audit. It is not VLM authority and not a PPO reward model.
    Human teacher rows are not multiplied by this score, but we still emit the
    audit block so downstream analysis can stratify demos/corrections.
    """
    present = bool(chunk_local.get("present"))
    e_end = _finite_float(chunk_local.get("end_combined_pose_distance_m"))
    if e_end is None:
        e_end = _finite_float(chunk_local.get("end_pos_error_m"))
    delta = _finite_float(chunk_local.get("delta_combined_pose_distance_m"))
    if delta is None:
        delta = _finite_float(chunk_local.get("delta_pos_error_m"))
    r_prox = math.exp(-e_end / 0.05) if e_end is not None else 0.0
    r_prog = max(0.0, min(1.0, delta / 0.05)) if delta is not None else 0.0
    r_inside = 1.0 if (chunk_local.get("inside_region_at_end") is True or chunk_local.get("hover_05cm_at_end") is True) else 0.0
    route_gate = present and e_end is not None
    if route_gate:
        r_route = max(0.05, min(1.0, 0.5 * r_prox + 0.4 * r_prog + 0.1 * r_inside))
    else:
        r_route = 0.0
    return {
        "reward_version": "tracevla_route_reward.v0.1",
        "reward_authority": "robot_trace_target_region_human_route_operator_label",
        "semantic_gate": True,
        "role_gate": sample_role in {"clean_demo", "human_correction", "model_success", "model_partial"},
        "safety_gate": sample_role in {"clean_demo", "human_correction", "model_success", "model_partial"},
        "route_gate": route_gate,
        "r_proximity_unvalidated": round(float(r_prox), 6),
        "r_progress_unvalidated": round(float(r_prog), 6),
        "r_inside_unvalidated": round(float(r_inside), 6),
        "R_route_unvalidated": round(float(r_route), 6),
    }


def _rw_weight(sample_role: str, reward: Mapping[str, Any], quarantined: bool, rank: int | None, route_components: Mapping[str, Any] | None = None) -> float:
    if quarantined:
        return 0.0
    w_base = {
        "clean_demo": 1.0,
        "human_correction": 1.5,
        "model_success": 0.8,
        "model_partial": 0.5,
    }.get(sample_role, 0.0)
    if sample_role in {"clean_demo", "human_correction"}:
        return w_base
    if sample_role in {"model_success", "model_partial"}:
        r_route = 0.0
        if isinstance(route_components, Mapping):
            r_route = float(route_components.get("R_route_unvalidated") or 0.0)
        return round(w_base * r_route, 6)
    # First pass does not train failures as negative flow targets.
    return 0.0


def _w_base(sample_role: str) -> float:
    return {
        "clean_demo": 1.0,
        "human_correction": 1.5,
        "model_success": 0.8,
        "model_partial": 0.5,
    }.get(sample_role, 0.0)


def _policy_loss_mask(sample_role: str, weight: float, chunk_local: Mapping[str, Any]) -> list[float]:
    if weight <= 0:
        return [0.0] * EXPECTED_HORIZON
    # Keep human teacher trajectories fully supervised. Only model rows may use
    # per-step route weighting so correction/demo targets are not suppressed.
    if sample_role in {"clean_demo", "human_correction"}:
        return [1.0] * EXPECTED_HORIZON
    per_step = chunk_local.get("per_step_combined_pose_distance_m")
    if not (isinstance(per_step, list) and len(per_step) == EXPECTED_HORIZON and all(isinstance(v, (int, float)) for v in per_step)):
        return [1.0] * EXPECTED_HORIZON
    raw = [math.exp(-float(v) / 0.075) for v in per_step]
    mean = sum(raw) / len(raw) if raw else 0.0
    if mean <= 0:
        return [1.0] * EXPECTED_HORIZON
    return [round(float(v / mean), 6) for v in raw]


AW_FMA_REWARD_VERSION = "tracevla_aw_fma_reward.v0.1"
AW_FMA_MIN_STRATUM_N = 4


def _aw_fma_reward_components(
    chunk_local: Mapping[str, Any],
    sample_role: str,
    quarantined: bool,
    quarantine_reasons: Sequence[str],
    rank: int | None,
    label: str | None,
) -> dict[str, Any]:
    """Compute AW-FMA reward: distance/progress-dominant with safety/role/quarantine gates."""
    safety_gate = not any("safety" in r or "clamped" in r for r in quarantine_reasons)
    role_gate = sample_role in {"clean_demo", "human_correction", "model_success", "model_partial", "model_failure"}
    quarantine_gate = not quarantined

    if sample_role == "clean_demo":
        return _aw_fma_human_anchor(1.0, sample_role, safety_gate, role_gate, quarantine_gate)
    if sample_role == "human_correction":
        return _aw_fma_human_anchor(0.95, sample_role, safety_gate, role_gate, quarantine_gate)

    present = bool(chunk_local.get("present"))
    e_end = _finite_float(chunk_local.get("end_combined_pose_distance_m"))
    if e_end is None:
        e_end = _finite_float(chunk_local.get("end_pos_error_m"))
    delta = _finite_float(chunk_local.get("delta_combined_pose_distance_m"))
    if delta is None:
        delta = _finite_float(chunk_local.get("delta_pos_error_m"))

    r_proximity = math.exp(-e_end / 0.05) if e_end is not None else 0.0
    r_progress = (0.5 + 0.5 * max(-1.0, min(1.0, delta / 0.05))) if delta is not None else 0.5
    r_terminal = 1.0 if (chunk_local.get("inside_region_at_end") is True or chunk_local.get("hover_05cm_at_end") is True) else 0.0

    r_stop_handoff = 0.0
    if label and "stop" in label:
        r_stop_handoff = 0.5

    r_operator_rank = 0.0
    if rank is not None:
        r_operator_rank = max(0.0, min(1.0, rank / 5.0))

    reward = (
        0.45 * r_proximity
        + 0.30 * r_progress
        + 0.15 * r_terminal
        + 0.05 * r_stop_handoff
        + 0.05 * r_operator_rank
    )

    gated = reward if (safety_gate and role_gate and quarantine_gate and present) else 0.0

    return {
        "aw_fma_reward": round(gated, 9),
        "aw_fma_reward_components": {
            "r_proximity": round(r_proximity, 9),
            "r_progress_signed01": round(r_progress, 9),
            "r_terminal": round(r_terminal, 9),
            "r_stop_handoff": round(r_stop_handoff, 9),
            "r_operator_rank": round(r_operator_rank, 9),
            "weights": [0.45, 0.30, 0.15, 0.05, 0.05],
        },
        "safety_gate": safety_gate,
        "role_gate": role_gate,
        "quarantine_gate": quarantine_gate,
        "present": present,
    }


def _aw_fma_human_anchor(
    anchor: float,
    sample_role: str,
    safety_gate: bool,
    role_gate: bool,
    quarantine_gate: bool,
) -> dict[str, Any]:
    return {
        "aw_fma_reward": round(anchor, 9),
        "aw_fma_reward_components": {
            "r_proximity": None,
            "r_progress_signed01": None,
            "r_terminal": None,
            "r_stop_handoff": None,
            "r_operator_rank": None,
            "weights": [0.45, 0.30, 0.15, 0.05, 0.05],
            "anchor_override": anchor,
            "anchor_role": sample_role,
        },
        "safety_gate": safety_gate,
        "role_gate": role_gate,
        "quarantine_gate": quarantine_gate,
        "present": True,
    }


def _aw_fma_returns_block(
    reward_info: Mapping[str, Any],
    sample_role: str,
    quarantined: bool,
    quarantine_reasons: Sequence[str],
    stratum_key: str,
) -> dict[str, Any]:
    """Build the full AW-FMA returns sub-block for a single chunk row."""
    aw_reward = float(reward_info.get("aw_fma_reward") or 0.0)
    safety_gate = bool(reward_info.get("safety_gate"))
    role_gate = bool(reward_info.get("role_gate"))
    quarantine_gate = bool(reward_info.get("quarantine_gate"))

    loss_eligible = (
        safety_gate
        and role_gate
        and quarantine_gate
        and sample_role != "model_failure"
        and aw_reward > 0
    )

    return {
        "aw_fma_reward": aw_reward,
        "aw_fma_return": aw_reward,
        "aw_fma_baseline": None,
        "aw_fma_advantage": None,
        "aw_fma_advantage_normalized": None,
        "aw_fma_weight": aw_reward if loss_eligible else 0.0,
        "aw_fma_reward_components": reward_info.get("aw_fma_reward_components"),
        "aw_fma_stratum_key": stratum_key,
        "aw_fma_stratum_backoff_level": 0,
        "aw_fma_stratum_size": None,
        "aw_fma_baseline_eligible": loss_eligible,
        "aw_fma_advantage_valid": False,
        "aw_fma_loss_eligible": loss_eligible,
        "aw_fma_version": AW_FMA_REWARD_VERSION,
        "aw_fma_vlm_authority_used": False,
        "aw_fma_reward_authority": "robot_trace_target_region_operator_label",
    }


def _aw_fma_stratum_key(config_id: str | None, block: str | None, component: str | None) -> str:
    parts = [str(config_id or "unknown"), str(block or "unknown"), str(component or "unknown")]
    return "|".join(parts)


def compute_aw_fma_baselines(
    chunks: Sequence[dict[str, Any]],
    *,
    min_n: int = AW_FMA_MIN_STRATUM_N,
    allow_all_rows_for_aw_fma_baseline: bool = False,
) -> None:
    """In-place dataset-level stratum baseline computation.

    Groups by stratum_key, computes mean reward of baseline-eligible rows.
    If fewer than min_n eligible rows exist in a stratum, advantage stays invalid.
    Does NOT use sample_role for grouping (avoids leaking outcome labels).

    Conservative split guard: when *allow_all_rows_for_aw_fma_baseline* is False
    (default), only rows whose ``split`` field equals ``"train"`` participate in
    baseline computation.  Rows without a train split keep reward/return fields
    for audit but baseline, advantage, and validity stay at their safe defaults.
    """
    strata: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(chunks):
        aw = row.get("returns", {})
        key = aw.get("aw_fma_stratum_key")
        if key and aw.get("aw_fma_baseline_eligible"):
            if not allow_all_rows_for_aw_fma_baseline and row.get("split") != "train":
                aw["aw_fma_weight"] = 0.0
                aw["aw_fma_loss_eligible"] = False
                continue
            strata[key].append(idx)

    for key, indices in strata.items():
        rewards = [float(chunks[i]["returns"]["aw_fma_reward"]) for i in indices]
        n = len(rewards)
        if n < min_n:
            for i in indices:
                chunks[i]["returns"]["aw_fma_stratum_size"] = n
            continue
        baseline = sum(rewards) / n
        std = math.sqrt(sum((r - baseline) ** 2 for r in rewards) / n) if n > 1 else 1.0
        std = max(std, 1e-8)
        for i in indices:
            r = float(chunks[i]["returns"]["aw_fma_reward"])
            adv = r - baseline
            adv_norm = adv / std
            chunks[i]["returns"]["aw_fma_baseline"] = round(baseline, 9)
            chunks[i]["returns"]["aw_fma_advantage"] = round(adv, 9)
            chunks[i]["returns"]["aw_fma_advantage_normalized"] = round(adv_norm, 9)
            chunks[i]["returns"]["aw_fma_stratum_size"] = n
            chunks[i]["returns"]["aw_fma_advantage_valid"] = True


def assign_splits(
    chunks: Sequence[dict[str, Any]],
    *,
    seed: int = 20260520,
    ratio: tuple[int, int, int] = (80, 10, 10),
) -> dict[str, str]:
    """Deterministic config-level train/val/heldout split assignment.

    Assigns splits at the config_id level so no config crosses splits.
    Returns the config→split mapping used.
    """
    configs: set[str] = set()
    for row in chunks:
        cid = row.get("config_id")
        if cid:
            configs.add(str(cid))
    total = sum(ratio)
    train_frac = ratio[0] / total
    val_frac = (ratio[0] + ratio[1]) / total
    config_splits: dict[str, str] = {}
    for cid in sorted(configs):
        h = hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        if bucket < train_frac:
            config_splits[cid] = "train"
        elif bucket < val_frac:
            config_splits[cid] = "val"
        else:
            config_splits[cid] = "heldout"
    for row in chunks:
        cid = row.get("config_id")
        row["split"] = config_splits.get(str(cid)) if cid else None
    return config_splits


def finalize_aw_fma_dataset(
    chunks: list[dict[str, Any]],
    *,
    fit_baselines: bool,
    min_stratum_n: int = AW_FMA_MIN_STRATUM_N,
    split_seed: int = 20260520,
    weight_beta: float = 1.0,
    weight_clip: float = 3.0,
    max_weight: float = 20.0,
) -> dict[str, Any]:
    """Dataset-level AW-FMA finalization: splits → baselines → weights → loss gating."""
    config_splits: dict[str, str] = {}
    if not fit_baselines:
        for row in chunks:
            ret = row.get("returns", {})
            ret["aw_fma_baseline"] = None
            ret["aw_fma_advantage"] = None
            ret["aw_fma_advantage_normalized"] = None
            ret["aw_fma_advantage_valid"] = False
            ret["aw_fma_weight"] = 0.0
            ret["aw_fma_loss_eligible"] = False
        return {"fit_baselines": False, "config_splits": config_splits}

    config_splits = assign_splits(chunks, seed=split_seed)
    compute_aw_fma_baselines(chunks, min_n=min_stratum_n)

    for row in chunks:
        ret = row.get("returns", {})
        split = row.get("split")
        sample_role = row.get("sample_role")
        quarantined = row.get("eligibility", {}).get("quarantined", False)

        can_be_loss_eligible = (
            split == "train"
            and ret.get("aw_fma_advantage_valid") is True
            and not quarantined
            and sample_role != "model_failure"
            and ret.get("aw_fma_vlm_authority_used") is not True
            and row.get("semantic_sidecar", {}).get("vlm_authority_used") is not True
        )

        if can_be_loss_eligible:
            adv_norm = float(ret.get("aw_fma_advantage_normalized") or 0.0)
            z = max(-weight_clip, min(weight_clip, adv_norm))
            w = min(max_weight, math.exp(weight_beta * z))
            ret["aw_fma_weight"] = round(w, 9)
            ret["aw_fma_loss_eligible"] = True
        else:
            ret["aw_fma_weight"] = 0.0
            ret["aw_fma_loss_eligible"] = False

    return {"fit_baselines": True, "config_splits": config_splits}


def _clip_l2_max(executed_chunk: Sequence[Sequence[float]], post_clamp_chunk: Sequence[Sequence[float]]) -> float:
    if len(executed_chunk) != len(post_clamp_chunk):
        return float("inf")
    diffs: list[float] = []
    for ea, pa in zip(executed_chunk, post_clamp_chunk):
        if len(ea) != len(pa):
            return float("inf")
        diffs.append(math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(ea, pa))))
    return max(diffs) if diffs else 0.0


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def compile_episode_action_chunks(ep: Path, *, repo_root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile one episode into trial, segment, and action-chunk rows."""
    ep = Path(ep)
    repo_root = Path(repo_root) if repo_root is not None else ep.parents[4]
    source = _infer_source(ep)
    meta = _read_json(ep / "episode_meta.json")
    feedback = _read_json(ep / "tracevla_trial_feedback.json")
    auto_score = _read_json(ep / "tracevla_auto_score.json")
    target_region = _read_json(ep / "target_region.json")
    trace = _read_jsonl(ep / "vla_action_trace.jsonl")
    label = feedback.get("label")
    rank = _label_rank(source.get("block"), label)
    trial_id = source.get("trial_id") or ep.name
    rel_ep = ep.relative_to(repo_root).as_posix() if ep.is_relative_to(repo_root) else ep.as_posix()

    query_rows = {r.get("query_index"): r for r in trace if r.get("type") == "policy_query"}
    exec_rows = [r for r in trace if r.get("type") == "executed_action"]
    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in exec_rows:
        if isinstance(row.get("query_index"), int):
            by_query[row["query_index"]].append(row)
    for rows in by_query.values():
        rows.sort(key=lambda r: r.get("action_index_within_chunk", -1))

    segments = meta.get("control_segments") or meta.get("phase_segments") or []
    segment_distance_by_id: dict[Any, dict[str, Any]] = {}
    segment_hover_by_id: dict[Any, dict[str, Any]] = {}
    segment_rows: list[dict[str, Any]] = []
    for seg in segments:
        role = _role_for_segment(source.get("block"), seg)
        control_source = seg.get("control_source")
        excluded = control_source in EXCLUDED_CONTROL_SOURCES or bool(seg.get("excluded_from_training"))
        seg_id = seg.get("segment_id", len(segment_rows))
        distance_score = _segment_distance_score(ep, seg, target_region)
        hover_score = _hover_distance_score(ep, seg, target_region)
        phase_gate = _phase_gated_progress(distance_score, hover_score)
        segment_distance_by_id[seg_id] = distance_score
        segment_hover_by_id[seg_id] = hover_score
        segment_rows.append({
            "schema": SEGMENT_SCHEMA_VERSION,
            "row_kind": "tracevla_rl_segment",
            "row_id": f"seg:{trial_id}:{seg_id}",
            "trial_id": trial_id,
            "episode_dir": rel_ep,
            "segment_id": seg_id,
            "phase": seg.get("phase"),
            "control_source": control_source,
            "policy_role": role,
            "start_frame": seg.get("start"),
            "end_frame": seg.get("end"),
            "attempt_idx": seg.get("attempt_idx"),
            "source_feedback_label": seg.get("source_feedback_label"),
            "distance_to_target_region": distance_score,
            "hover_above_target_region": hover_score,
            "phase_gated_progress": phase_gate,
            "excluded_from_training": excluded,
        })

    trial_row = {
        "schema": TRIAL_SCHEMA_VERSION,
        "row_kind": "tracevla_rl_trial",
        "trial_id": trial_id,
        "episode_dir": rel_ep,
        "config_id": source.get("config_id") or meta.get("config_id"),
        "block": source.get("block") or meta.get("block"),
        "component": source.get("component") or meta.get("component"),
        "label": label,
        "language_instruction": meta.get("language_instruction"),
        "value_rank": rank,
        "split": None,
        "num_frames": meta.get("num_frames"),
        "num_trace_rows": len(trace),
        "num_policy_queries": len(query_rows),
        "num_executed_actions": len(exec_rows),
        "had_correction": bool(meta.get("had_correction")),
        "had_full_demo": bool(meta.get("had_full_demo")),
        "target_region_present": bool(target_region.get("points")),
        "auto_score_present": bool(auto_score),
        "vlm_authority_used": False,
    }

    chunk_rows: list[dict[str, Any]] = []
    for query_index, query in sorted(query_rows.items()):
        if not isinstance(query_index, int):
            continue
        raw_chunk = query.get("raw_action_chunk")
        q_frame = query.get("observation_frame_index")
        seg = _segment_for_frame(segments, q_frame)
        if seg is None:
            seg = {}
        role = _role_for_segment(source.get("block"), seg)
        sample_role = _sample_role(seg, label, rank)
        executed = by_query.get(query_index, [])
        executed_chunk: list[list[float]] = []
        post_clamp_chunk: list[list[float]] = []
        gripper_cap_rows: list[dict[str, Any]] = []
        executed_frame_indices: list[int] = []
        for r in executed:
            ea = r.get("executed_action")
            pa = r.get("post_safety_clamp_action")
            raw_extra = r.get("extra")
            extra = raw_extra if isinstance(raw_extra, dict) else {}
            if extra.get("gripper_cap_enabled"):
                gripper_cap_rows.append(
                    {
                        "action_index_within_chunk": r.get("action_index_within_chunk"),
                        "frame_index": r.get("frame_index"),
                        "applied": bool(extra.get("gripper_cap_applied")),
                        "value_norm": extra.get("gripper_cap_value_norm"),
                        "raw_value": extra.get("gripper_cap_raw_value"),
                        "executed_value": extra.get("gripper_cap_executed_value"),
                        "component": extra.get("gripper_cap_component"),
                        "phase": extra.get("gripper_cap_phase"),
                        "reason": extra.get("gripper_cap_reason"),
                    }
                )
            if _numeric_vec(ea, EXPECTED_ACTION_DIM) and isinstance(ea, list):
                executed_chunk.append([float(v) for v in ea])
            if _numeric_vec(pa, EXPECTED_ACTION_DIM) and isinstance(pa, list):
                post_clamp_chunk.append([float(v) for v in pa])
            fi = r.get("frame_index")
            if isinstance(fi, int):
                executed_frame_indices.append(fi)
        has_full_raw = _is_chunk(raw_chunk)
        has_full_executed = len(executed_chunk) == EXPECTED_HORIZON
        has_full_post = len(post_clamp_chunk) == EXPECTED_HORIZON
        clip_l2 = 0.0
        clamped = False
        if has_full_executed and has_full_post:
            clip_l2 = _clip_l2_max(executed_chunk, post_clamp_chunk)
            clamped = clip_l2 > 0.01
        frame = _load_frame(ep / f"frame_{int(q_frame):04d}.pkl") if isinstance(q_frame, int) else None
        obs_state_7d = _frame_action(frame) if isinstance(frame, dict) else None
        frame_auto = bool(frame and frame.get("is_auto_motion"))
        control_source = seg.get("control_source")
        excluded_segment = control_source in EXCLUDED_CONTROL_SOURCES or bool(seg.get("excluded_from_training"))
        quarantine_reasons: list[str] = []
        if query.get("open_loop_horizon") != EXPECTED_HORIZON or query.get("chunk_length") != EXPECTED_HORIZON:
            quarantine_reasons.append("unexpected_horizon")
        if query.get("adapter_action_format") != "openpi_absolute_joint":
            quarantine_reasons.append("unexpected_action_format")
        if not has_full_raw:
            quarantine_reasons.append("raw_action_chunk_missing_or_bad_shape")
        if not has_full_executed:
            quarantine_reasons.append("executed_action_chunk_incomplete")
        if clamped:
            quarantine_reasons.append("safety_clamped_action")
        gripper_cap_applied = any(bool(r.get("applied")) for r in gripper_cap_rows)
        if gripper_cap_applied:
            quarantine_reasons.append("gripper_cap_applied_execution_only")
        if frame_auto or excluded_segment:
            quarantine_reasons.append("auto_motion_or_excluded_segment")
        if sample_role == "model_failure":
            quarantine_reasons.append("model_failure_no_positive_flow_target")
        if label in {"operator_uncertain_exclude", "unsafe_abort"}:
            quarantine_reasons.append(f"operator_excluded_label:{label}")
        reward = _score_reward_components(auto_score, label, rank, sample_role)
        seg_id_for_distance = seg.get("segment_id")
        segment_distance = segment_distance_by_id.get(seg_id_for_distance, _segment_distance_score(ep, seg, target_region))
        hover_scores = segment_hover_by_id.get(seg_id_for_distance, _hover_distance_score(ep, seg, target_region))
        phase_gate = _phase_gated_progress(segment_distance, hover_scores)
        chunk_local = _chunk_local_distance_features(
            ep,
            executed_frame_indices if len(executed_frame_indices) == EXPECTED_HORIZON else [],
            target_region,
            str(phase_gate.get("active_distance_target") or "hover_approach"),
        )
        reward["phase_gated_components"] = phase_gate
        route_components = _route_reward_components(chunk_local, sample_role)
        reward["route_reward_components"] = route_components
        quarantined = bool(quarantine_reasons)
        weight = _rw_weight(sample_role, reward, quarantined, rank, route_components)
        mask = _policy_loss_mask(sample_role, weight, chunk_local)
        aw_info = _aw_fma_reward_components(chunk_local, sample_role, quarantined, quarantine_reasons, rank, label)
        stratum_key = _aw_fma_stratum_key(trial_row["config_id"], trial_row["block"], trial_row["component"])
        aw_returns = _aw_fma_returns_block(aw_info, sample_role, quarantined, quarantine_reasons, stratum_key)
        chunk_rows.append({
            "schema": ACTION_CHUNK_SCHEMA_VERSION,
            "row_kind": "tracevla_rl_action_chunk",
            "row_id": f"chunk:{trial_id}:q{query_index:04d}",
            "trial_id": trial_id,
            "episode_dir": rel_ep,
            "config_id": trial_row["config_id"],
            "block": trial_row["block"],
            "component": trial_row["component"],
            "policy_role": role,
            "sample_role": sample_role,
            "segment_id": seg.get("segment_id"),
            "phase": seg.get("phase"),
            "control_source": control_source,
            "query_index": query_index,
            "observation": {
                **_frame_refs(ep, q_frame if isinstance(q_frame, int) else None),
                "prompt": query.get("prompt") or meta.get("language_instruction"),
                "query_obs_joints": (query.get("extra") or {}).get("query_obs_joints"),
                "state_7d": obs_state_7d,
                "query_ee_pos_quat": (query.get("extra") or {}).get("query_ee_pos_quat"),
                "base_fingerprint": (query.get("extra") or {}).get("query_base_fingerprint"),
                "wrist_fingerprint": (query.get("extra") or {}).get("query_wrist_fingerprint"),
            },
            "action_convention": {
                "model_type": query.get("model_type"),
                "adapter_action_format": query.get("adapter_action_format"),
                "external_action_dim": EXPECTED_ACTION_DIM,
                "horizon": EXPECTED_HORIZON,
                "internal_openpi_action_dim": 32,
                "delta_actions_first_six_joints": True,
                "gripper_absolute": True,
            },
            "actions": {
                "raw_action_chunk": raw_chunk if has_full_raw else None,
                "executed_action_chunk": executed_chunk if has_full_executed else executed_chunk,
                "post_safety_clamp_action_chunk": post_clamp_chunk if post_clamp_chunk else None,
                "gripper_cap": {
                    "enabled": bool(gripper_cap_rows),
                    "applied": bool(gripper_cap_applied),
                    "rows": gripper_cap_rows,
                    "num_applied": sum(1 for r in gripper_cap_rows if r.get("applied")),
                    "value_norm": next((r.get("value_norm") for r in gripper_cap_rows if r.get("value_norm") is not None), None),
                },
            },
            "masks": {
                "action_mask": [1] * len(raw_chunk) if has_full_raw else [],
                "policy_loss_mask": mask,
                "reward_pad_mask": [1] * len(executed_chunk),
            },
            "reward": reward,
            "distance_features": {
                "trial_final_to_target_region": _trial_final_block(auto_score, target_region),
                "segment_to_target_region": segment_distance,
                "hover_above_target_region": hover_scores,
                "phase_gated_progress": phase_gate,
                "chunk_local_to_active_target": chunk_local,
            },
            "returns": {
                "return_unvalidated": reward["scalar_reward_unvalidated"],
                "rank_weight_unvalidated": weight,
                "w_base_unvalidated": _w_base(sample_role),
                "R_route_unvalidated": route_components["R_route_unvalidated"],
                "advantage_valid": False,
                "advantage": None,
                **aw_returns,
            },
            "eligibility": {
                "usable_for_rw_fma_dry_run": weight > 0,
                "quarantined": quarantined,
                "quarantine_reasons": quarantine_reasons,
                "clip_l2_max": round(float(clip_l2), 9),
            },
            "semantic_sidecar": _vlmb_categorical_sidecar(sample_role, phase_gate, quarantined, quarantine_reasons),
            "provenance": {
                "vla_action_trace_sha256": _sha256_file(ep / "vla_action_trace.jsonl"),
                "episode_meta_sha256": _sha256_file(ep / "episode_meta.json"),
                "target_region_sha256": _sha256_file(ep / "target_region.json"),
                "tracevla_auto_score_sha256": _sha256_file(ep / "tracevla_auto_score.json"),
                "tracevla_trial_feedback_sha256": _sha256_file(ep / "tracevla_trial_feedback.json"),
            },
        })

    # Add high-value human correction / clean-demo chunks directly from saved
    # frame trajectories. These are not based on the VLM-B dataset and do not
    # require model policy_query rows.
    for seg in segments:
        seg_id = seg.get("segment_id")
        chunk_rows.extend(_human_segment_chunk_rows(
            ep=ep,
            trial_row=trial_row,
            segment=seg,
            segment_distance=segment_distance_by_id.get(seg_id, _segment_distance_score(ep, seg, target_region)),
            hover_scores=segment_hover_by_id.get(seg_id, _hover_distance_score(ep, seg, target_region)),
            label=label,
            rank=rank,
            target_region=target_region,
            rel_ep=rel_ep,
        ))
    return trial_row, segment_rows, chunk_rows


@dataclass(frozen=True)
class AwFmaOptions:
    fit_baselines: bool = False
    min_stratum_n: int = AW_FMA_MIN_STRATUM_N
    split_seed: int = 20260520
    weight_beta: float = 1.0
    weight_clip: float = 3.0
    max_weight: float = 20.0


@dataclass(frozen=True)
class CompileResult:
    trials: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    action_chunks: list[dict[str, Any]]
    report: dict[str, Any]
    aw_fma_options: AwFmaOptions | None = None
    aw_fma_finalization: dict[str, Any] | None = None


def discover_episode_dirs(input_root: Path, *, limit: int | None = None) -> list[Path]:
    eps = sorted(p.parent for p in Path(input_root).glob("config_*/**/vla_action_trace.jsonl"))
    return eps[:limit] if limit is not None else eps


def compile_action_chunk_dataset(
    input_root: Path,
    *,
    repo_root: Path | None = None,
    limit: int | None = None,
    aw_fma: AwFmaOptions | None = None,
) -> CompileResult:
    input_root = Path(input_root)
    repo_root = Path(repo_root) if repo_root is not None else input_root.parents[1]
    trials: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for ep in discover_episode_dirs(input_root, limit=limit):
        try:
            t, s, c = compile_episode_action_chunks(ep, repo_root=repo_root)
        except Exception as exc:
            failures.append({"episode_dir": ep.as_posix(), "error": repr(exc)})
            continue
        trials.append(t)
        segments.extend(s)
        chunks.extend(c)

    aw_fin: dict[str, Any] | None = None
    if aw_fma is not None:
        aw_fin = finalize_aw_fma_dataset(
            chunks,
            fit_baselines=aw_fma.fit_baselines,
            min_stratum_n=aw_fma.min_stratum_n,
            split_seed=aw_fma.split_seed,
            weight_beta=aw_fma.weight_beta,
            weight_clip=aw_fma.weight_clip,
            max_weight=aw_fma.max_weight,
        )
    else:
        finalize_aw_fma_dataset(chunks, fit_baselines=False)

    validation = validate_action_chunk_rows(chunks)
    report = build_report(trials, segments, chunks, failures, validation, aw_fma_options=aw_fma)
    return CompileResult(
        trials=trials,
        segments=segments,
        action_chunks=chunks,
        report=report,
        aw_fma_options=aw_fma,
        aw_fma_finalization=aw_fin,
    )


def validate_action_chunk_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    for idx, row in enumerate(rows):
        rid = row.get("row_id")
        if rid in ids:
            errors.append(f"duplicate_row_id:{rid}")
        ids.add(str(rid))
        if row.get("schema") != ACTION_CHUNK_SCHEMA_VERSION:
            errors.append(f"bad_schema:{idx}")
        if row.get("semantic_sidecar", {}).get("vlm_authority_used") is not False:
            errors.append(f"vlm_authority_used_not_false:{rid}")
        conv = row.get("action_convention") or {}
        if conv.get("horizon") != EXPECTED_HORIZON or conv.get("external_action_dim") != EXPECTED_ACTION_DIM:
            errors.append(f"bad_action_convention:{rid}")
        actions = row.get("actions") or {}
        raw = actions.get("raw_action_chunk")
        if raw is not None and not _is_chunk(raw):
            errors.append(f"raw_action_chunk_bad_shape:{rid}")
        mask = row.get("masks", {}).get("policy_loss_mask")
        if not (
            isinstance(mask, list)
            and len(mask) == EXPECTED_HORIZON
            and all(isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= 0 for v in mask)
        ):
            errors.append(f"bad_policy_loss_mask:{rid}")
        weight = row.get("returns", {}).get("rank_weight_unvalidated")
        if not isinstance(weight, (int, float)) or float(weight) < 0:
            errors.append(f"negative_or_missing_weight:{rid}")
        if float(weight or 0) > 0 and row.get("eligibility", {}).get("quarantined"):
            errors.append(f"positive_weight_quarantined:{rid}")
        if row.get("sample_role") == "model_failure" and float(weight or 0) > 0:
            errors.append(f"model_failure_positive_weight:{rid}")
    if not rows:
        warnings.append("no_action_chunk_rows")
    usable = sum(bool(r.get("eligibility", {}).get("usable_for_rw_fma_dry_run")) for r in rows)
    if rows and usable == 0:
        warnings.append("no_usable_rw_fma_rows")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "n_rows": len(rows), "n_usable_rw_fma_dry_run": usable}


def _build_aw_fma_report(
    chunks: Sequence[Mapping[str, Any]],
    aw_fma_options: Any | None,
) -> dict[str, Any]:
    """Build the AW-FMA sub-report for validation_report.json."""

    def _dist(values: Sequence[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "min": None, "p01": None, "p25": None, "p50": None, "p75": None, "p99": None, "max": None, "mean": None}
        vals = sorted(float(v) for v in values)
        n = len(vals)
        def pct(q: float) -> float:
            idx = min(n - 1, max(0, round(q * (n - 1))))
            return round(vals[idx], 6)
        return {"n": n, "min": round(vals[0], 6), "p01": pct(0.01), "p25": pct(0.25), "p50": pct(0.50), "p75": pct(0.75), "p99": pct(0.99), "max": round(vals[-1], 6), "mean": round(sum(vals) / n, 6)}

    enabled = aw_fma_options is not None
    fit = bool(getattr(aw_fma_options, "fit_baselines", False)) if aw_fma_options else False

    split_dist = dict(Counter(str(r.get("split") or "None") for r in chunks))
    reward_vals = [float(r.get("returns", {}).get("aw_fma_reward", 0)) for r in chunks]
    train_rewards = [float(r["returns"]["aw_fma_reward"]) for r in chunks if r.get("split") == "train"]
    val_rewards = [float(r["returns"]["aw_fma_reward"]) for r in chunks if r.get("split") == "val"]

    reward_by_role: dict[str, dict[str, Any]] = {}
    role_groups: dict[str, list[float]] = defaultdict(list)
    for r in chunks:
        role_groups[str(r.get("sample_role"))].append(float(r.get("returns", {}).get("aw_fma_reward", 0)))
    for role, vals in sorted(role_groups.items()):
        reward_by_role[role] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 6) if vals else 0.0, "p50": round(sorted(vals)[len(vals) // 2], 6) if vals else None}

    all_strata: dict[str, list[int]] = defaultdict(list)
    train_eligible_strata: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(chunks):
        key = r.get("returns", {}).get("aw_fma_stratum_key")
        if key:
            all_strata[key].append(idx)
            if r.get("returns", {}).get("aw_fma_baseline_eligible") and r.get("split") == "train":
                train_eligible_strata[key].append(idx)
    all_stratum_sizes = [len(indices) for indices in all_strata.values()]
    train_stratum_sizes = [len(indices) for indices in train_eligible_strata.values()]
    min_n = int(getattr(aw_fma_options, "min_stratum_n", AW_FMA_MIN_STRATUM_N)) if aw_fma_options else AW_FMA_MIN_STRATUM_N
    n_valid_strata = sum(1 for s in train_stratum_sizes if s >= min_n)
    n_too_small = sum(1 for s in train_stratum_sizes if s < min_n)
    baseline_vals = [float(r["returns"]["aw_fma_baseline"]) for r in chunks if r.get("returns", {}).get("aw_fma_baseline") is not None]

    adv_valid = [r for r in chunks if r.get("returns", {}).get("aw_fma_advantage_valid") is True]
    adv_invalid = [r for r in chunks if r.get("returns", {}).get("aw_fma_advantage_valid") is not True]
    adv_raw = [float(r["returns"]["aw_fma_advantage"]) for r in adv_valid if r["returns"].get("aw_fma_advantage") is not None]
    adv_norm = [float(r["returns"]["aw_fma_advantage_normalized"]) for r in adv_valid if r["returns"].get("aw_fma_advantage_normalized") is not None]

    loss_eligible = [r for r in chunks if r.get("returns", {}).get("aw_fma_loss_eligible") is True]
    not_eligible = [r for r in chunks if r.get("returns", {}).get("aw_fma_loss_eligible") is not True]
    eligible_by_role = dict(Counter(str(r.get("sample_role")) for r in loss_eligible))
    loss_weights = [float(r["returns"]["aw_fma_weight"]) for r in loss_eligible if r["returns"].get("aw_fma_weight", 0) > 0]

    heldout_loss = sum(1 for r in chunks if r.get("split") == "heldout" and r.get("returns", {}).get("aw_fma_loss_eligible") is True)
    no_split_loss = sum(1 for r in chunks if r.get("split") is None and r.get("returns", {}).get("aw_fma_loss_eligible") is True)
    model_failure_loss = sum(1 for r in chunks if r.get("sample_role") == "model_failure" and r.get("returns", {}).get("aw_fma_loss_eligible") is True)
    quarantined_loss = sum(1 for r in chunks if r.get("eligibility", {}).get("quarantined") and r.get("returns", {}).get("aw_fma_loss_eligible") is True)
    vlm_authority_violations = sum(
        1 for r in chunks
        if r.get("returns", {}).get("aw_fma_vlm_authority_used") is not False
        or r.get("semantic_sidecar", {}).get("vlm_authority_used") is not False
    )

    violations_clean = (heldout_loss == 0 and no_split_loss == 0 and model_failure_loss == 0 and quarantined_loss == 0 and vlm_authority_violations == 0)

    if not fit:
        rec = "AW_FMA_AUDIT_ONLY"
    elif violations_clean and len(loss_eligible) > 0:
        rec = "READY_FOR_ONE_BATCH_AW_FMA_SMOKE"
    else:
        rec = "NOT_READY_FOR_TRAINING"

    comp_r_prox = []
    comp_r_prog = []
    comp_r_term: Counter = Counter()
    comp_r_stop: Counter = Counter()
    comp_r_rank = []
    for r in chunks:
        comps = r.get("returns", {}).get("aw_fma_reward_components") or {}
        if comps.get("r_proximity") is not None:
            comp_r_prox.append(float(comps["r_proximity"]))
        if comps.get("r_progress_signed01") is not None:
            comp_r_prog.append(float(comps["r_progress_signed01"]))
        if comps.get("r_terminal") is not None:
            comp_r_term[float(comps["r_terminal"])] += 1
        if comps.get("r_stop_handoff") is not None:
            comp_r_stop[float(comps["r_stop_handoff"])] += 1
        if comps.get("r_operator_rank") is not None:
            comp_r_rank.append(float(comps["r_operator_rank"]))

    return {
        "enabled": enabled,
        "version": AW_FMA_REWARD_VERSION,
        "split_distribution": split_dist,
        "reward_distribution": {
            "all": _dist(reward_vals),
            "train": _dist(train_rewards),
            "val": _dist(val_rewards),
        },
        "reward_by_sample_role": reward_by_role,
        "baseline_stats": {
            "n_strata": len(all_strata),
            "n_strata_valid": n_valid_strata,
            "n_strata_too_small": n_too_small,
            "strata_sizes": _dist(all_stratum_sizes) if all_stratum_sizes else _dist([]),
            "baseline_values": _dist(baseline_vals),
            "all_rows_strata": {
                "n_strata": len(all_strata),
                "strata_sizes": _dist(all_stratum_sizes) if all_stratum_sizes else _dist([]),
            },
            "train_baseline_eligible_strata": {
                "n_strata": len(train_eligible_strata),
                "n_valid": n_valid_strata,
                "n_too_small": n_too_small,
                "strata_sizes": _dist(train_stratum_sizes) if train_stratum_sizes else _dist([]),
            },
        },
        "advantage_distribution": {
            "n_valid": len(adv_valid),
            "n_invalid": len(adv_invalid),
            "advantage_raw": _dist(adv_raw),
            "advantage_norm": _dist(adv_norm),
        },
        "loss_eligibility": {
            "n_loss_eligible": len(loss_eligible),
            "n_not_eligible": len(not_eligible),
            "eligible_by_role": eligible_by_role,
            "weight_distribution": _dist(loss_weights),
        },
        "violation_counts": {
            "heldout_loss_eligible_rows": heldout_loss,
            "no_split_loss_eligible_rows": no_split_loss,
            "model_failure_loss_eligible_rows": model_failure_loss,
            "quarantined_loss_eligible_rows": quarantined_loss,
            "vlm_authority_violation_rows": vlm_authority_violations,
        },
        "component_histograms": {
            "r_proximity": _dist(comp_r_prox),
            "r_progress_signed01": _dist(comp_r_prog),
            "r_terminal": dict(comp_r_term),
            "r_stop_handoff": dict(comp_r_stop),
            "r_operator_rank": _dist(comp_r_rank),
        },
        "recommendation": rec,
        "authority": {
            "vlm_authority_used_for_aw_fma_scalars": False,
            "reward_authority": "robot_trace_target_region_operator_label",
            "training_launch_authorized": False,
        },
    }


def build_report(
    trials: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]],
    validation: Mapping[str, Any],
    *,
    aw_fma_options: Any | None = None,
) -> dict[str, Any]:
    def hist(key: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return dict(Counter(str(r.get(key)) for r in rows))

    def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"p01": None, "p25": None, "p50": None, "p75": None, "p99": None}
        vals = sorted(float(v) for v in values)
        def pct(q: float) -> float:
            idx = min(len(vals) - 1, max(0, round(q * (len(vals) - 1))))
            return round(vals[idx], 6)
        return {"p01": pct(0.01), "p25": pct(0.25), "p50": pct(0.50), "p75": pct(0.75), "p99": pct(0.99)}

    weights = [float((r.get("returns") or {}).get("rank_weight_unvalidated") or 0.0) for r in chunks]
    positive_weights = [w for w in weights if w > 0]
    weights_by_role: dict[str, list[float]] = defaultdict(list)
    for r, w in zip(chunks, weights):
        weights_by_role[str(r.get("sample_role"))].append(w)
    nontrivial_masks = 0
    for r in chunks:
        mask = ((r.get("masks") or {}).get("policy_loss_mask") or [])
        if isinstance(mask, list) and any(isinstance(v, (int, float)) and float(v) not in {0.0, 1.0} for v in mask):
            nontrivial_masks += 1

    report = {
        "schema": ACTION_CHUNK_SCHEMA_VERSION,
        "recommendation": "READY_FOR_SMALL_RW_FMA_DRY_RUN" if validation.get("ok") and validation.get("n_usable_rw_fma_dry_run", 0) > 0 else "NOT_READY_FOR_TRAINING",
        "counts": {
            "trials": len(trials),
            "segments": len(segments),
            "action_chunks": len(chunks),
            "usable_for_rw_fma_dry_run": validation.get("n_usable_rw_fma_dry_run", 0),
            "compile_failures": len(failures),
        },
        "histograms": {
            "trial_block": hist("block", trials),
            "trial_component": hist("component", trials),
            "trial_label": hist("label", trials),
            "chunk_policy_role": hist("policy_role", chunks),
            "chunk_sample_role": hist("sample_role", chunks),
            "chunk_phase": hist("phase", chunks),
            "chunk_control_source": hist("control_source", chunks),
        },
        "rw_fma_weight_distribution": {
            "n_positive": len(positive_weights),
            "unique_values_count": len(set(round(w, 6) for w in positive_weights)),
            **_percentiles(positive_weights),
            "mean_per_role": {
                role: round(sum(vals) / len(vals), 6) if vals else 0.0
                for role, vals in sorted(weights_by_role.items())
            },
        },
        "policy_loss_mask_distribution": {
            "n_nontrivial_rows": nontrivial_masks,
        },
        "quarantine_reasons": dict(Counter(e for r in chunks for e in r.get("eligibility", {}).get("quarantine_reasons", []))),
        "validation": validation,
        "failures": list(failures[:25]),
        "authority": {
            "vlm_authority_used": False,
            "reward_authority": "robot_trace_target_region_operator_label_scaffold_only",
            "training_launch_authorized": False,
            "note": "This compiler creates read-only dry-run rows. Actual training still requires OpenPI dataloader/loss gate and calibration review.",
        },
    }
    report["aw_fma"] = _build_aw_fma_report(chunks, aw_fma_options)
    return report


def write_dataset(result: CompileResult, output_dir: Path, *, force: bool = False) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output dir is non-empty: {output_dir}; pass force=True")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "trials.jsonl", result.trials)
    _write_jsonl(output_dir / "segments.jsonl", result.segments)
    _write_jsonl(output_dir / "action_chunks.jsonl", result.action_chunks)
    (output_dir / "schema.json").write_text(json.dumps({
        "schema": ACTION_CHUNK_SCHEMA_VERSION,
        "trial_schema": TRIAL_SCHEMA_VERSION,
        "segment_schema": SEGMENT_SCHEMA_VERSION,
        "expected_horizon": EXPECTED_HORIZON,
        "expected_external_action_dim": EXPECTED_ACTION_DIM,
        "method": "JAX/OpenPI RW-FMA dry-run candidate dataset",
    }, indent=2, sort_keys=True))
    (output_dir / "validation_report.json").write_text(json.dumps(result.report, indent=2, sort_keys=True))
    (output_dir / "README.md").write_text(_readme(result.report))

    if result.aw_fma_options is not None:
        _write_aw_fma_sidecars(result, output_dir)


def _write_aw_fma_sidecars(result: CompileResult, output_dir: Path) -> None:
    aw_dir = output_dir / "aw_fma"
    aw_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(aw_dir / "aw_fma_action_chunks.jsonl", result.action_chunks)

    opts = result.aw_fma_options or AwFmaOptions()
    config_splits = (result.aw_fma_finalization or {}).get("config_splits", {})

    chunks_sha = hashlib.sha256()
    for row in result.action_chunks:
        chunks_sha.update(json.dumps(row, sort_keys=True).encode())

    rw_sha = hashlib.sha256()
    for row in result.action_chunks:
        rw_row = {k: v for k, v in row.items() if k != "split"}
        rw_sha.update(json.dumps(rw_row, sort_keys=True).encode())

    n_loss = sum(1 for r in result.action_chunks if r.get("returns", {}).get("aw_fma_loss_eligible") is True)
    n_adv_valid = sum(1 for r in result.action_chunks if r.get("returns", {}).get("aw_fma_advantage_valid") is True)
    n_train = sum(1 for r in result.action_chunks if r.get("split") == "train")
    n_val = sum(1 for r in result.action_chunks if r.get("split") == "val")
    n_heldout = sum(1 for r in result.action_chunks if r.get("split") == "heldout")

    aw_fma_report = result.report.get("aw_fma", {})
    rec = aw_fma_report.get("recommendation", "AW_FMA_AUDIT_ONLY")

    manifest = {
        "method": "AW-FMA candidate artifact",
        "training_launch_authorized": False,
        "action_chunk_schema": ACTION_CHUNK_SCHEMA_VERSION,
        "aw_fma_reward_version": AW_FMA_REWARD_VERSION,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cli_flags": {
            "aw_fma_fit_baselines": opts.fit_baselines,
            "aw_fma_min_stratum_n": opts.min_stratum_n,
            "aw_fma_split_seed": opts.split_seed,
            "aw_fma_weight_beta": opts.weight_beta,
            "aw_fma_weight_clip": opts.weight_clip,
            "aw_fma_max_weight": opts.max_weight,
        },
        "counts": {
            "total_rows": len(result.action_chunks),
            "loss_eligible": n_loss,
            "advantage_valid": n_adv_valid,
            "train_split": n_train,
            "val_split": n_val,
            "heldout_split": n_heldout,
        },
        "split_unit": "config_id",
        "baseline_group": "config_id|block|component",
        "weight_rule": "exp(beta * clamp(advantage_normalized, -clip, clip)); capped",
        "authority": {
            "vlm_authority_used_for_aw_fma_scalars": False,
            "reward_authority": "robot_trace_target_region_operator_label",
            "training_launch_authorized": False,
        },
        "violation_counts": aw_fma_report.get("violation_counts", {}),
        "recommendation": rec,
        "rw_fma_action_chunks_sha256": "sha256:" + rw_sha.hexdigest(),
        "aw_fma_action_chunks_sha256": "sha256:" + chunks_sha.hexdigest(),
    }
    (aw_dir / "aw_fma_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    split_manifest = {
        "split_seed": opts.split_seed,
        "split_unit": "config_id",
        "config_splits": config_splits,
        "counts": {"train": n_train, "val": n_val, "heldout": n_heldout},
    }
    (aw_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True))

    histograms = aw_fma_report.get("component_histograms", {})
    histograms["split_distribution"] = aw_fma_report.get("split_distribution", {})
    histograms["reward_distribution"] = aw_fma_report.get("reward_distribution", {})
    histograms["advantage_distribution"] = aw_fma_report.get("advantage_distribution", {})
    histograms["loss_eligibility"] = aw_fma_report.get("loss_eligibility", {})
    (aw_dir / "aw_fma_histograms.json").write_text(json.dumps(histograms, indent=2, sort_keys=True))


def _readme(report: Mapping[str, Any]) -> str:
    counts = report.get("counts", {})
    return f"""# TRACE-VLA RL action-chunk dry-run dataset

Schema: `{ACTION_CHUNK_SCHEMA_VERSION}`

This directory is a read-only offline compiler product for TRACE-VLA TRACE-VLA
Pi0.5/OpenPI RW-FMA preparation. It is not a training run and does not authorize
live robot control.

Counts:

- trials: {counts.get('trials')}
- segments: {counts.get('segments')}
- action chunks: {counts.get('action_chunks')}
- usable for RW-FMA dry run: {counts.get('usable_for_rw_fma_dry_run')}
- compile failures: {counts.get('compile_failures')}

Files to review in order:

1. `validation_report.json`
2. `schema.json`
3. `trials.jsonl`
4. `segments.jsonl`
5. `action_chunks.jsonl`

Authority rule: VLM-B fields are sidecars only. Numeric reward/weight fields here
are unvalidated scaffolds from robot/operator/target-region evidence and must be
reviewed before any real training launch.
"""
