"""EEF/TCP direction cosines and open-loop validation geometry.

Implements the validation-side geometry of app:valscore and app:blocker:
- eq:app_align — reference-direction alignment between the generated
  action-chunk direction u_j and a matched reference chunk direction u_ref,
  cosine clipped to [0, 1];
- the direction submetric — target-direction cosine of the predicted chunk
  (same form as the training feature e_dir, evaluated on the candidate's
  open-loop chunk);
- ``open_loop_submetrics`` — the full bounded submetric dict for one
  validation row from predicted open-loop TCP positions (the user
  inverse-normalizes actions and integrates them from the logged start pose
  with their robot's own action convention; this module consumes the
  resulting base-frame TCP positions only);
- ``with_no_regression`` — eq:app_reg via the geometric row score eq:app_vhat;
- ``blocker_flags_from_open_loop`` — the wrong-target / non-target-exclusion /
  invalid-orientation / unsafe flags of eq:app_bj from spherical declared
  regions, a calibrated workspace box, and the stop source.

Everything is robot-agnostic: positions are [x, y, z] in the robot base frame,
regions are operator-captured points with component-dependent tolerances.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from pacer_framework.evidence import EPS, d_ref_for, geometric_distances, r_target_for
from pacer_framework.validation import geometric_row_score, no_regression

Point = Sequence[float]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in vec))


def _sub(a: Point, b: Point) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def direction_cosine01(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine of two directions, clipped to [0, 1] (shared eq:app_align core)."""
    return _clip01(_dot(u, v) / (_norm(u) * _norm(v) + EPS))


def chunk_direction(positions: Sequence[Point]) -> list[float]:
    """Net displacement direction of a TCP chunk (endpoint minus start)."""
    if len(positions) < 1:
        raise ValueError("chunk_direction requires at least one position")
    return _sub(positions[-1], positions[0])


def _common_step_deltas(a: Sequence[Point], b: Sequence[Point]) -> tuple[list[list[float]], list[list[float]]]:
    common = min(len(a), len(b))
    if common < 2:
        raise ValueError("chunk trajectory alignment requires at least two positions in each chunk")
    a_steps = [_sub(a[i + 1], a[i]) for i in range(common - 1)]
    b_steps = [_sub(b[i + 1], b[i]) for i in range(common - 1)]
    return a_steps, b_steps


def _flatten(vectors: Sequence[Sequence[float]]) -> list[float]:
    return [float(x) for vec in vectors for x in vec]


def chunk_trajectory_alignment(
    predicted_positions: Sequence[Point],
    reference_positions: Sequence[Point],
) -> float:
    """Whole action-chunk EEF/TCP trajectory alignment.

    Unlike :func:`reference_alignment`, which compares only endpoint net
    displacement, this compares the sequence of per-step TCP displacements over
    the common valid horizon. This rewards a candidate for following the same
    approach path toward the target zone, not merely ending with the same net
    direction.
    """
    pred_steps, ref_steps = _common_step_deltas(predicted_positions, reference_positions)
    return direction_cosine01(_flatten(pred_steps), _flatten(ref_steps))


def reference_alignment(
    predicted_positions: Sequence[Point],
    reference_positions: Sequence[Point],
) -> float:
    """eq:app_align — clip(u_j . u_ref / (|u_j||u_ref| + eps), 0, 1)."""
    return direction_cosine01(chunk_direction(predicted_positions), chunk_direction(reference_positions))


def target_direction_cosine(positions: Sequence[Point], target_point: Point) -> float:
    """Direction submetric — cosine between chunk displacement and start-to-target."""
    return direction_cosine01(chunk_direction(positions), _sub(target_point, positions[0]))


def open_loop_submetrics(
    predicted_positions: Sequence[Point],
    *,
    target_point: Point,
    component: str | None = None,
    r_target: float | None = None,
    d_ref: float | None = None,
    orientation_ok: bool | None = None,
    terminal_region_member: bool | None = None,
    phase_ending: bool = False,
    stop_emitted: bool | None = None,
    reference_positions: Sequence[Point] | None = None,
    reference_alignment_mode: str = "endpoint",
) -> dict[str, float]:
    """Bounded submetrics for one validation row (app:valscore).

    Geometric submetrics use the same distance and terminal-region predicates
    as the training evidence, evaluated on the candidate-predicted chunk. The
    stop submetric is included only for phase-ending rows (it is omitted and
    the remaining coefficients renormalize otherwise), and the align submetric
    only when a matched reference chunk exists. By default align preserves the
    historical endpoint/net-direction definition. Pass
    ``reference_alignment_mode="trajectory"`` to score whole EEF/TCP action-
    chunk trajectory alignment over the common valid horizon.
    """
    if len(predicted_positions) < 1:
        raise ValueError("open_loop_submetrics requires at least one predicted position")
    r_t = float(r_target) if r_target is not None else r_target_for(component)
    d_r = float(d_ref) if d_ref is not None else d_ref_for(component, r_t)

    dist = geometric_distances(predicted_positions, target_point)
    d0, d_end, d_min = dist["d0"], dist["d_end"], dist["d_min"]
    if terminal_region_member is not None:
        terminal = 1.0 if bool(terminal_region_member) else 0.0
    else:
        terminal = 1.0 if (d_end <= r_t and orientation_ok is not False) else 0.0

    submetrics: dict[str, float] = {
        "progress": _clip01((d0 - d_min) / (d0 + EPS)),
        "proximity": _clip01(1.0 - d_end / d_r),
        "terminal": terminal,
        "direction": target_direction_cosine(predicted_positions, target_point),
    }
    if phase_ending:
        submetrics["stop"] = 1.0 if stop_emitted else 0.0
    if reference_positions is not None:
        if reference_alignment_mode in {"endpoint", "endpoint_net_direction", "net_direction"}:
            submetrics["align"] = reference_alignment(predicted_positions, reference_positions)
        elif reference_alignment_mode in {"trajectory", "whole_chunk", "whole_chunk_trajectory"}:
            submetrics["align"] = chunk_trajectory_alignment(predicted_positions, reference_positions)
        else:
            raise ValueError(f"unknown reference_alignment_mode: {reference_alignment_mode}")
    return submetrics


def with_no_regression(
    candidate_submetrics: Mapping[str, float],
    reference_submetrics: Mapping[str, float],
    *,
    delta_reg: float,
    submetric_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Append the eq:app_reg no-regression submetric to a candidate's submetrics.

    v-hat (eq:app_vhat) is computed for the candidate and for the reference
    policy under the same geometric coefficients, then compared with the fixed
    margin delta_reg.
    """
    v_candidate = geometric_row_score(candidate_submetrics, submetric_weights=submetric_weights)
    v_reference = geometric_row_score(reference_submetrics, submetric_weights=submetric_weights)
    out = dict(candidate_submetrics)
    out["no_regression"] = no_regression(v_candidate, v_reference, delta_reg=delta_reg)
    return out


def _region_radius(region: Mapping[str, Any]) -> float:
    radius = region.get("radius")
    if radius is None:
        radius = r_target_for(region.get("component"))
    return float(radius)


def _inside_region(point: Point, region: Mapping[str, Any]) -> bool:
    return _norm(_sub(point, region["point"])) <= _region_radius(region)


def _outside_workspace(point: Point, bounds: Mapping[str, Sequence[float]]) -> bool:
    lo, hi = bounds["min"], bounds["max"]
    return any(float(p) < float(l) or float(p) > float(h) for p, l, h in zip(point, lo, hi))


def blocker_flags_from_open_loop(
    predicted_positions: Sequence[Point],
    *,
    target_point: Point,
    component: str | None = None,
    r_target: float | None = None,
    non_target_regions: Sequence[Mapping[str, Any]] = (),
    workspace_bounds: Mapping[str, Sequence[float]] | None = None,
    orientation_ok: bool | None = None,
    stop_source: str | None = None,
    action_bounds_ok: bool = True,
) -> dict[str, bool]:
    """eq:app_bj blocker inputs from open-loop geometry (app:blocker).

    - non_target_exclusion: any predicted TCP along the chunk enters the
      declared region of a non-target component ({"point": [x,y,z],
      "radius": r} or a component-keyed default radius);
    - wrong_target: the predicted endpoint terminates inside such a region
      while outside the declared terminal region of the active target;
    - invalid_orientation: the endpoint violates the declared orientation
      tolerance (caller evaluates the tolerance; pass orientation_ok=False);
    - unsafe: the chunk leaves the calibrated workspace box, violates action
      bounds after inverse normalization (action_bounds_ok=False), or carries
      a stop source marked unsafe.
    """
    if len(predicted_positions) < 1:
        raise ValueError("blocker_flags_from_open_loop requires at least one predicted position")
    r_t = float(r_target) if r_target is not None else r_target_for(component)
    endpoint = predicted_positions[-1]
    inside_terminal = _norm(_sub(endpoint, target_point)) <= r_t and orientation_ok is not False

    path_in_non_target = any(
        _inside_region(point, region) for point in predicted_positions for region in non_target_regions
    )
    endpoint_in_non_target = any(_inside_region(endpoint, region) for region in non_target_regions)

    unsafe = bool(
        (workspace_bounds is not None and any(_outside_workspace(p, workspace_bounds) for p in predicted_positions))
        or not action_bounds_ok
        or str(stop_source or "") == "unsafe_abort"
    )
    return {
        "wrong_target": bool(endpoint_in_non_target and not inside_terminal),
        "non_target_exclusion": bool(path_in_non_target),
        "invalid_orientation": orientation_ok is False,
        "unsafe": unsafe,
    }
