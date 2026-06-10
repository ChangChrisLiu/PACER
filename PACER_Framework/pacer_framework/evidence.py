"""Bounded evidence vector e_k(i) over K = {prog, prox, term, dir, stop, op, prov}.

Implements app:features of the paper:
- eq:app_dist distances d_0, d_end, d_min and displacement Delta p over the
  valid mask prefix;
- eq:app_term terminal-region membership Z_term (position-only spherical form
  d_end <= r_target, with optional explicit region-membership override);
- eq:app_geom geometric features, each multiplied by the target-consistency
  indicator T_i;
- psi_op closed-vocabulary operator evidence (0.8 / 0.6 / 0.2 / 0; clean and
  correction rows carry the constant 1, which stratum centering removes);
- P_i provenance check (manifest, timing alignment, component identity, action
  mask all valid);
- S_i stop evidence (phase-ending row terminating with the expected stop or
  handoff event inside the declared terminal region);
- component-dependent tolerances r_target (0.005 m CPU, 0.010 m otherwise) and
  d_ref = 2 r_target (app:components).

Row schema (all blocks optional unless noted; see README "Row schema"):
  component, phase, role|sample_role, mask (per-horizon 0/1 list),
  evidence  — precomputed e_k dict; T_i is still applied to the geometric four,
  geometry  — tcp_positions [[x,y,z],...], target_point [x,y,z], r_target,
              d_ref, terminal_region_member, orientation_ok, target_id,
              or summary fields d0 / d_end / d_min / direction_cosine,
  stop      — {phase_ending, stop_event, inside_terminal},
  operator_label — closed-vocabulary label for autonomous rows,
  provenance — {manifest_valid, timing_aligned, component_identity_valid,
                action_mask_valid} (absent block = upstream-trusted = valid),
  target_meta — {component_id, phase_id, target_id, declared_target_id,
                 observed_target_id, geometry_record_present, consistency,
                 consistent}.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from pacer_framework.eta import EVIDENCE_KEYS
from pacer_framework.roles import normalize_role

EPS = 1e-6

# app:components — component-dependent hard tolerance and proximity normalizer.
R_TARGET_CPU_M = 0.005
R_TARGET_DEFAULT_M = 0.010

# psi_op closed-vocabulary lookup (app:features).
PSI_OP = {
    "success": 0.8,
    "near_but_not_accurate": 0.6,
    "wrong_orientation_or_wrong_location": 0.2,
    "totally_off_wrong_region_or_target": 0.0,
    "operator_uncertain_exclude": 0.0,
}

# Stop/handoff events accepted by S_i.
EXPECTED_STOP_EVENTS = frozenset({"model_stop_token", "handoff", "manual_stop_near_target"})

PROVENANCE_FLAGS = (
    "manifest_valid",
    "timing_aligned",
    "component_identity_valid",
    "action_mask_valid",
)

GEOMETRIC_KEYS = ("prog", "prox", "term", "dir")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def r_target_for(component: str | None) -> float:
    """Component-dependent position tolerance r_target (app:components)."""
    if str(component or "").strip().lower() == "cpu":
        return R_TARGET_CPU_M
    return R_TARGET_DEFAULT_M


def d_ref_for(component: str | None, r_target: float | None = None) -> float:
    """Proximity normalizer d_ref(c) = 2 r_target(c) (app:components)."""
    r = float(r_target) if r_target is not None else r_target_for(component)
    return 2.0 * r


def psi_op(role: str, operator_label: str | None) -> float:
    """Operator evidence e_op. Clean/correction rows carry the constant 1."""
    if role in {"clean", "correction"}:
        return 1.0
    return float(PSI_OP.get(str(operator_label or ""), 0.0))


def provenance_check(provenance: Mapping[str, Any] | None) -> float:
    """P_i = 1 when manifest, timing, component identity, and action mask are valid.

    An absent provenance block means the row came from a trusted upstream
    pipeline and is treated as valid; any explicit False flag zeroes P_i.
    """
    if provenance is None:
        return 1.0
    if not isinstance(provenance, Mapping):
        return 0.0
    return 1.0 if all(bool(provenance.get(k, True)) for k in PROVENANCE_FLAGS) else 0.0


def target_consistency(row: Mapping[str, Any]) -> tuple[float, list[str]]:
    """T_i in eq:app_geom — paper-strict target-metadata consistency.

    T_i = 0 when the component id, phase id, target id, or target-geometry
    record is missing or inconsistent with the declared task state. The
    declared target is never reassigned from the executed endpoint.
    """
    meta = row.get("target_meta") or {}
    geometry = row.get("geometry") or {}
    if not isinstance(meta, Mapping):
        meta = {}
    if not isinstance(geometry, Mapping):
        geometry = {}
    reasons: list[str] = []

    component = meta.get("component_id", row.get("component"))
    phase = meta.get("phase_id", row.get("phase"))
    target_id = meta.get("target_id", geometry.get("target_id"))
    geometry_present = bool(
        meta.get("geometry_record_present")
        or geometry.get("target_point") is not None
        or geometry.get("d_end") is not None
        or geometry.get("terminal_region_member") is not None
    )

    if component in (None, ""):
        reasons.append("missing_component_id")
    if phase in (None, ""):
        reasons.append("missing_phase_id")
    if target_id in (None, ""):
        reasons.append("missing_target_id")
    if not geometry_present:
        reasons.append("missing_target_geometry_record")

    declared = meta.get("declared_target_id")
    if declared is not None and target_id is not None and str(declared) != str(target_id):
        reasons.append("declared_target_id_mismatch")
    observed = meta.get("observed_target_id")
    anchor = declared if declared is not None else target_id
    if observed is not None and anchor is not None and str(observed) != str(anchor):
        reasons.append("observed_target_id_mismatch")
    if str(meta.get("consistency") or "") in {"wrong_target", "wrong_region"}:
        reasons.append("target_consistency_wrong")
    if meta.get("consistent") is False:
        reasons.append("target_marked_inconsistent")

    return (1.0 if not reasons else 0.0), reasons


def stop_evidence(stop: Mapping[str, Any] | None) -> float:
    """S_i — phase-ending row ends with expected stop/handoff inside terminal region.

    Non-phase-ending rows get the constant 0; within their strata this is
    removed by centering exactly as the paper states.
    """
    if not isinstance(stop, Mapping) or not bool(stop.get("phase_ending")):
        return 0.0
    event_ok = str(stop.get("stop_event") or "") in EXPECTED_STOP_EVENTS
    inside = bool(stop.get("inside_terminal"))
    return 1.0 if (event_ok and inside) else 0.0


def _valid_position_prefix(positions: Sequence[Sequence[float]], mask: Sequence[Any] | None) -> list[Sequence[float]]:
    """Restrict TCP positions to the valid prefix p_0..p_{h_end} (eq:app_dist).

    Positions may include the start point (len == len(mask) + 1) or be aligned
    one-to-one with horizons (len == len(mask)). Without a mask the whole
    sequence is the valid prefix.
    """
    pts = list(positions)
    if not mask:
        return pts
    valid = [i for i, m in enumerate(mask) if _as_float(m, 0.0) > 0.0]
    if not valid:
        return []
    h_end = valid[-1]
    if len(pts) == len(mask) + 1:
        return pts[: h_end + 2]
    return pts[: h_end + 1]


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in vec))


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def geometric_distances(
    positions: Sequence[Sequence[float]],
    target_point: Sequence[float],
) -> dict[str, Any]:
    """eq:app_dist — d_0, d_end, d_min, and Delta p over the given prefix."""
    if len(positions) == 0:
        raise ValueError("geometric_distances requires at least one TCP position")
    dists = [_norm(_sub(p, target_point)) for p in positions]
    return {
        "d0": dists[0],
        "d_end": dists[-1],
        "d_min": min(dists),
        "delta_p": _sub(positions[-1], positions[0]),
        "to_target": _sub(target_point, positions[0]),
    }


def _geometric_evidence(row: Mapping[str, Any]) -> dict[str, float]:
    """Compute the four geometric features (before the T_i factor)."""
    geometry = row.get("geometry") or {}
    if not isinstance(geometry, Mapping):
        geometry = {}
    component = row.get("component")
    r_target = _as_float(geometry.get("r_target"), r_target_for(component))
    d_ref = _as_float(geometry.get("d_ref"), d_ref_for(component, r_target))

    positions = geometry.get("tcp_positions")
    target_point = geometry.get("target_point")
    if positions and target_point is not None:
        prefix = _valid_position_prefix(positions, row.get("mask"))
        if not prefix:
            return {"prog": 0.0, "prox": 0.0, "term": 0.0, "dir": 0.0}
        dist = geometric_distances(prefix, target_point)
        d0, d_end, d_min = dist["d0"], dist["d_end"], dist["d_min"]
        delta_p, to_target = dist["delta_p"], dist["to_target"]
        direction = _clip01(_dot(delta_p, to_target) / (_norm(delta_p) * _norm(to_target) + EPS))
    else:
        # Summary form: distances precomputed upstream.
        d0 = _as_float(geometry.get("d0"), float("nan"))
        d_end = _as_float(geometry.get("d_end"), float("nan"))
        d_min = _as_float(geometry.get("d_min"), d_end)
        direction = _clip01(_as_float(geometry.get("direction_cosine"), 0.0))
        if not (math.isfinite(d0) and math.isfinite(d_end)):
            return {"prog": 0.0, "prox": 0.0, "term": 0.0, "dir": 0.0}

    prog = _clip01((d0 - d_min) / (d0 + EPS))
    prox = _clip01(1.0 - d_end / d_ref)
    if geometry.get("terminal_region_member") is not None:
        z_term = 1.0 if bool(geometry.get("terminal_region_member")) else 0.0
    else:
        orientation_ok = geometry.get("orientation_ok")
        z_term = 1.0 if (d_end <= r_target and orientation_ok is not False) else 0.0
    return {"prog": prog, "prox": prox, "term": z_term, "dir": direction}


def compute_evidence(row: Mapping[str, Any]) -> dict[str, float]:
    """Bounded evidence vector e_k(i) over K with the T_i factor applied.

    Precedence: a precomputed row["evidence"] dict is used as-is (clamped);
    otherwise the geometric features come from row["geometry"], stop evidence
    from row["stop"], operator evidence from the role / operator label, and
    provenance from row["provenance"]. T_i multiplies the geometric four in
    both paths (eq:app_geom).
    """
    role = normalize_role(row.get("role") or row.get("sample_role"), strict=False)
    t_i, _t_reasons = target_consistency(row)

    pre = row.get("evidence")
    if isinstance(pre, Mapping):
        evidence = {k: _clip01(_as_float(pre.get(k), 0.0)) for k in EVIDENCE_KEYS}
    else:
        geom = _geometric_evidence(row)
        evidence = {
            "prog": geom["prog"],
            "prox": geom["prox"],
            "term": geom["term"],
            "dir": geom["dir"],
            "stop": stop_evidence(row.get("stop")),
            "op": _clip01(psi_op(role, row.get("operator_label"))),
            "prov": provenance_check(row.get("provenance")),
        }

    for key in GEOMETRIC_KEYS:
        evidence[key] = t_i * evidence[key]
    return evidence
