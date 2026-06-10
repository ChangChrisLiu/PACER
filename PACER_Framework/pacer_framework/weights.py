"""Row weights: robust centering, fallback, weight equation, loss accounting.

Implements, paper-exactly:
- eq:app_center — training-only median b_str and robust scale
  sigma_str = max(1.4826 * MAD, sigma_min), sigma_min = 1e-6. No further
  fallback scale is used (the legacy PACER package adds a standard-deviation
  fallback when the MAD collapses; this reference implementation keeps the
  declared formula).
- eq:app_fallback — stratum (c, phi, rho) backs off to (c, phi), then to the
  global eligible training pool. Strata with at least one eligible training
  row always receive their own statistics.
- eq:weight — w_i = g_i * clip(max(f_rho, gamma_rho * exp((s - b)/(tau*sigma + eps))), 0, w_max).
  The role multiplier sits OUTSIDE the exponential and the lower clip is 0
  (no w_min; the legacy package historically carried one).
- eq:n_eff — effective weighted sample count over weighted masked horizons.
- eq:loss denominator — sum of w_i * m_{i,h} (reported as loss_normalizer; the
  actual VLA action loss is computed by the external training stack).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Mapping

from pacer_framework.eta import PaperEta, process_score
from pacer_framework.evidence import compute_evidence
from pacer_framework.gate import eligibility_gate
from pacer_framework.roles import normalize_role

EPS = 1e-6
SIGMA_MIN = 1e-6
# exp(60) ~ 1.1e26 already saturates any admissible w_max; capping the exponent
# keeps math.exp finite while leaving every clipped weight identical.
_MAX_EXPONENT = 60.0

PACER_FRAMEWORK_WEIGHT_SCHEMA = "pacer_framework_weights.v1"


def robust_center_scale(scores: list[float]) -> tuple[float, float]:
    """eq:app_center — (median, max(1.4826 * MAD, sigma_min))."""
    if not scores:
        return 0.0, 1.0
    center = median(scores)
    mad = median([abs(s - center) for s in scores])
    return center, max(1.4826 * mad, SIGMA_MIN)


def stratum_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """str(i) = (component, phase, role) — paper app:center.

    The implementation field name for the phase is "phase" here; legacy
    compiler rows call it "block".
    """
    role = normalize_role(row.get("role") or row.get("sample_role"), strict=False)
    phase = row.get("phase", row.get("block"))
    return (str(row.get("component", "unknown")), str(phase if phase is not None else "unknown"), role)


def row_weight(
    eta: PaperEta,
    role: str,
    score: float,
    center: float,
    scale: float,
    gate: bool,
) -> float:
    """eq:weight — gate-dominated clipped exponential advantage weight."""
    if not gate:
        return 0.0
    exponent = (score - center) / (eta.tau * scale + EPS)
    exponent = min(exponent, _MAX_EXPONENT)
    inner = max(eta.floor(role), eta.role_multiplier(role) * math.exp(exponent))
    return max(0.0, min(float(eta.w_max), inner))


def _mask_mass(row: Mapping[str, Any]) -> float:
    mask = row.get("mask") or []
    if not isinstance(mask, (list, tuple)):
        return 0.0
    total = 0.0
    for value in mask:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v > 0.0:
            total += 1.0
    return total


def compile_weights(
    rows: list[dict[str, Any]],
    eta: PaperEta,
    *,
    eta_id: str | None = None,
    train_split: str = "train",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute paper-faithful PACER weights for every row plus an audit manifest.

    Returns deep-copied rows carrying a "pacer_framework" block with the
    evidence vector, process score, stratum key and fallback level, centering
    statistics, gate outcome, and final weight. The manifest carries the
    pre-training audit quantities of app:audits.
    """
    bound_errors = eta.validate_bounds()
    if bound_errors:
        raise ValueError(f"eta outside declared bounds: {bound_errors}")
    eta_id = eta_id or f"eta_{eta.eta_hash()}"

    prepared: list[tuple[dict[str, Any], dict[str, float], float, bool, list[str], tuple[str, str, str]]] = []
    pools_full: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    pools_phase: dict[tuple[str, str], list[float]] = defaultdict(list)
    pool_global: list[float] = []

    for row in rows:
        evidence = compute_evidence(row)
        score = process_score(eta, evidence)
        gate, reasons = eligibility_gate(row, train_split=train_split)
        key = stratum_key(row)
        prepared.append((row, evidence, score, gate, reasons, key))
        if gate:
            pools_full[key].append(score)
            pools_phase[(key[0], key[1])].append(score)
            pool_global.append(score)

    stats_full = {k: robust_center_scale(v) for k, v in pools_full.items()}
    stats_phase = {k: robust_center_scale(v) for k, v in pools_phase.items()}
    stats_global = robust_center_scale(pool_global)

    out_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    role_weight_sum: defaultdict[str, float] = defaultdict(float)
    component_weight_sum: defaultdict[str, float] = defaultdict(float)
    weighted_mass_sum = 0.0
    weighted_mass_sq_sum = 0.0
    positive_weights: list[float] = []
    forbidden_positive = 0

    for row, evidence, score, gate, reasons, key in prepared:
        if key in stats_full:
            level, (center, scale) = "stratum", stats_full[key]
        elif (key[0], key[1]) in stats_phase:
            level, (center, scale) = "component_phase", stats_phase[(key[0], key[1])]
        elif pool_global:
            level, (center, scale) = "global", stats_global
        else:
            level, (center, scale) = "empty", (0.0, 1.0)

        role = key[2]
        weight = row_weight(eta, role, score, center, scale, gate)
        if weight > 0.0 and not gate:
            forbidden_positive += 1

        mass = _mask_mass(row)
        weighted_mass_sum += weight * mass
        weighted_mass_sq_sum += (weight ** 2) * mass

        new_row = json.loads(json.dumps(row))
        new_row["pacer_framework"] = {
            "schema": PACER_FRAMEWORK_WEIGHT_SCHEMA,
            "eta_id": eta_id,
            "eta_hash": eta.eta_hash(),
            "evidence": evidence,
            "process_score": score,
            "stratum": list(key),
            "stratum_level": level,
            "center": center,
            "scale": scale,
            "gate": bool(gate),
            "gate_reasons": reasons,
            "weight": weight,
        }
        out_rows.append(new_row)

        counters["rows"] += 1
        counters[f"role:{role}"] += 1
        counters[f"component:{key[0]}"] += 1
        level_counts[level] += 1
        if gate:
            counters["eligible"] += 1
        else:
            for reason in reasons:
                counters[f"ineligible:{reason}"] += 1
        if weight > 0.0:
            positive_weights.append(weight)
            role_weight_sum[role] += weight
            component_weight_sum[key[0]] += weight

    n_eff_horizons = (weighted_mass_sum ** 2) / (weighted_mass_sq_sum + EPS)
    n_eff_rows = (
        (sum(positive_weights) ** 2) / (sum(w * w for w in positive_weights) + EPS)
        if positive_weights
        else 0.0
    )
    manifest = {
        "schema": PACER_FRAMEWORK_WEIGHT_SCHEMA,
        "eta_id": eta_id,
        "eta": eta.to_dict(),
        "eta_hash": eta.eta_hash(),
        "counts": dict(counters),
        "num_strata": len(pools_full),
        "stratum_levels": dict(level_counts),
        "weight_min": min(positive_weights) if positive_weights else 0.0,
        "weight_max": max(positive_weights) if positive_weights else 0.0,
        "weight_sum": sum(positive_weights),
        "loss_normalizer": weighted_mass_sum,
        "n_eff_horizons": n_eff_horizons,
        "n_eff_rows": n_eff_rows,
        "forbidden_positive_weight_rows": forbidden_positive,
        "role_weight_sum": dict(role_weight_sum),
        "component_weight_sum": dict(component_weight_sum),
    }
    return out_rows, manifest
