"""Offline validation scoring, blocker, audits, and candidate selection.

Implements:
- eq:row_val — v_j = B_j * clip(sum lambda_m e_m / (sum lambda_m + eps), 0, 1)
  with absent submetrics omitted per row and the remaining coefficients
  renormalized (this is how non-phase-ending rows drop the stop submetric).
- eq:app_bj — blocker B_j over wrong-target, non-target exclusion, invalid
  orientation, and unsafe events; terminal-region failure is scored, never
  blocked.
- eq:app_vhat / eq:app_reg — geometric row score and the no-regression
  indicator against the reference policy.
- eq:j_val — component-balanced validation score over positive-scoring roles
  {clean, correction, auto_success, partial}; failure/excluded rows are
  audit-only.
- eq:app_feasibility — F(eta) = A_audit * A_wrong * A_safe * A_reg.
- eq:selection — argmax J_val among audited candidates, with fallback to the
  reference policy when no candidate is audited (select_candidate returns
  None in that case).

This module consumes already-computed submetric values and blocker flags from
an evaluator (the open-loop chunk rollout/geometry evaluator is external); it
owns the scoring and selection algebra only.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pacer_framework.roles import AUDIT_ONLY_ROLES, POSITIVE_IMITATION_ROLES, normalize_role

EPS = 1e-6

# lambda_m, nonnegative, fixed before ranking, shared by every candidate.
DEFAULT_SUBMETRIC_WEIGHTS: dict[str, float] = {
    "progress": 0.25,
    "proximity": 0.25,
    "terminal": 0.20,
    "direction": 0.15,
    "stop": 0.15,
    "align": 0.15,
    "no_regression": 0.10,
}

GEOMETRIC_SUBMETRICS = ("progress", "proximity", "terminal", "direction")

BLOCKER_FLAGS = ("wrong_target", "non_target_exclusion", "invalid_orientation", "unsafe")
WRONG_TARGET_REASONS = frozenset({"wrong_target", "non_target_exclusion", "invalid_orientation"})


@dataclass(frozen=True)
class RowScore:
    row_id: str
    component: str
    role: str
    positive_scoring: bool
    blocker_passed: bool
    score: float
    used_submetrics: tuple[str, ...] = field(default_factory=tuple)
    audit_reasons: tuple[str, ...] = field(default_factory=tuple)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def blocker(row: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """B_j (eq:app_bj) from precomputed evaluator flags."""
    reasons = tuple(flag for flag in BLOCKER_FLAGS if bool(row.get(flag)))
    return len(reasons) == 0, reasons


def _weighted_mean(
    submetrics: Mapping[str, Any],
    weights: Mapping[str, float],
) -> tuple[float, tuple[str, ...]]:
    numerator = 0.0
    denominator = 0.0
    used: list[str] = []
    for name, weight in weights.items():
        if not _finite(weight) or float(weight) < 0.0:
            continue
        value = submetrics.get(name)
        if not _finite(value):
            continue
        numerator += float(weight) * _clip01(float(value))
        denominator += float(weight)
        used.append(name)
    if denominator <= 0.0:
        return 0.0, tuple(used)
    return _clip01(numerator / (denominator + EPS)), tuple(used)


def row_score(
    row: Mapping[str, Any],
    *,
    submetric_weights: Mapping[str, float] | None = None,
) -> RowScore:
    """Row-level validation score v_j (eq:row_val) with role and blocker audit."""
    weights = dict(DEFAULT_SUBMETRIC_WEIGHTS if submetric_weights is None else submetric_weights)
    role = normalize_role(row.get("role") or row.get("sample_role"), strict=False)
    component = str(row.get("component") or "unknown")
    row_id = str(row.get("row_id") or row.get("id") or "unknown")

    passed, reasons = blocker(row)
    positive = role in POSITIVE_IMITATION_ROLES
    audit_reasons = list(reasons)
    if role in AUDIT_ONLY_ROLES:
        audit_reasons.append("audit_only_role")
    elif not positive:
        audit_reasons.append("non_positive_scoring_role")

    submetrics = row.get("submetrics", {})
    if not isinstance(submetrics, Mapping):
        submetrics = {}
    value, used = _weighted_mean(submetrics, weights)
    score = value if (positive and passed and used) else 0.0

    return RowScore(
        row_id=row_id,
        component=component,
        role=role,
        positive_scoring=positive,
        blocker_passed=passed,
        score=score,
        used_submetrics=used,
        audit_reasons=tuple(audit_reasons),
    )


def geometric_row_score(
    submetrics: Mapping[str, Any],
    *,
    submetric_weights: Mapping[str, float] | None = None,
) -> float:
    """v-hat (eq:app_vhat) — weighted mean over the geometric submetrics only."""
    weights = dict(DEFAULT_SUBMETRIC_WEIGHTS if submetric_weights is None else submetric_weights)
    geometric = {k: weights[k] for k in GEOMETRIC_SUBMETRICS if k in weights}
    value, _used = _weighted_mean(submetrics, geometric)
    return value


def no_regression(v_hat_candidate: float, v_hat_reference: float, *, delta_reg: float) -> float:
    """eq:app_reg — indicator that the candidate row score has not regressed."""
    return 1.0 if float(v_hat_candidate) >= float(v_hat_reference) - float(delta_reg) else 0.0


def component_balanced_j_val(
    rows: Sequence[Mapping[str, Any]],
    *,
    submetric_weights: Mapping[str, float] | None = None,
) -> tuple[float, list[RowScore], dict[str, Any]]:
    """J_val (eq:j_val) — mean over components of per-component mean row scores."""
    scores = [row_score(row, submetric_weights=submetric_weights) for row in rows]
    by_component: dict[str, list[float]] = defaultdict(list)
    audit_counts: dict[str, int] = defaultdict(int)
    for s in scores:
        if s.positive_scoring:
            by_component[s.component].append(s.score)
        for reason in s.audit_reasons:
            audit_counts[reason] += 1
    component_scores = {
        component: (sum(values) / len(values) if values else 0.0)
        for component, values in sorted(by_component.items())
    }
    j_val = sum(component_scores.values()) / len(component_scores) if component_scores else 0.0
    manifest = {
        "num_rows": len(rows),
        "num_positive_scoring_rows": sum(1 for s in scores if s.positive_scoring),
        "component_scores": component_scores,
        "audit_counts": dict(sorted(audit_counts.items())),
    }
    return _clip01(j_val), scores, manifest


def candidate_feasibility(
    *,
    weight_manifest: Mapping[str, Any],
    validation_scores: Sequence[RowScore],
    w_max: float,
    candidate_component_scores: Mapping[str, float] | None = None,
    reference_component_scores: Mapping[str, float] | None = None,
    delta_reg: float = 0.0,
    n_eff_min: float = 1.0,
) -> tuple[bool, dict[str, bool]]:
    """F(eta) = A_audit * A_wrong * A_safe * A_reg (eq:app_feasibility).

    A_audit reads the pre-training weight manifest (forbidden positive-weight
    count, finite loss denominator, max weight within the clip, non-collapsed
    n_eff). A_wrong / A_safe demand zero wrong-target / unsafe leakage over the
    validation rows. A_reg checks every reported component against the
    reference policy under the fixed margin; it passes vacuously when no
    reference scores are supplied.
    """
    loss_normalizer = float(weight_manifest.get("loss_normalizer", 0.0))
    checks: dict[str, bool] = {
        "A_audit_forbidden_zero": int(weight_manifest.get("forbidden_positive_weight_rows", 1)) == 0,
        "A_audit_finite_normalizer": math.isfinite(loss_normalizer) and loss_normalizer > 0.0,
        "A_audit_weight_within_clip": float(weight_manifest.get("weight_max", float("inf"))) <= float(w_max) + 1e-9,
        "A_audit_n_eff": float(weight_manifest.get("n_eff_horizons", 0.0)) >= float(n_eff_min),
        "A_wrong_zero_leakage": not any(
            set(s.audit_reasons) & WRONG_TARGET_REASONS for s in validation_scores
        ),
        "A_safe_zero_unsafe": not any("unsafe" in s.audit_reasons for s in validation_scores),
    }
    if reference_component_scores and candidate_component_scores is not None:
        checks["A_reg_no_regression"] = all(
            float(candidate_component_scores.get(component, 0.0)) >= float(ref) - float(delta_reg)
            for component, ref in reference_component_scores.items()
        )
    else:
        checks["A_reg_no_regression"] = True
    return all(checks.values()), checks


def select_candidate(
    j_vals: Mapping[str, float],
    feasible: Mapping[str, bool],
) -> str | None:
    """eq:selection — argmax J_val among audited candidates; None => fall back
    to the reference policy pi_theta0."""
    audited = {eta_id: score for eta_id, score in j_vals.items() if feasible.get(eta_id)}
    if not audited:
        return None
    return max(sorted(audited), key=lambda eta_id: audited[eta_id])


def evaluate_candidate(
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    weight_manifest: Mapping[str, Any],
    w_max: float,
    submetric_weights: Mapping[str, float] | None = None,
    reference_component_scores: Mapping[str, float] | None = None,
    delta_reg: float = 0.0,
    n_eff_min: float = 1.0,
) -> dict[str, Any]:
    """One-call post-training evaluation of a trained candidate.

    Computes J_val (eq:j_val) over the candidate's scored validation rows and
    the feasibility predicate F(eta) (eq:app_feasibility) against its
    pre-training weight manifest. Feed the per-candidate results to
    ``select_candidate`` / ``reporting.candidate_ranking_table``.
    """
    j_val, scores, manifest = component_balanced_j_val(validation_rows, submetric_weights=submetric_weights)
    feasible, checks = candidate_feasibility(
        weight_manifest=weight_manifest,
        validation_scores=scores,
        w_max=w_max,
        candidate_component_scores=manifest["component_scores"],
        reference_component_scores=reference_component_scores,
        delta_reg=delta_reg,
        n_eff_min=n_eff_min,
    )
    return {
        "j_val": j_val,
        "feasible": feasible,
        "checks": checks,
        "component_scores": manifest["component_scores"],
        "row_scores": scores,
        "manifest": manifest,
    }
