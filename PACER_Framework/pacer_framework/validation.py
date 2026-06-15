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

# Named validation/scoring profiles.  These profiles are scorer-side lambda_m
# choices over already-computed validation submetrics; they do NOT change the
# trained policy or PACER eta weighting.  The trajectory-primary profiles are
# useful when the diagnostic objective is process fidelity over the whole
# common-horizon EEF/TCP trajectory rather than endpoint/terminal matching.
SCORING_PROFILES: dict[str, dict[str, float]] = {
    "default_current": dict(DEFAULT_SUBMETRIC_WEIGHTS),
    "trajectory_direction_balanced_A": {
        "progress": 0.025,
        "proximity": 0.025,
        "terminal": 0.025,
        "direction": 0.300,
        "align": 0.600,
        "no_regression": 0.025,
    },
    "trajectory_direction_balanced_B": {
        "progress": 0.020,
        "proximity": 0.040,
        "terminal": 0.020,
        "direction": 0.300,
        "align": 0.590,
        "no_regression": 0.030,
    },
    "trajectory_primary_strong": {
        "progress": 0.030,
        "proximity": 0.040,
        "terminal": 0.030,
        "direction": 0.100,
        "align": 0.750,
        "no_regression": 0.050,
    },
    "trajectory_primary_stronger": {
        "progress": 0.025,
        "proximity": 0.035,
        "terminal": 0.025,
        "direction": 0.100,
        "align": 0.780,
        "no_regression": 0.035,
    },
}


def submetric_weights_for_profile(profile: str | Mapping[str, float] | None = None) -> dict[str, float]:
    """Return scorer weights for a named validation profile.

    Passing a mapping returns a shallow float-cast copy.  Passing None returns
    the default paper-strict weights.  Named trajectory profiles are intended
    for explicitly-labeled sensitivity/diagnostic tables unless frozen before a
    held-out evaluation.
    """
    if profile is None:
        return dict(DEFAULT_SUBMETRIC_WEIGHTS)
    if isinstance(profile, Mapping):
        return {str(k): float(v) for k, v in profile.items()}
    key = str(profile)
    if key not in SCORING_PROFILES:
        raise KeyError(f"unknown scoring profile {key!r}; available={sorted(SCORING_PROFILES)}")
    return dict(SCORING_PROFILES[key])


GEOMETRIC_SUBMETRICS = ("progress", "proximity", "terminal", "direction")

BLOCKER_FLAGS = ("wrong_target", "non_target_exclusion", "invalid_orientation", "unsafe")
WRONG_TARGET_REASONS = frozenset({"wrong_target", "non_target_exclusion", "invalid_orientation"})
VALIDATION_SUBMETRICS = (
    "progress",
    "proximity",
    "terminal",
    "direction",
    "stop",
    "align",
    "no_regression",
)
SUBMETRIC_ALIASES = {"reg": "no_regression"}


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


@dataclass(frozen=True)
class ScorerProfile:
    """Named validation-score contract for paper/appendix diagnostics.

    `SCORING_PROFILES` above remains the legacy direct lambda_m map. This
    richer profile object defines factor families and aggregation semantics for
    trajectory-primary diagnostics that were underspecified in the paper.
    """

    name: str
    family_weights: dict[str, float]
    outcome_weights: dict[str, float]
    trajectory_weights: dict[str, float]
    aggregation: str
    label: str


DEFAULT_ROBUST_SCORER_PROFILE = "trajectory_robust_component_role_q25_A_plus"


ROBUST_SCORER_PROFILES: dict[str, ScorerProfile] = {
    "trajectory_robust_component_role_q25_A": ScorerProfile(
        name="trajectory_robust_component_role_q25_A",
        family_weights={"outcome": 0.25, "trajectory": 0.65, "no_regression": 0.10},
        outcome_weights={"progress": 1.0, "proximity": 1.0, "terminal": 1.0},
        trajectory_weights={"direction": 0.20, "align": 0.80},
        aggregation="component_role_q25",
        label="trajectory_robust_sensitivity",
    ),
    "trajectory_robust_component_role_q25_B": ScorerProfile(
        name="trajectory_robust_component_role_q25_B",
        family_weights={"outcome": 0.20, "trajectory": 0.60, "no_regression": 0.20},
        outcome_weights={"progress": 1.0, "proximity": 1.0, "terminal": 1.0},
        trajectory_weights={"direction": 0.10, "align": 0.90},
        aggregation="component_role_q25",
        label="trajectory_robust_sensitivity",
    ),
    "trajectory_robust_component_role_q25_A_plus": ScorerProfile(
        name="trajectory_robust_component_role_q25_A_plus",
        family_weights={"outcome": 0.20, "trajectory": 0.70, "no_regression": 0.10},
        outcome_weights={"progress": 1.0, "proximity": 1.0, "terminal": 1.0},
        trajectory_weights={"direction": 0.15, "align": 0.85},
        aggregation="component_role_q25",
        label="trajectory_robust_sensitivity",
    ),
}


def scorer_profile(profile: str | ScorerProfile | None = None) -> ScorerProfile:
    if profile is None:
        profile = DEFAULT_ROBUST_SCORER_PROFILE
    if isinstance(profile, ScorerProfile):
        return profile
    key = str(profile)
    if key not in ROBUST_SCORER_PROFILES:
        raise KeyError(f"unknown robust scorer profile {key!r}; available={sorted(ROBUST_SCORER_PROFILES)}")
    return ROBUST_SCORER_PROFILES[key]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def blocker(row: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """B_j (eq:app_bj) from precomputed evaluator flags."""
    reasons = tuple(flag for flag in BLOCKER_FLAGS if bool(row.get(flag)))
    return len(reasons) == 0, reasons


def _canonical_submetrics(submetrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return paper-canonical validation submetric names.

    The paper writes the no-regression term as e_reg, while the code/reporting
    name is `no_regression`. Accept the paper alias but keep one canonical key
    so weights, manifests, and used_submetrics cannot silently diverge.
    """
    out: dict[str, Any] = {}
    for key, value in submetrics.items():
        canonical = SUBMETRIC_ALIASES.get(str(key), str(key))
        out[canonical] = value
    return out


def _weighted_mean(
    submetrics: Mapping[str, Any],
    weights: Mapping[str, float],
) -> tuple[float, tuple[str, ...]]:
    submetrics = _canonical_submetrics(submetrics)
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


def _family_weighted_row_score(row: Mapping[str, Any], profile: ScorerProfile) -> RowScore:
    """Row score for named factor-family profiles.

    This is deliberately separate from `row_score`: it makes the paper-facing
    trajectory-primary extension explicit as outcome/trajectory/no-regression
    families instead of many independent lambda_m terms.
    """
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
    submetrics = _canonical_submetrics(submetrics)
    outcome, outcome_used = _weighted_mean(submetrics, profile.outcome_weights)
    trajectory, trajectory_used = _weighted_mean(submetrics, profile.trajectory_weights)
    reg_value = submetrics.get("no_regression")
    no_regression_score = _clip01(float(reg_value)) if _finite(reg_value) else 0.0
    family_values = {
        "outcome": outcome,
        "trajectory": trajectory,
        "no_regression": no_regression_score,
    }
    value = 0.0
    denom = 0.0
    for family_name, weight in profile.family_weights.items():
        if _finite(weight) and float(weight) >= 0.0:
            value += float(weight) * family_values.get(family_name, 0.0)
            denom += float(weight)
    score = _clip01(value / (denom + EPS)) if denom > 0.0 else 0.0
    used = tuple(dict.fromkeys(tuple(outcome_used) + tuple(trajectory_used) + (("no_regression",) if _finite(reg_value) else tuple())))
    score = score if (positive and passed and used) else 0.0
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


def _lower_quartile(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    return ordered[int(0.25 * (len(ordered) - 1))]


def robust_profile_j_val(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: str | ScorerProfile | None = None,
) -> tuple[float, list[RowScore], dict[str, Any]]:
    """Trajectory-primary robust diagnostic score.

    This fills an underspecified paper space: when the diagnostic objective is
    whole-chunk trajectory/process fidelity, aggregate row scores over
    component×role cells and take the lower quartile (q25). That makes the score
    a consistency/robustness diagnostic rather than an easy-row mean, while the
    same blocker/role rules apply to every method including controls.
    """
    prof = scorer_profile(profile)
    scores = [_family_weighted_row_score(row, prof) for row in rows]
    by_cell: dict[str, list[float]] = defaultdict(list)
    by_component: dict[str, list[float]] = defaultdict(list)
    audit_counts: dict[str, int] = defaultdict(int)
    for s in scores:
        if s.positive_scoring:
            by_component[s.component].append(s.score)
            by_cell[f"{s.component}/{s.role}"].append(s.score)
        for reason in s.audit_reasons:
            audit_counts[reason] += 1
    component_scores = {k: sum(v) / len(v) for k, v in sorted(by_component.items()) if v}
    cell_scores = {k: sum(v) / len(v) for k, v in sorted(by_cell.items()) if v}
    if prof.aggregation == "component_role_q25":
        value = _lower_quartile(list(cell_scores.values()))
    else:
        raise ValueError(f"unsupported robust aggregation {prof.aggregation!r}")
    manifest = {
        "profile": prof.name,
        "label": prof.label,
        "aggregation": prof.aggregation,
        "family_weights": dict(prof.family_weights),
        "outcome_weights": dict(prof.outcome_weights),
        "trajectory_weights": dict(prof.trajectory_weights),
        "num_rows": len(rows),
        "num_positive_scoring_rows": sum(1 for s in scores if s.positive_scoring),
        "component_scores": component_scores,
        "cell_scores": cell_scores,
        "audit_counts": dict(sorted(audit_counts.items())),
    }
    return _clip01(value), scores, manifest


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
