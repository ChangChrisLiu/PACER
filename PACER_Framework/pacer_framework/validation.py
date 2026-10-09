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
from typing import Any, Mapping, Sequence, TypeGuard

from pacer_framework.roles import AUDIT_ONLY_ROLES, POSITIVE_IMITATION_ROLES, normalize_role
from pacer_framework.configuration import (
    DEFAULT_SUBMETRIC_WEIGHTS, GEOMETRIC_SUBMETRICS, SUBMETRIC_ALIASES,
    VALIDATION_SUBMETRICS, ScoringConfig, resolve_scoring_config,
    validate_submetric_weights,
)

EPS = 1e-6



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
        return validate_submetric_weights(profile)
    key = str(profile)
    if key not in SCORING_PROFILES:
        raise KeyError(f"unknown scoring profile {key!r}; available={sorted(SCORING_PROFILES)}")
    return dict(SCORING_PROFILES[key])




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
    no_regression_source: str = "not_scored"
    no_regression_cache_disagrees: bool = False
    blocker_inputs_complete: bool = False


@dataclass(frozen=True)
class ScorerProfile:
    """Named validation-score contract for paper/appendix diagnostics.

    Factor-family profiles are explicit compatibility diagnostics. The default
    direct-weight component-mean protocol uses ScoringConfig instead.
    """

    name: str
    family_weights: dict[str, float]
    outcome_weights: dict[str, float]
    trajectory_weights: dict[str, float]
    aggregation: str
    label: str


DEFAULT_SCORER_PROFILE = "component_balanced_mean"
# Compatibility name; new callers should use DEFAULT_SCORER_PROFILE.
DEFAULT_ROBUST_SCORER_PROFILE = DEFAULT_SCORER_PROFILE


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


def scorer_profile(
    profile: str | ScorerProfile | ScoringConfig | None = None,
) -> ScorerProfile | ScoringConfig:
    """Resolve the component-mean default or an explicitly named diagnostic."""
    if profile is None or profile == DEFAULT_SCORER_PROFILE:
        return ScoringConfig()
    if isinstance(profile, ScoringConfig):
        return profile
    if isinstance(profile, ScorerProfile):
        source = profile
    else:
        key = str(profile)
        if key not in ROBUST_SCORER_PROFILES:
            available = [DEFAULT_SCORER_PROFILE, *sorted(ROBUST_SCORER_PROFILES)]
            raise KeyError(f"unknown scorer profile {key!r}; available={available}")
        source = ROBUST_SCORER_PROFILES[key]
    return ScorerProfile(
        name=source.name, family_weights=dict(source.family_weights),
        outcome_weights=dict(source.outcome_weights),
        trajectory_weights=dict(source.trajectory_weights),
        aggregation=source.aggregation, label=source.label,
    )


def _finite(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip01(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("score normalization produced a non-finite value")
    return max(0.0, min(1.0, number))


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
    scoring_config: ScoringConfig | None = None,
) -> RowScore:
    """Score a row, recomputing no-regression when reference geometry is stored."""
    config = resolve_scoring_config(scoring_config, submetric_weights=submetric_weights)
    weights = config.submetric_weights
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
    for name, value in submetrics.items():
        if name in VALIDATION_SUBMETRICS and not _finite(value):
            raise ValueError(f"row {row_id!r}: submetric {name!r} must be a finite number; omit unavailable metrics")
    active_geometry = [key for key in GEOMETRIC_SUBMETRICS if weights.get(key, 0.0) > 0]
    missing_geometry = positive and bool(active_geometry) and not any(key in submetrics for key in active_geometry)
    if missing_geometry:
        audit_reasons.append("no_geometric_support")
    source = "not_scored"
    cache_disagrees = False
    if positive and passed and not missing_geometry:
        source = "unavailable"
        if weights.get("no_regression", 0.0) > 0:
            if "reference_submetrics" in row:
                reference = row["reference_submetrics"]
                if not isinstance(reference, Mapping):
                    raise ValueError("reference_submetrics must be a mapping of baseline geometric metrics")
                cached = submetrics.get("no_regression")
                submetrics["no_regression"] = no_regression_from_submetrics(
                    submetrics, reference, scoring_config=config,
                )
                cache_disagrees = _finite(cached) and abs(float(cached) - submetrics["no_regression"]) > 1e-9
                source = "reference_geometry"
            elif _finite(submetrics.get("no_regression")):
                source = "provided"
        declared_mode = row.get("reference_alignment_mode")
        if declared_mode is not None and weights.get("align", 0.0) > 0 and _finite(submetrics.get("align")):
            actual_mode = ScoringConfig(reference_alignment_mode=declared_mode).reference_alignment_mode
            if actual_mode != config.reference_alignment_mode:
                raise ValueError("cached alignment mode differs from scoring_config; rebuild submetrics from traces")
    value, used = _weighted_mean(submetrics, weights)
    score = value if (positive and passed and used and not missing_geometry) else 0.0

    return RowScore(
        row_id=row_id,
        component=component,
        role=role,
        positive_scoring=positive,
        blocker_passed=passed,
        blocker_inputs_complete=all(flag in row and isinstance(row[flag], bool) for flag in BLOCKER_FLAGS),
        score=score,
        used_submetrics=used,
        audit_reasons=tuple(audit_reasons),
        no_regression_source=source,
        no_regression_cache_disagrees=cache_disagrees,
    )


def geometric_row_score(
    submetrics: Mapping[str, Any],
    *,
    submetric_weights: Mapping[str, float] | None = None,
    scoring_config: ScoringConfig | None = None,
) -> float:
    """v-hat (eq:app_vhat) — weighted mean over the geometric submetrics only."""
    config = resolve_scoring_config(scoring_config, submetric_weights=submetric_weights)
    weights = config.submetric_weights
    geometric = {k: weights[k] for k in GEOMETRIC_SUBMETRICS if k in weights}
    value, _used = _weighted_mean(submetrics, geometric)
    return value


def no_regression(v_hat_candidate: float, v_hat_reference: float, *, delta_reg: float) -> float:
    """eq:app_reg — indicator that the candidate row score has not regressed."""
    return 1.0 if float(v_hat_candidate) >= float(v_hat_reference) - float(delta_reg) else 0.0


def no_regression_from_submetrics(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    scoring_config: ScoringConfig | None = None,
    submetric_weights: Mapping[str, float] | None = None,
    delta_reg: float | None = None,
) -> float:
    """Compare matched geometric support under one shared scoring configuration."""
    config = resolve_scoring_config(
        scoring_config, submetric_weights=submetric_weights, delta_reg=delta_reg,
    )
    candidate = _canonical_submetrics(candidate)
    reference = _canonical_submetrics(reference)
    active = [key for key in GEOMETRIC_SUBMETRICS if config.submetric_weights.get(key, 0.0) > 0]
    candidate_keys = {key for key in active if _finite(candidate.get(key))}
    reference_keys = {key for key in active if _finite(reference.get(key))}
    if not candidate_keys or candidate_keys != reference_keys:
        raise ValueError("no-regression requires matching, nonempty active geometric submetrics")
    return no_regression(
        geometric_row_score(candidate, scoring_config=config),
        geometric_row_score(reference, scoring_config=config),
        delta_reg=config.delta_reg,
    )


def component_balanced_j_val(
    rows: Sequence[Mapping[str, Any]],
    *,
    submetric_weights: Mapping[str, float] | None = None,
    scoring_config: ScoringConfig | None = None,
) -> tuple[float, list[RowScore], dict[str, Any]]:
    """J_val (eq:j_val) — mean over components of per-component mean row scores."""
    config = resolve_scoring_config(scoring_config, submetric_weights=submetric_weights)
    scores = [row_score(row, scoring_config=config) for row in rows]
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
    regression_sources: dict[str, int] = defaultdict(int)
    for scored in scores:
        regression_sources[scored.no_regression_source] += 1
    manifest = {
        "profile": "component_balanced_mean",
        "aggregation": "component_mean",
        "scoring_config": config.to_dict(),
        "scoring_config_hash": config.fingerprint,
        "candidate_feasibility_checked": False,
        "blocker_inputs_complete_rows": sum(s.blocker_inputs_complete for s in scores),
        "blocker_inputs_missing_or_invalid_rows": sum(not s.blocker_inputs_complete for s in scores),
        "alignment_mode_undeclared_scoring_rows": sum(
            s.positive_scoring and s.blocker_passed and "align" in s.used_submetrics
            and config.submetric_weights.get("align", 0.0) > 0
            and row.get("reference_alignment_mode") is None
            for row, s in zip(rows, scores)
        ),
        "no_regression_sources": dict(sorted(regression_sources.items())),
        "no_regression_recomputed": regression_sources.get("reference_geometry", 0),
        "no_regression_cached": regression_sources.get("provided", 0),
        "no_regression_cache_disagreements": sum(s.no_regression_cache_disagrees for s in scores),
        "cached_no_regression_under_custom_weights": (
            regression_sources.get("provided", 0)
            if dict(config.submetric_weights) != DEFAULT_SUBMETRIC_WEIGHTS else 0
        ),
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
        blocker_inputs_complete=all(flag in row and isinstance(row[flag], bool) for flag in BLOCKER_FLAGS),
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
    profile: str | ScorerProfile | ScoringConfig | None = None,
    scoring_config: ScoringConfig | None = None,
    submetric_weights: Mapping[str, float] | None = None,
) -> tuple[float, list[RowScore], dict[str, Any]]:
    """Compatibility entry point; component mean is the shared default.

    Named factor-family/lower-tail diagnostics remain explicit opt-ins. They
    keep their historical calculation and cannot be mixed with direct-weight
    overrides or ScoringConfig.
    """
    if profile is None or profile == DEFAULT_SCORER_PROFILE:
        return component_balanced_j_val(
            rows, scoring_config=scoring_config, submetric_weights=submetric_weights,
        )
    prof = scorer_profile(profile)
    if scoring_config is not None or submetric_weights is not None:
        raise ValueError("pass a profile or direct scoring configuration, not both")
    if isinstance(prof, ScoringConfig):
        return component_balanced_j_val(rows, scoring_config=prof)
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
    require_reference_component_scores: bool = True,
) -> tuple[bool, dict[str, bool]]:
    """F(eta) = A_audit * A_wrong * A_safe * A_reg (eq:app_feasibility).

    A_audit reads the pre-training weight manifest (forbidden positive-weight
    count, finite loss denominator, max weight within the clip, non-collapsed
    n_eff). A_wrong / A_safe demand zero wrong-target / unsafe leakage over the
    validation rows. A_reg requires matching component coverage and checks
    the reference policy under the fixed margin. Missing reference scores fail
    closed by default. A caller may explicitly disable the reference requirement
    for a labelled diagnostic, not a complete candidate-selection audit.
    """
    loss_normalizer = float(weight_manifest.get("loss_normalizer", 0.0))
    checks: dict[str, bool] = {
        "A_validation_has_scoring_rows": any(s.positive_scoring for s in validation_scores),
        "A_validation_blocker_inputs": all(s.blocker_inputs_complete for s in validation_scores),
        "A_validation_support": not any("no_geometric_support" in s.audit_reasons for s in validation_scores),
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
        support_matches = set(reference_component_scores) == set(candidate_component_scores)
        checks["A_reg_component_support"] = support_matches
        checks["A_reg_no_regression"] = support_matches and all(
            float(candidate_component_scores[component]) >= float(ref) - float(delta_reg)
            for component, ref in reference_component_scores.items()
        )
    elif require_reference_component_scores:
        checks["A_reg_no_regression"] = False
        checks["A_reg_reference_scores_present"] = False
    else:
        checks["A_reg_no_regression"] = True
        checks["A_reg_reference_audit_skipped"] = True
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


def _matched_reference_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    scoring_config: ScoringConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach matched baseline geometry without mutating caller-owned rows."""
    def key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
        role = normalize_role(row.get("role") or row.get("sample_role"), strict=False)
        if role not in POSITIVE_IMITATION_ROLES:
            return None
        row_id = row.get("row_id") or row.get("id")
        if not row_id:
            raise ValueError("reference_validation_rows requires stable row_id values on scoring rows")
        return str(row_id), str(row.get("component") or "unknown"), role

    reference_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in reference_rows:
        identity = key(row)
        if identity is not None:
            if identity in reference_index:
                raise ValueError("reference_validation_rows contains duplicate scoring row identities")
            reference_index[identity] = row
    candidate_keys = [identity for row in candidate_rows if (identity := key(row)) is not None]
    if len(candidate_keys) != len(set(candidate_keys)) or set(candidate_keys) != set(reference_index):
        raise ValueError("candidate and reference must have identical positive scoring row identities")
    active_geometry = [name for name in GEOMETRIC_SUBMETRICS
                       if scoring_config.submetric_weights.get(name, 0.0) > 0]

    def attach(row: dict[str, Any], reference: Any) -> None:
        if "reference_submetrics" in row:
            stored = row["reference_submetrics"]
            if not isinstance(stored, Mapping) or not isinstance(reference, Mapping):
                raise ValueError("conflicting reference representations; provide one reference geometry source")
            for name in active_geometry:
                if (name in stored) != (name in reference):
                    raise ValueError("conflicting reference representations; provide one reference geometry source")
                if name in stored:
                    left, right = stored[name], reference[name]
                    if not _finite(left) or not _finite(right) or float(left) != float(right):
                        raise ValueError("conflicting reference representations; provide one reference geometry source")
        row["reference_submetrics"] = reference

    prepared_candidates = []
    for original in candidate_rows:
        row = dict(original)
        identity = key(row)
        if identity is not None:
            attach(row, reference_index[identity].get("submetrics", {}))
        prepared_candidates.append(row)
    prepared_reference = []
    for original in reference_rows:
        row = dict(original)
        if key(row) is not None:
            attach(row, row.get("submetrics", {}))
        prepared_reference.append(row)
    return prepared_candidates, prepared_reference


def evaluate_candidate(
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    weight_manifest: Mapping[str, Any],
    w_max: float,
    submetric_weights: Mapping[str, float] | None = None,
    scoring_config: ScoringConfig | None = None,
    reference_component_scores: Mapping[str, float] | None = None,
    reference_validation_rows: Sequence[Mapping[str, Any]] | None = None,
    delta_reg: float | None = None,
    n_eff_min: float = 1.0,
    require_reference_component_scores: bool | None = None,
) -> dict[str, Any]:
    """Score and audit a candidate with one shared, explicit configuration.

    Matched reference rows are preferred: their component scores and every
    row-level no-regression comparison are rebuilt under the same coefficients.
    Precomputed reference component scores remain accepted as a caller-owned
    contract. Supply exactly one reference representation.
    """
    config = resolve_scoring_config(
        scoring_config, submetric_weights=submetric_weights, delta_reg=delta_reg,
        require_reference_component_scores=require_reference_component_scores,
    )
    if reference_validation_rows is not None and reference_component_scores is not None:
        raise ValueError("provide reference_validation_rows or reference_component_scores, not both")
    reference_manifest = None
    reference_scores: list[RowScore] = []
    if reference_validation_rows is not None:
        validation_rows, prepared_reference = _matched_reference_rows(
            validation_rows, reference_validation_rows, scoring_config=config,
        )
        _, reference_scores, reference_manifest = component_balanced_j_val(
            prepared_reference, scoring_config=config,
        )
        reference_component_scores = reference_manifest["component_scores"]
    if reference_component_scores is not None:
        if any(not _finite(v) or not 0.0 <= float(v) <= 1.0 for v in reference_component_scores.values()):
            raise ValueError("reference component scores must be finite bounded numbers")
    j_val, scores, manifest = component_balanced_j_val(validation_rows, scoring_config=config)
    feasible, checks = candidate_feasibility(
        weight_manifest=weight_manifest,
        validation_scores=scores,
        w_max=w_max,
        candidate_component_scores=manifest["component_scores"],
        reference_component_scores=reference_component_scores,
        delta_reg=config.delta_reg,
        n_eff_min=n_eff_min,
        require_reference_component_scores=config.require_reference_component_scores,
    )
    if reference_validation_rows is not None:
        checks["A_reg_reference_geometry_support"] = not any(
            "no_geometric_support" in s.audit_reasons for s in reference_scores
        )
        checks["A_reg_reference_blocker_inputs"] = all(s.blocker_inputs_complete for s in reference_scores)
        feasible = all(checks.values())
    manifest["candidate_feasibility_checked"] = True
    return {
        "j_val": j_val,
        "feasible": feasible,
        "checks": checks,
        "component_scores": manifest["component_scores"],
        "reference_component_scores": dict(reference_component_scores) if reference_component_scores is not None else None,
        "reference_manifest": reference_manifest,
        "row_scores": scores,
        "manifest": manifest,
    }
