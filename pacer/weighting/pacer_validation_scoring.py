"""Row-level PACER validation scoring utilities.

This module implements the offline validation equations used for PACER candidate
selection. It is intentionally small and dependency-free: callers provide the
already-computed per-row submetrics and blocker flags from their evaluator, and
this module applies the reviewer-facing row score and component-balanced J_val
aggregation.

It does not run model inference, train adapters, or touch robot hardware.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

POSITIVE_SCORING_ROLES = frozenset({"clean", "correction", "auto_success", "partial"})
AUDIT_ONLY_ROLES = frozenset({"failure", "excluded"})
# Compiler/manifest role vocabulary -> paper role vocabulary. Validation rows can
# come either from the paper-facing evaluator (paper names) or straight from the
# action-chunk compiler (sample_role names); both must score identically.
ROLE_ALIASES: dict[str, str] = {
    "clean_demo": "clean",
    "human_correction": "correction",
    "model_success": "auto_success",
    "model_partial": "partial",
    "model_failure": "failure",
}
# Submetric coefficients lambda_m, fixed before ranking. The applicable set M_j
# comprises the geometric primitives, stop/handoff correctness, reference-chunk
# alignment when a matched reference exists, and the no-regression term; absent
# submetrics are omitted per row and the remaining weights renormalize.
DEFAULT_SUBMETRIC_WEIGHTS: dict[str, float] = {
    "progress": 0.25,
    "proximity": 0.25,
    "terminal": 0.20,
    "direction": 0.15,
    "stop": 0.15,
    "align": 0.15,
    "no_regression": 0.10,
}


def normalize_role(value: object) -> str:
    """Map compiler sample_role names onto the paper role vocabulary."""
    role = str(value or "")
    return ROLE_ALIASES.get(role, role)


@dataclass(frozen=True)
class ValidationRowScore:
    row_id: str
    component: str
    role: str
    positive_scoring: bool
    blocker_passed: bool
    score: float
    used_submetrics: tuple[str, ...] = field(default_factory=tuple)
    audit_reasons: tuple[str, ...] = field(default_factory=tuple)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _row_bool(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    return bool(value) if isinstance(value, bool) else default


def blocker_passed(row: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Return B_j and audit reasons for a validation row.

    Wrong-target, non-target exclusion, invalid orientation, and unsafe events are
    hard blockers. Terminal-region failure is deliberately not a blocker; it
    should be represented as a normal submetric so valid partial progress can
    still receive credit.
    """
    reasons: list[str] = []
    if _row_bool(row, "wrong_target") or _row_bool(row, "non_target_exclusion"):
        reasons.append("wrong_target")
    if _row_bool(row, "invalid_orientation"):
        reasons.append("invalid_orientation")
    if _row_bool(row, "unsafe"):
        reasons.append("unsafe")
    return len(reasons) == 0, tuple(reasons)


def score_validation_row(
    row: Mapping[str, Any],
    *,
    submetric_weights: Mapping[str, float] | None = None,
) -> ValidationRowScore:
    """Compute the row-level PACER validation score v_j.

    Expected row fields:
    - row_id: stable row identifier, optional
    - component: component id/name
    - role: one of clean, correction, auto_success, partial, failure, excluded
    - submetrics: mapping from metric name to value in [0, 1]
    - wrong_target / non_target_exclusion / invalid_orientation / unsafe: blockers

    Only positive-scoring roles contribute to J_val. Failure/excluded rows are
    returned with score zero and audit reasons so callers can check leakage.
    Missing submetrics are omitted and the remaining weights are renormalized;
    this is how non-phase-ending rows omit stop without zeroing the row.
    """
    weights = dict(DEFAULT_SUBMETRIC_WEIGHTS if submetric_weights is None else submetric_weights)
    role = normalize_role(row.get("role") or row.get("sample_role"))
    component = str(row.get("component") or "unknown")
    row_id = str(row.get("row_id") or row.get("id") or "unknown")

    passed, reasons = blocker_passed(row)
    positive = role in POSITIVE_SCORING_ROLES
    if role in AUDIT_ONLY_ROLES:
        reasons = (*reasons, "audit_only_role")
    elif not positive:
        reasons = (*reasons, "non_positive_scoring_role")

    submetrics = row.get("submetrics", {})
    if not isinstance(submetrics, Mapping):
        submetrics = {}

    numerator = 0.0
    denominator = 0.0
    used: list[str] = []
    for name, weight in weights.items():
        if not _finite_number(weight) or float(weight) < 0:
            continue
        value = submetrics.get(name)
        if not _finite_number(value):
            continue
        value_f = float(value)  # type: ignore[arg-type]
        weight_f = float(weight)  # type: ignore[arg-type]
        numerator += weight_f * _clip01(value_f)
        denominator += weight_f
        used.append(name)

    if not positive or not passed or denominator <= 0.0:
        score = 0.0
    else:
        score = _clip01(numerator / denominator)

    return ValidationRowScore(
        row_id=row_id,
        component=component,
        role=role,
        positive_scoring=positive,
        blocker_passed=passed,
        score=score,
        used_submetrics=tuple(used),
        audit_reasons=reasons,
    )


def component_balanced_j_val(
    rows: Sequence[Mapping[str, Any]],
    *,
    submetric_weights: Mapping[str, float] | None = None,
) -> tuple[float, list[ValidationRowScore], dict[str, Any]]:
    """Compute component-balanced J_val over positive-scoring validation rows."""
    row_scores = [score_validation_row(row, submetric_weights=submetric_weights) for row in rows]
    by_component: dict[str, list[float]] = defaultdict(list)
    audit_counts: dict[str, int] = defaultdict(int)

    for score in row_scores:
        if score.positive_scoring:
            by_component[score.component].append(score.score)
        for reason in score.audit_reasons:
            audit_counts[reason] += 1

    component_scores = {
        component: (sum(values) / len(values) if values else 0.0)
        for component, values in sorted(by_component.items())
    }
    if component_scores:
        j_val = sum(component_scores.values()) / len(component_scores)
    else:
        j_val = 0.0

    manifest = {
        "num_rows": len(rows),
        "num_positive_scoring_rows": sum(1 for s in row_scores if s.positive_scoring),
        "component_scores": component_scores,
        "audit_counts": dict(sorted(audit_counts.items())),
    }
    return _clip01(j_val), row_scores, manifest
