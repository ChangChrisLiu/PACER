"""Compatibility imports for the canonical component-balanced row scorer.

Both installation options provide pacer_framework. Runtime adapters and the
portable framework therefore share configuration, alias handling, blockers,
normalization and reference comparisons rather than maintaining two scorers.
"""
from __future__ import annotations

from pacer_framework.configuration import ScoringConfig, DEFAULT_SUBMETRIC_WEIGHTS
from pacer_framework.roles import AUDIT_ONLY_ROLES, POSITIVE_IMITATION_ROLES
from pacer_framework.roles import COMPILER_TO_PAPER as ROLE_ALIASES
from pacer_framework.roles import normalize_role as _normalize_role
from pacer_framework.validation import (
    RowScore as ValidationRowScore,
    blocker as blocker_passed,
    component_balanced_j_val,
    row_score as score_validation_row,
)

POSITIVE_SCORING_ROLES = POSITIVE_IMITATION_ROLES


def normalize_role(value: object) -> str:
    """Preserve the adapter's non-strict role-normalization interface."""
    return _normalize_role(value, strict=False)


__all__ = [
    "AUDIT_ONLY_ROLES", "DEFAULT_SUBMETRIC_WEIGHTS", "POSITIVE_SCORING_ROLES",
    "ROLE_ALIASES", "ScoringConfig", "ValidationRowScore", "blocker_passed",
    "component_balanced_j_val", "normalize_role", "score_validation_row",
]
