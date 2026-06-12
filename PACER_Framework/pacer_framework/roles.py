"""Process roles, label mapping, and strict success (paper §PACER + app:roles, app:labels).

Implements:
- eq:process_role — role taxonomy R = {clean, correction, auto_success, partial,
  failure, excluded}.
- app:roles deterministic mapping from trace source and operator label to role.
- eq:app_strict — strict hardware success indicator.
"""
from __future__ import annotations

PAPER_ROLES = ("clean", "correction", "auto_success", "partial", "failure", "excluded")

# Compiler / legacy-view vocabulary used by PACER action-chunk rows.
COMPILER_TO_PAPER = {
    "clean_demo": "clean",
    "human_correction": "correction",
    "model_success": "auto_success",
    "model_partial": "partial",
    "model_failure": "failure",
    "excluded": "excluded",
}

POSITIVE_IMITATION_ROLES = frozenset({"clean", "correction", "auto_success", "partial"})
AUDIT_ONLY_ROLES = frozenset({"failure", "excluded"})

# Closed-vocabulary operator outcome labels (app:labels).
OPERATOR_LABELS = (
    "success",
    "near_but_not_accurate",
    "wrong_orientation_or_wrong_location",
    "totally_off_wrong_region_or_target",
    "operator_uncertain_exclude",
)

_LABEL_TO_ROLE = {
    "success": "auto_success",
    "near_but_not_accurate": "partial",
    "wrong_orientation_or_wrong_location": "failure",
    "totally_off_wrong_region_or_target": "failure",
    "operator_uncertain_exclude": "excluded",
}

# Stop sources excluded from strict success (eq:app_strict).
STRICT_EXCLUDED_STOP_SOURCES = frozenset({"timeout", "unsafe_abort", "intervention"})


def normalize_role(value: object, *, strict: bool = True) -> str:
    """Return the paper role name for either vocabulary.

    strict=True raises on unknown names so typos cannot silently become a
    blocked role; strict=False passes unknown names through for diagnostics.
    """
    role = str(value or "")
    if role in PAPER_ROLES:
        return role
    if role in COMPILER_TO_PAPER:
        return COMPILER_TO_PAPER[role]
    if strict:
        raise ValueError(f"unknown PACER role name: {role!r}")
    return role


def role_from_source_and_label(*, source: str | None = None, operator_label: str | None = None) -> str:
    """Deterministic source/label -> role mapping (Table app_rolemap).

    Trace source dominates: supplementary teleoperated demonstrations are clean
    and operator recoveries are corrections regardless of any autonomous label.
    Autonomous rollouts map through the closed-vocabulary operator label.
    """
    src = str(source or "")
    if src in {"clean", "clean_demo", "supplementary_demo", "teleop_demo"}:
        return "clean"
    if src in {"correction", "human_correction", "operator_recovery"}:
        return "correction"
    label = str(operator_label or "")
    if label not in _LABEL_TO_ROLE:
        raise ValueError(f"unknown operator label: {label!r}")
    return _LABEL_TO_ROLE[label]


def strict_success(operator_label: str, stop_source: str | None) -> bool:
    """eq:app_strict — label must be success AND stop source not excluded."""
    if str(operator_label) != "success":
        return False
    return str(stop_source or "") not in STRICT_EXCLUDED_STOP_SOURCES
