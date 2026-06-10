"""Eligibility gate g_i = G_split * G_mask * G_prov * G_safe * G_role (eq:app_gate).

The gate requires that the row is in the training split, has nonzero valid mask
mass, has trusted provenance, is safe, and has a role outside
{failure, excluded}. Wrong-target outcome labels map to the failure role and
are therefore blocked through G_role; target-metadata inconsistency acts inside
evidence construction (the T_i factor), not as an extra gate factor.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from pacer_framework.evidence import provenance_check
from pacer_framework.roles import AUDIT_ONLY_ROLES, normalize_role


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def mask_has_mass(mask: Sequence[Any] | None) -> bool:
    """G_mask — per-horizon mask exists, is binary-valid, and has nonzero mass.

    Negative or non-numeric entries invalidate the mask; an all-zero mask has
    no valid mass and blocks the row entirely (app:mask).
    """
    if mask is None or not isinstance(mask, (list, tuple)) or len(mask) == 0:
        return False
    values = [_as_float(v, -1.0) for v in mask]
    return all(v >= 0.0 for v in values) and any(v > 0.0 for v in values)


def row_is_unsafe(row: Mapping[str, Any]) -> bool:
    """G_safe — unsafe abort, manual safety stop, or quarantine blocks the row."""
    safety = row.get("safety") or {}
    if not isinstance(safety, Mapping):
        return True
    return bool(
        safety.get("unsafe")
        or safety.get("manual_safety_stop")
        or safety.get("quarantined")
    )


def eligibility_gate(
    row: Mapping[str, Any],
    *,
    train_split: str = "train",
    require_train_split: bool = True,
) -> tuple[bool, list[str]]:
    """Evaluate g_i; returns (gate, reasons) with one named reason per failed factor."""
    reasons: list[str] = []
    if require_train_split and row.get("split") != train_split:
        reasons.append("gate_split_not_train")
    if not mask_has_mass(row.get("mask")):
        reasons.append("gate_mask_no_valid_mass")
    if provenance_check(row.get("provenance")) <= 0.0:
        reasons.append("gate_provenance_untrusted")
    if row_is_unsafe(row):
        reasons.append("gate_unsafe")
    role = normalize_role(row.get("role") or row.get("sample_role"), strict=False)
    if role in AUDIT_ONLY_ROLES:
        reasons.append("gate_role_blocked_from_positive_imitation")
    return len(reasons) == 0, reasons
