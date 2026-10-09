"""Executable data contract for robot/VLA-agnostic PACER records.

Two record types cross the framework boundary:

1. TRAINING ROW — one compiled action-chunk row of the post-training buffer
   (paper app:roles: x_i = (o_i, a_{i,1:H}, m_{i,1:H}, c_i, phi_i, t_i_act,
   q_i, z_i, rho_i)). The observation/action tensors stay with the user's
   training stack; PACER needs the typed metadata listed in
   ``REQUIRED_TRAINING_FIELDS`` plus an evidence source (precomputed
   ``evidence`` or a ``geometry`` block).
2. VALIDATION ROW — one scored validation record (paper app:valscore):
   bounded submetrics from the user's open-loop evaluator plus blocker flags.

``validate_training_row`` / ``validate_validation_row`` return a list of
human-readable errors (empty = contract satisfied). ``training_row_warnings``
flags conditions that are legal but zero out evidence (e.g. T_i = 0).
Templates return fully populated example records.
See docs/DATA_CONTRACT.md for the prose specification.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from pacer_framework.eta import EVIDENCE_KEYS
from pacer_framework.configuration import GEOMETRIC_SUBMETRICS
from pacer_framework.evidence import PROVENANCE_FLAGS, target_consistency
from pacer_framework.roles import normalize_role
from pacer_framework.validation import BLOCKER_FLAGS, SUBMETRIC_ALIASES, VALIDATION_SUBMETRICS

SPLITS = ("train", "val", "heldout")

REQUIRED_TRAINING_FIELDS = ("row_id", "component", "phase", "split", "mask")
POSITIVE_ROLES_NEED_EVIDENCE = ("clean", "correction", "auto_success", "partial")
SAFETY_FLAGS = ("unsafe", "manual_safety_stop", "quarantined")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_point3(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(_is_number(v) for v in value)
    )


def _check_role(row: Mapping[str, Any], errors: list[str]) -> str | None:
    raw = row.get("role") or row.get("sample_role")
    if raw in (None, ""):
        errors.append("missing role (set 'role' or 'sample_role')")
        return None
    try:
        return normalize_role(raw)
    except ValueError:
        errors.append(f"unknown role name {raw!r} (paper or compiler vocabulary required)")
        return None


def validate_training_row(row: Mapping[str, Any]) -> list[str]:
    """Return contract violations for a training row (empty list = valid)."""
    errors: list[str] = []
    if not isinstance(row, Mapping):
        return ["row must be a mapping"]

    for field in ("row_id", "component", "phase"):
        if row.get(field) in (None, ""):
            errors.append(f"missing required field '{field}'")
    if row.get("split") not in SPLITS:
        errors.append(f"split must be one of {SPLITS}, got {row.get('split')!r}")
    role = _check_role(row, errors)

    mask = row.get("mask")
    if not isinstance(mask, (list, tuple)) or len(mask) == 0:
        errors.append("mask must be a non-empty per-horizon list")
    else:
        for h, value in enumerate(mask):
            if not _is_number(value) or float(value) not in (0.0, 1.0):
                errors.append(f"mask[{h}] must be 0 or 1, got {value!r}")
                break

    evidence = row.get("evidence")
    geometry = row.get("geometry")
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            errors.append("evidence must be a mapping over the seven evidence keys")
        else:
            unknown = sorted(set(evidence) - set(EVIDENCE_KEYS))
            if unknown:
                errors.append(f"unknown evidence keys {unknown}; allowed: {list(EVIDENCE_KEYS)}")
            for key, value in evidence.items():
                if key in EVIDENCE_KEYS and (not _is_number(value) or not 0.0 <= float(value) <= 1.0):
                    errors.append(f"evidence[{key!r}] must be a number in [0, 1], got {value!r}")
    if geometry is not None:
        if not isinstance(geometry, Mapping):
            errors.append("geometry must be a mapping")
        else:
            positions = geometry.get("tcp_positions")
            has_trajectory = positions is not None
            if has_trajectory:
                if not isinstance(positions, (list, tuple)) or len(positions) < 1 or not all(
                    _is_point3(p) for p in positions
                ):
                    errors.append("geometry.tcp_positions must be a list of [x, y, z] points")
                if not _is_point3(geometry.get("target_point")):
                    errors.append("geometry.target_point must be [x, y, z] when tcp_positions is given")
            else:
                if not (_is_number(geometry.get("d0")) and _is_number(geometry.get("d_end"))):
                    errors.append("geometry needs tcp_positions+target_point or d0+d_end summaries")
    if role in POSITIVE_ROLES_NEED_EVIDENCE and evidence is None and geometry is None:
        errors.append(f"role {role!r} is positive-imitation and requires an 'evidence' or 'geometry' block")

    for block_name, allowed in (("provenance", PROVENANCE_FLAGS), ("safety", SAFETY_FLAGS)):
        block = row.get(block_name)
        if block is not None:
            if not isinstance(block, Mapping):
                errors.append(f"{block_name} must be a mapping of boolean flags")
            else:
                for key, value in block.items():
                    if key in allowed and not isinstance(value, bool):
                        errors.append(f"{block_name}[{key!r}] must be boolean, got {value!r}")
    stop = row.get("stop")
    if stop is not None and not isinstance(stop, Mapping):
        errors.append("stop must be a mapping {phase_ending, stop_event, inside_terminal}")
    if row.get("rank_weight") is not None and not _is_number(row.get("rank_weight")):
        errors.append("rank_weight must be numeric (fixed-geometry baseline input)")
    return errors


def training_row_warnings(row: Mapping[str, Any]) -> list[str]:
    """Legal-but-consequential conditions (row still validates)."""
    warnings: list[str] = []
    t_i, reasons = target_consistency(row)
    if t_i == 0.0:
        warnings.append(
            "target consistency T_i = 0 (" + ", ".join(reasons) + "): geometric evidence will be zeroed"
        )
    role = normalize_role(row.get("role") or row.get("sample_role"), strict=False)
    if role in {"auto_success", "partial", "failure", "excluded"} and not row.get("operator_label"):
        warnings.append("autonomous row without operator_label: operator evidence e_op defaults to 0")
    if row.get("collection_config") in (None, ""):
        warnings.append("missing collection_config: row cannot enter the case-group split")
    return warnings


def validate_validation_row(row: Mapping[str, Any], *, strict_paper: bool = False) -> list[str]:
    """Return contract violations for a validation-scoring row.

    In default mode this checks that any supplied values are well-formed. In
    ``strict_paper`` mode it also requires every audit/blocker dimension to be
    explicitly present and requires the paper-critical `align` and
    no-regression (`no_regression`, alias `reg`) submetrics, or reference
    geometry from which no-regression can be derived under the scorer config. Use
    strict mode before reporting a result as paper-framework `J_val` rather
    than as a reduced diagnostic.
    """
    errors: list[str] = []
    if not isinstance(row, Mapping):
        return ["row must be a mapping"]
    for field in ("row_id", "component"):
        if row.get(field) in (None, ""):
            errors.append(f"missing required field '{field}'")
    _check_role(row, errors)
    submetrics = row.get("submetrics")
    if not isinstance(submetrics, Mapping) or len(submetrics) == 0:
        errors.append("submetrics must be a non-empty mapping of bounded metric values")
    else:
        allowed = set(VALIDATION_SUBMETRICS) | set(SUBMETRIC_ALIASES)
        canonical_keys = {SUBMETRIC_ALIASES.get(str(k), str(k)) for k in submetrics}
        for key, value in submetrics.items():
            canonical = SUBMETRIC_ALIASES.get(str(key), str(key))
            if str(key) not in allowed and canonical not in VALIDATION_SUBMETRICS:
                errors.append(
                    f"unknown validation submetric {key!r}; allowed: {list(VALIDATION_SUBMETRICS)} plus aliases {SUBMETRIC_ALIASES}"
                )
                continue
            if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"submetrics[{key!r}] must be a number in [0, 1], got {value!r}")
        if strict_paper:
            for required in ("align", "no_regression"):
                if required not in canonical_keys:
                    reference = row.get("reference_submetrics")
                    derivable = required == "no_regression" and isinstance(reference, Mapping) and any(
                        key in submetrics and _is_number(reference.get(key))
                        for key in GEOMETRIC_SUBMETRICS
                    )
                    if not derivable:
                        errors.append(f"strict paper validation row requires submetric '{required}' or reference geometry")
    if "reference_submetrics" in row:
        reference = row["reference_submetrics"]
        if not isinstance(reference, Mapping):
            errors.append("reference_submetrics must be a mapping")
        else:
            for key, value in reference.items():
                canonical = SUBMETRIC_ALIASES.get(str(key), str(key))
                if canonical not in VALIDATION_SUBMETRICS:
                    errors.append(f"unknown reference submetric {key!r}")
                elif not _is_number(value) or not 0.0 <= float(value) <= 1.0:
                    errors.append(f"reference_submetrics[{key!r}] must be a bounded number")
    if "reference_alignment_mode" in row:
        mode = row["reference_alignment_mode"]
        if not isinstance(mode, str) or mode not in {
            "trajectory", "whole_chunk", "whole_chunk_trajectory",
            "endpoint", "endpoint_net_direction", "net_direction",
        }:
            errors.append("reference_alignment_mode must identify trajectory or endpoint alignment")
    for flag in BLOCKER_FLAGS:
        if flag in row and not isinstance(row[flag], bool):
            errors.append(f"blocker flag {flag!r} must be boolean")
        if strict_paper and flag not in row:
            errors.append(f"missing blocker flag '{flag}' for strict paper validation")
    return errors


def training_row_template(role: str = "partial") -> dict[str, Any]:
    """A fully populated, contract-valid training row for any robot/VLA."""
    return {
        "row_id": "trace_0001:chunk_03",
        "component": "widget",
        "phase": "approach",
        "split": "train",
        "collection_config": "cfg_a",
        "role": role,
        "operator_label": "near_but_not_accurate" if role in {"partial", "auto_success"} else None,
        "mask": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        "geometry": {
            "tcp_positions": [[0.30, 0.00, 0.20], [0.30, 0.00, 0.12], [0.30, 0.00, 0.06]],
            "target_point": [0.30, 0.00, 0.05],
            "target_id": "widget_0",
            "orientation_ok": True,
        },
        "target_meta": {"target_id": "widget_0", "declared_target_id": "widget_0"},
        "stop": {"phase_ending": True, "stop_event": "model_stop_token", "inside_terminal": True},
        "provenance": {
            "manifest_valid": True,
            "timing_aligned": True,
            "component_identity_valid": True,
            "action_mask_valid": True,
        },
        "safety": {"unsafe": False, "manual_safety_stop": False, "quarantined": False},
        "rank_weight": 1.0,
    }


def validation_row_template() -> dict[str, Any]:
    """A fully populated, contract-valid validation-scoring row."""
    return {
        "row_id": "val_0001",
        "component": "widget",
        "role": "partial",
        "submetrics": {
            "progress": 0.8,
            "proximity": 0.6,
            "terminal": 0.0,
            "direction": 0.9,
            "align": 0.85,
            "no_regression": 1.0,
        },
        "wrong_target": False,
        "non_target_exclusion": False,
        "invalid_orientation": False,
        "unsafe": False,
    }
