"""The eight paper baselines (Table app_baselines + eq:app_fixed_geometry).

Modes:
- no_post_train        — pi_theta0; zero training weight everywhere.
- vanilla_post_sft     — clean + correction rows at weight 1. The PACER
  eligibility gate is NOT used as a row-level inclusion rule (gate column
  "no"); the shared filters that all methods apply — train split, per-horizon
  mask validity, and safety exclusions (app:baselines, "All methods use the
  same parser validity filters, safety exclusions, and per-horizon masks") —
  still hold.
- uniform_all_eligible — every gate-eligible row at weight 1.
- outcome_only         — gate-eligible clean + auto_success rows at weight 1.
- clean_only           — gate-eligible supplementary clean rows at weight 1.
- correction_only      — gate-eligible correction rows at weight 1.
- fixed_geometry       — w_i = g_i * max(0, r_rank) using the frozen
  prior-pipeline rank scalar carried by the row (eq:app_fixed_geometry).
- pacer_selected       — full PACER weighting w_i(eta*) via weights.compile_weights.

Manifests report the Table app_baselines accounting: positively weighted train
rows, valid horizons summed over those rows, and nominal-start rows (positive
rows whose role is clean, auto_success, or partial).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Mapping

from pacer_framework.eta import PaperEta
from pacer_framework.gate import eligibility_gate, mask_has_mass, row_is_unsafe
from pacer_framework.roles import normalize_role
from pacer_framework.weights import compile_weights

BASELINE_MODES = (
    "no_post_train",
    "vanilla_post_sft",
    "uniform_all_eligible",
    "outcome_only",
    "clean_only",
    "correction_only",
    "fixed_geometry",
    "pacer_selected",
)

_MODE_ROLES = {
    "vanilla_post_sft": frozenset({"clean", "correction"}),
    "outcome_only": frozenset({"clean", "auto_success"}),
    "clean_only": frozenset({"clean"}),
    "correction_only": frozenset({"correction"}),
}

NOMINAL_START_ROLES = frozenset({"clean", "auto_success", "partial"})

PACER_FRAMEWORK_BASELINE_SCHEMA = "pacer_framework_baseline_view.v1"


def _rank_weight(row: Mapping[str, Any]) -> float:
    value = row.get("rank_weight")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, out)


def baseline_weight(
    row: Mapping[str, Any],
    mode: str,
    *,
    train_split: str = "train",
) -> tuple[float, list[str]]:
    """Per-row baseline weight and ineligibility reasons (pacer_selected excluded)."""
    if mode not in BASELINE_MODES or mode == "pacer_selected":
        raise ValueError(f"baseline_weight does not handle mode {mode!r}; use compile_baseline")
    role = normalize_role(row.get("role") or row.get("sample_role"), strict=False)

    if mode == "no_post_train":
        return 0.0, ["mode_no_post_train"]

    if mode == "vanilla_post_sft":
        reasons: list[str] = []
        if row.get("split") != train_split:
            reasons.append("non_train_split")
        if not mask_has_mass(row.get("mask")):
            reasons.append("no_valid_mask_mass")
        if row_is_unsafe(row):
            reasons.append("unsafe_or_quarantined")
        if role not in _MODE_ROLES[mode]:
            reasons.append(f"mode_excludes_role:{role}")
        return (1.0, []) if not reasons else (0.0, reasons)

    gate, gate_reasons = eligibility_gate(row, train_split=train_split)
    if not gate:
        return 0.0, gate_reasons

    if mode == "uniform_all_eligible":
        return 1.0, []
    if mode in _MODE_ROLES:
        if role not in _MODE_ROLES[mode]:
            return 0.0, [f"mode_excludes_role:{role}"]
        return 1.0, []
    if mode == "fixed_geometry":
        weight = _rank_weight(row)
        return (weight, []) if weight > 0.0 else (0.0, ["non_positive_rank_weight"])
    raise ValueError(f"unhandled baseline mode {mode!r}")


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


def compile_baseline(
    rows: list[dict[str, Any]],
    mode: str,
    *,
    eta: PaperEta | None = None,
    eta_id: str | None = None,
    train_split: str = "train",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize a baseline weighting view plus a Table-app_baselines manifest."""
    if mode not in BASELINE_MODES:
        raise ValueError(f"unknown baseline mode {mode!r}; choices={list(BASELINE_MODES)}")

    if mode == "pacer_selected":
        if eta is None:
            raise ValueError("pacer_selected requires the selected PaperEta")
        weighted_rows, weight_manifest = compile_weights(rows, eta, eta_id=eta_id, train_split=train_split)
        out_rows = []
        for row in weighted_rows:
            row["pacer_framework_baseline"] = {
                "schema": PACER_FRAMEWORK_BASELINE_SCHEMA,
                "mode": mode,
                "weight": row["pacer_framework"]["weight"],
                "reasons": row["pacer_framework"]["gate_reasons"],
            }
            out_rows.append(row)
        manifest = _baseline_manifest(out_rows, mode=mode, uses_gate=True, weight_rule="w_i(eta_star)")
        manifest["pacer_weight_manifest"] = weight_manifest
        return out_rows, manifest

    out_rows = []
    for row in rows:
        weight, reasons = baseline_weight(row, mode, train_split=train_split)
        new_row = json.loads(json.dumps(row))
        new_row["pacer_framework_baseline"] = {
            "schema": PACER_FRAMEWORK_BASELINE_SCHEMA,
            "mode": mode,
            "weight": float(weight),
            "reasons": reasons,
        }
        out_rows.append(new_row)
    uses_gate = mode not in {"no_post_train", "vanilla_post_sft"}
    rule = {
        "no_post_train": "none",
        "vanilla_post_sft": "1 on clean+correction (shared filters only)",
        "uniform_all_eligible": "1 on gate-eligible",
        "outcome_only": "1 on gate-eligible clean+auto_success",
        "clean_only": "1 on gate-eligible clean",
        "correction_only": "1 on gate-eligible correction",
        "fixed_geometry": "g_i * max(0, r_rank)",
    }[mode]
    return out_rows, _baseline_manifest(out_rows, mode=mode, uses_gate=uses_gate, weight_rule=rule)


def _baseline_manifest(rows: list[Mapping[str, Any]], *, mode: str, uses_gate: bool, weight_rule: str) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    train_rows = 0
    valid_horizons = 0.0
    nominal_start_rows = 0
    for row in rows:
        info = row.get("pacer_framework_baseline", {})
        weight = float(info.get("weight", 0.0))
        role = normalize_role(row.get("role") or row.get("sample_role"), strict=False)
        counters["rows"] += 1
        counters[f"role:{role}"] += 1
        if weight > 0.0:
            train_rows += 1
            valid_horizons += _mask_mass(row)
            if role in NOMINAL_START_ROLES:
                nominal_start_rows += 1
        else:
            for reason in info.get("reasons", []) or []:
                counters[f"ineligible:{reason}"] += 1
    return {
        "schema": PACER_FRAMEWORK_BASELINE_SCHEMA,
        "mode": mode,
        "uses_pacer_gate": uses_gate,
        "weight_rule": weight_rule,
        "train_rows": train_rows,
        "valid_horizons": valid_horizons,
        "nominal_start_rows": nominal_start_rows,
        "counts": dict(counters),
    }
