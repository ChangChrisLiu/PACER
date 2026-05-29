"""Reviewer-safe PACER BO validation-output contract.

Candidate training jobs should write one ``scores.json`` with this schema after
running a fixed validation set. This module validates the JSON shape and the
authority rules only; it does not run model inference or hardware.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

PACER_BO_EVAL_SCHEMA_VERSION = "pacer_bo_eval_scores.v0.1"

REQUIRED_BO_SCORE_KEYS = frozenset(
    {
        "clean_demo_action_alignment",
        "correction_route_alignment",
        "failed_state_progress",
        "tcp_direction_cosine",
        "action_vector_direction_cosine",
        "stop_handoff_correctness",
        "wrong_target_or_unsafe_leakage",
        "component_macro_score",
        "ram_score",
        "connector_score",
        "no_regression_score",
        "stop_token_emission_score",
        "stop_token_timing_score",
        "approach_axis_alignment",
        "approach_lateral_drift_score",
        "J_B_val",
    }
)

_COSINE_SCORE_KEYS = {"tcp_direction_cosine", "action_vector_direction_cosine"}

RECOMMENDED_J_B_VAL_WEIGHTS = {
    "component_macro_score": 0.18,
    "clean_demo_action_alignment": 0.10,
    "correction_route_alignment": 0.12,
    "failed_state_progress": 0.10,
    "tcp_direction_cosine": 0.10,
    "stop_handoff_correctness": 0.08,
    "stop_token_emission_score": 0.12,
    "stop_token_timing_score": 0.08,
    "approach_axis_alignment": 0.07,
    "approach_lateral_drift_score": 0.05,
}


def compute_recommended_j_b_val(scores: Mapping[str, Any]) -> float:
    """Compute a VLM-free Stage-B validation objective for PACER selection.

    This intentionally includes the two failure modes observed on the previous
    RW-FMA 8499 checkpoint: weak stop-token emission and angled/lateral
    approach. Action-vector direction remains diagnostic and is not included.
    Wrong-target or unsafe leakage is a hard failure for candidate ranking.
    """
    leakage = scores.get("wrong_target_or_unsafe_leakage", 0.0)
    if _finite_number(leakage) and float(leakage) > 0.0:
        return 0.0
    total = 0.0
    weight_sum = 0.0
    for key, weight in RECOMMENDED_J_B_VAL_WEIGHTS.items():
        value = scores.get(key)
        if key == "tcp_direction_cosine" and value is not None:
            # Convert cosine [-1,1] into score [0,1].
            value = (float(value) + 1.0) / 2.0
        if not _finite_number(value):
            continue
        v = max(0.0, min(1.0, float(value)))
        total += float(weight) * v
        weight_sum += float(weight)
    if weight_sum <= 0:
        return 0.0
    return max(0.0, min(1.0, total / weight_sum))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _score_in_range(key: str, value: Any) -> bool:
    if value is None and key in _COSINE_SCORE_KEYS:
        return True
    if not _finite_number(value):
        return False
    lower = -1.0 if key in _COSINE_SCORE_KEYS else 0.0
    return lower <= float(value) <= 1.0


def validate_pacer_bo_eval_report(report: Mapping[str, Any]) -> list[str]:
    """Return validation errors for a PACER BO candidate ``scores.json``."""
    errors: list[str] = []
    if report.get("schema") != PACER_BO_EVAL_SCHEMA_VERSION:
        errors.append("bad_schema")
    if report.get("split") != "val":
        errors.append("split_must_be_val")

    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping):
        errors.append("candidate_must_be_object")
    else:
        for key in ("eta_id", "eta_hash", "checkpoint_ref", "training_run_id"):
            if not candidate.get(key):
                errors.append(f"missing_candidate:{key}")

    scores = report.get("scores")
    if not isinstance(scores, Mapping):
        errors.append("scores_must_be_object")
    else:
        for key in sorted(REQUIRED_BO_SCORE_KEYS):
            if key not in scores:
                errors.append(f"missing_score:{key}")
            elif not _score_in_range(key, scores.get(key)):
                errors.append(f"bad_score:{key}")

    # Keep the reported BO objective tied to the source-of-truth objective so
    # weak stop-token emission and angled/lateral approach cannot be logged but
    # ignored by candidate ranking.
    if isinstance(scores, Mapping) and _finite_number(scores.get("wrong_target_or_unsafe_leakage")):
        if float(scores["wrong_target_or_unsafe_leakage"]) != 0.0:
            errors.append("wrong_target_or_unsafe_leakage_must_be_zero")
    if isinstance(scores, Mapping) and _finite_number(scores.get("J_B_val")):
        recommended = compute_recommended_j_b_val(scores)
        if abs(float(scores["J_B_val"]) - recommended) > 1e-6:
            errors.append("j_b_val_mismatch_recommended_objective")

    authority = report.get("metric_authority")
    if not isinstance(authority, Mapping):
        errors.append("metric_authority_must_be_object")
    else:
        if authority.get("external_semantic_scalar_authority_used") is not False:
            errors.append("external_semantic_scalar_authority_used_must_be_false")
        if authority.get("external_semantic_action_authority_used") is not False:
            errors.append("external_semantic_action_authority_used_must_be_false")
        if authority.get("direction_metric_for_J_B_val") not in {"tcp_direction_cosine", "none"}:
            errors.append("direction_metric_for_J_B_val_must_be_tcp_or_none")
        if authority.get("action_vector_direction_is_diagnostic_only") is not True:
            errors.append("action_vector_direction_must_remain_diagnostic_only")
        if not authority.get("coordinate_convention"):
            errors.append("missing_coordinate_convention")

    gates = report.get("safety_gates")
    if not isinstance(gates, Mapping):
        errors.append("safety_gates_must_be_object")
    else:
        if gates.get("passed") is not True:
            errors.append("safety_gates_passed_must_be_true_for_bo_selection")
        violations = gates.get("violations", [])
        if not isinstance(violations, list):
            errors.append("safety_gates_violations_must_be_list")
        elif violations:
            errors.append("safety_gates_violations_must_be_empty_for_bo_selection")
        for key in ("forbidden_positive_weight_rows", "heldout_loss_eligible_rows"):
            value = gates.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                errors.append(f"safety_gate_count_must_be_zero:{key}")

    component_scores = report.get("component_scores", {})
    if not isinstance(component_scores, Mapping):
        errors.append("component_scores_must_be_object")
    return errors


def pacer_bo_eval_json_schema() -> dict[str, Any]:
    """Return a dependency-free JSON Schema-style description for scores.json."""
    score_properties = {
        key: {
            "type": ["number", "null"] if key in _COSINE_SCORE_KEYS else "number",
            "minimum": -1.0 if key in _COSINE_SCORE_KEYS else 0.0,
            "maximum": 1.0,
        }
        for key in sorted(REQUIRED_BO_SCORE_KEYS)
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PACER BO validation scores",
        "type": "object",
        "required": ["schema", "candidate", "split", "scores", "metric_authority", "safety_gates"],
        "properties": {
            "schema": {"const": PACER_BO_EVAL_SCHEMA_VERSION},
            "candidate": {
                "type": "object",
                "required": ["eta_id", "eta_hash", "checkpoint_ref", "training_run_id"],
                "properties": {
                    "eta_id": {"type": "string"},
                    "eta_hash": {"type": "string"},
                    "checkpoint_ref": {"type": "string"},
                    "training_run_id": {"type": "string"},
                },
            },
            "split": {"const": "val"},
            "scores": {
                "type": "object",
                "required": sorted(REQUIRED_BO_SCORE_KEYS),
                "properties": score_properties,
            },
            "component_scores": {"type": "object"},
            "metric_authority": {
                "type": "object",
                "required": [
                    "external_semantic_scalar_authority_used",
                    "external_semantic_action_authority_used",
                    "direction_metric_for_J_B_val",
                    "action_vector_direction_is_diagnostic_only",
                    "coordinate_convention",
                ],
                "properties": {
                    "external_semantic_scalar_authority_used": {"const": False},
                    "external_semantic_action_authority_used": {"const": False},
                    "direction_metric_for_J_B_val": {"enum": ["tcp_direction_cosine", "none"]},
                    "action_vector_direction_is_diagnostic_only": {"const": True},
                    "coordinate_convention": {"type": "string"},
                },
            },
            "safety_gates": {
                "type": "object",
                "required": ["passed", "violations", "forbidden_positive_weight_rows", "heldout_loss_eligible_rows"],
                "properties": {
                    "passed": {"const": True},
                    "violations": {"type": "array", "maxItems": 0},
                    "forbidden_positive_weight_rows": {"const": 0},
                    "heldout_loss_eligible_rows": {"const": 0},
                },
            },
        },
    }
