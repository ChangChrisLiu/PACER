"""Local PACER validation-score adapter for offline checkpoint alignment smokes.

This module does not run robot hardware and does not claim real success. It takes
a deterministic offline alignment evaluation, such as the legacy 8499 checkpoint
clean-demo alignment JSON, and maps it into the PACER `validation/scores.json`
contract so the metric matrix, J_B_val computation, authority fields, and safety
gates can be verified locally before HPRC.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from tracevla.stage_b_rl.pacer_eval_contract import (
    PACER_BO_EVAL_SCHEMA_VERSION,
    compute_recommended_j_b_val,
    validate_pacer_bo_eval_report,
)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _cos_to_score(v: Any) -> float:
    if not _finite(v):
        return 0.0
    return _clip01((float(v) + 1.0) / 2.0)


def _rows(evals: Iterable[Mapping[str, Any]], mode: str | None = None) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for item in evals:
        if mode is not None and item.get("mode") != mode:
            continue
        rs = item.get("rows") or []
        if isinstance(rs, list):
            out.extend(r for r in rs if isinstance(r, Mapping))
    return out


def _mean_finite(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [float(v) for v in values if _finite(v)]
    if not vals:
        return default
    return float(mean(vals))


def _l2_score(values: Iterable[Any], *, scale: float = 0.20) -> float:
    """Convert small L2 errors into [0,1] scores with a conservative scale."""
    m = _mean_finite(values, default=scale)
    return _clip01(1.0 - m / max(scale, 1e-9))


def _has_wrong_target_or_unsafe_row(rows: Iterable[Mapping[str, Any]]) -> bool:
    wrong_values = {"wrong_target", "wrong_region", "wrong_target_abort"}
    for row in rows:
        target_consistency = str(row.get("target_consistency") or row.get("target_match") or "")
        stop_label = str(row.get("operator_stop_label") or row.get("stop_source") or "")
        if target_consistency in wrong_values or stop_label in wrong_values:
            return True
        if row.get("unsafe") is True or row.get("safety_violation") is True:
            return True
    return False


def build_pacer_scores_from_alignment_eval(
    payload: Mapping[str, Any],
    *,
    eta_id: str,
    eta_hash: str,
    training_run_id: str,
) -> dict[str, Any]:
    """Build a contract-valid PACER scores report from offline alignment JSON.

    The 8499 alignment file reports joint/action-chunk agreement against initial
    SFT and Stage-B clean-demo rows. These are not full robot validation metrics,
    so unsupported quantities are filled conservatively from available alignment,
    stop/gripper telemetry, and safety defaults. The report is marked as a local
    smoke via `evaluation_scope`.
    """
    evals = payload.get("evals") or []
    if not isinstance(evals, list):
        evals = []

    all_rows = _rows(evals)
    clean_rows = _rows(evals, "stageb_clean_demo") or all_rows
    initial_rows = _rows(evals, "original_initial_sft")

    clean_chunk_cos = _mean_finite((r.get("chunk_cosine_joint_delta") for r in clean_rows), default=0.0)
    clean_final_cos = _mean_finite((r.get("final_step_cosine_joint_delta") for r in clean_rows), default=clean_chunk_cos)
    all_chunk_cos = _mean_finite((r.get("chunk_cosine_joint_delta") for r in all_rows), default=clean_chunk_cos)
    initial_chunk_cos = _mean_finite((r.get("chunk_cosine_joint_delta") for r in initial_rows), default=all_chunk_cos)

    clean_l2 = _l2_score((r.get("mean_l2_joint_abs") for r in clean_rows), scale=0.20)
    final_l2 = _l2_score((r.get("final_l2_joint_abs") for r in clean_rows), scale=0.20)
    gripper_l2 = _l2_score((r.get("mean_l2_gripper_abs") for r in clean_rows), scale=0.50)

    direction_score = _cos_to_score(clean_chunk_cos)
    final_direction_score = _cos_to_score(clean_final_cos)
    initial_direction_score = _cos_to_score(initial_chunk_cos)
    wrong_target_or_unsafe_leakage = 1.0 if _has_wrong_target_or_unsafe_row(all_rows) else 0.0

    clean_demo_alignment = _clip01(0.45 * direction_score + 0.35 * clean_l2 + 0.20 * final_l2)
    # Legacy alignment JSON has no human-correction route eval; use clean-demo
    # alignment as a conservative runnable placeholder rather than claiming a
    # separate correction metric exists.
    correction_route_alignment = _clip01(0.80 * clean_demo_alignment)
    failed_state_progress = _clip01(0.50 * direction_score + 0.50 * final_direction_score)
    stop_handoff_correctness = _clip01(gripper_l2)
    stop_token_emission_score = _clip01(gripper_l2)
    stop_token_timing_score = _clip01(0.5 * gripper_l2 + 0.5 * final_l2)
    approach_axis_alignment = _clip01(final_direction_score)
    approach_lateral_drift_score = _clip01(final_l2)
    no_regression_score = _clip01(initial_direction_score)
    component_macro_score = _clip01(mean([clean_demo_alignment, failed_state_progress, no_regression_score]))

    scores = {
        "clean_demo_action_alignment": clean_demo_alignment,
        "correction_route_alignment": correction_route_alignment,
        "failed_state_progress": failed_state_progress,
        "tcp_direction_cosine": float(clean_chunk_cos) if _finite(clean_chunk_cos) else 0.0,
        "action_vector_direction_cosine": float(clean_chunk_cos) if _finite(clean_chunk_cos) else 0.0,
        "stop_handoff_correctness": stop_handoff_correctness,
        "wrong_target_or_unsafe_leakage": wrong_target_or_unsafe_leakage,
        "component_macro_score": component_macro_score,
        "ram_score": component_macro_score,
        "connector_score": component_macro_score,
        "no_regression_score": no_regression_score,
        "stop_token_emission_score": stop_token_emission_score,
        "stop_token_timing_score": stop_token_timing_score,
        "approach_axis_alignment": approach_axis_alignment,
        "approach_lateral_drift_score": approach_lateral_drift_score,
        "J_B_val": 0.0,
    }
    scores["J_B_val"] = compute_recommended_j_b_val(scores)

    report = {
        "schema": PACER_BO_EVAL_SCHEMA_VERSION,
        "split": "val",
        "candidate": {
            "eta_id": eta_id,
            "eta_hash": eta_hash,
            "checkpoint_ref": str(payload.get("checkpoint") or "unknown_checkpoint"),
            "training_run_id": training_run_id,
        },
        "scores": scores,
        "component_scores": {
            "local_alignment_smoke": component_macro_score,
            "rows_evaluated": len(all_rows),
            "clean_rows_evaluated": len(clean_rows),
            "initial_rows_evaluated": len(initial_rows),
        },
        "metric_authority": {
            "external_semantic_scalar_authority_used": False,
            "external_semantic_action_authority_used": False,
            "direction_metric_for_J_B_val": "tcp_direction_cosine",
            "action_vector_direction_is_diagnostic_only": True,
            "coordinate_convention": "offline_joint_delta_alignment_from_8499_smoke_not_robot_tcp_success",
        },
        "safety_gates": {
            "passed": wrong_target_or_unsafe_leakage == 0.0,
            "violations": ["wrong_target_or_unsafe_leakage"] if wrong_target_or_unsafe_leakage > 0.0 else [],
            "forbidden_positive_weight_rows": 0,
            "heldout_loss_eligible_rows": 0,
        },
        "evaluation_scope": "local_offline_8499_alignment_smoke_not_hardware_success",
        "notes": [
            "This verifies the PACER validation matrix and J_B_val contract locally.",
            "It is not a substitute for the real HPRC evaluator over frozen validation states.",
        ],
    }
    errors = validate_pacer_bo_eval_report(report)
    if errors:
        raise ValueError(f"generated invalid PACER scores report: {errors}")
    return report


def write_pacer_scores_json(
    *,
    alignment_eval_path: str | Path,
    output_path: str | Path,
    eta_id: str,
    eta_hash: str,
    training_run_id: str,
) -> dict[str, Any]:
    alignment_eval_path = Path(alignment_eval_path)
    output_path = Path(output_path)
    payload = json.loads(alignment_eval_path.read_text())
    report = build_pacer_scores_from_alignment_eval(
        payload,
        eta_id=eta_id,
        eta_hash=eta_hash,
        training_run_id=training_run_id,
    )
    report["source_alignment_eval"] = str(alignment_eval_path)
    # Recompute validation after adding harmless provenance field.
    errors = validate_pacer_bo_eval_report(report)
    if errors:
        raise ValueError(f"invalid PACER scores report after provenance: {errors}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
