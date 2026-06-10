import json
from pathlib import Path

import pytest

from pacer.weighting.pacer_eval_contract import validate_pacer_bo_eval_report
from pacer.weighting.pacer_local_validation_eval import (
    build_pacer_scores_from_alignment_eval,
    write_pacer_scores_json,
)


def _alignment_payload():
    return {
        "checkpoint": "/tmp/8499",
        "config": "pi05_droid_ur5e_pacer_rwfma_lora_10hz",
        "evals": [
            {
                "mode": "original_initial_sft",
                "n_evaluated": 2,
                "mean_chunk_cosine_joint_delta": 0.50,
                "median_chunk_cosine_joint_delta": 0.50,
                "mean_l2_joint_abs": 0.10,
                "rows": [
                    {"chunk_cosine_joint_delta": 0.60, "final_step_cosine_joint_delta": 0.70, "mean_l2_joint_abs": 0.09, "final_l2_joint_abs": 0.08, "mean_l2_gripper_abs": 0.20},
                    {"chunk_cosine_joint_delta": 0.40, "final_step_cosine_joint_delta": 0.50, "mean_l2_joint_abs": 0.11, "final_l2_joint_abs": 0.10, "mean_l2_gripper_abs": 0.25},
                ],
            },
            {
                "mode": "pacer_clean_demo",
                "n_evaluated": 2,
                "mean_chunk_cosine_joint_delta": 0.80,
                "median_chunk_cosine_joint_delta": 0.80,
                "mean_l2_joint_abs": 0.05,
                "rows": [
                    {"chunk_cosine_joint_delta": 0.90, "final_step_cosine_joint_delta": 0.90, "mean_l2_joint_abs": 0.04, "final_l2_joint_abs": 0.04, "mean_l2_gripper_abs": 0.05},
                    {"chunk_cosine_joint_delta": 0.70, "final_step_cosine_joint_delta": 0.80, "mean_l2_joint_abs": 0.06, "final_l2_joint_abs": 0.05, "mean_l2_gripper_abs": 0.10},
                ],
            },
        ],
    }


def test_alignment_eval_converts_to_contract_valid_pacer_scores():
    report = build_pacer_scores_from_alignment_eval(
        _alignment_payload(),
        eta_id="fixed_rw_fma_8499_local_smoke",
        eta_hash="legacy8499",
        training_run_id="fixed_rw_fma_8499_local_validation_smoke",
    )

    assert report["schema"] == "pacer_bo_eval_scores.v0.1"
    assert report["split"] == "val"
    assert report["candidate"]["checkpoint_ref"] == "/tmp/8499"
    assert report["scores"]["action_vector_direction_cosine"] == 0.8
    assert report["scores"]["tcp_direction_cosine"] == 0.8
    assert 0.0 <= report["scores"]["J_B_val"] <= 1.0
    assert validate_pacer_bo_eval_report(report) == []


def test_alignment_eval_marks_wrong_target_rows_as_ineligible_for_selection():
    payload = _alignment_payload()
    payload["evals"][1]["rows"][0]["target_consistency"] = "wrong_target"

    with pytest.raises(ValueError, match="wrong_target_or_unsafe_leakage"):
        build_pacer_scores_from_alignment_eval(
            payload,
            eta_id="fixed_rw_fma_8499_local_smoke",
            eta_hash="legacy8499",
            training_run_id="fixed_rw_fma_8499_local_validation_smoke",
        )


def test_write_pacer_scores_json_round_trips_and_records_source(tmp_path: Path):
    src = tmp_path / "alignment.json"
    out = tmp_path / "validation" / "scores.json"
    src.write_text(json.dumps(_alignment_payload()))

    report = write_pacer_scores_json(
        alignment_eval_path=src,
        output_path=out,
        eta_id="fixed_rw_fma_8499_local_smoke",
        eta_hash="legacy8499",
        training_run_id="fixed_rw_fma_8499_local_validation_smoke",
    )

    loaded = json.loads(out.read_text())
    assert loaded == report
    assert loaded["source_alignment_eval"] == str(src)
    assert validate_pacer_bo_eval_report(loaded) == []
