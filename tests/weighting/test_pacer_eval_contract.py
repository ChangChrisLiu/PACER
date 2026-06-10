from pacer.weighting.pacer_eval_contract import (
    PACER_BO_EVAL_SCHEMA_VERSION,
    REQUIRED_BO_SCORE_KEYS,
    compute_recommended_j_b_val,
    pacer_bo_eval_json_schema,
    validate_pacer_bo_eval_report,
)


def _valid_report():
    report = {
        "schema": PACER_BO_EVAL_SCHEMA_VERSION,
        "candidate": {
            "eta_id": "rw_like",
            "eta_hash": "abc123",
            "checkpoint_ref": "checkpoint://candidate",
            "training_run_id": "candidate_001",
        },
        "split": "val",
        "scores": {
            "clean_demo_action_alignment": 0.81,
            "correction_route_alignment": 0.72,
            "failed_state_progress": 0.68,
            "tcp_direction_cosine": 0.63,
            "action_vector_direction_cosine": 0.59,
            "stop_handoff_correctness": 0.9,
            "wrong_target_or_unsafe_leakage": 0.0,
            "component_macro_score": 0.74,
            "ram_score": 0.66,
            "connector_score": 0.62,
            "no_regression_score": 0.93,
            "stop_token_emission_score": 0.78,
            "stop_token_timing_score": 0.74,
            "approach_axis_alignment": 0.69,
            "approach_lateral_drift_score": 0.71,
            "J_B_val": 0.76,
        },
        "component_scores": {
            "ram": {"component_macro_score": 0.66},
            "connector": {"component_macro_score": 0.62},
        },
        "metric_authority": {
            "external_semantic_scalar_authority_used": False,
            "external_semantic_action_authority_used": False,
            "direction_metric_for_J_B_val": "tcp_direction_cosine",
            "action_vector_direction_is_diagnostic_only": True,
            "coordinate_convention": "eef_delta_in_base_frame",
        },
        "safety_gates": {
            "passed": True,
            "violations": [],
            "forbidden_positive_weight_rows": 0,
            "heldout_loss_eligible_rows": 0,
        },
    }
    report["scores"]["J_B_val"] = compute_recommended_j_b_val(report["scores"])
    return report


def test_eval_contract_accepts_reviewer_safe_scores_json():
    report = _valid_report()
    assert validate_pacer_bo_eval_report(report) == []

    schema = pacer_bo_eval_json_schema()
    assert schema["properties"]["scores"]["required"] == sorted(REQUIRED_BO_SCORE_KEYS)
    assert schema["properties"]["metric_authority"]["properties"]["direction_metric_for_J_B_val"]["enum"] == [
        "tcp_direction_cosine",
        "none",
    ]


def test_eval_contract_blocks_external_semantic_scalar_or_action_authority():
    report = _valid_report()
    report["metric_authority"]["external_semantic_scalar_authority_used"] = True
    assert "external_semantic_scalar_authority_used_must_be_false" in validate_pacer_bo_eval_report(report)

    report = _valid_report()
    report["metric_authority"]["external_semantic_action_authority_used"] = True
    assert "external_semantic_action_authority_used_must_be_false" in validate_pacer_bo_eval_report(report)


def test_eval_contract_keeps_action_vector_direction_diagnostic_only():
    report = _valid_report()
    report["metric_authority"]["direction_metric_for_J_B_val"] = "action_vector_direction_cosine"
    report["metric_authority"]["action_vector_direction_is_diagnostic_only"] = False
    errors = validate_pacer_bo_eval_report(report)
    assert "direction_metric_for_J_B_val_must_be_tcp_or_none" in errors
    assert "action_vector_direction_must_remain_diagnostic_only" in errors


def test_eval_contract_requires_stop_and_approach_scores_for_rw_fma_8499_failure_modes():
    report = _valid_report()
    del report["scores"]["stop_token_emission_score"]
    del report["scores"]["approach_axis_alignment"]

    errors = validate_pacer_bo_eval_report(report)

    assert "missing_score:stop_token_emission_score" in errors
    assert "missing_score:approach_axis_alignment" in errors


def test_recommended_j_b_val_penalizes_weak_stop_and_angled_approach():
    good = _valid_report()["scores"]
    good = dict(good)
    good["J_B_val"] = compute_recommended_j_b_val(good)
    bad = dict(good)
    bad["stop_token_emission_score"] = 0.1
    bad["stop_token_timing_score"] = 0.2
    bad["approach_axis_alignment"] = 0.2
    bad["approach_lateral_drift_score"] = 0.2

    assert compute_recommended_j_b_val(good) > compute_recommended_j_b_val(bad)
    assert 0.0 <= compute_recommended_j_b_val(bad) <= 1.0


def test_eval_contract_rejects_j_b_val_that_ignores_stop_and_approach_terms():
    report = _valid_report()
    for key in [
        "stop_token_emission_score",
        "stop_token_timing_score",
        "approach_axis_alignment",
        "approach_lateral_drift_score",
    ]:
        report["scores"][key] = 0.0
    report["scores"]["J_B_val"] = 1.0

    errors = validate_pacer_bo_eval_report(report)

    assert "j_b_val_mismatch_recommended_objective" in errors


def test_eval_contract_rejects_wrong_target_leakage_and_zeroes_objective():
    report = _valid_report()
    report["scores"]["wrong_target_or_unsafe_leakage"] = 1.0
    report["scores"]["J_B_val"] = compute_recommended_j_b_val(report["scores"])

    errors = validate_pacer_bo_eval_report(report)

    assert report["scores"]["J_B_val"] == 0.0
    assert "wrong_target_or_unsafe_leakage_must_be_zero" in errors


def test_eval_contract_requires_val_split_j_b_val_and_safety_gate():
    report = _valid_report()
    report["split"] = "heldout"
    del report["scores"]["J_B_val"]
    report["safety_gates"]["passed"] = False
    errors = validate_pacer_bo_eval_report(report)
    assert "split_must_be_val" in errors
    assert "missing_score:J_B_val" in errors
    assert "safety_gates_passed_must_be_true_for_bo_selection" in errors
