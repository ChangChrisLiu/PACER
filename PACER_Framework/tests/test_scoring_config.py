"""Configuration behavior through the exported framework API."""
import json
import pytest

from pacer_framework import ScoringConfig, row_score, open_loop_submetrics, with_no_regression, evaluate_candidate


def test_scoring_config_roundtrips_an_editable_json_file(tmp_path):
    config = ScoringConfig(
        submetric_weights={"progress": 2, "align": 1},
        delta_reg=0.05,
    )
    path = tmp_path / "scoring.json"
    config.save(path)
    restored = ScoringConfig.load(path)

    assert dict(restored.submetric_weights) == {"progress": 2.0, "align": 1.0}
    assert restored.reference_alignment_mode == "trajectory"
    assert restored.delta_reg == 0.05
    assert restored.fingerprint == config.fingerprint
    assert restored.require_reference_component_scores is True
    assert json.loads(path.read_text())["submetric_weights"] == {"progress": 2.0, "align": 1.0}


def test_changed_coefficients_recompute_no_regression_from_reference_geometry():
    row = {
        "row_id": "example", "component": "widget", "role": "partial",
        "submetrics": {"progress": 1.0, "proximity": 0.0, "no_regression": 1.0},
        "reference_submetrics": {"progress": 0.0, "proximity": 1.0},
    }
    progress_first = ScoringConfig(submetric_weights={"progress": 4, "proximity": 1, "no_regression": 2})
    proximity_first = ScoringConfig(submetric_weights={"progress": 1, "proximity": 4, "no_regression": 2})
    assert row_score(row, scoring_config=progress_first).score > 0.8
    assert row_score(row, scoring_config=proximity_first).score < 0.2
    assert row["submetrics"]["no_regression"] == 1.0


def test_one_config_controls_whole_chunk_alignment_and_reference_comparison():
    prediction = [[0, 0, 0], [1, 1, 0], [2, -1, 0], [3, 0, 0]]
    reference = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    default = open_loop_submetrics(prediction, target_point=[3, 0, 0], reference_positions=reference)
    endpoint = open_loop_submetrics(
        prediction, target_point=[3, 0, 0], reference_positions=reference,
        scoring_config=ScoringConfig(reference_alignment_mode="endpoint"),
    )
    assert default["align"] < 0.6
    assert endpoint["align"] > 0.99
    config = ScoringConfig(submetric_weights={"progress": 1, "proximity": 4, "no_regression": 2})
    value = with_no_regression(
        {"progress": 1.0, "proximity": 0.0},
        {"progress": 0.0, "proximity": 1.0}, scoring_config=config,
    )
    assert value["no_regression"] == 0.0


def test_candidate_and_reference_use_one_config_and_missing_reference_fails_closed():
    config = ScoringConfig(submetric_weights={"progress": 1, "no_regression": 1})
    candidate = [{"row_id": "sample", "component": "widget", "role": "clean",
                  "submetrics": {"progress": 0.8, "no_regression": 0.0}}]
    reference = [{"row_id": "sample", "component": "widget", "role": "clean",
                  "submetrics": {"progress": 0.5}}]
    for row in candidate + reference:
        row.update({"wrong_target": False, "non_target_exclusion": False,
                    "invalid_orientation": False, "unsafe": False})
    weight_manifest = {"forbidden_positive_weight_rows": 0, "loss_normalizer": 10,
                       "weight_max": 1, "n_eff_horizons": 10}
    result = evaluate_candidate(validation_rows=candidate, reference_validation_rows=reference,
                                weight_manifest=weight_manifest, w_max=2, scoring_config=config)
    assert result["feasible"] is True
    assert result["j_val"] > 0.89
    assert result["reference_component_scores"]["widget"] > 0.74
    assert result["manifest"]["scoring_config_hash"] == config.fingerprint
    assert result["reference_manifest"]["scoring_config_hash"] == config.fingerprint
    assert candidate[0]["submetrics"]["no_regression"] == 0.0

    missing = evaluate_candidate(validation_rows=candidate, weight_manifest=weight_manifest, w_max=2)
    assert missing["feasible"] is False
    mismatched = evaluate_candidate(validation_rows=candidate, weight_manifest=weight_manifest, w_max=2,
                                    reference_component_scores={"missing": 0.0})
    assert mismatched["feasible"] is False


@pytest.mark.parametrize("kwargs", [
    {"submetric_weights": {}},
    {"submetric_weights": {"progress": 0}},
    {"submetric_weights": {"unknown": 1}},
    {"submetric_weights": {"progress": -1}},
    {"submetric_weights": {"progress": True}},
    {"submetric_weights": {"progress": float("nan")}},
    {"submetric_weights": {"progress": 1e308, "align": 1e308}},
    {"submetric_weights": {"no_regression": 1}},
    {"reference_alignment_mode": []},
    {"delta_reg": -1},
    {"require_reference_component_scores": "yes"},
])
def test_invalid_protocol_configuration_fails_loudly(kwargs):
    with pytest.raises(ValueError):
        ScoringConfig(**kwargs)


def test_invalid_provided_metrics_and_missing_expected_geometry_do_not_win():
    base = {"row_id": "row", "component": "widget", "role": "clean"}
    with pytest.raises(ValueError):
        row_score({**base, "submetrics": {"terminal": True}})
    missing = row_score({**base, "submetrics": {"align": 0.95, "no_regression": 1.0}})
    assert missing.score == 0.0
    assert "no_geometric_support" in missing.audit_reasons
    diagnostic = row_score({**base, "submetrics": {"align": 0.95}},
                           scoring_config=ScoringConfig(submetric_weights={"align": 1}))
    assert diagnostic.score > 0.94


def test_reference_support_mismatch_is_not_renormalized_into_a_comparison():
    config = ScoringConfig(submetric_weights={"progress": 1, "proximity": 1, "no_regression": 1})
    row = {"row_id": "row", "component": "widget", "role": "clean",
           "submetrics": {"progress": 0.8, "proximity": 0.2},
           "reference_submetrics": {"progress": 0.5}}
    with pytest.raises(ValueError, match="matching"):
        row_score(row, scoring_config=config)


def test_installed_demo_uses_custom_config_for_candidate_and_reference(tmp_path):
    from pacer_framework.demo import main

    config = ScoringConfig(submetric_weights={"progress": 2, "proximity": 1, "terminal": 1,
                                              "direction": 1, "align": 2, "no_regression": 1})
    result = main(out_dir=tmp_path, scoring_config=config)
    assert result["scoring_config_hash"] == config.fingerprint
    assert result["selected"] == "terminal_stop_heavy"
    for evaluation in result["evaluations"].values():
        assert evaluation["manifest"]["scoring_config_hash"] == config.fingerprint
        assert evaluation["reference_manifest"]["scoring_config_hash"] == config.fingerprint
        assert evaluation["manifest"]["no_regression_recomputed"] > 0
        assert evaluation["manifest"]["no_regression_cached"] == 0
    saved = json.loads((tmp_path / "evaluation.json").read_text())
    assert saved["synthetic_demo"] is True
    assert saved["scoring_config_hash"] == config.fingerprint


def test_validation_contract_accepts_reference_geometry_and_checks_its_types():
    from pacer_framework import validate_validation_row

    row = {"row_id": "sample", "component": "widget", "role": "clean",
           "submetrics": {"progress": 0.8, "align": 0.9},
           "reference_submetrics": {"progress": 0.5},
           "reference_alignment_mode": "trajectory",
           "wrong_target": False, "non_target_exclusion": False,
           "invalid_orientation": False, "unsafe": False}
    assert validate_validation_row(row, strict_paper=True) == []
    row["reference_submetrics"] = {"progress": True}
    assert validate_validation_row(row)
    row["reference_submetrics"] = {"progress": 0.5}
    row["reference_alignment_mode"] = "unknown"
    assert validate_validation_row(row)


def test_full_candidate_audit_does_not_infer_safe_flags_from_missing_inputs():
    row = {"row_id": "sample", "component": "widget", "role": "clean",
           "submetrics": {"progress": 1.0}}
    manifest = {"forbidden_positive_weight_rows": 0, "loss_normalizer": 10,
                "weight_max": 1, "n_eff_horizons": 10}
    result = evaluate_candidate(validation_rows=[row], weight_manifest=manifest, w_max=2,
                                reference_component_scores={"widget": 0.0})
    assert result["feasible"] is False
    assert result["checks"]["A_validation_blocker_inputs"] is False


def test_conflicting_embedded_and_matched_baseline_geometry_is_rejected():
    flags = {"wrong_target": False, "non_target_exclusion": False,
             "invalid_orientation": False, "unsafe": False}
    candidate = {"row_id": "sample", "component": "widget", "role": "clean",
                 "submetrics": {"progress": 0.5},
                 "reference_submetrics": {"progress": 0.9}, **flags}
    reference = {"row_id": "sample", "component": "widget", "role": "clean",
                 "submetrics": {"progress": 0.1}, **flags}
    manifest = {"forbidden_positive_weight_rows": 0, "loss_normalizer": 10,
                "weight_max": 1, "n_eff_horizons": 10}
    config = ScoringConfig(submetric_weights={"progress": 1, "no_regression": 1})
    with pytest.raises(ValueError, match="reference"):
        evaluate_candidate(validation_rows=[candidate], reference_validation_rows=[reference],
                           weight_manifest=manifest, w_max=2, scoring_config=config)


def test_explicit_diagnostic_reference_opt_out_is_marked_as_skipped():
    row = {"row_id": "sample", "component": "widget", "role": "clean",
           "submetrics": {"progress": 1.0}, "wrong_target": False,
           "non_target_exclusion": False, "invalid_orientation": False, "unsafe": False}
    manifest = {"forbidden_positive_weight_rows": 0, "loss_normalizer": 10,
                "weight_max": 1, "n_eff_horizons": 10}
    result = evaluate_candidate(validation_rows=[row], weight_manifest=manifest, w_max=2,
                                require_reference_component_scores=False)
    assert result["feasible"] is True
    assert result["checks"]["A_reg_reference_audit_skipped"] is True


def test_nonfinite_prediction_geometry_cannot_become_a_high_score():
    from pacer_framework import blocker_flags_from_open_loop

    positions = [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]
    with pytest.raises(ValueError):
        open_loop_submetrics(positions, target_point=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        blocker_flags_from_open_loop(positions, target_point=[1.0, 0.0, 0.0])
