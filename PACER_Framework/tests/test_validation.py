import pytest

from pacer_framework.validation import (
    DEFAULT_SUBMETRIC_WEIGHTS,
    candidate_feasibility,
    component_balanced_j_val,
    geometric_row_score,
    no_regression,
    row_score,
    select_candidate,
)


def val_row(component="ram", role="partial", **over):
    row = {
        "row_id": f"{component}:{role}",
        "component": component,
        "role": role,
        "submetrics": {"progress": 0.8, "proximity": 0.6, "terminal": 0.4, "direction": 1.0},
    }
    row.update(over)
    return row


def test_default_lambda_includes_align_and_no_regression():
    for name in ("progress", "proximity", "terminal", "direction", "stop", "align", "no_regression"):
        assert name in DEFAULT_SUBMETRIC_WEIGHTS


def test_row_score_renormalizes_over_applicable_submetrics():
    score = row_score(val_row())
    expected = (0.25 * 0.8 + 0.25 * 0.6 + 0.20 * 0.4 + 0.15 * 1.0) / (0.25 + 0.25 + 0.20 + 0.15)
    assert score.score == pytest.approx(expected, abs=1e-4)
    assert set(score.used_submetrics) == {"progress", "proximity", "terminal", "direction"}

    full = val_row()
    full["submetrics"].update({"stop": 1.0, "align": 1.0, "no_regression": 1.0})
    score_full = row_score(full)
    assert {"stop", "align", "no_regression"} <= set(score_full.used_submetrics)


def test_blocker_flags_zero_rows_but_terminal_failure_does_not():
    ok = row_score(val_row())
    assert ok.score > 0.0 and ok.blocker_passed
    zero_term = val_row()
    zero_term["submetrics"]["terminal"] = 0.0
    assert row_score(zero_term).score > 0.0
    for flag in ("wrong_target", "non_target_exclusion", "invalid_orientation", "unsafe"):
        blocked = row_score(val_row(**{flag: True}))
        assert blocked.score == 0.0
        assert blocked.blocker_passed is False
        assert flag in blocked.audit_reasons


def test_roles_audit_only_and_compiler_vocabulary():
    for role in ("failure", "excluded", "model_failure"):
        score = row_score(val_row(role=role))
        assert score.score == 0.0
        assert score.positive_scoring is False
        assert "audit_only_role" in score.audit_reasons
    compiler = val_row()
    compiler.pop("role")
    compiler["sample_role"] = "human_correction"
    score = row_score(compiler)
    assert score.role == "correction" and score.positive_scoring


def test_geometric_row_score_and_no_regression():
    submetrics = {"progress": 1.0, "proximity": 1.0, "terminal": 1.0, "direction": 1.0, "stop": 0.0}
    assert geometric_row_score(submetrics) == pytest.approx(1.0, abs=1e-4)
    assert no_regression(0.70, 0.80, delta_reg=0.05) == 0.0
    assert no_regression(0.76, 0.80, delta_reg=0.05) == 1.0


def test_component_balanced_j_val_averages_components_not_rows():
    rows = [
        val_row(component="ram", role="clean", submetrics={"progress": 1.0}),
        val_row(component="ram", role="correction", submetrics={"progress": 1.0}),
        val_row(component="cpu", role="partial", submetrics={"progress": 0.0}),
        val_row(component="cpu", role="failure", submetrics={"progress": 1.0}),
    ]
    j_val, scores, manifest = component_balanced_j_val(rows, submetric_weights={"progress": 1.0})
    assert j_val == pytest.approx(0.5, abs=1e-4)
    assert manifest["component_scores"]["cpu"] == 0.0
    assert manifest["component_scores"]["ram"] == pytest.approx(1.0, abs=1e-4)
    assert manifest["num_positive_scoring_rows"] == 3
    assert len(scores) == 4


def _good_manifest(**over):
    manifest = {
        "forbidden_positive_weight_rows": 0,
        "loss_normalizer": 20.0,
        "weight_max": 2.0,
        "n_eff_horizons": 20.0,
    }
    manifest.update(over)
    return manifest


def test_candidate_feasibility_eq_app_feasibility():
    _, clean_scores, _ = component_balanced_j_val([val_row()])
    feasible, checks = candidate_feasibility(
        weight_manifest=_good_manifest(),
        validation_scores=clean_scores,
        w_max=4.5,
    )
    assert feasible and all(checks.values())

    _, leak_scores, _ = component_balanced_j_val([val_row(wrong_target=True)])
    feasible, checks = candidate_feasibility(
        weight_manifest=_good_manifest(), validation_scores=leak_scores, w_max=4.5
    )
    assert not feasible and checks["A_wrong_zero_leakage"] is False

    _, unsafe_scores, _ = component_balanced_j_val([val_row(unsafe=True)])
    feasible, checks = candidate_feasibility(
        weight_manifest=_good_manifest(), validation_scores=unsafe_scores, w_max=4.5
    )
    assert not feasible and checks["A_safe_zero_unsafe"] is False

    feasible, checks = candidate_feasibility(
        weight_manifest=_good_manifest(n_eff_horizons=0.5),
        validation_scores=clean_scores,
        w_max=4.5,
        n_eff_min=1.0,
    )
    assert not feasible and checks["A_audit_n_eff"] is False

    feasible, checks = candidate_feasibility(
        weight_manifest=_good_manifest(weight_max=9.0),
        validation_scores=clean_scores,
        w_max=4.5,
    )
    assert not feasible and checks["A_audit_weight_within_clip"] is False

    feasible, checks = candidate_feasibility(
        weight_manifest=_good_manifest(),
        validation_scores=clean_scores,
        w_max=4.5,
        candidate_component_scores={"ram": 0.70},
        reference_component_scores={"ram": 0.80},
        delta_reg=0.05,
    )
    assert not feasible and checks["A_reg_no_regression"] is False

    feasible, _ = candidate_feasibility(
        weight_manifest=_good_manifest(),
        validation_scores=clean_scores,
        w_max=4.5,
        candidate_component_scores={"ram": 0.76},
        reference_component_scores={"ram": 0.80},
        delta_reg=0.05,
    )
    assert feasible


def test_select_candidate_argmax_with_reference_fallback():
    j_vals = {"a": 0.70, "b": 0.90, "c": 0.95}
    feasible = {"a": True, "b": True, "c": False}
    assert select_candidate(j_vals, feasible) == "b"
    assert select_candidate(j_vals, {k: False for k in j_vals}) is None
    # Deterministic tie-break by sorted candidate id.
    assert select_candidate({"a": 0.5, "b": 0.5}, {"a": True, "b": True}) == "a"
