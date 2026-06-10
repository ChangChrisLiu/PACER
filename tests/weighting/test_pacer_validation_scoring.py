from tracevla.weighting.pacer_validation_scoring import (
    DEFAULT_SUBMETRIC_WEIGHTS,
    component_balanced_j_val,
    score_validation_row,
)


def _row(component="ram", role="clean", **kwargs):
    row = {
        "row_id": f"{component}:{role}",
        "component": component,
        "role": role,
        "submetrics": {
            "progress": 0.8,
            "proximity": 0.6,
            "terminal": 0.4,
            "direction": 1.0,
            "stop": 0.5,
        },
    }
    row.update(kwargs)
    return row


def test_row_score_renormalizes_when_stop_submetric_is_omitted():
    row = _row()
    row["submetrics"].pop("stop")

    score = score_validation_row(row)

    expected = (0.25 * 0.8 + 0.25 * 0.6 + 0.20 * 0.4 + 0.15 * 1.0) / (0.25 + 0.25 + 0.20 + 0.15)
    assert score.used_submetrics == ("progress", "proximity", "terminal", "direction")
    assert abs(score.score - expected) < 1e-9


def test_wrong_target_and_unsafe_are_hard_blockers_but_terminal_failure_is_not():
    partial = _row(role="partial")
    partial["submetrics"]["terminal"] = 0.0
    assert score_validation_row(partial).score > 0.0

    wrong = _row(role="partial", wrong_target=True)
    wrong_score = score_validation_row(wrong)
    assert wrong_score.score == 0.0
    assert wrong_score.blocker_passed is False
    assert "wrong_target" in wrong_score.audit_reasons

    unsafe = _row(role="partial", unsafe=True)
    unsafe_score = score_validation_row(unsafe)
    assert unsafe_score.score == 0.0
    assert "unsafe" in unsafe_score.audit_reasons


def test_failure_and_excluded_validation_rows_are_audit_only():
    for role in ["failure", "excluded"]:
        score = score_validation_row(_row(role=role))
        assert score.score == 0.0
        assert score.positive_scoring is False
        assert "audit_only_role" in score.audit_reasons


def test_component_balanced_j_val_averages_components_not_rows():
    rows = [
        _row(component="ram", role="clean", submetrics={"progress": 1.0}),
        _row(component="ram", role="correction", submetrics={"progress": 1.0}),
        _row(component="cpu", role="partial", submetrics={"progress": 0.0}),
        _row(component="cpu", role="failure", submetrics={"progress": 1.0}),
    ]

    j_val, row_scores, manifest = component_balanced_j_val(rows, submetric_weights={"progress": 1.0})

    assert j_val == 0.5
    assert manifest["component_scores"] == {"cpu": 0.0, "ram": 1.0}
    assert manifest["num_positive_scoring_rows"] == 3
    assert manifest["audit_counts"] == {"audit_only_role": 1}
    assert len(row_scores) == 4


def test_validation_scoring_accepts_compiler_role_names():
    row = {
        "row_id": "compiler-row",
        "component": "ram",
        "sample_role": "human_correction",
        "submetrics": {"progress": 1.0, "proximity": 1.0},
    }

    score = score_validation_row(row)

    assert score.role == "correction"
    assert score.positive_scoring is True
    assert score.score == 1.0


def test_default_submetric_weights_include_alignment_and_no_regression():
    assert "align" in DEFAULT_SUBMETRIC_WEIGHTS
    assert "no_regression" in DEFAULT_SUBMETRIC_WEIGHTS
