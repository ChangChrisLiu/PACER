import pytest

from pacer_framework.roles import (
    PAPER_ROLES,
    normalize_role,
    role_from_source_and_label,
    strict_success,
)


def test_normalize_role_accepts_both_vocabularies():
    assert normalize_role("clean") == "clean"
    assert normalize_role("clean_demo") == "clean"
    assert normalize_role("human_correction") == "correction"
    assert normalize_role("model_success") == "auto_success"
    assert normalize_role("model_partial") == "partial"
    assert normalize_role("model_failure") == "failure"
    assert normalize_role("excluded") == "excluded"
    for role in PAPER_ROLES:
        assert normalize_role(role) == role


def test_normalize_role_strictness():
    with pytest.raises(ValueError):
        normalize_role("not_a_role")
    assert normalize_role("not_a_role", strict=False) == "not_a_role"


def test_role_from_source_and_label_matches_table_app_rolemap():
    assert role_from_source_and_label(source="clean_demo") == "clean"
    assert role_from_source_and_label(source="human_correction") == "correction"
    assert role_from_source_and_label(operator_label="success") == "auto_success"
    assert role_from_source_and_label(operator_label="near_but_not_accurate") == "partial"
    assert role_from_source_and_label(operator_label="wrong_orientation_or_wrong_location") == "failure"
    assert role_from_source_and_label(operator_label="totally_off_wrong_region_or_target") == "failure"
    assert role_from_source_and_label(operator_label="operator_uncertain_exclude") == "excluded"
    # Source dominates the autonomous label.
    assert role_from_source_and_label(source="human_correction", operator_label="success") == "correction"
    with pytest.raises(ValueError):
        role_from_source_and_label(operator_label="mystery_label")


def test_strict_success_eq_app_strict():
    assert strict_success("success", "model_stop_token") is True
    assert strict_success("success", "manual_stop_near_target") is True
    assert strict_success("success", "timeout") is False
    assert strict_success("success", "unsafe_abort") is False
    assert strict_success("success", "intervention") is False
    assert strict_success("near_but_not_accurate", "model_stop_token") is False
