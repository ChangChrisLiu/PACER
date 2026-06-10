from pacer_framework.schema import (
    training_row_template,
    training_row_warnings,
    validate_training_row,
    validate_validation_row,
    validation_row_template,
)


def test_templates_satisfy_their_own_contract():
    for role in ("clean", "correction", "auto_success", "partial", "failure", "excluded"):
        assert validate_training_row(training_row_template(role)) == [], role
    assert validate_validation_row(validation_row_template()) == []


def test_missing_required_fields_are_reported():
    row = training_row_template()
    row.pop("component")
    row["split"] = "test"  # not a valid split
    errors = validate_training_row(row)
    assert any("component" in e for e in errors)
    assert any("split" in e for e in errors)


def test_role_must_be_known_in_either_vocabulary():
    row = training_row_template()
    row["role"] = "mystery"
    assert any("unknown role" in e for e in validate_training_row(row))
    row["role"] = "human_correction"  # compiler vocabulary is fine
    assert validate_training_row(row) == []
    row.pop("role")
    assert any("missing role" in e for e in validate_training_row(row))


def test_mask_must_be_binary_per_horizon():
    for bad in ([], [1, 2], [1, -1], [0.5], "not-a-list"):
        row = training_row_template()
        row["mask"] = bad
        assert validate_training_row(row), bad


def test_evidence_block_is_checked_against_the_seven_keys():
    row = training_row_template()
    row.pop("geometry")
    row["evidence"] = {"prog": 0.5, "bogus": 1.0}
    errors = validate_training_row(row)
    assert any("unknown evidence keys" in e for e in errors)
    row["evidence"] = {"prog": 1.5}
    assert any("[0, 1]" in e for e in validate_training_row(row))
    row["evidence"] = {"prog": 0.5, "prov": 1.0}
    assert validate_training_row(row) == []


def test_positive_roles_require_an_evidence_source_but_failure_does_not():
    row = training_row_template("partial")
    row.pop("geometry")
    assert any("requires an 'evidence' or 'geometry'" in e for e in validate_training_row(row))
    failure = training_row_template("failure")
    failure.pop("geometry")
    assert validate_training_row(failure) == []


def test_geometry_block_accepts_trajectory_or_summary_form():
    row = training_row_template()
    row["geometry"] = {"tcp_positions": [[0.1, 0.0, 0.0]]}  # missing target_point
    assert any("target_point" in e for e in validate_training_row(row))
    row["geometry"] = {"d0": 0.10, "d_end": 0.02}
    assert validate_training_row(row) == []


def test_validation_row_contract():
    row = validation_row_template()
    row["submetrics"]["progress"] = 1.2
    assert any("[0, 1]" in e for e in validate_validation_row(row))
    row = validation_row_template()
    row["wrong_target"] = "yes"
    assert any("must be boolean" in e for e in validate_validation_row(row))
    row = validation_row_template()
    row["submetrics"] = {}
    assert any("submetrics" in e for e in validate_validation_row(row))


def test_warnings_surface_t_zero_and_split_readiness():
    row = training_row_template()
    row.pop("target_meta")
    row["geometry"].pop("target_id")
    warnings = training_row_warnings(row)
    assert any("T_i = 0" in w for w in warnings)
    row = training_row_template("auto_success")
    row["operator_label"] = None
    row.pop("collection_config")
    warnings = training_row_warnings(row)
    assert any("operator_label" in w for w in warnings)
    assert any("collection_config" in w for w in warnings)
