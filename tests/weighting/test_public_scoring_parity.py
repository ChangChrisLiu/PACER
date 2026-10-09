"""Runtime adapters expose the same scorer as the portable framework."""
from pacer_framework import ScoringConfig
from pacer_framework.validation import component_balanced_j_val as canonical_score
from pacer_framework.validation import DEFAULT_SUBMETRIC_WEIGHTS as canonical_defaults
from pacer.weighting.pacer_validation_scoring import component_balanced_j_val, DEFAULT_SUBMETRIC_WEIGHTS


def test_runtime_and_framework_share_parameters_and_edge_row_semantics():
    config = ScoringConfig(submetric_weights={"progress": 1, "no_regression": 1})
    rows = [
        {"row_id": "clean", "component": "widget", "role": "clean",
         "submetrics": {"progress": 0.8, "reg": 0.0},
         "reference_submetrics": {"progress": 0.5}},
        {"row_id": "blocked", "component": "widget", "role": "partial",
         "submetrics": {"progress": 1.0}, "unsafe": 1},
        {"row_id": "excluded", "component": "socket", "role": "failure",
         "submetrics": {"progress": 1.0}},
    ]
    assert DEFAULT_SUBMETRIC_WEIGHTS is canonical_defaults
    assert component_balanced_j_val(rows, scoring_config=config) == canonical_score(rows, scoring_config=config)
