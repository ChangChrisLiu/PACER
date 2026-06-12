import pytest

from pacer_framework.alignment import (
    blocker_flags_from_open_loop,
    chunk_trajectory_alignment,
    direction_cosine01,
    open_loop_submetrics,
    reference_alignment,
    target_direction_cosine,
    with_no_regression,
)


def test_direction_cosine01_clips_to_unit_interval():
    assert direction_cosine01([1, 0, 0], [2, 0, 0]) == pytest.approx(1.0, abs=1e-4)
    assert direction_cosine01([1, 0, 0], [-1, 0, 0]) == 0.0  # opposite clips at 0
    assert direction_cosine01([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0, abs=1e-9)
    assert direction_cosine01([0, 0, 0], [1, 0, 0]) == 0.0  # degenerate -> eps denominator


def test_reference_alignment_eq_app_align():
    predicted = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    reference_same = [[5.0, 5.0, 5.0], [5.2, 5.0, 5.0]]
    reference_opposite = [[5.0, 5.0, 5.0], [4.8, 5.0, 5.0]]
    assert reference_alignment(predicted, reference_same) == pytest.approx(1.0, abs=1e-4)
    assert reference_alignment(predicted, reference_opposite) == 0.0


def test_chunk_trajectory_alignment_distinguishes_path_from_endpoint_direction():
    reference = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]
    same_endpoint_but_zigzag = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, -1.0, 0.0],
        [3.0, 0.0, 0.0],
    ]
    matched = [
        [5.0, 2.0, 0.0],
        [6.0, 2.0, 0.0],
        [7.0, 2.0, 0.0],
        [8.0, 2.0, 0.0],
    ]

    # Endpoint/net direction cannot distinguish these two paths.
    assert reference_alignment(same_endpoint_but_zigzag, reference) == pytest.approx(1.0, abs=1e-4)

    # Whole chunk alignment rewards matching the demonstrated per-step path instead.
    assert chunk_trajectory_alignment(matched, reference) == pytest.approx(1.0, abs=1e-4)
    assert chunk_trajectory_alignment(same_endpoint_but_zigzag, reference) < 0.60


def test_chunk_trajectory_alignment_uses_common_horizon_for_short_chunks():
    reference = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    predicted_short = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]
    assert chunk_trajectory_alignment(predicted_short, reference) == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError, match="at least two"):
        chunk_trajectory_alignment([[0.0, 0.0, 0.0]], reference)


def test_target_direction_cosine_reads_chunk_displacement():
    toward = [[0.2, 0.0, 0.0], [0.15, 0.0, 0.0]]
    away = [[0.2, 0.0, 0.0], [0.25, 0.0, 0.0]]
    target = [0.1, 0.0, 0.0]
    assert target_direction_cosine(toward, target) == pytest.approx(1.0, abs=1e-3)
    assert target_direction_cosine(away, target) == 0.0


def test_open_loop_submetrics_match_training_predicates():
    predicted = [[0.20, 0.0, 0.0], [0.105, 0.0, 0.0]]
    sub = open_loop_submetrics(predicted, target_point=[0.10, 0.0, 0.0], component="widget")
    assert sub["terminal"] == 1.0  # d_end = 0.005 <= 0.010
    assert sub["proximity"] == pytest.approx(0.75, abs=1e-6)
    assert sub["progress"] == pytest.approx(0.95, abs=1e-4)
    assert sub["direction"] == pytest.approx(1.0, abs=1e-3)
    assert "stop" not in sub  # non-phase-ending rows omit the stop submetric
    assert "align" not in sub

    sub = open_loop_submetrics(
        predicted,
        target_point=[0.10, 0.0, 0.0],
        component="widget",
        phase_ending=True,
        stop_emitted=True,
        reference_positions=[[0.30, 0.0, 0.0], [0.20, 0.0, 0.0]],
    )
    assert sub["stop"] == 1.0
    assert sub["align"] == pytest.approx(1.0, abs=1e-3)
    sub = open_loop_submetrics(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, -1.0, 0.0], [3.0, 0.0, 0.0]],
        target_point=[3.0, 0.0, 0.0],
        component="widget",
        reference_positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        reference_alignment_mode="trajectory",
    )
    assert sub["align"] < 0.60
    sub = open_loop_submetrics(
        predicted, target_point=[0.10, 0.0, 0.0], component="widget",
        phase_ending=True, stop_emitted=False,
    )
    assert sub["stop"] == 0.0


def test_with_no_regression_appends_eq_app_reg_indicator():
    good = {"progress": 1.0, "proximity": 1.0, "terminal": 1.0, "direction": 1.0}
    weak = {"progress": 0.5, "proximity": 0.5, "terminal": 0.0, "direction": 0.5}
    out = with_no_regression(good, good, delta_reg=0.0)
    assert out["no_regression"] == 1.0
    out = with_no_regression(weak, good, delta_reg=0.1)
    assert out["no_regression"] == 0.0
    assert "no_regression" not in weak  # input not mutated


NON_TARGET = {"point": [0.0, 0.0, 0.0], "radius": 0.02}


def test_blocker_flags_wrong_target_and_exclusion():
    # Path passes through the non-target region and terminates inside it,
    # far from the active target -> both flags.
    flags = blocker_flags_from_open_loop(
        [[0.10, 0.0, 0.0], [0.01, 0.0, 0.0]],
        target_point=[0.50, 0.0, 0.0],
        component="widget",
        non_target_regions=[NON_TARGET],
    )
    assert flags["wrong_target"] is True
    assert flags["non_target_exclusion"] is True
    assert flags["invalid_orientation"] is False
    assert flags["unsafe"] is False

    # Passing near the non-target region early but terminating at the active
    # target: exclusion only, no wrong-target endpoint.
    flags = blocker_flags_from_open_loop(
        [[0.01, 0.0, 0.0], [0.50, 0.0, 0.0]],
        target_point=[0.50, 0.0, 0.0],
        component="widget",
        non_target_regions=[NON_TARGET],
    )
    assert flags["wrong_target"] is False
    assert flags["non_target_exclusion"] is True

    # Clean approach: no flags.
    flags = blocker_flags_from_open_loop(
        [[0.40, 0.0, 0.0], [0.50, 0.0, 0.0]],
        target_point=[0.50, 0.0, 0.0],
        component="widget",
        non_target_regions=[NON_TARGET],
    )
    assert not any(flags.values())


def test_blocker_flags_orientation_and_unsafe_predicates():
    positions = [[0.40, 0.0, 0.0], [0.50, 0.0, 0.0]]
    target = [0.50, 0.0, 0.0]
    flags = blocker_flags_from_open_loop(positions, target_point=target, orientation_ok=False)
    assert flags["invalid_orientation"] is True

    bounds = {"min": [0.0, -0.1, -0.1], "max": [0.45, 0.1, 0.1]}
    flags = blocker_flags_from_open_loop(positions, target_point=target, workspace_bounds=bounds)
    assert flags["unsafe"] is True  # endpoint leaves the calibrated workspace

    flags = blocker_flags_from_open_loop(positions, target_point=target, stop_source="unsafe_abort")
    assert flags["unsafe"] is True
    flags = blocker_flags_from_open_loop(positions, target_point=target, action_bounds_ok=False)
    assert flags["unsafe"] is True
