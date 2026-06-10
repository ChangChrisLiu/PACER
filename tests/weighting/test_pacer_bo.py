import json
from pathlib import Path

import numpy as np

from pacer.weighting.pacer_bo import (
    BayesianOptimizationResult,
    NoCandidateProposalsError,
    expected_improvement,
    fit_gp_surrogate,
    load_bo_observations,
    propose_next_etas,
    write_bo_proposal_artifacts,
)
from pacer.weighting.pacer_bo_weights import PacerEta
from pacer.weighting.pacer_eval_contract import compute_recommended_j_b_val


def _score_report(j: float) -> dict:
    scores = {
        "clean_demo_action_alignment": j,
        "correction_route_alignment": j,
        "failed_state_progress": j,
        "tcp_direction_cosine": 2 * j - 1,
        "action_vector_direction_cosine": 2 * j - 1,
        "stop_handoff_correctness": j,
        "wrong_target_or_unsafe_leakage": 0.0,
        "component_macro_score": j,
        "ram_score": j,
        "connector_score": j,
        "no_regression_score": j,
        "stop_token_emission_score": j,
        "stop_token_timing_score": j,
        "approach_axis_alignment": j,
        "approach_lateral_drift_score": j,
        "J_B_val": 0.0,
    }
    scores["J_B_val"] = compute_recommended_j_b_val(scores)
    return {
        "schema": "pacer_bo_eval_scores.v0.1",
        "split": "val",
        "candidate": {"eta_id": "unit", "eta_hash": "unit", "checkpoint_ref": "ckpt", "training_run_id": "run"},
        "scores": scores,
        "component_scores": {},
        "metric_authority": {
            "external_semantic_scalar_authority_used": False,
            "external_semantic_action_authority_used": False,
            "direction_metric_for_J_B_val": "tcp_direction_cosine",
            "action_vector_direction_is_diagnostic_only": True,
            "coordinate_convention": "unit_test",
        },
        "safety_gates": {"passed": True, "violations": [], "forbidden_positive_weight_rows": 0, "heldout_loss_eligible_rows": 0},
    }


def test_load_bo_observations_pairs_eta_configs_with_valid_scores(tmp_path: Path):
    root = tmp_path / "obs"
    candidate = root / "eta_progress_heavy"
    candidate.mkdir(parents=True)
    eta = PacerEta(progress=0.55, proximity=0.20, terminal=0.15, stop=0.05)
    (candidate / "eta_config.json").write_text(json.dumps(eta.to_dict()))
    score_path = candidate / "validation" / "scores.json"
    score_path.parent.mkdir()
    score_path.write_text(json.dumps(_score_report(0.73)))

    observations = load_bo_observations(root)

    assert len(observations) == 1
    assert observations[0].eta_id == "eta_progress_heavy"
    assert observations[0].score_path == score_path
    assert observations[0].eta.free_vector()["progress"] == 0.55
    assert 0.0 <= observations[0].x_normalized[0] <= 1.0
    assert observations[0].j_b_val == _score_report(0.73)["scores"]["J_B_val"]


def test_gp_surrogate_predicts_mean_and_positive_uncertainty():
    observations = []
    for i, x in enumerate(np.linspace(0.0, 1.0, 6)):
        eta = PacerEta.from_normalized([x, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4])
        observations.append((eta, 0.4 + 0.3 * x))

    surrogate = fit_gp_surrogate(observations, random_state=7)
    mean, std = surrogate.predict(np.array([[0.5] + [0.4] * 7]), return_std=True)

    assert mean.shape == (1,)
    assert std.shape == (1,)
    assert float(std[0]) >= 0.0


def test_expected_improvement_is_positive_for_uncertain_better_region():
    mu = np.array([0.8, 0.4])
    sigma = np.array([0.1, 0.1])

    ei = expected_improvement(mu, sigma, y_best=0.6, xi=0.01)

    assert ei[0] > ei[1]
    assert ei[0] > 0.0
    assert ei[1] >= 0.0


def test_propose_next_etas_excludes_observed_points_and_sorts_by_acquisition():
    observed = []
    for x, score in [(0.0, 0.45), (0.25, 0.55), (0.5, 0.62), (0.75, 0.60), (1.0, 0.58)]:
        observed.append((PacerEta.from_normalized([x] * 8), score))

    result = propose_next_etas(observed, n_candidates=3, random_state=11, n_search_samples=256, min_distance=1e-3)

    assert isinstance(result, BayesianOptimizationResult)
    assert result.surrogate_model == "GaussianProcessRegressor"
    assert result.acquisition_rule == "expected_improvement"
    assert len(result.proposals) == 3
    acq = [p.acquisition_value for p in result.proposals]
    assert acq == sorted(acq, reverse=True)
    observed_x = {tuple(np.round(eta.to_normalized(), 8)) for eta, _ in observed}
    for proposal in result.proposals:
        assert tuple(np.round(proposal.eta.to_normalized(), 8)) not in observed_x
        for dim, value in proposal.eta.free_vector().items():
            lo, hi = PacerEta.SEARCH_RANGES[dim]
            assert lo <= value <= hi


def test_write_bo_proposal_artifacts_round_trips_json(tmp_path: Path):
    observed = [(PacerEta.from_normalized([0.1] * 8), 0.4), (PacerEta.from_normalized([0.9] * 8), 0.7)]
    result = propose_next_etas(observed, n_candidates=2, random_state=3, n_search_samples=64)

    out = write_bo_proposal_artifacts(result, tmp_path)

    payload = json.loads((out / "next_eta_candidates.json").read_text())
    assert payload["schema"] == "pacer_bo_proposals.v0.1"
    assert payload["surrogate_model"] == "GaussianProcessRegressor"
    assert payload["acquisition_rule"] == "expected_improvement"
    assert "kernel_fitted" in payload
    assert "fit_warnings" in payload
    assert "min_distance" in payload
    assert len(payload["proposals"]) == 2
    for item in payload["proposals"]:
        assert (out / item["eta_config_path"]).exists()
        assert item["nearest_observed_eta_id"] is not None
        assert item["nearest_observed_distance"] >= payload["min_distance"]


def test_propose_next_etas_reports_candidate_pool_exhaustion():
    observed = [(PacerEta.from_normalized([0.5] * 8), 0.7), (PacerEta.from_normalized([0.6] * 8), 0.71)]

    try:
        propose_next_etas(observed, n_candidates=1, random_state=3, n_search_samples=0, min_distance=10.0)
    except NoCandidateProposalsError as exc:
        assert "No candidate proposals remain" in str(exc)
    else:
        raise AssertionError("expected NoCandidateProposalsError")
