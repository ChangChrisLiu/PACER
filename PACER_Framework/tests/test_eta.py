import pytest

from pacer_framework.eta import CANDIDATE_POOL, FREE_BOUNDS, PaperEta, process_score

# Table app_candidates, transcribed cell by cell from the PACER paper appendix.
PAPER_CANDIDATE_TABLE = {
    "rw_like":                 (0.30, 0.45, 0.15, 0.05, 1.20, 0.60, 1.00, 4.50),
    "progress_heavy":          (0.55, 0.20, 0.15, 0.05, 1.20, 0.60, 1.00, 4.50),
    "hover_proximity_heavy":   (0.20, 0.60, 0.10, 0.05, 1.20, 0.60, 1.00, 4.50),
    "terminal_stop_heavy":     (0.20, 0.25, 0.35, 0.15, 1.20, 0.60, 1.00, 4.50),
    "correction_heavy":        (0.30, 0.45, 0.15, 0.05, 1.80, 0.70, 1.00, 4.50),
    "conservative":            (0.30, 0.45, 0.15, 0.05, 1.20, 0.40, 1.50, 2.50),
    "ram_connector_recovery":  (0.45, 0.30, 0.10, 0.05, 1.70, 0.90, 1.00, 5.00),
    "balanced_low_clip":       (0.30, 0.45, 0.15, 0.05, 1.20, 0.60, 0.85, 3.50),
}


def test_candidate_pool_matches_paper_table_exactly():
    assert set(CANDIDATE_POOL) == set(PAPER_CANDIDATE_TABLE)
    for name, expected in PAPER_CANDIDATE_TABLE.items():
        eta = CANDIDATE_POOL[name]
        actual = (
            eta.beta_prog, eta.beta_prox, eta.beta_term, eta.beta_stop,
            eta.gamma_correction, eta.gamma_partial, eta.tau, eta.w_max,
        )
        assert actual == expected, name


def test_fixed_protocol_fields_match_eq_eta_fixed():
    assert PaperEta.FREE_FIELDS == (
        "beta_prog", "beta_prox", "beta_term", "beta_stop",
        "gamma_correction", "gamma_partial", "tau", "w_max",
    )
    assert PaperEta.FIXED_FIELDS == (
        "beta_dir", "beta_op", "beta_prov",
        "gamma_clean", "gamma_auto_success", "gamma_failure", "gamma_excluded",
        "f_clean", "f_corr",
    )
    for name, eta in CANDIDATE_POOL.items():
        assert eta.beta_dir == 0.0, name
        assert eta.beta_op == 0.05, name
        assert eta.beta_prov == 0.0, name
        assert eta.gamma_clean == 1.00, name
        assert eta.gamma_auto_success == 1.00, name
        assert eta.gamma_failure == 0.0, name
        assert eta.gamma_excluded == 0.0, name
        assert eta.f_clean == 0.80, name
        assert eta.f_corr == 0.65, name


def test_process_score_uses_raw_coefficients_without_renormalization():
    eta = CANDIDATE_POOL["ram_connector_recovery"]
    all_ones = {k: 1.0 for k in ("prog", "prox", "term", "dir", "stop", "op", "prov")}
    # Raw sum: 0.45 + 0.30 + 0.10 + 0.05 + beta_op 0.05 + 0 + 0 = 0.95, NOT 1.0.
    assert process_score(eta, all_ones) == pytest.approx(0.95, abs=1e-12)
    assert process_score(eta, {"prog": 1.0}) == pytest.approx(0.45, abs=1e-12)
    # dir/prov carry zero coefficient in this paper.
    assert process_score(eta, {"dir": 1.0, "prov": 1.0}) == 0.0


def test_validate_bounds_against_table_app_eta_free():
    for name, eta in CANDIDATE_POOL.items():
        assert eta.validate_bounds() == [], name
    bad = PaperEta(beta_prog=0.05)
    errors = bad.validate_bounds()
    assert any("beta_prog" in e for e in errors)
    assert FREE_BOUNDS["beta_stop"] == (0.00, 0.20)


def test_role_multiplier_and_floor_mapping():
    eta = CANDIDATE_POOL["correction_heavy"]
    assert eta.role_multiplier("clean") == 1.00
    assert eta.role_multiplier("correction") == 1.80
    assert eta.role_multiplier("auto_success") == 1.00
    assert eta.role_multiplier("partial") == 0.70
    assert eta.role_multiplier("failure") == 0.0
    assert eta.role_multiplier("excluded") == 0.0
    assert eta.floor("clean") == 0.80
    assert eta.floor("correction") == 0.65
    assert eta.floor("partial") == 0.0
