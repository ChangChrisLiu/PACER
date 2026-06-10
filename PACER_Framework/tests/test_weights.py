import math

import pytest

from pacer_framework.eta import CANDIDATE_POOL, PaperEta
from pacer_framework.weights import (
    SIGMA_MIN,
    compile_weights,
    robust_center_scale,
    row_weight,
)


def paper_row(role, *, component="ram", phase="approach", split="train", prog=0.5, mask=None, **over):
    row = {
        "row_id": f"{component}:{phase}:{role}:{prog}",
        "component": component,
        "phase": phase,
        "split": split,
        "role": role,
        "mask": mask if mask is not None else [1] * 10,
        "safety": {"unsafe": False, "manual_safety_stop": False, "quarantined": False},
        "target_meta": {"target_id": f"{component}_0", "geometry_record_present": True},
        "evidence": {"prog": prog, "prox": 0.0, "term": 0.0, "dir": 0.0, "stop": 0.0, "op": 0.0, "prov": 1.0},
    }
    row.update(over)
    return row


def test_robust_center_scale_is_median_mad_with_sigma_floor():
    center, scale = robust_center_scale([0.2, 0.5, 0.8])
    assert center == pytest.approx(0.5)
    assert scale == pytest.approx(1.4826 * 0.3, rel=1e-9)
    # Degenerate MAD floors at sigma_min with NO standard-deviation fallback.
    center, scale = robust_center_scale([0.5, 0.5, 0.9])
    assert center == pytest.approx(0.5)
    assert scale == SIGMA_MIN


def test_row_weight_matches_eq_weight_hand_computation():
    eta = PaperEta()  # tau=1, w_max=4.5, gamma_partial=0.6
    sigma = 1.4826 * 0.3
    w = row_weight(eta, "partial", score=0.8, center=0.5, scale=sigma, gate=True)
    expected = min(4.5, max(0.0, 0.6 * math.exp(0.3 / (sigma + 1e-6))))
    assert w == pytest.approx(expected, rel=1e-9)
    assert w == pytest.approx(1.1779, abs=2e-3)
    # Role multiplier sits outside the exponential: the median row gets gamma.
    assert row_weight(eta, "partial", 0.5, 0.5, sigma, True) == pytest.approx(0.6, rel=1e-9)


def test_floors_apply_only_to_clean_and_correction():
    eta = PaperEta()
    sigma = SIGMA_MIN
    # Strongly negative advantage drives exp() to ~0.
    assert row_weight(eta, "clean", 0.0, 1.0, sigma, True) == pytest.approx(0.80)
    assert row_weight(eta, "correction", 0.0, 1.0, sigma, True) == pytest.approx(0.65)
    assert row_weight(eta, "partial", 0.0, 1.0, sigma, True) == pytest.approx(0.0, abs=1e-12)
    assert row_weight(eta, "auto_success", 0.0, 1.0, sigma, True) == pytest.approx(0.0, abs=1e-12)


def test_gate_dominates_floor_and_upper_clip_handles_overflow():
    eta = PaperEta()
    assert row_weight(eta, "clean", 1.0, 0.0, SIGMA_MIN, gate=False) == 0.0
    # Collapsed scale + positive advantage saturates at w_max (overflow-guarded).
    assert row_weight(eta, "partial", 1.0, 0.0, SIGMA_MIN, gate=True) == eta.w_max


def test_compile_weights_stratum_statistics_and_fallback_levels():
    eta = PaperEta()  # rw_like: beta_prog = 0.30
    rows = [
        paper_row("partial", prog=0.0),
        paper_row("partial", prog=0.5),
        paper_row("partial", prog=1.0),
        paper_row("failure", prog=1.0),                      # same (c, phi): falls back to component_phase
        paper_row("excluded", component="cpu", phase="lever"),  # no (c, phi) pool: global
    ]
    out, manifest = compile_weights(rows, eta, eta_id="unit")
    info = {r["row_id"]: r["pacer_framework"] for r in out}

    # Scores are 0.30 * prog -> [0.0, 0.15, 0.30]; med 0.15, sigma = 1.4826 * 0.15.
    sigma = 1.4826 * 0.15
    top = info["ram:approach:partial:1.0"]
    assert top["stratum_level"] == "stratum"
    assert top["process_score"] == pytest.approx(0.30, rel=1e-9)
    assert top["weight"] == pytest.approx(0.6 * math.exp(0.15 / (sigma + 1e-6)), rel=1e-6)
    mid = info["ram:approach:partial:0.5"]
    assert mid["weight"] == pytest.approx(0.6, rel=1e-6)
    bottom = info["ram:approach:partial:0.0"]
    assert bottom["weight"] == pytest.approx(0.6 * math.exp(-0.15 / (sigma + 1e-6)), rel=1e-6)

    failure = info["ram:approach:failure:1.0"]
    assert failure["weight"] == 0.0
    assert failure["stratum_level"] == "component_phase"
    excluded = info["cpu:lever:excluded:0.5"]
    assert excluded["weight"] == 0.0
    assert excluded["stratum_level"] == "global"

    assert manifest["num_strata"] == 1
    assert manifest["forbidden_positive_weight_rows"] == 0
    assert manifest["counts"]["eligible"] == 3


def test_n_eff_and_loss_normalizer_over_weighted_masked_horizons():
    eta = PaperEta()
    rows = [
        paper_row("clean", prog=1.0, row_id="clean_a"),
        paper_row("clean", prog=1.0, row_id="clean_b"),
    ]
    out, manifest = compile_weights(rows, eta, eta_id="unit")
    weights = [r["pacer_framework"]["weight"] for r in out]
    assert weights == [pytest.approx(1.0, rel=1e-9)] * 2  # z=0 -> gamma_clean * e^0 = 1
    assert manifest["loss_normalizer"] == pytest.approx(20.0, rel=1e-6)
    assert manifest["n_eff_horizons"] == pytest.approx(20.0, rel=1e-4)


def test_zero_mass_mask_blocks_row_entirely():
    eta = PaperEta()
    out, manifest = compile_weights([paper_row("clean", mask=[0] * 10)], eta, eta_id="unit")
    pf = out[0]["pacer_framework"]
    assert pf["weight"] == 0.0
    assert "gate_mask_no_valid_mass" in pf["gate_reasons"]
    assert manifest["counts"].get("eligible", 0) == 0


def test_compile_weights_rejects_out_of_bounds_eta():
    with pytest.raises(ValueError):
        compile_weights([paper_row("clean")], PaperEta(beta_prog=0.9), eta_id="unit")


def test_candidate_pool_etas_all_compile():
    rows = [paper_row("partial", prog=p) for p in (0.0, 0.5, 1.0)]
    for name, eta in CANDIDATE_POOL.items():
        _out, manifest = compile_weights(rows, eta, eta_id=name)
        assert manifest["forbidden_positive_weight_rows"] == 0, name
        assert manifest["weight_max"] <= eta.w_max + 1e-9, name
