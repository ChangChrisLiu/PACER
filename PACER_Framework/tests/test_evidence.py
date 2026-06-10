import pytest

from pacer_framework.evidence import (
    compute_evidence,
    d_ref_for,
    provenance_check,
    psi_op,
    r_target_for,
    stop_evidence,
    target_consistency,
)


def geometry_row(positions, *, component="ram", mask=None, **extra):
    row = {
        "row_id": "geo",
        "component": component,
        "phase": "approach",
        "split": "train",
        "role": "partial",
        "mask": mask if mask is not None else [1] * (len(positions) - 1),
        "operator_label": "near_but_not_accurate",
        "geometry": {
            "tcp_positions": positions,
            "target_point": [0.10, 0.0, 0.0],
            "target_id": f"{component}_0",
        },
    }
    row.update(extra)
    return row


def test_component_dependent_tolerances():
    assert r_target_for("cpu") == 0.005
    assert r_target_for("CPU") == 0.005
    assert r_target_for("ram") == 0.010
    assert d_ref_for("cpu") == 0.010
    assert d_ref_for("ram") == 0.020


def test_geometric_features_on_an_approach_chunk():
    ev = compute_evidence(geometry_row([[0.20, 0.0, 0.0], [0.14, 0.0, 0.0]]))
    # d0 = 0.10, d_end = d_min = 0.04 (eq:app_dist).
    assert ev["prog"] == pytest.approx(0.6, abs=1e-4)
    assert ev["prox"] == 0.0  # 1 - 0.04/0.02 clips to 0
    assert ev["term"] == 0.0  # 0.04 > r_target = 0.01
    assert ev["dir"] == pytest.approx(1.0, abs=1e-3)
    assert ev["op"] == pytest.approx(0.6)  # near_but_not_accurate
    assert ev["prov"] == 1.0  # absent provenance block = trusted


def test_terminal_entry_and_proximity_grading():
    ev = compute_evidence(geometry_row([[0.20, 0.0, 0.0], [0.105, 0.0, 0.0]]))
    assert ev["term"] == 1.0  # d_end = 0.005 <= 0.010
    assert ev["prox"] == pytest.approx(0.75, abs=1e-6)  # 1 - 0.005/0.02
    assert ev["prog"] == pytest.approx(0.95, abs=1e-4)


def test_cpu_uses_tighter_tolerance():
    ev = compute_evidence(geometry_row([[0.20, 0.0, 0.0], [0.105, 0.0, 0.0]], component="cpu"))
    assert ev["term"] == 1.0  # d_end = 0.005 <= r_target(cpu) = 0.005
    assert ev["prox"] == pytest.approx(0.5, abs=1e-6)  # d_ref(cpu) = 0.010


def test_target_inconsistency_zeroes_geometric_evidence_only():
    row = geometry_row([[0.20, 0.0, 0.0], [0.105, 0.0, 0.0]])
    row["target_meta"] = {"target_id": "ram_0", "consistency": "wrong_target"}
    ev = compute_evidence(row)
    assert ev["prog"] == ev["prox"] == ev["term"] == ev["dir"] == 0.0
    assert ev["op"] == pytest.approx(0.6)
    assert ev["prov"] == 1.0


def test_missing_target_id_is_paper_strict_t_zero():
    row = geometry_row([[0.20, 0.0, 0.0], [0.105, 0.0, 0.0]])
    row["geometry"].pop("target_id")
    t_i, reasons = target_consistency(row)
    assert t_i == 0.0
    assert "missing_target_id" in reasons
    ev = compute_evidence(row)
    assert ev["term"] == 0.0 and ev["prog"] == 0.0


def test_precomputed_evidence_still_gets_t_factor():
    row = {
        "row_id": "pre",
        "component": "ram",
        "phase": "approach",
        "role": "auto_success",
        "split": "train",
        "mask": [1] * 10,
        "target_meta": {"target_id": "ram_0", "geometry_record_present": True, "consistency": "wrong_target"},
        "evidence": {"prog": 1.0, "prox": 1.0, "term": 1.0, "dir": 1.0, "stop": 1.0, "op": 0.8, "prov": 1.0},
    }
    ev = compute_evidence(row)
    assert ev["prog"] == ev["prox"] == ev["term"] == ev["dir"] == 0.0
    assert ev["stop"] == 1.0 and ev["op"] == 0.8 and ev["prov"] == 1.0


def test_mask_prefix_limits_geometry_to_valid_horizons():
    toward = [[0.20 - 0.02 * i, 0.0, 0.0] for i in range(6)]  # p0..p5 toward 0.10
    away = [[0.50, 0.0, 0.0]] * 5  # masked-out garbage tail
    row = geometry_row(toward + away, mask=[1] * 5 + [0] * 5)
    ev = compute_evidence(row)
    assert ev["dir"] == pytest.approx(1.0, abs=1e-3)  # garbage tail excluded
    assert ev["prog"] == pytest.approx(1.0, abs=1e-3)  # d_min = d_end = 0.0 at p5? no: p5 = 0.10 -> d=0; prog=(0.1-0)/0.1


def test_psi_op_lookup():
    assert psi_op("clean", None) == 1.0
    assert psi_op("correction", "totally_off_wrong_region_or_target") == 1.0
    assert psi_op("auto_success", "success") == 0.8
    assert psi_op("partial", "near_but_not_accurate") == 0.6
    assert psi_op("failure", "wrong_orientation_or_wrong_location") == 0.2
    assert psi_op("failure", "totally_off_wrong_region_or_target") == 0.0
    assert psi_op("excluded", "operator_uncertain_exclude") == 0.0
    assert psi_op("partial", "unknown_label") == 0.0


def test_provenance_check_requires_all_four_flags():
    assert provenance_check(None) == 1.0
    assert provenance_check({}) == 1.0
    assert provenance_check({"manifest_valid": True, "timing_aligned": True,
                             "component_identity_valid": True, "action_mask_valid": True}) == 1.0
    for flag in ("manifest_valid", "timing_aligned", "component_identity_valid", "action_mask_valid"):
        assert provenance_check({flag: False}) == 0.0, flag


def test_stop_evidence_semantics():
    assert stop_evidence({"phase_ending": True, "stop_event": "model_stop_token", "inside_terminal": True}) == 1.0
    assert stop_evidence({"phase_ending": True, "stop_event": "handoff", "inside_terminal": True}) == 1.0
    assert stop_evidence({"phase_ending": False, "stop_event": "model_stop_token", "inside_terminal": True}) == 0.0
    assert stop_evidence({"phase_ending": True, "stop_event": "timeout", "inside_terminal": True}) == 0.0
    assert stop_evidence({"phase_ending": True, "stop_event": "model_stop_token", "inside_terminal": False}) == 0.0
    assert stop_evidence(None) == 0.0
