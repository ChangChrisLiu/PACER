import pytest

from pacer_framework.baselines import BASELINE_MODES, compile_baseline
from pacer_framework.eta import CANDIDATE_POOL


def row(role, *, rank_weight=1.0, split="train", **over):
    out = {
        "row_id": f"row:{role}",
        "component": "ram",
        "phase": "approach",
        "split": split,
        "role": role,
        "mask": [1] * 10,
        "rank_weight": rank_weight,
        "safety": {"unsafe": False, "manual_safety_stop": False, "quarantined": False},
        "target_meta": {"target_id": "ram_0", "geometry_record_present": True},
        "evidence": {"prog": 0.5, "prox": 0.5, "term": 0.0, "dir": 0.0, "stop": 0.0, "op": 0.5, "prov": 1.0},
    }
    out.update(over)
    return out


def full_buffer():
    return [
        row("clean", rank_weight=1.0),
        row("correction", rank_weight=1.5),
        row("partial", rank_weight=0.4),
        row("auto_success", rank_weight=0.8),
        row("failure", rank_weight=0.9),
        row("excluded", rank_weight=0.9),
    ]


def positive_roles(rows):
    return {
        r["role"]
        for r in rows
        if r["pacer_framework_baseline"]["weight"] > 0
    }


def test_paper_baseline_mode_list():
    assert BASELINE_MODES == (
        "no_post_train",
        "vanilla_post_sft",
        "uniform_all_eligible",
        "outcome_only",
        "clean_only",
        "correction_only",
        "fixed_geometry",
        "pacer_selected",
    )


def test_baseline_modes_select_table_app_baselines_role_sets():
    expected = {
        "no_post_train": set(),
        "vanilla_post_sft": {"clean", "correction"},
        "uniform_all_eligible": {"clean", "correction", "partial", "auto_success"},
        "outcome_only": {"clean", "auto_success"},
        "clean_only": {"clean"},
        "correction_only": {"correction"},
        "fixed_geometry": {"clean", "correction", "partial", "auto_success"},
    }
    for mode, roles in expected.items():
        out, manifest = compile_baseline(full_buffer(), mode)
        assert positive_roles(out) == roles, mode
        assert manifest["mode"] == mode


def test_vanilla_post_sft_skips_pacer_gate_but_keeps_shared_filters():
    untrusted = row("clean", provenance={"manifest_valid": False})
    out, _ = compile_baseline([untrusted], "vanilla_post_sft")
    assert out[0]["pacer_framework_baseline"]["weight"] == 1.0  # gate column "no"
    out, _ = compile_baseline([untrusted], "uniform_all_eligible")
    assert out[0]["pacer_framework_baseline"]["weight"] == 0.0  # gate column "yes"

    unsafe = row("clean", safety={"unsafe": True})
    out, _ = compile_baseline([unsafe], "vanilla_post_sft")
    assert out[0]["pacer_framework_baseline"]["weight"] == 0.0  # shared safety exclusion
    masked = row("correction", mask=[0] * 10)
    out, _ = compile_baseline([masked], "vanilla_post_sft")
    assert out[0]["pacer_framework_baseline"]["weight"] == 0.0  # shared mask filter
    val_row = row("clean", split="val")
    out, _ = compile_baseline([val_row], "vanilla_post_sft")
    assert out[0]["pacer_framework_baseline"]["weight"] == 0.0  # split discipline


def test_fixed_geometry_uses_frozen_rank_weight():
    out, _ = compile_baseline(full_buffer(), "fixed_geometry")
    by_role = {r["role"]: r["pacer_framework_baseline"] for r in out}
    assert by_role["clean"]["weight"] == 1.0
    assert by_role["correction"]["weight"] == 1.5
    assert by_role["partial"]["weight"] == 0.4
    assert by_role["failure"]["weight"] == 0.0  # gate dominates the rank weight
    zero_rank = row("clean", rank_weight=0.0)
    out, _ = compile_baseline([zero_rank], "fixed_geometry")
    info = out[0]["pacer_framework_baseline"]
    assert info["weight"] == 0.0 and "non_positive_rank_weight" in info["reasons"]


def test_pacer_selected_delegates_to_compile_weights():
    eta = CANDIDATE_POOL["terminal_stop_heavy"]
    out, manifest = compile_baseline(full_buffer(), "pacer_selected", eta=eta, eta_id="terminal_stop_heavy")
    for r in out:
        assert r["pacer_framework_baseline"]["weight"] == r["pacer_framework"]["weight"]
    assert positive_roles(out) == {"clean", "correction", "partial", "auto_success"}
    assert manifest["pacer_weight_manifest"]["forbidden_positive_weight_rows"] == 0
    with pytest.raises(ValueError):
        compile_baseline(full_buffer(), "pacer_selected")


def test_manifest_accounting_matches_table_app_baselines_columns():
    rows = [
        row("clean", row_id="c1"),
        row("clean", row_id="c2"),
        row("correction", row_id="k1"),
        row("partial", row_id="p1"),
        row("failure", row_id="f1"),
    ]
    _out, manifest = compile_baseline(rows, "uniform_all_eligible")
    assert manifest["train_rows"] == 4
    assert manifest["valid_horizons"] == 40.0
    assert manifest["nominal_start_rows"] == 3  # correction rows are not nominal-start


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        compile_baseline(full_buffer(), "not_a_mode")
