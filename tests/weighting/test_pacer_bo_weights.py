import json
from pathlib import Path

from pacer.weighting.pacer_bo_weights import (
    ABLATION_WEIGHT_MODES,
    DEFAULT_ETA_POOL,
    PACER_ABLATION_WEIGHT_SCHEMA_VERSION,
    PacerEta,
    compute_ablation_weights,
    compute_pacer_weights,
    evidence_components,
    freeze_config_splits,
    hard_gate,
    raw_score,
    target_match,
    _robust_stats,
)


def _row(*, role="clean_demo", eligible=True, component="ram", config="config_001"):
    rank_weight = {
        "clean_demo": 1.0,
        "human_correction": 1.5,
        "model_partial": 0.4,
        "model_success": 0.8,
        "model_failure": 0.0,
    }.get(role, 1.0)
    return {
        "row_id": f"{config}:{component}:{role}",
        "config_id": config,
        "component": component,
        "block": "corrector_only",
        "sample_role": role,
        "split": "train",
        "returns": {
            "aw_fma_loss_eligible": eligible,
            "aw_fma_vlm_authority_used": False,
            "rank_weight_unvalidated": rank_weight,
            "aw_fma_reward_components": {
                "r_progress_signed01": 0.8,
                "r_proximity": 0.7,
                "r_terminal": 0.4,
                "r_stop_handoff": 0.3,
                "r_operator_rank": 0.9,
            },
        },
        "eligibility": {"quarantined": False},
        "semantic_sidecar": {"vlm_authority_used": False},
        "masks": {"policy_loss_mask": [1] * 10},
    }


def test_hard_gate_blocks_model_failure_and_quarantine():
    fail = _row(role="model_failure")
    ok, reasons = hard_gate(fail)
    assert not ok
    assert "model_failure_no_positive_imitation" in reasons

    q = _row()
    q["eligibility"]["quarantined"] = True
    ok, reasons = hard_gate(q)
    assert not ok
    assert "quarantined" in reasons


def test_freeze_config_splits_assigns_non_train_rows():
    rows = [_row(config=f"config_{i:03d}") for i in range(1, 5)]
    out, manifest = freeze_config_splits(rows)
    assert manifest["split_counts"]["train"] == 2
    assert manifest["split_counts"]["val"] == 1
    assert manifest["split_counts"]["heldout"] == 1
    assert {r["split"] for r in out} == {"train", "val", "heldout"}


def test_compute_pacer_weights_zeroes_val_and_forbidden_rows():
    rows = [
        _row(role="clean_demo", config="config_001"),
        _row(role="human_correction", config="config_001"),
        _row(role="model_partial", config="config_001"),
        _row(role="model_failure", config="config_001"),
        _row(role="clean_demo", config="config_002"),
    ]
    rows[-1]["split"] = "val"
    out, manifest = compute_pacer_weights(rows, PacerEta(), eta_id="unit")
    by_role = {r["row_id"]: r for r in out}
    assert by_role["config_001:ram:model_failure"]["returns"]["pacer_weight"] == 0.0
    assert by_role["config_002:ram:clean_demo"]["returns"]["pacer_weight"] == 0.0
    assert by_role["config_001:ram:clean_demo"]["returns"]["pacer_weight"] >= 0.8
    assert by_role["config_001:ram:human_correction"]["returns"]["pacer_weight"] >= 0.65
    assert manifest["safety_leakage"] == {}
    assert manifest["effective_sample_count"] > 0


def test_manifest_json_serializable(tmp_path: Path):
    out, manifest = compute_pacer_weights([_row()], PacerEta(), eta_id="unit")
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    assert json.loads(p.read_text())["eta_id"] == "unit"
    assert out[0]["returns"]["pacer_schema"] == "pacer_bo_lc_fma_weights.v0.1"


def _human_correction_without_aw_components(row_id: str, *, start: float, end: float, terminal: bool = False):
    row = _row(role="human_correction", component="ram", config="config_001")
    row["row_id"] = row_id
    row["trial_id"] = row_id
    row["returns"].pop("aw_fma_reward_components", None)
    row["returns"]["rank_weight_unvalidated"] = 1.0
    row["distance_features"] = {
        "chunk_local_to_active_target": {
            "present": True,
            "start_combined_pose_distance_m": start,
            "end_combined_pose_distance_m": end,
            "delta_combined_pose_distance_m": start - end,
            "start_pos_error_m": start,
            "end_pos_error_m": end,
            "delta_pos_error_m": start - end,
            "inside_region_at_end": terminal,
            "hover_05cm_at_end": terminal,
            "inside_region_first_t": 9 if terminal else None,
            "hover_05cm_first_t": 9 if terminal else None,
        }
    }
    return row


def test_human_correction_pacer_evidence_uses_chunk_local_trajectory_when_aw_components_missing():
    weak = _human_correction_without_aw_components("weak_correction", start=0.20, end=0.19)
    strong = _human_correction_without_aw_components("strong_correction", start=0.20, end=0.02, terminal=True)

    weak_evidence = evidence_components(weak)
    strong_evidence = evidence_components(strong)

    assert strong_evidence["progress"] > weak_evidence["progress"]
    assert strong_evidence["proximity"] > weak_evidence["proximity"]
    assert strong_evidence["terminal"] > weak_evidence["terminal"]

    out, _manifest = compute_pacer_weights([weak, strong], PacerEta(), eta_id="unit")
    by_id = {row["row_id"]: row for row in out}

    assert by_id["strong_correction"]["returns"]["pacer_raw_evidence_score"] > by_id["weak_correction"]["returns"]["pacer_raw_evidence_score"]
    assert by_id["strong_correction"]["returns"]["pacer_weight"] > by_id["weak_correction"]["returns"]["pacer_weight"]


def test_hard_gate_allows_legacy_rows_without_aw_fma_loss_eligible_when_other_gates_pass():
    row = _row(role="clean_demo")
    del row["returns"]["aw_fma_loss_eligible"]

    ok, reasons = hard_gate(row)

    assert ok is True
    assert "not_aw_fma_loss_eligible" not in reasons


def test_hard_gate_blocks_explicit_false_loss_eligibility():
    row = _row(role="clean_demo", eligible=False)

    ok, reasons = hard_gate(row)

    assert ok is False
    assert "not_aw_fma_loss_eligible" in reasons


def test_evidence_components_exposes_full_paper_vector_and_zeroes_wrong_target_geometry():
    row = _row(role="model_success")
    row["target"] = {
        "active_target_id": "cpu_fan",
        "geometry_target_id": "cpu_fan",
        "observed_target_id": "ram",
        "target_consistency": "wrong_target",
    }
    row["returns"]["aw_fma_reward_components"].update(
        {
            "r_progress_signed01": 1.0,
            "r_proximity": 1.0,
            "r_terminal": 1.0,
            "r_direction": 1.0,
            "r_provenance": 1.0,
        }
    )

    target_ok, target_reasons = target_match(row)
    comps = evidence_components(row)

    assert target_ok is False
    assert "target_consistency_wrong" in target_reasons
    assert set(comps) == {"progress", "proximity", "terminal", "direction", "stop", "operator", "provenance"}
    assert comps["progress"] == 0.0
    assert comps["proximity"] == 0.0
    assert comps["terminal"] == 0.0
    assert comps["direction"] == 0.0
    assert comps["operator"] > 0.0
    assert comps["provenance"] == 1.0


def test_all_zero_policy_loss_mask_fails_hard_gate():
    row = _row(role="clean_demo")
    row["masks"]["policy_loss_mask"] = [0] * 10

    ok, reasons = hard_gate(row)

    assert ok is False
    assert "invalid_policy_loss_mask" in reasons


def test_pacer_weight_uses_raw_beta_coefficients_not_unit_normalized():
    row = _row(role="model_success")
    row["returns"]["aw_fma_reward_components"].update(
        {
            "r_progress_signed01": 1.0,
            "r_proximity": 0.0,
            "r_terminal": 0.0,
            "r_stop_handoff": 0.0,
            "r_operator_rank": 0.0,
        }
    )
    eta = PacerEta(progress=0.45, proximity=0.30, terminal=0.10, stop=0.05, operator=0.05)

    assert abs(sum(eta.components().values()) - 0.95) < 1e-12
    assert raw_score(row, eta) == 0.45


def test_pacer_weight_applies_role_multiplier_outside_centered_exponential():
    rows = []
    for idx, progress in enumerate((0.0, 0.5, 1.0)):
        row = _row(role="model_partial")
        row["row_id"] = f"partial_{idx}"
        row["returns"]["aw_fma_reward_components"].update(
            {
                "r_progress_signed01": progress,
                "r_proximity": 0.0,
                "r_terminal": 0.0,
                "r_stop_handoff": 0.0,
                "r_operator_rank": 0.0,
            }
        )
        rows.append(row)

    out, _manifest = compute_pacer_weights(rows, PacerEta(), eta_id="unit")
    by_id = {row["row_id"]: row for row in out}

    assert by_id["partial_1"]["returns"]["pacer_weight"] == PacerEta().partial_multiplier


def test_runtime_robust_stats_use_mad_floor_without_standard_deviation_fallback():
    center, scale = _robust_stats([0.5, 0.5, 0.9])

    assert center == 0.5
    assert scale == 1e-6


def test_pacer_weight_saturates_large_advantage_instead_of_overflowing():
    rows = []
    for idx, progress in enumerate((0.5, 0.5, 0.9)):
        row = _row(role="model_success")
        row["row_id"] = f"outlier_{idx}"
        row["returns"]["aw_fma_reward_components"].update(
            {
                "r_progress_signed01": progress,
                "r_proximity": 0.0,
                "r_terminal": 0.0,
                "r_stop_handoff": 0.0,
                "r_operator_rank": 0.0,
            }
        )
        rows.append(row)

    eta = PacerEta(progress=1.0, proximity=0.0, terminal=0.0, stop=0.0, operator=0.0, w_max=3.0)
    out, _manifest = compute_pacer_weights(rows, eta, eta_id="unit")
    by_id = {row["row_id"]: row for row in out}

    assert by_id["outlier_2"]["returns"]["pacer_weight"] == 3.0
    assert by_id["outlier_2"]["returns"]["pacer_weight_unclipped"] == 3.0


def test_freeze_config_splits_falls_back_to_trial_groups_when_single_config():
    rows = []
    for trial_idx in range(5):
        row = _row(role="clean_demo", config="config_only")
        row["trial_id"] = f"trial_{trial_idx:03d}"
        row["row_id"] = f"trial_{trial_idx:03d}:row"
        rows.append(row)

    split_rows, manifest = freeze_config_splits(rows)

    assert manifest["strategy"] == "trial_id_holdout_fallback"
    assert set(manifest["split_counts"]) == {"train", "val", "heldout"}
    by_trial = {}
    for row in split_rows:
        by_trial.setdefault(row["trial_id"], row["split"])
        assert by_trial[row["trial_id"]] == row["split"]


def test_ablation_factory_preserves_rows_and_zeroes_non_train():
    rows = [
        _row(role="clean_demo", config="config_001"),
        _row(role="human_correction", config="config_001"),
        _row(role="model_partial", config="config_001"),
        _row(role="model_success", config="config_001"),
        _row(role="model_failure", config="config_001"),
        _row(role="clean_demo", config="config_002"),
        _row(role="human_correction", config="config_003"),
    ]
    rows[-2]["split"] = "val"
    rows[-1]["split"] = "heldout"

    for mode in sorted(ABLATION_WEIGHT_MODES):
        out, manifest = compute_ablation_weights(rows, mode=mode, eta=PacerEta(), eta_id="unit", random_seed=11)
        assert [r["row_id"] for r in out] == [r["row_id"] for r in rows]
        assert [r["split"] for r in out] == [r["split"] for r in rows]
        assert manifest["schema"] == PACER_ABLATION_WEIGHT_SCHEMA_VERSION
        assert manifest["ablation_mode"] == mode
        assert manifest["safety_leakage"] == {}

        non_train = [r for r in out if r["split"] != "train"]
        assert non_train
        for row in non_train:
            ret = row["returns"]
            assert ret["loss_weight"] == 0.0
            assert ret["pacer_ablation_weight"] == 0.0
            assert "non_train_split" in ret["pacer_ablation_ineligible_reasons"]


def test_default_eta_pool_keeps_operator_as_fixed_protocol_dimension():
    for eta_id, eta in DEFAULT_ETA_POOL.items():
        assert eta.operator == PacerEta().operator, eta_id
        assert "operator" not in PacerEta.FREE_DIMS
        assert "operator" in PacerEta.FIXED_DIMS


def test_ablation_modes_select_expected_training_rows():
    rows = [
        _row(role="clean_demo"),
        _row(role="human_correction"),
        _row(role="model_partial"),
        _row(role="model_success"),
        _row(role="model_failure"),
    ]

    expected_roles = {
        "clean_demo_only": {"clean_demo"},
        "uniform_replay": {"clean_demo", "human_correction", "model_partial", "model_success"},
        "correction_only": {"human_correction"},
        "outcome_only": {"clean_demo", "model_success"},
        "fixed_geometry": {"clean_demo", "human_correction", "model_partial", "model_success"},
        "random_weight": {"clean_demo", "human_correction", "model_partial", "model_success"},
        "pacer_eta": {"clean_demo", "human_correction", "model_partial", "model_success"},
    }

    for mode, roles in expected_roles.items():
        out, _manifest = compute_ablation_weights(rows, mode=mode, eta=PacerEta(), eta_id="unit", random_seed=19)
        positive_roles = {r["sample_role"] for r in out if r["returns"]["loss_weight"] > 0}
        assert positive_roles == roles


def test_ablation_modes_never_assign_positive_weight_to_excluded_rows():
    rows = [_row(role="excluded")]

    for mode in sorted(ABLATION_WEIGHT_MODES):
        out, manifest = compute_ablation_weights(rows, mode=mode, eta=PacerEta(), eta_id="unit")
        assert out[0]["returns"]["loss_weight"] == 0.0
        assert out[0]["returns"]["pacer_ablation_weight"] == 0.0
        assert "excluded_role_no_positive_imitation" in out[0]["returns"]["pacer_ablation_ineligible_reasons"]
        assert manifest["safety_leakage"] == {}


def test_fixed_geometry_and_random_ablation_weights_are_auditable():
    rows = [_row(role="clean_demo"), _row(role="human_correction"), _row(role="model_partial")]

    fixed, fixed_manifest = compute_ablation_weights(rows, mode="fixed_geometry")
    by_role = {r["sample_role"]: r for r in fixed}
    assert by_role["clean_demo"]["returns"]["loss_weight"] == 1.0
    assert by_role["human_correction"]["returns"]["loss_weight"] == 1.5
    assert by_role["model_partial"]["returns"]["loss_weight"] == 0.4
    assert fixed_manifest["weight_field"] == "returns.loss_weight"

    random_a, _ = compute_ablation_weights(rows, mode="random_weight", random_seed=123)
    random_b, _ = compute_ablation_weights(rows, mode="random_weight", random_seed=123)
    random_c, _ = compute_ablation_weights(rows, mode="random_weight", random_seed=456)
    weights_a = [r["returns"]["loss_weight"] for r in random_a]
    weights_b = [r["returns"]["loss_weight"] for r in random_b]
    weights_c = [r["returns"]["loss_weight"] for r in random_c]
    assert weights_a == weights_b
    assert weights_a != weights_c
    assert all(0.05 <= w <= 1.0 for w in weights_a)
