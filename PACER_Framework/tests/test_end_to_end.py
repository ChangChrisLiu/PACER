"""End-to-end journey of a generic robot/VLA user, from standard records to
weighted training views, candidate evaluation, selection, baselines, and the
paper-style comparison tables. No private integration specifics anywhere: the robot is
"anybot" with two task targets ("widget", "socket")."""
import json

from pacer_framework.alignment import (
    blocker_flags_from_open_loop,
    open_loop_submetrics,
    with_no_regression,
)
from pacer_framework.baselines import BASELINE_MODES, compile_baseline
from pacer_framework.eta import CANDIDATE_POOL
from pacer_framework.export import export_training_view, read_jsonl
from pacer_framework.reporting import (
    candidate_ranking_table,
    paired_component_bootstrap,
    success_table,
)
from pacer_framework.schema import validate_training_row, validate_validation_row
from pacer_framework.split import assign_case_group_split
from pacer_framework.validation import component_balanced_j_val, evaluate_candidate, select_candidate
from pacer_framework.weights import compile_weights

TARGETS = {"widget": [0.40, 0.10, 0.10], "socket": [0.20, -0.15, 0.08]}
ROLE_END_HEIGHT = {  # final TCP height above the target by trace quality
    "clean": 0.004,
    "correction": 0.006,
    "auto_success": 0.005,
    "partial": 0.018,
    "failure": 0.150,
}
ROLE_LABEL = {"auto_success": "success", "partial": "near_but_not_accurate",
              "failure": "totally_off_wrong_region_or_target"}
ROLE_RANK = {"clean": 1.0, "correction": 1.4, "auto_success": 0.9, "partial": 0.5, "failure": 0.7}


def approach(target, end_height, *, n=4, start_height=0.20):
    return [
        [target[0], target[1], target[2] + start_height + (end_height - start_height) * i / (n - 1)]
        for i in range(n)
    ]


def trace_row(component, config, role):
    target = TARGETS[component]
    positions = approach(target, ROLE_END_HEIGHT[role])
    phase_ending = role in {"clean", "auto_success"}
    return {
        "row_id": f"{config}:{role}",
        "component": component,
        "phase": "approach",
        "split": "train",  # provisional; the split utility overwrites it
        "collection_config": config,
        "role": role,
        "operator_label": ROLE_LABEL.get(role),
        "mask": [1, 1, 1],
        "geometry": {
            "tcp_positions": positions,
            "target_point": target,
            "target_id": f"{component}_0",
            "orientation_ok": True,
        },
        "target_meta": {"target_id": f"{component}_0", "declared_target_id": f"{component}_0"},
        "stop": {
            "phase_ending": phase_ending,
            "stop_event": "model_stop_token" if phase_ending else "timeout",
            "inside_terminal": phase_ending,
        },
        "provenance": {
            "manifest_valid": True,
            "timing_aligned": True,
            "component_identity_valid": True,
            "action_mask_valid": True,
        },
        "safety": {"unsafe": False, "manual_safety_stop": False, "quarantined": False},
        "rank_weight": ROLE_RANK[role],
    }


def build_buffer():
    rows = []
    for component in TARGETS:
        for k in range(4):
            config = f"{component}_cfg{k}"
            for role in ROLE_END_HEIGHT:
                rows.append(trace_row(component, config, role))
    return rows


def predicted_positions(component, quality):
    target = TARGETS[component]
    end_height = {"good": 0.004, "weak": 0.030, "reference": 0.050}[quality]
    return approach(target, end_height)


def candidate_validation_rows(val_rows, quality, *, reference_scores=False):
    """Open-loop scoring of one trained candidate on the validation split."""
    other = {"widget": "socket", "socket": "widget"}
    out = []
    for row in val_rows:
        component = row["component"]
        predicted = predicted_positions(component, quality)
        phase_ending = bool(row["stop"]["phase_ending"])
        submetrics = open_loop_submetrics(
            predicted,
            target_point=TARGETS[component],
            component=component,
            phase_ending=phase_ending,
            stop_emitted=(quality == "good"),
            reference_positions=row["geometry"]["tcp_positions"],
        )
        if not reference_scores:
            reference = open_loop_submetrics(
                predicted_positions(component, "reference"),
                target_point=TARGETS[component],
                component=component,
            )
            submetrics = with_no_regression(submetrics, reference, delta_reg=0.05)
        flags = blocker_flags_from_open_loop(
            predicted,
            target_point=TARGETS[component],
            component=component,
            non_target_regions=[{"point": TARGETS[other[component]], "radius": 0.02}],
            workspace_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        )
        out.append({"row_id": f"val:{row['row_id']}:{quality}", "component": component,
                    "role": row["role"], "submetrics": submetrics, **flags})
    return out


def test_generic_user_end_to_end(tmp_path):
    # 1. Record traces and validate them against the executable contract.
    buffer_rows = build_buffer()
    for row in buffer_rows:
        assert validate_training_row(row) == [], row["row_id"]

    # 2. Configuration-disjoint case-group split under a fixed seed.
    split_rows, split_manifest = assign_case_group_split(buffer_rows, seed=20260610)
    assert split_manifest["split_counts"]["val"] > 0
    for component in TARGETS:
        assert len(split_manifest["plan"][component]["val_configs"]) == 1

    # 3. Compile PACER weights per candidate and export trainer-ready views.
    candidates = {name: CANDIDATE_POOL[name] for name in ("terminal_stop_heavy", "progress_heavy")}
    manifests = {}
    for name, eta in candidates.items():
        weighted, manifest = compile_weights(split_rows, eta, eta_id=name)
        manifests[name] = manifest
        assert manifest["forbidden_positive_weight_rows"] == 0
        paths = export_training_view(weighted, manifest, tmp_path / name)
        loaded = read_jsonl(paths["rows"])
        positive = [r for r in loaded if r["loss_weight"] > 0]
        assert positive and all(r["split"] == "train" for r in positive)
        roles = {r["role"] for r in positive}
        assert "failure" not in roles and "excluded" not in roles

    # 4. Post-training: score each candidate's predictions on the val split.
    val_rows = [r for r in split_rows if r["split"] == "val" and r["role"] != "failure"]
    assert val_rows
    quality_by_candidate = {"terminal_stop_heavy": "good", "progress_heavy": "weak"}
    reference_rows = candidate_validation_rows(val_rows, "reference", reference_scores=True)
    _ref_j, _scores, ref_manifest = component_balanced_j_val(reference_rows)

    evaluations = {}
    for name, quality in quality_by_candidate.items():
        rows = candidate_validation_rows(val_rows, quality)
        for row in rows:
            assert validate_validation_row(row) == [], row["row_id"]
        evaluations[name] = evaluate_candidate(
            validation_rows=rows,
            weight_manifest=manifests[name],
            w_max=candidates[name].w_max,
            reference_component_scores=ref_manifest["component_scores"],
            delta_reg=0.05,
        )

    assert evaluations["terminal_stop_heavy"]["feasible"]
    assert evaluations["progress_heavy"]["feasible"]
    assert evaluations["terminal_stop_heavy"]["j_val"] > evaluations["progress_heavy"]["j_val"]

    # 5. Selection (eq:selection) and the ranking table (Table app_ranking).
    j_vals = {name: ev["j_val"] for name, ev in evaluations.items()}
    feasible = {name: ev["feasible"] for name, ev in evaluations.items()}
    selected = select_candidate(j_vals, feasible)
    assert selected == "terminal_stop_heavy"
    ranking = candidate_ranking_table(j_vals, feasible, selected)
    assert ranking[0]["candidate"] == selected and ranking[0]["selected"]

    # 6. All eight paper baselines from the same buffer.
    baseline_manifests = {}
    for mode in BASELINE_MODES:
        eta = candidates[selected] if mode == "pacer_selected" else None
        _rows, manifest = compile_baseline(split_rows, mode, eta=eta, eta_id=selected)
        baseline_manifests[mode] = manifest
    assert baseline_manifests["no_post_train"]["train_rows"] == 0
    assert baseline_manifests["correction_only"]["nominal_start_rows"] == 0
    assert baseline_manifests["uniform_all_eligible"]["train_rows"] >= baseline_manifests["clean_only"]["train_rows"]
    assert baseline_manifests["pacer_selected"]["pacer_weight_manifest"]["eta_id"] == selected

    # 7. Held-out comparison exactly as reported (Wilson CI + paired bootstrap).
    held_out = {
        "pacer_selected": {"widget": (18, 20), "socket": (17, 20)},
        "vanilla_post_sft": {"widget": (11, 20), "socket": (9, 20)},
    }
    table = success_table(held_out)
    by_method = {row["method"]: row for row in table}
    assert by_method["pacer_selected"]["successes"] == 35
    diff = paired_component_bootstrap(
        held_out["pacer_selected"], held_out["vanilla_post_sft"], n_boot=500, seed=5
    )
    assert diff["difference_pp"] == 37.5
    assert diff["ci95_pp"][0] > 0.0

    # 8. Everything above came from JSON-able standard records.
    json.dumps(split_manifest)
    json.dumps(baseline_manifests)
    json.dumps(ranking)
