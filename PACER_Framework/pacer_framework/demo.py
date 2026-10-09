"""Runnable end-to-end PACER demo for a generic robot + VLA.

Synthesizes a small post-training buffer for an imaginary arm ("anybot") with
two task targets, then walks the full paper pipeline:

  records -> contract validation -> case-group split -> candidate weights ->
  exported training views -> open-loop validation scoring -> J_val +
  feasibility -> selection -> the eight baselines -> held-out comparison table.

Run:  python -m pacer_framework.demo
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from pacer_framework import (
    ScoringConfig,
    BASELINE_MODES,
    CANDIDATE_POOL,
    assign_case_group_split,
    blocker_flags_from_open_loop,
    candidate_ranking_table,
    compile_baseline,
    compile_weights,
    component_balanced_j_val,
    evaluate_candidate,
    export_training_view,
    open_loop_submetrics,
    paired_component_bootstrap,
    select_candidate,
    success_table,
    validate_training_row,
)

TARGETS = {"widget": [0.40, 0.10, 0.10], "socket": [0.20, -0.15, 0.08]}
END_HEIGHT = {"clean": 0.004, "correction": 0.006, "auto_success": 0.005, "partial": 0.018, "failure": 0.15}
LABELS = {"auto_success": "success", "partial": "near_but_not_accurate",
          "failure": "totally_off_wrong_region_or_target"}


def approach(target, end_height, n=4, start=0.20):
    return [[target[0], target[1], target[2] + start + (end_height - start) * i / (n - 1)] for i in range(n)]


def build_buffer():
    rows = []
    for component, target in TARGETS.items():
        for k in range(4):
            for role, end in END_HEIGHT.items():
                phase_ending = role in {"clean", "auto_success"}
                rows.append({
                    "row_id": f"{component}_cfg{k}:{role}",
                    "component": component, "phase": "approach", "split": "train",
                    "collection_config": f"{component}_cfg{k}", "role": role,
                    "operator_label": LABELS.get(role), "mask": [1, 1, 1],
                    "geometry": {"tcp_positions": approach(target, end), "target_point": target,
                                 "target_id": f"{component}_0", "orientation_ok": True},
                    "target_meta": {"target_id": f"{component}_0", "declared_target_id": f"{component}_0"},
                    "stop": {"phase_ending": phase_ending,
                             "stop_event": "model_stop_token" if phase_ending else "timeout",
                             "inside_terminal": phase_ending},
                    "safety": {"unsafe": False, "manual_safety_stop": False, "quarantined": False},
                    "rank_weight": {"clean": 1.0, "correction": 1.4, "auto_success": 0.9,
                                    "partial": 0.5, "failure": 0.7}[role],
                })
    return rows


def score_candidate(val_rows, quality, scoring_config: ScoringConfig):
    other = {"widget": "socket", "socket": "widget"}
    end = {"good": 0.004, "weak": 0.030, "reference": 0.050}[quality]
    out = []
    for row in val_rows:
        component = row["component"]
        predicted = approach(TARGETS[component], end)
        submetrics = open_loop_submetrics(
            predicted, target_point=TARGETS[component], component=component,
            phase_ending=bool(row["stop"]["phase_ending"]), stop_emitted=(quality == "good"),
            reference_positions=row["geometry"]["tcp_positions"],
            scoring_config=scoring_config,
        )
        flags = blocker_flags_from_open_loop(
            predicted, target_point=TARGETS[component], component=component,
            non_target_regions=[{"point": TARGETS[other[component]], "radius": 0.02}],
            workspace_bounds={"min": [-1, -1, -1], "max": [1, 1, 1]},
        )
        out.append({"row_id": f"val:{row['row_id']}", "component": component,
                    "role": row["role"], "submetrics": submetrics,
                    "reference_alignment_mode": scoring_config.reference_alignment_mode, **flags})
    return out


def main(
    out_dir: str | Path | None = "demo_output",
    *,
    scoring_config: ScoringConfig | None = None,
) -> dict:
    config = scoring_config if scoring_config is not None else ScoringConfig()
    rows = build_buffer()
    contract_errors = [e for row in rows for e in validate_training_row(row)]
    split_rows, split_manifest = assign_case_group_split(rows, seed=20260610)

    candidates = {n: CANDIDATE_POOL[n] for n in ("terminal_stop_heavy", "progress_heavy")}
    manifests = {}
    for name, eta in candidates.items():
        weighted, manifest = compile_weights(split_rows, eta, eta_id=name)
        manifests[name] = manifest
        if out_dir is not None:
            export_training_view(weighted, manifest, Path(out_dir) / name)

    val_rows = [r for r in split_rows if r["split"] == "val" and r["role"] != "failure"]
    reference_rows = score_candidate(val_rows, "reference", config)
    quality = {"terminal_stop_heavy": "good", "progress_heavy": "weak"}
    evaluations = {
        name: evaluate_candidate(
            validation_rows=score_candidate(val_rows, quality[name], config),
            weight_manifest=manifests[name], w_max=candidates[name].w_max,
            reference_validation_rows=reference_rows, scoring_config=config,
        )
        for name in candidates
    }
    j_vals = {n: e["j_val"] for n, e in evaluations.items()}
    feasible = {n: e["feasible"] for n, e in evaluations.items()}
    selected = select_candidate(j_vals, feasible)
    ranking = candidate_ranking_table(j_vals, feasible, selected)

    baselines = {}
    for mode in BASELINE_MODES:
        eta = candidates[selected] if mode == "pacer_selected" and selected is not None else None
        effective_mode = "no_post_train" if mode == "pacer_selected" and selected is None else mode
        _r, manifest = compile_baseline(split_rows, effective_mode, eta=eta, eta_id=selected)
        baselines[mode] = {"train_rows": manifest["train_rows"],
                           "nominal_start_rows": manifest["nominal_start_rows"]}

    held_out = {"pacer_selected": {"widget": (18, 20), "socket": (17, 20)},
                "vanilla_post_sft": {"widget": (11, 20), "socket": (9, 20)}}
    table = success_table(held_out)
    bootstrap = paired_component_bootstrap(
        held_out["pacer_selected"], held_out["vanilla_post_sft"], n_boot=2000, seed=7)

    print("synthetic demo only: replace these toy records/counts with your robot's logs")
    print(f"contract errors: {len(contract_errors)}")
    print(f"split: {split_manifest['split_counts']}")
    print("ranking:")
    for entry in ranking:
        marker = " <- selected" if entry["selected"] else ""
        print(f"  {entry['candidate']:24s} J_val={entry['j_val']:.4f} feasible={entry['feasible']}{marker}")
    print("baselines (train_rows / nominal):")
    for mode, info in baselines.items():
        print(f"  {mode:22s} {info['train_rows']:3d} / {info['nominal_start_rows']:3d}")
    for row in table:
        print(f"  {row['method']:18s} {row['successes']}/{row['trials']}  CI95 {row['ci95_pct']}")
    print(f"PACER - vanilla: {bootstrap['difference_pp']:.1f} pp, CI95 {bootstrap['ci95_pp']}")

    result = {"contract_errors": contract_errors, "split_manifest": split_manifest,
              "selected": selected, "ranking": ranking, "evaluations": evaluations,
              "baselines": baselines, "success_table": table, "bootstrap": bootstrap,
              "synthetic_demo": True, "scoring_config": config.to_dict(),
              "scoring_config_hash": config.fingerprint,
              "reference_fallback": selected is None}
    if out_dir is not None:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation.json").write_text(
            json.dumps(result, indent=2, default=asdict, allow_nan=False) + "\n", encoding="utf-8",
        )
    print(f"scoring configuration: {config.fingerprint}")
    return result


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic, no-hardware PACER workflow")
    parser.add_argument("--scoring-config", type=Path, help="Editable JSON scoring configuration")
    parser.add_argument("--out-dir", type=Path, default=Path("demo_output"))
    args = parser.parse_args(argv)
    try:
        config = ScoringConfig.load(args.scoring_config) if args.scoring_config is not None else ScoringConfig()
        main(out_dir=args.out_dir, scoring_config=config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
