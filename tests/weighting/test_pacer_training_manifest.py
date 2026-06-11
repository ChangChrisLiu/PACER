import json
from pathlib import Path

from pacer.weighting.training_manifest import (
    PACER_TRAINING_RUN_MANIFEST_SCHEMA,
    build_pacer_training_run_plan,
    validate_training_run_plan,
)


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_build_training_plan_uses_same_merged_base_for_sftpp_and_ablation_runs(tmp_path: Path):
    unmerged = _touch(tmp_path / "openpi/checkpoints/original_sft_unmerged/49999/_CHECKPOINT_METADATA")
    merged = _touch(tmp_path / "openpi/checkpoints/original_sft_merged/49999/_CHECKPOINT_METADATA")
    views_root = tmp_path / "views"
    for mode in ["clean_demo_only", "correction_only", "fixed_geometry", "pacer_eta"]:
        _touch(views_root / mode / "action_chunks.jsonl", "{}\n")
        _touch(views_root / mode / "pacer_weight_manifest.json", json.dumps({"ablation_mode": mode, "weight_field": "returns.loss_weight", "safety_leakage": {}}))
    output_root = tmp_path / "runs"

    plan = build_pacer_training_run_plan(
        unmerged_original_sft_checkpoint=unmerged.parent,
        merged_original_sft_checkpoint=merged.parent,
        weighted_views_root=views_root,
        output_root=output_root,
        methods=["clean_demo_only", "correction_only", "fixed_geometry", "pacer_eta"],
        cluster_openpi_root="/scratch/$USER/openpi",
        cluster_robot_runtime_root="/scratch/$USER/PACER",
        steps=100,
    )

    assert plan["schema"] == PACER_TRAINING_RUN_MANIFEST_SCHEMA
    assert plan["sftpp_demo_run"]["base_checkpoint_type"] == "merged_original_sft"
    assert plan["sftpp_demo_run"]["base_checkpoint_path"] == str(merged.parent)
    assert plan["sftpp_demo_run"]["comparison_role"] == "same_base_demonstration_addition_baseline"
    assert plan["sftpp_demo_run"]["evidence_sources"] == ["clean_demo_only", "correction_only"]

    runs = {run["method"]: run for run in plan["ablation_lora_runs"]}
    assert set(runs) == {"clean_demo_only", "correction_only", "fixed_geometry", "pacer_eta"}
    for method, run in runs.items():
        assert run["base_checkpoint_type"] == "merged_original_sft"
        assert run["base_checkpoint_path"] == str(merged.parent)
        assert run["weighted_view_path"] == str(views_root / method / "action_chunks.jsonl")
        assert run["loss_weight_field"] == "returns.loss_weight"
        assert run["trainable_adapter"] == "new_lora"
        assert run["cluster_slurm_path"].endswith("train_job.slurm")

    assert validate_training_run_plan(plan) == []


def test_build_training_plan_accepts_legacy_hprc_parameter_aliases(tmp_path: Path):
    merged = _touch(tmp_path / "openpi/checkpoints/original_sft_merged/49999/_CHECKPOINT_METADATA")
    views_root = tmp_path / "views"
    for mode in ["clean_demo_only", "correction_only"]:
        _touch(views_root / mode / "action_chunks.jsonl", "{}\n")
        _touch(views_root / mode / "pacer_weight_manifest.json", json.dumps({"ablation_mode": mode, "weight_field": "returns.loss_weight", "safety_leakage": {}}))

    plan = build_pacer_training_run_plan(
        unmerged_original_sft_checkpoint=None,
        merged_original_sft_checkpoint=merged.parent,
        weighted_views_root=views_root,
        output_root=tmp_path / "runs",
        methods=["clean_demo_only"],
        hprc_openpi_root="/scratch/legacy/openpi",
        hprc_robot_runtime_root="/scratch/legacy/PACER",
    )

    assert plan["cluster"] == {"openpi_root": "/scratch/legacy/openpi", "robot_runtime_root": "/scratch/legacy/PACER"}


def test_materialize_training_plan_accepts_legacy_hprc_slurm_path(tmp_path: Path):
    from pacer.weighting.training_manifest import materialize_training_run_plan

    run_dir = tmp_path / "legacy_run"
    plan = {
        "schema": PACER_TRAINING_RUN_MANIFEST_SCHEMA,
        "cluster": {"openpi_root": "/scratch/openpi", "robot_runtime_root": "/scratch/PACER"},
        "sftpp_demo_run": {
            "method": "legacy",
            "local_run_dir": str(run_dir),
            "hprc_slurm_path": str(run_dir / "train_job.slurm"),
            "steps": 1,
        },
        "ablation_lora_runs": [],
    }

    written = materialize_training_run_plan(plan, tmp_path)

    assert Path(written["sftpp_demo_run"]["slurm"]).exists()


def test_materialized_slurm_placeholder_uses_run_manifest_path(tmp_path: Path):
    unmerged = _touch(tmp_path / "openpi/checkpoints/original_sft_unmerged/49999/_CHECKPOINT_METADATA")
    merged = _touch(tmp_path / "openpi/checkpoints/original_sft_merged/49999/_CHECKPOINT_METADATA")
    views_root = tmp_path / "views"
    for mode in ["clean_demo_only", "correction_only"]:
        _touch(views_root / mode / "action_chunks.jsonl", "{}\n")
        _touch(views_root / mode / "pacer_weight_manifest.json", json.dumps({"ablation_mode": mode, "weight_field": "returns.loss_weight", "safety_leakage": {}}))
    output_root = tmp_path / "runs"
    plan = build_pacer_training_run_plan(
        unmerged_original_sft_checkpoint=unmerged.parent,
        merged_original_sft_checkpoint=merged.parent,
        weighted_views_root=views_root,
        output_root=output_root,
        methods=["clean_demo_only", "correction_only"],
    )

    from pacer.weighting.training_manifest import materialize_training_run_plan

    written = materialize_training_run_plan(plan, output_root)
    slurm = Path(written["sftpp_demo_run"]["slurm"]).read_text()

    assert "#SBATCH --gres=gpu:2" in slurm
    assert "module load WebProxy" in slurm
    assert "UV_FROZEN=1" in slurm
    assert "DATASET_DIR=" in slurm
    assert "NORM_FILE=" in slurm
    assert 'if [ ! -d "$DATASET_DIR" ]' in slurm
    assert 'if [ ! -f "$NORM_FILE" ]' in slurm
    assert "Run exports locally and upload before submitting this job." in slurm
    assert "json.load" not in slurm
    assert "export_rwfma_chunks_to_lerobot" not in slurm


def test_training_plan_validation_checks_weight_manifest_method_and_safety(tmp_path: Path):
    base = _touch(tmp_path / "merged/49999/_CHECKPOINT_METADATA").parent
    unmerged = _touch(tmp_path / "unmerged/49999/_CHECKPOINT_METADATA").parent
    view = _touch(tmp_path / "views/fixed_geometry/action_chunks.jsonl", "{}\n")
    _touch(tmp_path / "views/clean_demo_only/action_chunks.jsonl", "{}\n")
    _touch(tmp_path / "views/correction_only/action_chunks.jsonl", "{}\n")
    bad_manifest = _touch(
        tmp_path / "views/fixed_geometry/pacer_weight_manifest.json",
        json.dumps({"ablation_mode": "random_weight", "weight_field": "returns.loss_weight", "safety_leakage": {}}),
    )
    plan = build_pacer_training_run_plan(
        unmerged_original_sft_checkpoint=unmerged,
        merged_original_sft_checkpoint=base,
        weighted_views_root=tmp_path / "views",
        output_root=tmp_path / "runs",
        methods=["fixed_geometry"],
    )

    errors = validate_training_run_plan(plan)

    assert "weight_manifest_mode_mismatch:fixed_geometry" in errors
    bad_manifest.write_text(json.dumps({"ablation_mode": "fixed_geometry", "weight_field": "returns.loss_weight", "safety_leakage": {"x": 1}}))
    errors = validate_training_run_plan(plan)
    assert "weight_manifest_safety_leakage_not_empty:fixed_geometry" in errors
    bad_manifest.unlink()
    errors = validate_training_run_plan(plan)
    assert "missing_weight_manifest:fixed_geometry" in errors


def test_training_plan_validation_rejects_missing_files_and_wrong_base_type(tmp_path: Path):
    plan = {
        "schema": PACER_TRAINING_RUN_MANIFEST_SCHEMA,
        "sftpp_demo_run": {
            "base_checkpoint_type": "unmerged_original_sft",
            "base_checkpoint_path": str(tmp_path / "missing_sft"),
            "evidence_sources": ["clean_demo_only", "correction_only"],
        },
        "ablation_lora_runs": [
            {
                "method": "fixed_geometry",
                "base_checkpoint_type": "unmerged_original_sft",
                "base_checkpoint_path": str(tmp_path / "missing_merged"),
                "weighted_view_path": str(tmp_path / "missing_view.jsonl"),
                "loss_weight_field": "returns.loss_weight",
                "trainable_adapter": "new_lora",
            }
        ],
    }

    errors = validate_training_run_plan(plan)

    assert "sftpp_base_must_be_merged_original_sft" in errors
    assert "missing_sftpp_base_checkpoint" in errors
    assert "ablation_base_must_be_merged_original_sft:fixed_geometry" in errors
    assert "missing_ablation_base_checkpoint:fixed_geometry" in errors
    assert "missing_weighted_view:fixed_geometry" in errors
