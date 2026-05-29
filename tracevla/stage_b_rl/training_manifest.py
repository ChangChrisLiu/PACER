"""PACER/SFT++ training-run manifest utilities.

This module is offline-only. It records how to combine original SFT checkpoints
with Stage-B weighted views without launching training or touching hardware.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

PACER_TRAINING_RUN_MANIFEST_SCHEMA = "tracevla_training_run_manifest.v0.1"
DEFAULT_ABLATION_METHODS = (
    "clean_demo_only",
    "uniform_replay",
    "correction_only",
    "outcome_only",
    "fixed_rw_fma",
    "random_weight",
    "tracevla_eta_rw_like",
    "tracevla_eta_progress_heavy",
    "tracevla_eta_hover_proximity_heavy",
    "tracevla_eta_terminal_stop_heavy",
    "tracevla_eta_correction_heavy",
    "tracevla_eta_conservative",
    "tracevla_eta_ram_connector_recovery",
    "tracevla_eta_balanced_low_clip",
)


def _path_exists(path: str | Path | None) -> bool:
    return bool(path) and Path(str(path)).exists()


def _slurm_text(
    *,
    method: str,
    sea_vla_root: str,
    openpi_root: str,
    steps: int,
    scratch_root: str = "${SCRATCH:-/path/to/scratch}",
    http_proxy: str = "",
    https_proxy: str = "",
) -> str:
    repo_id = f"ChangChrisLiu/ur5e_stage_b_{method}_10hz"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=pacer_{method}
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module load WebProxy

export SCRATCH={scratch_root}
export SEA_VLA_ROOT={sea_vla_root}
export OPENPI_ROOT={openpi_root}
export UV_FROZEN=1
export UV_CACHE_DIR=$SCRATCH/.cache/uv
export HF_HOME=$SCRATCH/.cache/huggingface
export HF_LEROBOT_HOME=$SCRATCH
export OPENPI_DATA_HOME=$SCRATCH/openpi_data
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export http_proxy={http_proxy}
export https_proxy={https_proxy}

RUN_DIR=$(pwd)
REPO_ID="{repo_id}"

# ── Stage 1: Verify dataset exists (fail fast) ──
echo ">>> Stage 1: Verify dataset for {method}"
DATASET_DIR="$SCRATCH/$REPO_ID"
if [ ! -d "$DATASET_DIR" ]; then
  echo "ERROR: Dataset not found at $DATASET_DIR"
  echo "Run exports locally and upload before submitting this job."
  exit 1
fi
echo "Dataset verified: $DATASET_DIR"

# ── Stage 1.5: Verify norm_stats exists ──
echo ">>> Stage 1.5: Verify norm_stats for {method}"
NORM_FILE="$OPENPI_ROOT/assets/pi05_droid_ur5e_stage_b_rwfma_lora_10hz/$REPO_ID/norm_stats.json"
if [ ! -f "$NORM_FILE" ]; then
  echo "ERROR: norm_stats not found at $NORM_FILE"
  exit 1
fi
echo "norm_stats verified: $NORM_FILE"

# ── Stage 2: Train LoRA via OpenPI ──
echo ">>> Stage 2: Train LoRA for {method} ({steps} steps)"
cd "$OPENPI_ROOT"
uv run python scripts/train.py pi05_droid_ur5e_stage_b_rwfma_lora_10hz \\
  --data.repo-id "$REPO_ID" \\
  --exp-name "tracevla_{method}_${{SLURM_JOB_ID:-local}}" \\
  --checkpoint-base-dir "$RUN_DIR/openpi_lora_output" \\
  --num-train-steps {steps} \\
  --batch-size 32 \\
  --fsdp-devices 2 \\
  --log-interval 100 \\
  --save-interval 1000 \\
  --keep-period 1000 \\
  --overwrite \\
  --no-wandb-enabled

# ── Stage 3: Eval placeholder ──
echo ">>> Stage 3: Eval placeholder for {method}"
python3 -c "
import json, pathlib
scores = dict(schema='pacer_bo_eval_scores.v0.1', method='{method}', status='EVAL_NOT_YET_IMPLEMENTED', run_dir='$RUN_DIR')
out = pathlib.Path('$RUN_DIR/validation/scores.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(scores, indent=2))
print(json.dumps(scores, indent=2))
"

echo ">>> DONE: {method}"
"""


def build_pacer_training_run_plan(
    *,
    unmerged_original_sft_checkpoint: str | Path | None,
    merged_original_sft_checkpoint: str | Path,
    stage_b_views_root: str | Path,
    output_root: str | Path,
    methods: Iterable[str] = DEFAULT_ABLATION_METHODS,
    hprc_openpi_root: str = "/scratch/$USER/openpi",
    hprc_sea_vla_root: str = "/scratch/$USER/SEA-VLA",
    steps: int = 8500,
) -> dict[str, Any]:
    """Build a reviewer-safe local/HPRC training plan without launching jobs."""
    stage_b_views_root = Path(stage_b_views_root)
    output_root = Path(output_root)
    method_list = list(methods)
    sftpp_dir = output_root / "sftpp_clean_correction_demo"
    ablation_root = output_root / "ablation_lora_runs"

    sftpp_demo_run = {
        "run_id": "sftpp_clean_correction_demo",
        "method": "sftpp_clean_correction_demo",
        "base_checkpoint_type": "merged_original_sft",
        "base_checkpoint_path": str(Path(merged_original_sft_checkpoint)),
        "stage_b_sources": ["clean_demo_only", "correction_only"],
        "stage_b_view_paths": [
            str(stage_b_views_root / "clean_demo_only" / "action_chunks.jsonl"),
            str(stage_b_views_root / "correction_only" / "action_chunks.jsonl"),
        ],
        "loss_weight_field": "returns.loss_weight",
        "trainable_adapter": "new_lora",
        "comparison_role": "same_base_demonstration_addition_baseline",
        "legacy_unmerged_original_sft_checkpoint": str(Path(unmerged_original_sft_checkpoint)) if unmerged_original_sft_checkpoint else None,
        "local_run_dir": str(sftpp_dir),
        "hprc_slurm_path": str(sftpp_dir / "train_job.slurm"),
        "validation_scores_path": str(sftpp_dir / "validation" / "scores.json"),
        "eval_schema": "tracevla_bo_eval_scores.v0.1",
        "steps": int(steps),
    }

    ablation_runs: list[dict[str, Any]] = []
    for method in method_list:
        run_dir = ablation_root / method
        ablation_runs.append(
            {
                "run_id": f"ablation_{method}",
                "method": method,
                "base_checkpoint_type": "merged_original_sft",
                "base_checkpoint_path": str(Path(merged_original_sft_checkpoint)),
                "stage_b_view_path": str(stage_b_views_root / method / "action_chunks.jsonl"),
                "stage_b_weight_manifest": str(stage_b_views_root / method / "tracevla_weight_manifest.json"),
                "loss_weight_field": "returns.loss_weight",
                "trainable_adapter": "new_lora",
                "local_run_dir": str(run_dir),
                "new_lora_output_dir": str(run_dir / "openpi_lora_output"),
                "hprc_slurm_path": str(run_dir / "train_job.slurm"),
                "validation_scores_path": str(run_dir / "validation" / "scores.json"),
                "eval_schema": "tracevla_bo_eval_scores.v0.1",
                "steps": int(steps),
            }
        )

    return {
        "schema": PACER_TRAINING_RUN_MANIFEST_SCHEMA,
        "protocol": "same_base_sftpp_demo_plus_merged_sft_ablation_loras",
        "notes": [
            "Main-table SFT++ starts from the same merged original SFT checkpoint as the weighting ablations and adds Stage-B clean/correction demonstrations.",
            "Ablation runs also start from the same merged original SFT checkpoint and train a fresh LoRA per method.",
            "Any unmerged-original SFT++ run is legacy/diagnostic only, not a strict main-table ablation.",
            "RW-FMA 8499 observed weak stop-token emission and angled approach; eval must include stop/approach scores.",
        ],
        "hprc": {"openpi_root": hprc_openpi_root, "sea_vla_root": hprc_sea_vla_root},
        "sftpp_demo_run": sftpp_demo_run,
        "ablation_lora_runs": ablation_runs,
    }


def validate_training_run_plan(plan: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("schema") != PACER_TRAINING_RUN_MANIFEST_SCHEMA:
        errors.append("bad_schema")

    sftpp = plan.get("sftpp_demo_run")
    if not isinstance(sftpp, Mapping):
        errors.append("missing_sftpp_demo_run")
    else:
        if sftpp.get("base_checkpoint_type") != "merged_original_sft":
            errors.append("sftpp_base_must_be_merged_original_sft")
        if not _path_exists(sftpp.get("base_checkpoint_path")):
            errors.append("missing_sftpp_base_checkpoint")
        for src, view in zip(sftpp.get("stage_b_sources", []), sftpp.get("stage_b_view_paths", []), strict=False):
            if not _path_exists(view):
                errors.append(f"missing_sftpp_stage_b_view:{src}")
        if sftpp.get("trainable_adapter") != "new_lora":
            errors.append("sftpp_must_train_new_lora")
        if sftpp.get("loss_weight_field") != "returns.loss_weight":
            errors.append("sftpp_wrong_loss_weight_field")

    runs = plan.get("ablation_lora_runs")
    if not isinstance(runs, list) or not runs:
        errors.append("missing_ablation_lora_runs")
    else:
        for run in runs:
            if not isinstance(run, Mapping):
                errors.append("bad_ablation_run")
                continue
            method = str(run.get("method") or "unknown")
            if run.get("base_checkpoint_type") != "merged_original_sft":
                errors.append(f"ablation_base_must_be_merged_original_sft:{method}")
            if not _path_exists(run.get("base_checkpoint_path")):
                errors.append(f"missing_ablation_base_checkpoint:{method}")
            if not _path_exists(run.get("stage_b_view_path")):
                errors.append(f"missing_stage_b_view:{method}")
            manifest_path = run.get("stage_b_weight_manifest")
            if not _path_exists(manifest_path):
                errors.append(f"missing_stage_b_weight_manifest:{method}")
            else:
                try:
                    manifest = json.loads(Path(str(manifest_path)).read_text())
                except Exception:
                    errors.append(f"bad_stage_b_weight_manifest_json:{method}")
                else:
                    expected_mode = "tracevla_eta" if method.startswith("tracevla_eta_") else method
                    if manifest.get("ablation_mode") != expected_mode:
                        errors.append(f"weight_manifest_mode_mismatch:{method}")
                    if manifest.get("weight_field") != "returns.loss_weight":
                        errors.append(f"weight_manifest_wrong_weight_field:{method}")
                    if manifest.get("safety_leakage", {}) != {}:
                        errors.append(f"weight_manifest_safety_leakage_not_empty:{method}")
            if run.get("loss_weight_field") != "returns.loss_weight":
                errors.append(f"wrong_loss_weight_field:{method}")
            if run.get("trainable_adapter") != "new_lora":
                errors.append(f"ablation_must_train_new_lora:{method}")
    return errors


def materialize_training_run_plan(plan: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Write per-run manifests and Slurm skeletons. Does not submit jobs."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    hprc = plan.get("hprc", {}) if isinstance(plan.get("hprc"), Mapping) else {}
    sea_root = str(hprc.get("sea_vla_root", "/scratch/$USER/SEA-VLA"))
    openpi_root = str(hprc.get("openpi_root", "/scratch/$USER/openpi"))

    def write_run(run: Mapping[str, Any]) -> dict[str, str]:
        run_dir = Path(str(run["local_run_dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "validation").mkdir(exist_ok=True)
        manifest_path = run_dir / "training_run_manifest.json"
        manifest_path.write_text(json.dumps(run, indent=2, sort_keys=True))
        slurm_path = Path(str(run["hprc_slurm_path"]))
        slurm_path.parent.mkdir(parents=True, exist_ok=True)
        slurm_path.write_text(
            _slurm_text(
                method=str(run.get("method", "unknown")),
                sea_vla_root=sea_root,
                openpi_root=openpi_root,
                steps=int(run.get("steps", 0)),
            )
        )
        return {"run_dir": str(run_dir), "manifest": str(manifest_path), "slurm": str(slurm_path)}

    written = {"sftpp_demo_run": write_run(plan["sftpp_demo_run"]), "ablation_lora_runs": []}
    for run in plan.get("ablation_lora_runs", []):
        written["ablation_lora_runs"].append(write_run(run))
    (output_root / "training_run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    (output_root / "materialized_manifest.json").write_text(json.dumps(written, indent=2, sort_keys=True))
    return written
