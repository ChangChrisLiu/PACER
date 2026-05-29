"""Writer for rollout-only model evaluation trials.

Layout:
    <output_root>/<run_id>/<model_id>/<config_id>/<component>/trial_NNN/
        frame_0000.pkl ...
        episode_meta.json
        rollout_eval_feedback.json
        action_trace.jsonl
        rollout_eval_summary.json
    <output_root>/<run_id>/
        eval_session_meta.json
        models.json
        progress.json   (managed by EvalProgressTracker)
        summary.csv
        summary.jsonl
"""
from __future__ import annotations

import csv
import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .eval_feedback import EvalFeedback, is_excluded, is_strict_success
from .eval_plan import EvalTrialSpec, ModelSpec


SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model_id",
    "model_type",
    "checkpoint_path",
    "unnorm_key",
    "config_id",
    "component",
    "trial_index",
    "eval_trial_id",
    "started_at_ts",
    "ended_at_ts",
    "wall_seconds",
    "stop_source",
    "steps",
    "operator_label",
    "operator_note",
    "operator_auto_assigned",
    "strict_success",
    "excluded",
    "final_tcp_x",
    "final_tcp_y",
    "final_tcp_z",
    "final_tcp_rx",
    "final_tcp_ry",
    "final_tcp_rz",
    "fps",
    "max_steps",
    "open_loop_horizon",
    "inference_obs_mode",
    "inference_obs_max_wait_ms",
    "gripper_cap_enabled",
    "gripper_cap_value_norm",
    "gripper_cap_components",
    "episode_dir",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _final_tcp_components(final_tcp: Iterable[float] | None) -> dict[str, Any]:
    keys = ("final_tcp_x", "final_tcp_y", "final_tcp_z",
            "final_tcp_rx", "final_tcp_ry", "final_tcp_rz")
    out: dict[str, Any] = {k: None for k in keys}
    if final_tcp is None:
        return out
    arr = list(final_tcp)
    for i, k in enumerate(keys):
        if i < len(arr):
            try:
                out[k] = float(arr[i])
            except Exception:
                out[k] = None
    return out


class TrialAlreadySavedError(RuntimeError):
    """Raised when save_trial is called for an eval_trial_id that has already
    been recorded in summary.jsonl. Prevents resume-time duplication."""


class RolloutEvalWriter:
    """Save rollout-only trials in per-run/per-model trees."""

    def __init__(self, output_root: str | Path, run_id: str):
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.run_dir = self.output_root / f"run_{run_id}" if not run_id.startswith("run_") else self.output_root / run_id
        # Allow the operator to pass either "20260601_v1" or "run_20260601_v1";
        # the on-disk layout always uses a `run_` prefix.
        if not run_id.startswith("run_"):
            self.run_dir = self.output_root / f"run_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def run_root(self) -> Path:
        return self.run_dir

    def model_dir(self, model_id: str) -> Path:
        return self.run_dir / model_id

    def trial_dir(self, trial: EvalTrialSpec) -> Path:
        return (
            self.run_dir
            / trial.model_id
            / trial.config_id
            / trial.component
            / f"trial_{trial.trial_index:03d}"
        )

    def summary_csv_path(self) -> Path:
        return self.run_dir / "summary.csv"

    def summary_jsonl_path(self) -> Path:
        return self.run_dir / "summary.jsonl"

    def progress_path(self) -> Path:
        return self.run_dir / "progress.json"

    # ------------------------------------------------------------------
    # Session-level writes
    # ------------------------------------------------------------------

    def save_session_meta(self, meta: dict[str, Any]) -> Path:
        path = self.run_dir / "eval_session_meta.json"
        write_json(path, meta)
        return path

    def save_models_registry(self, models: Iterable[ModelSpec]) -> Path:
        path = self.run_dir / "models.json"
        write_json(path, {"models": [m.to_dict() for m in models]})
        return path

    # ------------------------------------------------------------------
    # Trial save
    # ------------------------------------------------------------------

    def _completed_ids(self) -> set[str]:
        path = self.summary_jsonl_path()
        if not path.exists():
            return set()
        ids: set[str] = set()
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and row.get("eval_trial_id"):
                    ids.add(row["eval_trial_id"])
        return ids

    def save_trial(
        self,
        *,
        trial: EvalTrialSpec,
        model: ModelSpec,
        frames: list[dict[str, Any]],
        action_trace: Iterable[dict[str, Any]] | None,
        feedback: EvalFeedback,
        stop_source: str,
        steps: int,
        started_at_ts: float,
        ended_at_ts: float,
        wall_seconds: float,
        final_tcp: list[float] | None,
        runner_args: dict[str, Any],
    ) -> Path:
        if trial.eval_trial_id in self._completed_ids():
            raise TrialAlreadySavedError(
                f"trial already saved: {trial.eval_trial_id}"
            )

        episode_dir = self.trial_dir(trial)
        episode_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            with (episode_dir / f"frame_{i:04d}.pkl").open("wb") as f:
                pickle.dump(frame, f)

        episode_meta = {
            **trial.to_dict(),
            "model_id": model.model_id,
            "model_type": model.model_type,
            "checkpoint_path": model.checkpoint_path,
            "unnorm_key": model.unnorm_key,
            "openpi_base": model.openpi_base,
            "server_host": model.server_host,
            "server_port": model.server_port,
            "language_instruction": runner_args.get("language_instruction", ""),
            "camera_mode": runner_args.get("inference_obs_mode"),
            "fps": runner_args.get("fps"),
            "record_hz": runner_args.get("fps"),
            "max_steps": runner_args.get("max_steps"),
            "open_loop_horizon": runner_args.get("open_loop_horizon"),
            "inference_obs_mode": runner_args.get("inference_obs_mode"),
            "inference_obs_max_wait_ms": runner_args.get("inference_obs_max_wait_ms"),
            "gripper_cap_enabled": runner_args.get("gripper_cap_enabled"),
            "gripper_cap_value_norm": runner_args.get("gripper_cap_value_norm"),
            "gripper_cap_components": runner_args.get("gripper_cap_components"),
            "stop_source": stop_source,
            "steps": int(steps),
            "wall_seconds": float(wall_seconds),
            "started_at_ts": float(started_at_ts),
            "ended_at_ts": float(ended_at_ts),
            "final_tcp": list(final_tcp) if final_tcp is not None else None,
            "num_frames": len(frames),
            "schema_version": "tracevla_rollout_eval.v0.1",
            "trial_mode": "rollout_eval",
        }
        write_json(episode_dir / "episode_meta.json", episode_meta)
        write_json(
            episode_dir / "rollout_eval_feedback.json",
            feedback.to_dict(),
        )

        if action_trace is not None:
            trace_path = episode_dir / "action_trace.jsonl"
            if trace_path.exists():
                trace_path.unlink()
            for row in action_trace:
                append_jsonl(trace_path, row)

        strict = is_strict_success(label=feedback.label, stop_source=stop_source)
        excluded = is_excluded(label=feedback.label, stop_source=stop_source)
        summary_row: dict[str, Any] = {
            "run_id": trial.run_id,
            "model_id": trial.model_id,
            "model_type": model.model_type,
            "checkpoint_path": model.checkpoint_path,
            "unnorm_key": model.unnorm_key,
            "config_id": trial.config_id,
            "component": trial.component,
            "trial_index": trial.trial_index,
            "eval_trial_id": trial.eval_trial_id,
            "started_at_ts": float(started_at_ts),
            "ended_at_ts": float(ended_at_ts),
            "wall_seconds": float(wall_seconds),
            "stop_source": stop_source,
            "steps": int(steps),
            "operator_label": feedback.label,
            "operator_note": feedback.note,
            "operator_auto_assigned": bool(feedback.auto_assigned),
            "strict_success": bool(strict),
            "excluded": bool(excluded),
            "fps": runner_args.get("fps"),
            "max_steps": runner_args.get("max_steps"),
            "open_loop_horizon": runner_args.get("open_loop_horizon"),
            "inference_obs_mode": runner_args.get("inference_obs_mode"),
            "inference_obs_max_wait_ms": runner_args.get("inference_obs_max_wait_ms"),
            "gripper_cap_enabled": runner_args.get("gripper_cap_enabled"),
            "gripper_cap_value_norm": runner_args.get("gripper_cap_value_norm"),
            "gripper_cap_components": ",".join(runner_args.get("gripper_cap_components") or []),
            "episode_dir": str(episode_dir),
            **_final_tcp_components(final_tcp),
        }
        # Persist a per-trial sidecar summary too (handy for downstream tools
        # that prefer one JSON per trial over scanning summary.jsonl).
        write_json(episode_dir / "rollout_eval_summary.json", summary_row)

        # Append to run-level summary.jsonl and summary.csv. Both are
        # append-only; resume duplication is blocked by the _completed_ids
        # check above.
        append_jsonl(self.summary_jsonl_path(), summary_row)
        self._append_csv_row(summary_row)

        return episode_dir

    def _append_csv_row(self, row: dict[str, Any]) -> None:
        path = self.summary_csv_path()
        new_file = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            if new_file:
                writer.writeheader()
            # Only include known columns; unknown keys are dropped to keep the
            # CSV schema stable.
            writer.writerow({k: row.get(k) for k in SUMMARY_COLUMNS})


def aggregate_summary(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the headline rollup from in-memory summary rows.

    Returns:
        - `n_total`: every trial saved
        - `n_excluded`: operator_uncertain or unsafe_abort
        - `n_strict_success`: success-label AND not unsafe_abort
        - `strict_success_rate`: n_strict_success / (n_total - n_excluded)
        - per-model / per-component / per-config breakdowns (same fields)
    """
    rows = list(rows)
    n_total = len(rows)
    n_strict = sum(1 for r in rows if r.get("strict_success"))
    n_excluded = sum(1 for r in rows if r.get("excluded"))
    denom = max(n_total - n_excluded, 0)
    rate = (n_strict / denom) if denom > 0 else None

    def _by(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            k = str(r.get(key, ""))
            groups.setdefault(k, []).append(r)
        out: dict[str, dict[str, Any]] = {}
        for k, group in groups.items():
            g_excluded = sum(1 for r in group if r.get("excluded"))
            g_strict = sum(1 for r in group if r.get("strict_success"))
            g_denom = max(len(group) - g_excluded, 0)
            out[k] = {
                "n_total": len(group),
                "n_excluded": g_excluded,
                "n_strict_success": g_strict,
                "strict_success_rate": (g_strict / g_denom) if g_denom > 0 else None,
            }
        return out

    return {
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_strict_success": n_strict,
        "strict_success_rate": rate,
        "by_model": _by("model_id"),
        "by_component": _by("component"),
        "by_config": _by("config_id"),
    }
