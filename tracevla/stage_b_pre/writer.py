"""Writers for Stage-B-pre sidecars and manifests."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

try:
    from src.data.episode_buffer import rebuild_phase_segments  # SEA-VLA integration path
except Exception:  # pragma: no cover - public PACER package fallback
    def rebuild_phase_segments(*args, **kwargs):
        return []

from .config import LABELS_REQUIRING_NOTE
from .feedback_compatibility import IncompatibleFeedbackError, assert_compatible
from .schemas import Feedback, PlannerScore, TargetRegion, TrialSpec


class InvalidFeedbackForWrite(ValueError):
    """Raised by the writer when a feedback record cannot be saved cleanly.

    Includes the underlying reason: empty note where required, or
    incompatible `(block, stop_source, label)` triple.
    """


def _validate_feedback_for_write(
    trial: TrialSpec,
    feedback: Feedback,
    metadata: dict[str, Any],
) -> None:
    """Defense-in-depth gate before saving feedback as clean accepted data.

    The collection prompt already filters the menu, but operators can be
    re-prompted, scripts can be patched mid-collection, or trials can be
    saved via direct-call paths. This re-check enforces the binding
    requirement from CONSENSUS_PHASE_A.md / LOKI_RECONCILIATION_V1_1.md §C1:
    "impossible pairs cannot be saved as clean accepted feedback".
    """
    if feedback.label in LABELS_REQUIRING_NOTE and not feedback.note.strip():
        raise InvalidFeedbackForWrite(
            f"feedback label {feedback.label!r} requires a non-empty note; "
            f"trial={trial.trial_id}"
        )
    stop_source = metadata.get("stop_source")
    if stop_source is None:
        # No stop_source recorded -> caller is saving a non-rollout trial
        # (target-region-only or aborted before run). Skip the compatibility
        # gate; the eligibility validator downstream will handle these.
        return
    try:
        assert_compatible(trial.block, stop_source, feedback.label)
    except IncompatibleFeedbackError as exc:
        raise InvalidFeedbackForWrite(
            f"trial={trial.trial_id}: {exc}"
        ) from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


class StageBPreWriter:
    """Save Stage-B-pre trials in config/block/component/trial layout."""

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def config_root(self, config_id: str) -> Path:
        return self.output_root / config_id

    def trial_dir(self, trial: TrialSpec) -> Path:
        return self.config_root(trial.config_id) / trial.block / trial.component / trial.trial_id

    def trial_parent(self, trial: TrialSpec) -> Path:
        # Kept for backward compatibility with early tests/review scripts.
        return self.config_root(trial.config_id) / trial.block / trial.component

    def save_session_meta(self, config_id: str, meta: dict[str, Any]) -> None:
        write_json(self.config_root(config_id) / "session_meta.json", meta)

    def save_trial(
        self,
        trial: TrialSpec,
        frames: list[dict[str, Any]],
        metadata: dict[str, Any],
        target_region: TargetRegion | None = None,
        feedback: Feedback | None = None,
        score: PlannerScore | dict[str, Any] | None = None,
        action_trace: Iterable[dict[str, Any]] | None = None,
    ) -> Path:
        if feedback is not None:
            _validate_feedback_for_write(trial, feedback, metadata)
        episode_dir = self.trial_dir(trial)
        episode_dir.mkdir(parents=True, exist_ok=True)
        phase_segments = rebuild_phase_segments(frames)

        saved_count = 0
        import pickle
        for i, frame in enumerate(frames):
            with (episode_dir / f"frame_{i:04d}.pkl").open("wb") as f:
                pickle.dump(frame, f)
            saved_count += 1

        meta = {
            **metadata,
            **trial.to_dict(),
            "component": trial.component,
            "phase_segments": phase_segments,
            "num_frames": len(frames),
            "num_frames_saved": saved_count,
            "fps": metadata.get("fps", 10),
            "record_hz": metadata.get("record_hz", 10),
        }
        write_json(episode_dir / "episode_meta.json", meta)
        if target_region is not None:
            write_json(episode_dir / "target_region.json", target_region.to_dict())
        if feedback is not None:
            write_json(episode_dir / "stage_b_trial_feedback.json", feedback.to_dict())
        if score is not None:
            write_json(episode_dir / "stage_b_auto_score.json", score if isinstance(score, dict) else score.to_dict())
        if action_trace is not None:
            trace_path = episode_dir / "vla_action_trace.jsonl"
            if trace_path.exists():
                trace_path.unlink()
            for row in action_trace:
                append_jsonl(trace_path, row)
        append_jsonl(
            self.config_root(trial.config_id) / "stage_b_pre_manifest.jsonl",
            {
                **trial.to_dict(),
                "episode_dir": str(episode_dir),
                "num_frames": len(frames),
                "feedback_label": feedback.label if feedback else None,
            },
        )
        return episode_dir
