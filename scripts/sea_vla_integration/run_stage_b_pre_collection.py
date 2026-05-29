#!/usr/bin/env python3
# PACER pre-publication integration script.
# This script is copied from the SEA-VLA robotics stack and expects the robot/camera/OpenPI
# adapters from SEA-VLA to be importable. It is intentionally included as an integration
# reference, not as a standalone hardware driver.

"""Stage-B-pre SFT rollout collection entrypoint.

This script provides the operator workflow described in
the internal SEA-VLA Stage-B-pre implementation plan.

Hardware modes intentionally keep the physical joystick mapping unchanged. The
script interprets L25 differently by state: target/failure pose capture rather
than normal run_collection arming.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

# Make src imports work when called from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SEA_VLA_IMPORT_ERROR = None
try:
    from src.agents.joystick_agent import JoystickAgent, load_home_pose
    from src.agents.safety import SafetyMonitor
    from src.agents.vla_agent import VLAAgent
    from src.comms.camera_node import ZMQClientCamera
    from src.comms.robot_node import ZMQClientRobot
    from src.data.episode_buffer import rebuild_phase_segments
    from src.skills.csv_skill_executor import CSVSkillExecutor
except ModuleNotFoundError as exc:  # pragma: no cover - integration-only fallback
    _SEA_VLA_IMPORT_ERROR = exc
    JoystickAgent = SafetyMonitor = VLAAgent = ZMQClientCamera = ZMQClientRobot = CSVSkillExecutor = None
    load_home_pose = None

    def rebuild_phase_segments(*args: Any, **kwargs: Any) -> list[Any]:
        return []


def _require_sea_vla_runtime() -> None:
    if _SEA_VLA_IMPORT_ERROR is not None:
        raise SystemExit(
            "This is a PACER pre-publication SEA-VLA integration reference. "
            "Live collection requires the full SEA-VLA robot runtime on PYTHONPATH. "
            f"Original import error: {_SEA_VLA_IMPORT_ERROR}"
        )
from pacer.stage_b_pre.config import (
    COMPONENT_SEQUENCE,
    CORRECTOR_FEEDBACK_LABELS,
    DEFAULT_FPS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRIAL_COUNTS,
    MODEL_SEGMENT_FEEDBACK_LABEL_SUFFIX,
    MODEL_SEGMENT_LIVE_CONTROLS,
    MODEL_SEGMENT_OPERATOR_CONTROLS,
    PLANNER_FEEDBACK_LABELS,
    PLANNER_ONLY_CORRECTION_EXEMPT_LABELS,
    PLANNER_SKILL_CORRECTION_EXEMPT_LABELS,
    PLANNER_SKILL_FAILURE_LABELS,
    PLANNER_SKILL_FEEDBACK_LABELS,
    TASK_INSTRUCTIONS,
)
from pacer.stage_b_pre.feedback import prompt_label, wait_for_enter
from pacer.stage_b_pre.gripper_verifier import GripperVerification, verify_gripper
from pacer.stage_b_pre.inference_runner import GripperCapConfig, TracedChunkRunner
from pacer.stage_b_pre.scoring import score_model_rollout_performance, score_planner_final_pose
from pacer.stage_b_pre.schemas import Feedback, TargetPoint, TargetRegion, TrialSpec
from pacer.stage_b_pre.teleop_capture import TeleopCaptureController, TeleopThread
from pacer.stage_b_pre.trial_plan import (
    COLLECTION_ORDER_CHOICES,
    COLLECTION_ORDERS,
    COMPONENT_MAJOR_PLANNER_THEN_CORRECTOR,
    DEFAULT_COLLECTION_ORDER,
    LEGACY_BLOCK_MAJOR,
    ProgressTracker,
    SessionMeta,
    VALID_PHASE_BLOCKS,
    build_trial_plan,
    canonical_collection_order,
    normalize_config_id,
)
from pacer.stage_b_pre.reference_cache import (
    CorrectorStartPoseCache,
    PlannerReferenceCache,
    build_planner_reference_binding,
    build_planner_skill_coverage_metadata,
    resolve_planner_skill_target_group,
    resolve_target_group,
)
from pacer.stage_b_pre.writer import StageBPreWriter, write_json

PLANNER_ONLY_FAILURE_LABELS = {
    # V0.7 failure labels.
    "near_but_not_accurate",
    "wrong_orientation_or_wrong_location",
    "totally_off",
    "totally_off_wrong_region_or_target",
    # Legacy V0.6 failure labels (still recognised).
    "near_miss_stop_token",
    "near_miss_manual_stop",
    "near_miss_timeout",
    "wrong_target",
    "bad_orientation",
    "no_stop_timeout",
    "manual_stop_bad",
}
# Stage-B0 V0.1: unsafe_abort is not eligible for recovery/full-demo human
# recording in this pass. The full RTDE stop performed for unsafe_abort can
# leave the control script inactive, so a follow-up joystick segment cannot
# be guaranteed safe without a documented revive step (out of scope here).
RECOVERY_INELIGIBLE_LABELS = {"unsafe_abort"}
CPU_FINISH_COMPONENT = "cpu"

# V0.7 paths to the per-section button maps. Default map drives normal
# collection; correction map drops skills/zones and routes R34 ->
# finish_recording for human recovery / clean-demo segments.
DEFAULT_BUTTON_MAP_PATH = "configs/button_mapping.json"
CORRECTION_BUTTON_MAP_PATH = "configs/button_mapping_stage_b_correction.json"

# V0.7 schema versions stamped into every trial's metadata. Bump when the
# recording timeline or feedback-label semantics change in a non-backwards-
# compatible way.
RECORDING_TIMELINE_VERSION = "stage_b0_v07"
FEEDBACK_SCHEMA_VERSION = "stage_b0_v07"
# V1.0 corrector_only protocol identifier — single rollout per trial.
#
# Replaces the V0.9 ``stage_b0_corrector_v09_3_attempts`` semantics. Under
# the V1.0 protocol the operator runs DIFFERENT TRIALS, not multiple
# attempts within one trial: each corrector_only trial executes a single
# model rollout (up to the configured ``--max-steps``, currently 600) and
# the post-trial reset returns the robot to the per-component temporary
# ``failure_start`` pose rather than the global configured home. See
# the internal corrector-only single-trial redesign note.
CORRECTOR_PROTOCOL_VERSION = "stage_b0_corrector_v10_single_trial"
CORRECTOR_ROLLOUTS_PER_TRIAL = 1
# Compatibility aliases for older tests/loaders that still read the historical
# V0.9 field names. New metadata also exposes rollout-specific names.
CORRECTOR_RETRY_PROTOCOL_VERSION = CORRECTOR_PROTOCOL_VERSION
CORRECTOR_MAX_AUTO_ATTEMPTS = CORRECTOR_ROLLOUTS_PER_TRIAL
# Match scripts/run_collection.py TELEOP_GRIPPER_SPEED. This keeps human
# correction teleop finger-controllable after skills/model rollouts that may
# leave the Robotiq SPE register at a faster value.
TELEOP_GRIPPER_SPEED = 80
POST_TRIAL_LIFT_Z_M = 0.10

# Stage-B0 V0.1 (2026-05-20) home-skip tolerances. Anything tighter than the
# UR5e joint repeatability spec (~0.1 mrad) gives false negatives; loosen
# only if a wider tolerance is justified by the operator workflow.
HOME_JOINT_TOL_RAD = 0.02
HOME_GRIPPER_TOL = 12  # raw 0..255 units

# Fix 1 (2026-05-20) corrector start cache: accepting a cached failure_start
# only counts as physical reuse when the robot is actually at the cached
# joints within this tolerance. Otherwise the script forces the
# adjust/recapture flow so saved metadata never claims "at cached pose"
# while the robot is elsewhere.
CORRECTOR_START_JOINT_TOL_RAD = 0.05

# Stage-B0 V0.1 (2026-05-20) corrector_only manual_stop semantics version.
CORRECTOR_MANUAL_STOP_SCHEMA = "stage_b0_corrector_manual_stop.v0.1"
# Stage-B0 V1.0 (2026-05-21) borderline classifier for the
# `model_stop_token + verifier_not_success` case: the model emitted the
# stop token but the physical verifier did not auto-pass (e.g. ram obs=227).
# The operator must classify the outcome before any human-correction
# recording so a model-stop-token win is not silently downgraded to forced
# human correction.
CORRECTOR_MODEL_STOP_SCHEMA = "stage_b0_corrector_model_stop_borderline.v0.1"
# Stage-B0 V1.0 release gate (2026-05-21): after any corrector_success
# (auto OR operator-confirmed), the operator confirms it is safe to open
# the gripper before any scripted post-trial motion.  Schema version for
# the timeline event and metadata field.
CORRECTOR_RELEASE_GATE_SCHEMA = "stage_b0_corrector_release.v0.1"
# Stage-B corrector gripper verifier family: physical gripper observation is
# authoritative, while cmd/action[6] is telemetry because values near 255 may
# encode synthesized stop-token intent. Current strict contact thresholds:
# ram obs<227; connector/cpu_fan/graphic_card obs<220; cpu obs<165.
CORRECTOR_UNCALIBRATED_COMPONENTS: frozenset[str] = frozenset()
HUMAN_SEGMENT_DECISIONS_SCHEMA = "stage_b0_human_segment_decisions.v0.1"

V07_MANUAL_STOP_FEEDBACK_LABELS = [
    "stop_token_should_emit_here",
    "near_but_not_accurate",
    "wrong_orientation_or_wrong_location",
    "totally_off_wrong_region_or_target",
    "operator_uncertain_exclude",
]
# Stage-B0 V0.5 (2026-05-21): compact V0.7-only planner_only menu for the
# `model_stop_token` exit. The legacy V0.6 stop-token labels
# (success_stop_token / near_miss_stop_token / wrong_target /
# bad_orientation) remain valid in `feedback_compatibility.py` so saved
# V0.6 trials still load, but they are no longer presented to the live
# operator. See `the internal Stage-B0 config_001 issue-A review notes`
# feedback_rtde_investigation_20260521.md` for the operator-confused
# 9-item menu the user hit on 2026-05-21.
V07_PLANNER_ONLY_MODEL_STOP_TOKEN_FEEDBACK_LABELS = [
    "success",
    "near_but_not_accurate",
    "wrong_orientation_or_wrong_location",
    "totally_off",
    "operator_uncertain_exclude",
]
V07_PLANNER_SKILL_MODEL_STOP_COMPLETED_LABELS = [
    "success_skill_completed",
    "near_but_not_accurate",
    "wrong_orientation_or_wrong_location",
    "totally_off_wrong_region_or_target",
    "operator_uncertain_exclude",
]


def _planner_only_feedback_labels_for_prompt(stop_source: str) -> list[str]:
    if stop_source == "manual_stop":
        return V07_MANUAL_STOP_FEEDBACK_LABELS
    if stop_source == "model_stop_token":
        return V07_PLANNER_ONLY_MODEL_STOP_TOKEN_FEEDBACK_LABELS
    return PLANNER_FEEDBACK_LABELS


def _planner_skill_feedback_labels_for_prompt(stop_source: str, skill_completed: bool) -> list[str]:
    if stop_source == "manual_stop":
        return V07_MANUAL_STOP_FEEDBACK_LABELS
    if stop_source == "model_stop_token" and skill_completed:
        return V07_PLANNER_SKILL_MODEL_STOP_COMPLETED_LABELS
    return PLANNER_SKILL_FEEDBACK_LABELS


# ---------------------------------------------------------------------------
# V0.7 frame stamping + classification helpers
# ---------------------------------------------------------------------------

_HUMAN_CONTROL_SOURCES = frozenset({
    "human_correction_from_failure_pose",
    "human_demo_from_home",
    "human_corrector_demo_from_temporary_home",
})
_AUTO_MOTION_CONTROL_SOURCES = frozenset({
    "csv_skill_executor",
    "fixed_skill_execution",
    "scripted_move_home",
})


def _classify_control_source(control_source: str) -> dict[str, bool]:
    """Return the V0.7 is_human / is_model / is_auto_motion triple."""
    if control_source in _HUMAN_CONTROL_SOURCES:
        return {"is_human": True, "is_model": False, "is_auto_motion": False}
    if control_source in _AUTO_MOTION_CONTROL_SOURCES:
        return {"is_human": False, "is_model": False, "is_auto_motion": True}
    # Default: anything starting with "model_" is model output.
    if control_source.startswith("model_"):
        return {"is_human": False, "is_model": True, "is_auto_motion": False}
    return {"is_human": False, "is_model": False, "is_auto_motion": False}


def _stamp_frame(
    frame: dict[str, Any],
    *,
    phase: str,
    control_source: str,
    segment_id: int,
    segment_order: int,
    source_block: str,
    attempt_idx: int | None = None,
) -> None:
    """Stamp V0.7 metadata on a frame in-place.

    The 5 canonical labels (phase, control_source, segment_id,
    segment_order, source_block) plus the derived is_human / is_model /
    is_auto_motion booleans give downstream loaders a deterministic way
    to filter frames by source without re-parsing string heuristics.
    """
    frame["phase"] = phase
    frame["control_source"] = control_source
    frame["segment_id"] = int(segment_id)
    frame["segment_order"] = int(segment_order)
    frame["source_block"] = source_block
    frame["attempt_idx"] = attempt_idx
    frame.update(_classify_control_source(control_source))


# ---------------------------------------------------------------------------
# V0.8 schema-quality helpers
# ---------------------------------------------------------------------------


def _stamp_segment_frames(
    frames: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    segment_id: int,
    source_block: str,
    attempt_idx: int | None = None,
) -> None:
    """Stamp V0.7 trial-level fields onto an already-recorded frame range.

    Frames in `frames[start..end]` are expected to already carry
    `phase` and `control_source` from their source writer (model runner,
    skill recorder, or joystick segment recorder). This helper adds:

    - `segment_id` (trial-local, monotonically assigned by the caller),
    - `segment_order` (0-based within the segment),
    - `source_block` (the planner_only / planner_skill / corrector_only
      bucket the segment belongs to, optionally failure-qualified for
      human recovery/demo segments such as planner_only_failure or
      planner_skill_failure),
    - `attempt_idx` (None unless the caller is implementing the future
      corrector-only N-auto-attempt loop),
    - the derived `is_human` / `is_model` / `is_auto_motion` booleans.

    No `phase` / `control_source` rewrite — those are authoritative from
    the source writer and must not be overwritten by V0.8 stamping.
    """
    if start > end:
        return
    for order, idx in enumerate(range(start, end + 1)):
        frame = frames[idx]
        # Preserve whatever phase / control_source the source writer set.
        control_source = frame.get("control_source", "")
        frame["segment_id"] = int(segment_id)
        frame["segment_order"] = int(order)
        frame["source_block"] = source_block
        frame["attempt_idx"] = attempt_idx
        frame.update(_classify_control_source(control_source))


def _build_action_trace_linkage(
    action_trace: list[dict[str, Any]] | None,
    num_model_frames: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Index action_trace rows by their referenced model frame index.

    Returns `(policy_query_by_frame, executed_action_by_frame)`. Frame
    indices outside `[0, num_model_frames)` are ignored so we never link
    a model frame to a row that references a frame the trial did not
    keep. `None` / missing fields are treated as "no linkage".
    """
    pq: dict[int, list[int]] = {}
    ea: dict[int, list[int]] = {}
    if not action_trace:
        return pq, ea
    for ai, row in enumerate(action_trace):
        if not isinstance(row, dict):
            continue
        rtype = row.get("type")
        if rtype == "policy_query":
            obs_idx = row.get("observation_frame_index")
            if isinstance(obs_idx, int) and 0 <= obs_idx < num_model_frames:
                pq.setdefault(obs_idx, []).append(ai)
        elif rtype == "executed_action":
            fr_idx = row.get("frame_index")
            if isinstance(fr_idx, int) and 0 <= fr_idx < num_model_frames:
                ea.setdefault(fr_idx, []).append(ai)
    return pq, ea


def _attach_action_trace_indices(
    frames: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    pq_by_frame: dict[int, list[int]],
    ea_by_frame: dict[int, list[int]],
) -> None:
    """Attach per-frame action_trace row indices to a model frame range.

    Only frames that have at least one referenced row get the field set,
    so loaders can use `policy_query_trace_indices in frame` as a
    presence check.
    """
    if start > end:
        return
    for order, idx in enumerate(range(start, end + 1)):
        frame = frames[idx]
        pq = pq_by_frame.get(order)
        ea = ea_by_frame.get(order)
        if pq:
            frame["policy_query_trace_indices"] = list(pq)
        if ea:
            frame["executed_action_trace_indices"] = list(ea)


def _make_control_segment(
    *,
    phase: str,
    control_source: str,
    source_block: str,
    segment_id: int,
    start: int,
    end: int,
    attempt_idx: int | None = None,
    source_feedback_label: str | None = None,
) -> dict[str, Any]:
    """Construct a V0.8 control_segments entry.

    Existing V0.7 keys (`phase`, `control_source`, `source_block`,
    `source_feedback_label`, `start`, `end`) are preserved; the V0.8
    additions are `segment_id` and `attempt_idx`. Loaders can use
    `segment_id` to align with `frame.segment_id` and rebuild per-source
    frame ranges without re-deriving them from `phase_segments`.
    """
    seg = {
        "phase": phase,
        "control_source": control_source,
        "source_block": source_block,
        "segment_id": int(segment_id),
        "attempt_idx": attempt_idx,
        "start": int(start),
        "end": int(end),
    }
    if source_feedback_label is not None:
        seg["source_feedback_label"] = source_feedback_label
    return seg


def _make_move_home_event(
    *,
    event_type: str,
    inserted_after_frame: int,
    reason: str,
    control_source: str = "scripted_move_home",
    start_timestamp: float | None = None,
    end_timestamp: float | None = None,
    entry_tcp: list[float] | None = None,
    entry_joints: list[float] | None = None,
    exit_tcp: list[float] | None = None,
    exit_joints: list[float] | None = None,
) -> dict[str, Any]:
    """Build a no-frame timeline event describing a scripted move-home.

    V0.7 default: move-home segments are NOT recorded as frames because
    cameras+obs server liveness isn't guaranteed across the blocking
    move_joints call. Boundary-only sidesteps the risk that auto-motion
    frames would be misused as human demonstrations downstream.
    """
    return {
        "event_type": event_type,
        "control_source": control_source,
        "frames_recorded": False,
        "inserted_after_frame": int(inserted_after_frame),
        "reason": reason,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "entry_tcp": entry_tcp,
        "entry_joints": entry_joints,
        "exit_tcp": exit_tcp,
        "exit_joints": exit_joints,
        "is_auto_motion": True,
        "excluded_from_training": True,
    }


def _v07_correction_required(block: str, label: str) -> bool:
    """Return True if the V0.7 protocol requires mandatory correction.

    Exempt sets cover successes, the stop-token-supervision positive label,
    the universal escape hatch, and unsafe_abort. Everything else is a
    mandatory recovery+demo failure.
    """
    if block == "planner_only":
        return label not in PLANNER_ONLY_CORRECTION_EXEMPT_LABELS
    if block == "planner_skill":
        return label not in PLANNER_SKILL_CORRECTION_EXEMPT_LABELS
    return False


def _float_list(values: Any, limit: int | None = None) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if limit is not None:
        arr = arr[:limit]
    return [float(x) for x in arr]


def _image_fingerprint(img: Any, *, max_samples: int = 4096) -> str:
    """Return a cheap deterministic image fingerprint from sampled pixels."""
    arr = np.asarray(img)
    flat = arr.reshape(-1)
    if flat.size > max_samples:
        idx = np.linspace(0, flat.size - 1, max_samples, dtype=np.int64)
        sample = np.ascontiguousarray(flat[idx])
    else:
        sample = np.ascontiguousarray(flat)
    digest = hashlib.blake2s(digest_size=8)
    digest.update(str(tuple(arr.shape)).encode("ascii"))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(sample.tobytes())
    return digest.hexdigest()


def _build_query_diag_fn_for_test():
    """Build the per-policy-query diagnostics closure.

    The returned function only inspects the already-fetched observation passed
    by TracedChunkRunner. It must not read live cameras or robot clients.
    """

    def query_diag_fn(query_obs):
        diag: dict[str, Any] = {}
        try:
            if not isinstance(query_obs, dict):
                return {"query_diag_error": f"non_dict_obs: {type(query_obs).__name__}"}

            scalar_fields = (
                ("timestamp", "query_obs_timestamp"),
                ("wrist_timestamp", "query_wrist_ts"),
                ("base_timestamp", "query_base_ts"),
                ("waited_s", "query_waited_s"),
                ("obs_lookup_drift_ms", "query_obs_lookup_drift_ms"),
            )
            for src_key, dst_key in scalar_fields:
                if src_key in query_obs and query_obs[src_key] is not None:
                    diag[dst_key] = float(query_obs[src_key])

            joints = query_obs.get("joint_positions")
            if joints is not None:
                diag["query_obs_joints"] = _float_list(joints, limit=6)
            ee = query_obs.get("ee_pos_quat")
            if ee is not None:
                diag["query_ee_pos_quat"] = _float_list(ee)

            for cam_key, field_prefix in (("wrist_rgb", "query_wrist"), ("base_rgb", "query_base")):
                img = query_obs.get(cam_key)
                if img is None:
                    continue
                arr = np.asarray(img)
                diag[f"{field_prefix}_shape"] = list(arr.shape)
                diag[f"{field_prefix}_mean"] = float(arr.mean()) if arr.size else 0.0
                diag[f"{field_prefix}_std"] = float(arr.std()) if arr.size else 0.0
                diag[f"{field_prefix}_fingerprint"] = _image_fingerprint(arr)
        except Exception as exc:
            diag["query_diag_error"] = repr(exc)
        return diag

    return query_diag_fn


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SEA-VLA Stage-B-pre rollout collection")
    ap.add_argument("--config-id", required=True, help="Configuration number/id, e.g. 1 or config_001")
    ap.add_argument(
        "--phase-block",
        default="all",
        choices=list(VALID_PHASE_BLOCKS),
        help=(
            "Phase subset to collect. 'all' = planner-side then corrector_only; "
            "'planner_side' = planner_only + planner_skill only; the three single-"
            "block options remain available for back-compat."
        ),
    )
    ap.add_argument(
        "--collection-order",
        default=DEFAULT_COLLECTION_ORDER,
        choices=list(COLLECTION_ORDER_CHOICES),
        help=(
            "Trial sweep order. 'legacy_block_major' (alias 'block_major') is "
            "the pre-V0.1 default; 'component_major_planner_then_corrector' "
            "sweeps planner_only then planner_skill per component, then "
            "corrector_only across components (Stage-B0 V0.1, 2026-05-20)."
        ),
    )
    ap.add_argument(
        "--require-planner-reference",
        action="store_true",
        help=(
            "Force planner_skill fail-closed when the planner reference cache "
            "has no inherited target_region. Under the V0.1 default "
            "(component_major) this is already on; this flag only matters for "
            "legacy_block_major debug runs that want to opt in."
        ),
    )
    ap.add_argument("--components", default=",".join(COMPONENT_SEQUENCE), help="Comma-separated component subset")
    ap.add_argument("--planner-skill-count", type=int, default=DEFAULT_TRIAL_COUNTS["planner_skill"])
    ap.add_argument("--planner-only-count", type=int, default=DEFAULT_TRIAL_COUNTS["planner_only"])
    ap.add_argument("--corrector-only-count", type=int, default=DEFAULT_TRIAL_COUNTS["corrector_only"])
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--robot-host", default="127.0.0.1")
    ap.add_argument("--camera-host", default="127.0.0.1")
    ap.add_argument("--robot-port", type=int, default=6000)
    ap.add_argument("--obs-port", type=int, default=6002)
    ap.add_argument("--wrist-camera-port", type=int, default=5000)
    ap.add_argument("--base-camera-port", type=int, default=5001)
    ap.add_argument("--planner-server-port", type=int, default=8000)
    ap.add_argument("--corrector-server-port", type=int, default=8001)
    ap.add_argument("--model-type", default="openpi", choices=["openpi", "openvla", "openvla_oft"])
    ap.add_argument("--openpi-base", default="droid")
    ap.add_argument("--unnorm-key", default="ur5e_vla_planner_10hz")
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--open-loop-horizon", type=int, default=10)
    ap.add_argument(
        "--planner-gripper-cap",
        action="store_true",
        help=(
            "Enable the Stage-C0 execution-only OpenPI planner gripper cap "
            "for cpu_fan, ram, connector, and graphic_card. Raw stop-token "
            "detection still runs before the cap."
        ),
    )
    ap.add_argument("--planner-gripper-cap-value", type=float, default=180.0 / 255.0)
    ap.add_argument(
        "--planner-gripper-cap-components",
        default="cpu_fan,ram,connector,graphic_card",
        help="Comma-separated components eligible for --planner-gripper-cap; CPU remains exempt by default.",
    )
    ap.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    ap.add_argument("--inference-obs-mode", default="and", choices=["and", "parallel"])
    ap.add_argument("--inference-obs-max-wait-ms", type=float, default=50.0)
    ap.add_argument("--operator", default="operator")
    ap.add_argument("--direction-bin", default="unspecified")
    ap.add_argument("--location-bin", default="unspecified")
    ap.add_argument("--dry-run", action="store_true", help="Print plan and write session metadata; no hardware")
    ap.add_argument("--confirm-hardware", action="store_true", help="Required for live robot control")
    ap.add_argument("--restart", action="store_true", help="Ignore existing progress.json")
    ap.add_argument("--disable-safety", action="store_true")
    ap.add_argument("--planner-skill-correction",
        default="ask",
        choices=["ask", "never", "always"],
        help=(
            "After a planner_skill failure feedback label, optionally record a "
            "human recovery_correction segment from the failure pose. 'ask' "
            "prompts y/N; 'never' skips; 'always' records without prompting. "
            "Note: unsafe_abort is never eligible for recovery in this pass "
            "(V0.1 §9) — the full-stop already executed for unsafe_abort can "
            "leave RTDE control inactive, so this mode does not apply to it."
        ),
    )
    ap.add_argument(
        "--planner-only-correction",
        default="ask",
        choices=["ask", "never", "always"],
        help=(
            "After a planner_only failure feedback label, optionally record a "
            "human recovery segment from the stopped/failure pose."
        ),
    )
    ap.add_argument(
        "--post-failure-full-demo",
        default="ask",
        choices=["ask", "never", "always"],
        help=(
            "After a planner/planner_skill failure correction, optionally move "
            "home and record a clean full human demonstration from home."
        ),
    )
    ap.add_argument(
        "--planner-skill-setup",
        default="auto_home",
        choices=["auto_home", "manual"],
        help=(
            "planner_skill start procedure. 'auto_home' mirrors the working "
            "scripts/run_planner_skill.py path: move to configured home, settle, "
            "fresh-observation gate, then infer. 'manual' preserves the older "
            "joystick setup gate for special diagnostics."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        _require_sea_vla_runtime()
    # Resolve block_major → legacy_block_major before propagating.
    args.collection_order = canonical_collection_order(args.collection_order)
    config_id = normalize_config_id(args.config_id)
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    counts = {
        "planner_skill": args.planner_skill_count,
        "planner_only": args.planner_only_count,
        "corrector_only": args.corrector_only_count,
    }
    plan = build_trial_plan(
        config_id,
        args.phase_block,
        components,
        counts,
        collection_order=args.collection_order,
    )
    writer = StageBPreWriter(args.output_root)
    session_meta = SessionMeta(
        config_id=config_id,
        component_sequence=list(COMPONENT_SEQUENCE),
        active_components=components,
        trial_counts=counts,
        fps=args.fps,
        robot_port=args.robot_port,
        obs_port=args.obs_port,
        planner_server_port=args.planner_server_port,
        corrector_server_port=args.corrector_server_port,
        operator=args.operator,
        direction_bin=args.direction_bin,
        location_bin=args.location_bin,
        collection_order=args.collection_order,
        gripper_cap_config=GripperCapConfig(
            enabled=bool(args.planner_gripper_cap),
            value_norm=float(args.planner_gripper_cap_value),
            components=tuple(c.strip() for c in args.planner_gripper_cap_components.split(",") if c.strip()),
        ).to_dict(),
    )
    writer.save_session_meta(config_id, session_meta.to_dict())
    progress_path = writer.config_root(config_id) / "progress.json"
    if args.restart and progress_path.exists():
        progress_path.unlink()
    progress = ProgressTracker(progress_path)

    print_plan(plan, args)
    if args.dry_run:
        return
    if not args.confirm_hardware:
        raise SystemExit("Refusing live robot control without --confirm-hardware. Run --dry-run first.")

    runner = LiveCollector(args, writer, progress)
    runner.run(plan)


def print_plan(plan: list[TrialSpec], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for trial in plan:
        counts[trial.block] = counts.get(trial.block, 0) + 1
    print("=" * 72)
    print("SEA-VLA Stage-B-pre collection plan")
    print(f"config_id: {normalize_config_id(args.config_id)}")
    print(f"components: {args.components}")
    print(f"blocks: {counts}")
    print(f"record_hz / inference_fps: {args.fps} Hz")
    print(f"output_root: {args.output_root}")
    print("fixed order: " + " -> ".join(COMPONENT_SEQUENCE))
    print("=" * 72)
    for trial in plan[:20]:
        print(f"{trial.global_index:03d}: {trial.trial_id}")
    if len(plan) > 20:
        print(f"... {len(plan) - 20} more trials")


# ---------------------------------------------------------------------------
# Stage-B0 V0.9 corrector_only operator UX helpers (2026-05-20).
#
# Extracted as module-level helpers so the rollout banner and the
# corrector_only per-attempt orchestration are unit-testable via capsys
# without instantiating the adapter / runner pipeline.
#
# Behaviour vs. the legacy inline prints:
#   - corrector_only suppresses the "use the feedback menu" hint because
#     V0.9 corrector_only has no operator menu — the trial label is picked
#     programmatically from auto_success / unsafe / fallback.
#   - per-attempt banner / result / retry / final lines make it obvious in
#     the terminal that three corrector attempts actually ran.
# ---------------------------------------------------------------------------


def _print_model_segment_banner(*, block: str, component: str, phase: str) -> None:
    """Print the per-rollout banner used by `_run_model_segment`.

    For `block == "corrector_only"` we print only the live-key controls
    (s / q / no-key) and suppress every feedback-menu / feedback-label
    instruction. V0.9 corrector_only chooses the trial label
    programmatically (unsafe_abort / corrector_success /
    human_corrected_then_save), so any "choose the feedback label" /
    "use the feedback menu" hint would mislead the operator into looking
    for a menu that does not exist. Planner_only / planner_skill still
    have an operator menu, so they keep the full controls + menu hint.
    """
    print(f"\nMODEL ROLLOUT START: block={block} component={component} phase={phase}")
    print(MODEL_SEGMENT_LIVE_CONTROLS)
    if block != "corrector_only":
        print(MODEL_SEGMENT_FEEDBACK_LABEL_SUFFIX)
        print(
            "After rollout stops, use the feedback menu to classify the "
            "first decisive outcome/source.\n"
        )


def _print_corrector_attempt_banner(
    *, trial_id: str, component: str, attempt_idx: int, max_attempts: int
) -> None:
    print(
        f"\n[CORRECTOR ROLLOUT {attempt_idx}/{max_attempts}] "
        f"trial={trial_id} component={component}"
    )


def _print_corrector_attempt_result(
    *,
    attempt_idx: int,
    outcome: str,
    stop_source: str,
    steps: int,
    verification: GripperVerification,
) -> None:
    print(
        f"[CORRECTOR ROLLOUT {attempt_idx} RESULT] "
        f"outcome={outcome} stop_source={stop_source} steps={steps} "
        f"cmd={verification.cmd} obs={verification.obs} "
        f"gap={verification.gap} rule={verification.rule}"
    )


def _print_corrector_retry(*, next_attempt_idx: int, max_attempts: int) -> None:
    print(f"[CORRECTOR RETRY] launching attempt {next_attempt_idx}/{max_attempts}")


def _print_corrector_final(
    *,
    unsafe: bool,
    auto_success: bool,
    attempts_run: int,
    max_attempts: int,
    operator_confirmed_success: bool = False,
    operator_uncertain_exclude: bool = False,
    operator_chose_record_correction: bool = False,
) -> None:
    if unsafe:
        state = "unsafe_abort"
    elif auto_success:
        state = "auto_success"
    elif operator_confirmed_success:
        state = "operator_confirmed_success"
    elif operator_uncertain_exclude:
        state = "operator_uncertain_exclude"
    elif operator_chose_record_correction:
        state = "operator_selected_human_correction"
    else:
        state = "rollout_failed -> human_correction_required"
    print(f"[CORRECTOR FINAL] {state}  rollouts_run={attempts_run}/{max_attempts}")


# ---------------------------------------------------------------------------
# Stage-B0 V0.1 (2026-05-20) module-level helpers.
#
# These helpers are module-level so the test suite can exercise the human-
# segment skip semantics, corrector manual_stop classification menu, and
# home-skip decision without instantiating a hardware-bound LiveCollector.
# ---------------------------------------------------------------------------


def _decide_human_segment(
    collector: Any,
    *,
    segment: str,
    label: str,
    mode_attr: str,
    operator: str | None = None,
    input_fn: Any = None,
    forced: bool = False,
) -> dict[str, Any]:
    """Decide whether to record an optional human segment and return metadata.

    Behavior under Stage-B0 V0.1 / Fix 1:

    * ``forced=True`` (V0.7 mandatory-correction gate) → ``[Y/n]`` prompt
      where empty defaults to record; ``n``/``no`` still skips. This is the
      V0.1 relaxation: even formerly required segments are skippable, and
      the decision is captured in metadata.
    * ``mode == "never"`` → skipped_by_mode (no prompt).
    * ``mode == "ask"``   → ``[y/N]``, empty defaults to skip.
    * ``mode == "always"`` → ``[Y/n]``, empty defaults to record.
    * ``EOFError`` (no stdin) → explicit skip with ``noninteractive_eof``.

    The return dict matches the V0.1 ``human_segment_decisions`` schema and
    can be appended directly to ``episode_meta.json``. The caller decides
    whether to actually record a segment based on ``decision == "recorded"``.
    """
    raw_mode = getattr(collector.args, mode_attr, "ask")
    op = operator if operator is not None else getattr(collector.args, "operator", "operator")
    fn = input_fn if input_fn is not None else input

    effective_mode = raw_mode
    if forced and raw_mode != "always":
        effective_mode = "always"

    base = {
        "phase": segment,
        "source_feedback_label": label,
        "mode": raw_mode,
        "effective_mode": effective_mode,
        "forced_by_v07": bool(forced),
        "operator": op,
        "note": "",
    }

    if effective_mode == "never":
        return {**base, "decision": "skipped_by_mode", "reason": "mode_never"}

    if effective_mode == "always":
        prompt = f"Record {segment} segment for label '{label}'? [Y/n]: "
    else:  # ask (or anything unrecognized treated as ask)
        prompt = f"Record {segment} segment for label '{label}'? [y/N]: "

    try:
        ans = fn(prompt).strip().lower()
    except EOFError:
        return {
            **base,
            "decision": "skipped_by_operator",
            "reason": "noninteractive_eof",
        }

    if effective_mode == "always":
        if ans in {"n", "no"}:
            return {
                **base,
                "decision": "skipped_by_operator",
                "reason": "operator_typed_n_or_no",
            }
        return {**base, "decision": "recorded", "reason": "operator_confirmed_default_yes"}

    # ask
    if ans in {"y", "yes"}:
        return {**base, "decision": "recorded", "reason": "operator_typed_y_or_yes"}
    return {
        **base,
        "decision": "skipped_by_operator",
        "reason": "operator_default_no_under_ask_mode",
    }


def _classify_corrector_manual_stop(
    *,
    component: str,
    attempt_idx: int,
    calibrated: bool,
    input_fn: Any = None,
) -> dict[str, Any]:
    """Stage-B0 V1.0 corrector_only manual_stop classification.

    Operator chooses one of:

      1. Good pose/grasp. The corrector should have emitted a stop token here.
      2. Failed rollout; record human correction in this same trial.
      3. Uncertain/exclude.
      4. Unsafe/abort.

    Choice 1 saves ``corrector_success`` with the
    operator-confirmed-success metadata. Choice 2 records a human correction
    in the same saved trial and saves ``human_corrected_then_save``. Choice 3
    labels the trial ``operator_uncertain_exclude`` with a required note.
    Choice 4 routes into the safety/abort path.

    Returns a dict with the keys used by the corrector_only collector path:

    .. code-block:: text

        {
          "schema_version": ...,
          "choice": int,
          "component": str,
          "attempt_idx": int,
          "stop_retry_loop": bool,
          "feedback_label": str,
          "requires_reset_confirm": bool,
          "manual_stop_semantics": str,
          "operator_confirmed_success": bool,
          "stop_token_supervision": bool,
          "corrector_uncalibrated_component": bool,
          "note": str,
        }
    """
    fn = input_fn if input_fn is not None else input
    print(
        "\n[CORRECTOR MANUAL STOP] You pressed 's' during corrector_only.\n"
        "Choose the reason:\n"
        "  1. Good pose/grasp. The corrector should have emitted a stop token here.\n"
        "  2. Failed rollout; record human correction now in this same trial.\n"
        "  3. Uncertain/exclude this trial.\n"
        "  4. Unsafe/abort."
    )
    try:
        raw = fn("Choice [1/2/3/4]: ").strip()
    except EOFError:
        # No interactive stdin -> conservative default: option 3 (exclude).
        return {
            "schema_version": CORRECTOR_MANUAL_STOP_SCHEMA,
            "choice": 3,
            "component": component,
            "attempt_idx": int(attempt_idx),
            "stop_retry_loop": True,
            "feedback_label": "operator_uncertain_exclude",
            "requires_reset_confirm": False,
            "manual_stop_semantics": "operator_uncertain_exclude",
            "operator_confirmed_success": False,
            "stop_token_supervision": False,
            "corrector_uncalibrated_component": not bool(calibrated),
            "note": "noninteractive_eof",
        }

    try:
        choice = int(raw)
    except ValueError:
        print(f"[CORRECTOR MANUAL STOP] unrecognized choice {raw!r}; defaulting to 3 (exclude).")
        choice = 3

    if choice == 1:
        return {
            "schema_version": CORRECTOR_MANUAL_STOP_SCHEMA,
            "choice": 1,
            "component": component,
            "attempt_idx": int(attempt_idx),
            "stop_retry_loop": True,
            "feedback_label": "corrector_success",
            "requires_reset_confirm": False,
            "manual_stop_semantics": "good_pose_stop_token_should_emit_here",
            "operator_confirmed_success": True,
            "stop_token_supervision": True,
            "corrector_uncalibrated_component": not bool(calibrated),
            "note": "",
        }
    if choice == 2:
        return {
            "schema_version": CORRECTOR_MANUAL_STOP_SCHEMA,
            "choice": 2,
            "component": component,
            "attempt_idx": int(attempt_idx),
            "stop_retry_loop": True,
            "feedback_label": "human_corrected_then_save",
            "requires_reset_confirm": False,
            "manual_stop_semantics": "failed_record_human_correction",
            "operator_confirmed_success": False,
            "stop_token_supervision": False,
            "corrector_uncalibrated_component": not bool(calibrated),
            "note": "",
        }
    if choice == 4:
        return {
            "schema_version": CORRECTOR_MANUAL_STOP_SCHEMA,
            "choice": 4,
            "component": component,
            "attempt_idx": int(attempt_idx),
            "stop_retry_loop": True,
            "feedback_label": "unsafe_abort",
            "requires_reset_confirm": False,
            "manual_stop_semantics": "unsafe_abort",
            "operator_confirmed_success": False,
            "stop_token_supervision": False,
            "corrector_uncalibrated_component": not bool(calibrated),
            "note": "",
        }

    # Option 3 (default for unknown / explicit 3): requires a note.
    note = ""
    try:
        note = fn("Note (required for operator_uncertain_exclude): ").strip()
    except EOFError:
        note = "noninteractive_eof"
    return {
        "schema_version": CORRECTOR_MANUAL_STOP_SCHEMA,
        "choice": 3,
        "component": component,
        "attempt_idx": int(attempt_idx),
        "stop_retry_loop": True,
        "feedback_label": "operator_uncertain_exclude",
        "requires_reset_confirm": False,
        "manual_stop_semantics": "operator_uncertain_exclude",
        "operator_confirmed_success": False,
        "stop_token_supervision": False,
        "corrector_uncalibrated_component": not bool(calibrated),
        "note": note,
    }


def _classify_corrector_model_stop_borderline(
    *,
    component: str,
    attempt_idx: int,
    verification: GripperVerification,
    input_fn: Any = None,
) -> dict[str, Any]:
    """Stage-B0 V1.0 corrector_only model_stop_token borderline classifier.

    Invoked when ``result.stop_source == "model_stop_token"`` AND the physical
    gripper verifier did NOT auto-pass (``verification.success is not True``).
    The model emitted a clean stop token, but the verifier either failed (e.g.
    ram ``obs == 227``) or produced no signal — the operator must classify the
    outcome before any human-correction recording.

    Operator choices:

      1. Accept as ``corrector_success`` (operator-confirmed). The stop token
         was emitted and the final state is acceptable.
      2. Reject; record a human correction in the SAME trial ->
         ``human_corrected_then_save``.
      3. Uncertain / exclude this trial -> ``operator_uncertain_exclude``;
         no human correction, no scripted post-trial motion.
      4. Unsafe / abort -> ``unsafe_abort``.

    The decision dict mirrors the manual_stop classifier shape and adds the
    verifier payload (cmd/obs/rule/success) so downstream loaders can audit
    why the borderline menu fired (especially the ram-obs-227 case).
    """
    fn = input_fn if input_fn is not None else input
    print(
        "\n[CORRECTOR MODEL_STOP_TOKEN BORDERLINE]\n"
        f"  component={component} attempt_idx={attempt_idx}\n"
        f"  verifier: cmd={verification.cmd} obs={verification.obs} "
        f"rule={verification.rule} success={verification.success}\n"
        "  The model emitted a stop token, but the physical verifier did NOT\n"
        "  auto-pass.  Classify the outcome:\n"
        "    1. Accept as corrector_success (operator-confirmed).\n"
        "    2. Reject; record human correction now in this same trial.\n"
        "    3. Uncertain/exclude this trial.\n"
        "    4. Unsafe/abort."
    )
    try:
        raw = fn("Choice [1/2/3/4]: ").strip()
    except EOFError:
        # No interactive stdin -> conservative default: option 3 (exclude).
        return {
            "schema_version": CORRECTOR_MODEL_STOP_SCHEMA,
            "choice": 3,
            "component": component,
            "attempt_idx": int(attempt_idx),
            "feedback_label": "operator_uncertain_exclude",
            "model_stop_semantics": "model_stop_token_uncertain_exclude",
            "operator_confirmed_success": False,
            "operator_chose_record_correction": False,
            "operator_uncertain_exclude": True,
            "operator_confirmed_unsafe": False,
            "verifier_success": verification.success,
            "verifier_cmd": verification.cmd,
            "verifier_obs": verification.obs,
            "verifier_gap": verification.gap,
            "verifier_rule": verification.rule,
            "verifier_event": verification.event,
            "verifier_description": verification.description,
            "note": "noninteractive_eof",
        }

    try:
        choice = int(raw)
    except ValueError:
        print(
            f"[CORRECTOR MODEL_STOP_TOKEN BORDERLINE] unrecognized choice "
            f"{raw!r}; defaulting to 3 (exclude)."
        )
        choice = 3

    base = {
        "schema_version": CORRECTOR_MODEL_STOP_SCHEMA,
        "component": component,
        "attempt_idx": int(attempt_idx),
        "verifier_success": verification.success,
        "verifier_cmd": verification.cmd,
        "verifier_obs": verification.obs,
        "verifier_gap": verification.gap,
        "verifier_rule": verification.rule,
        "verifier_event": verification.event,
        "verifier_description": verification.description,
    }
    if choice == 1:
        return {
            **base,
            "choice": 1,
            "feedback_label": "corrector_success",
            "model_stop_semantics": "model_stop_token_accepted_by_operator",
            "operator_confirmed_success": True,
            "operator_chose_record_correction": False,
            "operator_uncertain_exclude": False,
            "operator_confirmed_unsafe": False,
            "note": "",
        }
    if choice == 2:
        return {
            **base,
            "choice": 2,
            "feedback_label": "human_corrected_then_save",
            "model_stop_semantics": "model_stop_token_rejected_record_human_correction",
            "operator_confirmed_success": False,
            "operator_chose_record_correction": True,
            "operator_uncertain_exclude": False,
            "operator_confirmed_unsafe": False,
            "note": "",
        }
    if choice == 4:
        return {
            **base,
            "choice": 4,
            "feedback_label": "unsafe_abort",
            "model_stop_semantics": "model_stop_token_unsafe_abort",
            "operator_confirmed_success": False,
            "operator_chose_record_correction": False,
            "operator_uncertain_exclude": False,
            "operator_confirmed_unsafe": True,
            "note": "",
        }

    # Option 3 (default for unknown / explicit 3): requires a note.
    note = ""
    try:
        note = fn("Note (required for operator_uncertain_exclude): ").strip()
    except EOFError:
        note = "noninteractive_eof"
    return {
        **base,
        "choice": 3,
        "feedback_label": "operator_uncertain_exclude",
        "model_stop_semantics": "model_stop_token_uncertain_exclude",
        "operator_confirmed_success": False,
        "operator_chose_record_correction": False,
        "operator_uncertain_exclude": True,
        "operator_confirmed_unsafe": False,
        "note": note,
    }


def _move_home_with_event(collector: Any) -> dict[str, Any] | None:
    """Move to configured home with Stage-B0 V0.1 "already-home" skip.

    When current joints (best-effort read) are within ``HOME_JOINT_TOL_RAD``
    of ``collector.home_joints`` and the gripper observation is within
    ``HOME_GRIPPER_TOL`` of ``collector.home_gripper``, skip both the +Z
    lift and the joint-home move, and return a ``move_home_skipped``
    timeline event. Otherwise perform the legacy lift-then-joint-home
    sequence (preserving fail-closed behavior on TCP/lift read failures).
    """
    # NB: ``home_joints`` is set by ``LiveCollector.__init__`` via
    # ``load_home_pose()``, which returns a NumPy ndarray.  The previous
    # ``getattr(..., []) or []`` pattern triggered
    # ``ValueError: The truth value of an array with more than one element
    # is ambiguous`` when the attribute was the ndarray (live-collection
    # crash 2026-05-21).  Resolve explicitly with ``is None`` + ``np.asarray``
    # so any array-like (ndarray, list, tuple) is normalised to a flat list.
    _raw_home_joints = getattr(collector, "home_joints", None)
    if _raw_home_joints is None:
        home_joints: list[float] = []
    else:
        home_joints = np.asarray(_raw_home_joints, dtype=float).reshape(-1).tolist()
    skip_event: dict[str, Any] | None = None
    if home_joints:
        try:
            obs = collector.obs_client.get_observations()
            cur_joints = list(obs["joint_positions"])[: len(home_joints)]
            errors = [
                abs(float(cur_joints[i]) - float(home_joints[i]))
                for i in range(len(home_joints))
            ]
            max_err = max(errors) if errors else float("inf")
            gripper_raw = obs.get("gripper_position", 0.0)
            try:
                gripper_arr = np.asarray(gripper_raw, dtype=float).reshape(-1)
                gripper_norm = float(gripper_arr[0]) if gripper_arr.size else 0.0
            except Exception:
                gripper_norm = float(gripper_raw) if isinstance(gripper_raw, (int, float)) else 0.0
            home_gripper = float(getattr(collector, "home_gripper", 0))
            gripper_err = abs(int(round(gripper_norm * 255 if 0.0 <= gripper_norm <= 1.0 else gripper_norm)) - int(round(home_gripper)))
            if max_err <= HOME_JOINT_TOL_RAD and gripper_err <= HOME_GRIPPER_TOL:
                # Verify TCP readability too before declaring already-home.
                # If the TCP socket is dead, we'd want to surface that
                # rather than silently skip motion the operator expects.
                _ = collector.robot.get_tcp_pose_raw()
                ts = time.time()
                skip_event = {
                    "event_type": "move_home_skipped",
                    "control_source": "scripted_move_home",
                    "reason": "already_at_home_within_tolerance",
                    "home_joint_error_max_rad": float(max_err),
                    "home_gripper_error": int(gripper_err),
                    "joint_tolerance_rad": float(HOME_JOINT_TOL_RAD),
                    "gripper_tolerance": int(HOME_GRIPPER_TOL),
                    "start_timestamp": ts,
                    "end_timestamp": ts,
                }
                print(
                    "[MOVE HOME] already at home within tolerance; skipping lift + joint-home."
                )
                return skip_event
        except Exception as exc:
            # State unreadable -> fall through to legacy fail-closed lift+home.
            print(f"[MOVE HOME] could not probe current state ({exc!r}); falling back to lift+home.")

    print("Moving to configured home position...")
    try:
        current_tcp = np.asarray(collector.robot.get_tcp_pose_raw(), dtype=float).copy()
        lift_tcp = current_tcp.copy()
        lift_tcp[2] += POST_TRIAL_LIFT_Z_M
        print(f"Lifting TCP +Z by {POST_TRIAL_LIFT_Z_M:.2f} m before home...")
        collector.robot.move_linear(lift_tcp, speed=0.05, accel=0.1, asynchronous=False)
    except Exception as exc:
        raise RuntimeError(
            f"pre-home vertical lift failed ({exc!r}); automatic joint-home "
            f"not attempted — operator must inspect the cell before continuing."
        ) from exc
    collector.robot.move_joints(list(collector.home_joints), speed=0.5, accel=0.3)
    try:
        collector.robot.set_gripper(collector.home_gripper)
    except Exception:
        pass
    return None


class LiveCollector:
    def __init__(self, args: argparse.Namespace, writer: StageBPreWriter, progress: ProgressTracker):
        self.args = args
        self.writer = writer
        self.progress = progress
        self.robot = ZMQClientRobot(port=args.robot_port, host=args.robot_host)
        self.obs_client = ZMQClientRobot(port=args.obs_port, host=args.robot_host)
        self.cameras = {
            "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.camera_host, camera_name="wrist"),
            "base": ZMQClientCamera(port=args.base_camera_port, host=args.camera_host, camera_name="base"),
        }
        self.agent = JoystickAgent(button_map_path="configs/button_mapping.json")
        # Align Stage-B-pre manual teleop with run_collection.py: slow Robotiq
        # finger speed enough for controlled correction demos.
        try:
            self.robot.set_gripper_speed(TELEOP_GRIPPER_SPEED)
        except Exception:
            pass
        self.teleop = TeleopCaptureController(self.agent, self.robot)
        self.teleop_thread = TeleopThread(self.teleop)
        self.teleop_thread.start()
        self.home_joints, self.home_gripper = load_home_pose(Path("configs"))
        self.safety = None if args.disable_safety else SafetyMonitor()
        self.skill_executor = self._build_skill_executor()

    def _build_skill_executor(self) -> CSVSkillExecutor:
        skills_root = Path("configs") / "skills"
        skill_csvs = {p.stem: str(p) for p in skills_root.glob("*.csv") if p.stem in COMPONENT_SEQUENCE}
        rel_path = skills_root / "relative_counts.json"
        exec_path = skills_root / "execution_config.json"
        relative_counts = json.loads(rel_path.read_text()) if rel_path.exists() else {}
        execution_configs = json.loads(exec_path.read_text()) if exec_path.exists() else {}
        missing = [c for c in COMPONENT_SEQUENCE if c not in skill_csvs]
        if missing:
            print(f"[WARN] Missing skill CSVs for: {missing}")
        return CSVSkillExecutor(
            skill_csvs=skill_csvs,
            robot_client=self.robot,
            obs_client=self.obs_client,
            relative_counts=relative_counts,
            execution_configs=execution_configs,
            move_speed=0.15,
        )

    def run(self, plan: list[TrialSpec]) -> None:
        completed_this_session = 0
        try:
            for trial in plan:
                if self.progress.is_completed(trial):
                    continue
                if completed_this_session and completed_this_session % 30 == 0:
                    wait_for_enter("Mandatory break: check robot/cables/scene, then press Enter to continue...")
                print("\n" + "=" * 72)
                print(f"TRIAL {trial.global_index}: {trial.trial_id}")
                print("=" * 72)
                try:
                    if trial.block == "planner_only":
                        self._run_planner_only(trial)
                    elif trial.block == "corrector_only":
                        self._run_corrector_only(trial)
                    elif trial.block == "planner_skill":
                        self._run_planner_skill_no_corrector(trial)
                    else:
                        raise ValueError(trial.block)
                    self.progress.mark_completed(trial)
                    completed_this_session += 1
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    self.progress.mark_failed(trial, repr(exc))
                    print(f"[ERROR] trial failed: {exc!r}")
                    wait_for_enter("Fix/reset scene if needed, then Enter to continue or Ctrl+C to quit...")
        finally:
            self.teleop.set_active(False)
            self.teleop_thread.stop()
            try:
                self.agent.close()
            except Exception:
                pass

    def _run_planner_only(self, trial: TrialSpec) -> None:
        # V0.7 startup timing markers. The script can't directly measure
        # adapter init / first infer outside _run_model_segment; we record the
        # outer boundaries here and the runner records query_inference_latency_s.
        startup_timing: dict[str, float | None] = {
            "trial_start_ts": time.time(),
            "setup_start_ts": None,
            "setup_home_done_ts": None,
            "fresh_obs_wait_start_ts": None,
            "fresh_obs_wait_end_ts": None,
            "adapter_init_start_ts": None,
            "adapter_init_end_ts": None,
        }
        startup_timing["setup_start_ts"] = time.time()
        # Stage-B0 V0.1: capture target_region only at group boundaries; reuse
        # the planner reference cache for subsequent trials in the same group.
        target_region, planner_reference_binding = self._resolve_planner_only_target_region(trial)
        pre_trial_setup_skip_event = self._move_home()
        startup_timing["setup_home_done_ts"] = time.time()
        result, final_tcp = self._run_model_segment(trial, phase="teleop", server_port=self.args.planner_server_port)
        score = score_planner_final_pose(final_tcp, target_region, result.stop_source, result.steps)
        feedback = prompt_label(
            _planner_only_feedback_labels_for_prompt(result.stop_source),
            f"Planner-only feedback (stop_source={result.stop_source}, steps={result.steps})",
            block=trial.block,
            stop_source=result.stop_source,
        )
        frames = list(result.frames)
        control_segments: list[dict[str, Any]] = []
        timeline_events: list[dict[str, Any]] = []
        if pre_trial_setup_skip_event is not None:
            timeline_events.append(pre_trial_setup_skip_event)
        human_segment_decisions: list[dict[str, Any]] = []
        had_correction = False
        had_full_demo = False
        next_segment_id = 0
        # V0.8 schema-quality: stamp model-rollout frames with segment_id /
        # segment_order / source_block / attempt_idx and the is_*/auto booleans,
        # then emit a control_segments entry covering the model range so
        # downstream loaders see model frames as first-class segments.
        if frames:
            model_end = len(frames) - 1
            _stamp_segment_frames(
                frames,
                start=0,
                end=model_end,
                segment_id=next_segment_id,
                source_block="planner_only",
                attempt_idx=None,
            )
            pq_by_frame, ea_by_frame = _build_action_trace_linkage(
                getattr(result, "action_trace", None), num_model_frames=len(result.frames)
            )
            _attach_action_trace_indices(
                frames, start=0, end=model_end,
                pq_by_frame=pq_by_frame, ea_by_frame=ea_by_frame,
            )
            model_phase = frames[0].get("phase", "teleop")
            model_control_source = frames[0].get("control_source", f"model_{model_phase}")
            control_segments.append(
                _make_control_segment(
                    phase=model_phase,
                    control_source=model_control_source,
                    source_block="planner_only",
                    segment_id=next_segment_id,
                    start=0,
                    end=model_end,
                )
            )
            next_segment_id += 1
        # Stage-B0 V0.1: the V0.7 mandatory-correction gate has been relaxed
        # to "strongly suggested but skippable". `_decide_human_segment`
        # presents the operator with [Y/n] when correction is required or
        # the legacy mode is `always`, and writes the decision into
        # `human_segment_decisions` for audit. Skipped recordings do not
        # produce a fake control_segments entry.
        correction_required_v07 = _v07_correction_required("planner_only", feedback.label)
        legacy_correction_mode = getattr(self.args, "planner_only_correction", "ask")
        legacy_full_demo_mode = getattr(self.args, "post_failure_full_demo", "ask")
        eligible_for_human_segments = (
            (correction_required_v07 or feedback.label in PLANNER_ONLY_FAILURE_LABELS)
            and feedback.label not in RECOVERY_INELIGIBLE_LABELS
            and legacy_correction_mode != "never"
        )
        # Stage-B0 V0.1 / Fix 1: even V0.7 mandatory-correction is now
        # default-yes-but-skippable. The operator sees [Y/n], empty defaults
        # to record, 'n' skips and the skip is recorded in metadata.
        if eligible_for_human_segments:
            correction_decision = _decide_human_segment(
                self,
                segment="recovery_correction",
                label=feedback.label,
                mode_attr="planner_only_correction",
                forced=correction_required_v07,
            )
            human_segment_decisions.append(correction_decision)
            record_correction_now = correction_decision["decision"] == "recorded"
        else:
            human_segment_decisions.append(
                {
                    "phase": "recovery_correction",
                    "source_feedback_label": feedback.label,
                    "mode": legacy_correction_mode,
                    "operator": getattr(self.args, "operator", "operator"),
                    "decision": "ineligible",
                    "reason": (
                        "label_in_recovery_ineligible"
                        if feedback.label in RECOVERY_INELIGIBLE_LABELS
                        else "label_not_in_failure_set"
                    ),
                    "note": "",
                }
            )
            record_correction_now = False
        if record_correction_now:
            correction_start = len(frames)
            correction_frames = self._record_human_joystick_segment(
                trial,
                phase="recovery_correction",
                control_source="human_correction_from_failure_pose",
            )
            if correction_frames:
                frames.extend(correction_frames)
                correction_end = len(frames) - 1
                _stamp_segment_frames(
                    frames,
                    start=correction_start,
                    end=correction_end,
                    segment_id=next_segment_id,
                    source_block="planner_only_failure",
                    attempt_idx=None,
                )
                control_segments.append(
                    _make_control_segment(
                        phase="recovery_correction",
                        control_source="human_correction_from_failure_pose",
                        source_block="planner_only_failure",
                        segment_id=next_segment_id,
                        start=correction_start,
                        end=correction_end,
                        source_feedback_label=feedback.label,
                    )
                )
                next_segment_id += 1
                had_correction = True
        if eligible_for_human_segments and legacy_full_demo_mode != "never":
            full_demo_decision = _decide_human_segment(
                self,
                segment="clean_full_demo",
                label=feedback.label,
                mode_attr="post_failure_full_demo",
                forced=correction_required_v07,
            )
            human_segment_decisions.append(full_demo_decision)
            record_full_demo_now = (
                full_demo_decision["decision"] == "recorded"
                and self._pre_clean_demo_safety_checkpoint(feedback.label)
            )
        else:
            human_segment_decisions.append(
                {
                    "phase": "clean_full_demo",
                    "source_feedback_label": feedback.label,
                    "mode": legacy_full_demo_mode,
                    "operator": getattr(self.args, "operator", "operator"),
                    "decision": "ineligible" if not eligible_for_human_segments else "skipped_by_mode",
                    "reason": (
                        "mode_never" if legacy_full_demo_mode == "never"
                        else "label_in_recovery_ineligible"
                        if feedback.label in RECOVERY_INELIGIBLE_LABELS
                        else "label_not_in_failure_set"
                    ),
                    "note": "",
                }
            )
            record_full_demo_now = False
        if record_full_demo_now:
            print("Moving home before clean full demonstration from home...")
            # V0.7: record the between-segments move-home as a no-frame
            # timeline event so downstream loaders can reconstruct the
            # complete trial timeline.
            move_home_start = time.time()
            between_skip = self._move_home()
            timeline_events.append(_make_move_home_event(
                event_type="move_home_between_segments",
                inserted_after_frame=len(frames) - 1,
                reason="recovery_to_clean_demo",
                start_timestamp=move_home_start,
                end_timestamp=time.time(),
            ))
            if between_skip is not None:
                timeline_events.append(between_skip)
            demo_start = len(frames)
            demo_frames = self._record_human_joystick_segment(
                trial,
                phase="clean_full_demo",
                control_source="human_demo_from_home",
            )
            if demo_frames:
                frames.extend(demo_frames)
                demo_end = len(frames) - 1
                _stamp_segment_frames(
                    frames,
                    start=demo_start,
                    end=demo_end,
                    segment_id=next_segment_id,
                    source_block="planner_only_failure",
                    attempt_idx=None,
                )
                control_segments.append(
                    _make_control_segment(
                        phase="clean_full_demo",
                        control_source="human_demo_from_home",
                        source_block="planner_only_failure",
                        segment_id=next_segment_id,
                        start=demo_start,
                        end=demo_end,
                        source_feedback_label=feedback.label,
                    )
                )
                next_segment_id += 1
                had_full_demo = True
        # V0.7: post-trial scripted move-home is always represented as a
        # no-frame timeline event, even when no correction segments ran.
        post_trial_move_start = time.time()
        # NB: the actual self._move_home() call happens at the end of this
        # method; we record both bracketing timestamps to match.
        timeline_events.append(_make_move_home_event(
            event_type="move_home_post_trial",
            inserted_after_frame=len(frames) - 1,
            reason="post_trial_return_home",
            start_timestamp=post_trial_move_start,
            end_timestamp=None,  # filled below
        ))
        # V0.7 blocker-fix #2: complete the post-trial move-home BEFORE
        # saving the trial, so the persisted episode_meta.json carries a
        # non-null move_home_post_trial.end_timestamp. Previously the home
        # happened after save_trial() and the in-memory mutation was lost.
        post_trial_skip_event = self._move_home()
        if timeline_events and timeline_events[-1].get("event_type") == "move_home_post_trial":
            timeline_events[-1]["end_timestamp"] = time.time()
        if post_trial_skip_event is not None:
            timeline_events.append(post_trial_skip_event)
        metadata = self._base_metadata(trial, stop_source=result.stop_source, control_segments=control_segments)
        metadata.update({
            "had_correction": had_correction,
            "had_full_demo": had_full_demo,
            "recording_timeline_version": RECORDING_TIMELINE_VERSION,
            "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
            "timeline_events": timeline_events,
            "startup_timing": startup_timing,
            "planner_reference": planner_reference_binding,
            "human_segment_decisions": list(human_segment_decisions),
        })
        self.writer.save_trial(
            trial,
            frames,
            metadata,
            target_region=target_region,
            feedback=feedback,
            score=score,
            action_trace=result.action_trace,
        )

    def _run_corrector_only(self, trial: TrialSpec) -> None:
        """V1.0 corrector_only: ONE model rollout per saved trial.

        Replaces the V0.9 3-auto-attempt loop. The quantity being measured
        is whether a single corrector rollout can complete the task within
        the configured max_steps (currently 600). Multiple opportunities
        live in DIFFERENT TRIALS, not attempts within one trial. See
        the internal corrector-only single-trial redesign note.

        Behaviour:

        1. Resolve failure_start via the cache + confirm-start gate
           (``_resolve_corrector_failure_start``) — unchanged. The gate
           still refuses to start a rollout when the robot is not at the
           cached failure_start joints within ``CORRECTOR_START_JOINT_TOL_RAD``.
        2. Run ONE corrector model rollout.
      3. Execute one model rollout. For corrector_only:
         * unsafe_abort -> ``unsafe_abort``.
         * manual_stop -> show the Stage-B0 V0.1 classifier menu:
           option 1 -> ``corrector_success``;
           option 2 -> record human correction in the SAME trial ->
           ``human_corrected_then_save``;
           option 3 -> ``operator_uncertain_exclude``;
           option 4 -> ``unsafe_abort``.
         * model_stop_token + verifier-not-True -> show the borderline
           classifier menu: accept as success, record human correction,
           exclude/uncertain, or unsafe-abort. The stop token is treated as
           emitted, but physical success is still operator/verifier-gated.
         * verifier success -> ``corrector_success``.
         * other verifier failed/unknown exits -> record human correction in
           the SAME trial -> ``human_corrected_then_save``.
        4. Post-trial reset only runs for ``corrector_success`` after the
           release gate confirms that it is safe to open the gripper. It
           returns the robot to the per-component *temporary*
           ``failure_start`` pose via ``_return_to_corrector_failure_start`` —
           NOT the configured global home. ``human_corrected_then_save``,
           unsafe, uncertain/excluded, and release-declined paths deliberately
           skip the scripted return; the next start-cache prompt handles the
           next pose.
        """
        failure_point, failure_start_source = self._resolve_corrector_failure_start(trial)

        combined_frames: list[dict[str, Any]] = []
        combined_action_trace: list[dict[str, Any]] = []
        control_segments: list[dict[str, Any]] = []
        timeline_events: list[dict[str, Any]] = []
        corrector_attempts: list[dict[str, Any]] = []
        manual_stop_decisions: list[dict[str, Any]] = []
        # V1.0 (2026-05-21) model_stop_token borderline classifier decisions.
        # Populated when the model emitted the stop token but the physical
        # verifier did NOT auto-pass (e.g. ram obs=227); the operator's
        # choice is recorded here for audit.
        model_stop_token_decisions: list[dict[str, Any]] = []
        next_segment_id = 0

        auto_success = False
        unsafe = False
        operator_confirmed_success = False
        operator_uncertain_exclude = False
        operator_confirmed_unsafe = False
        operator_chose_record_correction = False
        operator_accepted_model_stop_token = False
        manual_stop_semantics: str | None = None
        model_stop_token_semantics: str | None = None
        stop_token_supervision = False
        corrector_uncalibrated_component = (
            trial.component in CORRECTOR_UNCALIBRATED_COMPONENTS
        )
        terminal_stop_source = "not_run"

        attempt_idx = 1
        _print_corrector_attempt_banner(
            trial_id=trial.trial_id,
            component=trial.component,
            attempt_idx=attempt_idx,
            max_attempts=CORRECTOR_MAX_AUTO_ATTEMPTS,
        )
        result, _final_tcp = self._run_model_segment(
            trial,
            phase="correction",
            server_port=self.args.corrector_server_port,
        )
        terminal_stop_source = result.stop_source

        attempt_frame_start = len(combined_frames)
        attempt_trace_start = len(combined_action_trace)
        attempt_action_trace = list(getattr(result, "action_trace", []) or [])
        combined_action_trace.extend(attempt_action_trace)
        attempt_frames = list(getattr(result, "frames", []) or [])
        combined_frames.extend(attempt_frames)
        attempt_frame_end = len(combined_frames) - 1

        attempt_segment_id: int | None = None
        if attempt_frame_end >= attempt_frame_start:
            _stamp_segment_frames(
                combined_frames,
                start=attempt_frame_start,
                end=attempt_frame_end,
                segment_id=next_segment_id,
                source_block="corrector_only",
                attempt_idx=attempt_idx,
            )
            pq_local, ea_local = _build_action_trace_linkage(
                attempt_action_trace, num_model_frames=len(attempt_frames)
            )
            _attach_action_trace_indices(
                combined_frames,
                start=attempt_frame_start,
                end=attempt_frame_end,
                pq_by_frame=pq_local,
                ea_by_frame=ea_local,
            )
            seg_phase = combined_frames[attempt_frame_start].get("phase", "correction")
            seg_control_source = combined_frames[attempt_frame_start].get(
                "control_source", "model_correction"
            )
            control_segments.append(
                _make_control_segment(
                    phase=seg_phase,
                    control_source=seg_control_source,
                    source_block="corrector_only",
                    segment_id=next_segment_id,
                    start=attempt_frame_start,
                    end=attempt_frame_end,
                    attempt_idx=attempt_idx,
                )
            )
            attempt_segment_id = next_segment_id
            next_segment_id += 1

        verification = self._verify_corrector_attempt(trial, result, attempt_idx)
        outcome = (
            "auto_success" if verification.success is True
            else "unknown" if verification.success is None
            else "failed"
        )
        attempt_steps = int(getattr(result, "steps", 0) or 0)
        _print_corrector_attempt_result(
            attempt_idx=attempt_idx,
            outcome=outcome,
            stop_source=result.stop_source,
            steps=attempt_steps,
            verification=verification,
        )
        corrector_attempts.append({
            "attempt_idx": attempt_idx,
            "segment_id": attempt_segment_id,
            "start_frame": attempt_frame_start if attempt_frame_end >= attempt_frame_start else None,
            "end_frame": attempt_frame_end if attempt_frame_end >= attempt_frame_start else None,
            "action_trace_start": attempt_trace_start,
            "action_trace_end": len(combined_action_trace) - 1,
            "stop_source": result.stop_source,
            "steps": attempt_steps,
            "verification": verification.to_dict(),
            "outcome": outcome,
        })

        if result.stop_source == "unsafe_abort":
            unsafe = True
        elif result.stop_source == "manual_stop":
            decision = _classify_corrector_manual_stop(
                component=trial.component,
                attempt_idx=attempt_idx,
                calibrated=not corrector_uncalibrated_component,
            )
            manual_stop_decisions.append(decision)
            choice = decision["choice"]
            if choice == 1:
                operator_confirmed_success = True
                manual_stop_semantics = decision["manual_stop_semantics"]
                stop_token_supervision = bool(decision["stop_token_supervision"])
            elif choice == 2:
                # V1.0: option 2 no longer triggers an auto-retry. The
                # trial is failed and a human correction is recorded in
                # this same trial -> ``human_corrected_then_save``.
                operator_chose_record_correction = True
                manual_stop_semantics = decision["manual_stop_semantics"]
            elif choice == 3:
                operator_uncertain_exclude = True
                manual_stop_semantics = decision["manual_stop_semantics"]
            elif choice == 4:
                operator_confirmed_unsafe = True
                unsafe = True
                manual_stop_semantics = decision["manual_stop_semantics"]
                # Operator-classified unsafe must save with
                # terminal_stop_source = "unsafe_abort" so the writer's
                # compatibility table accepts the (corrector_only,
                # stop_source, label) triple.
                terminal_stop_source = "unsafe_abort"
        elif (
            result.stop_source == "model_stop_token"
            and verification.success is not True
        ):
            # V1.0 (2026-05-21): model emitted a stop token but the physical
            # verifier did NOT auto-pass (e.g. ram obs=227, the live operator
            # report case).  Surface a borderline classifier so the operator
            # decides whether to accept the stop token as success, record a
            # human correction, exclude, or abort -- instead of silently
            # falling through to forced human correction.
            decision_msb = _classify_corrector_model_stop_borderline(
                component=trial.component,
                attempt_idx=attempt_idx,
                verification=verification,
            )
            model_stop_token_decisions.append(decision_msb)
            choice = decision_msb["choice"]
            if choice == 1:
                operator_confirmed_success = True
                operator_accepted_model_stop_token = True
                model_stop_token_semantics = decision_msb["model_stop_semantics"]
            elif choice == 2:
                operator_chose_record_correction = True
                model_stop_token_semantics = decision_msb["model_stop_semantics"]
            elif choice == 3:
                operator_uncertain_exclude = True
                model_stop_token_semantics = decision_msb["model_stop_semantics"]
            elif choice == 4:
                operator_confirmed_unsafe = True
                unsafe = True
                model_stop_token_semantics = decision_msb["model_stop_semantics"]
                # Same writer-compatibility constraint as manual_stop choice 4:
                # save with terminal_stop_source=unsafe_abort.
                terminal_stop_source = "unsafe_abort"
        elif verification.success is True:
            auto_success = True

        _print_corrector_final(
            unsafe=unsafe,
            auto_success=auto_success,
            attempts_run=attempt_idx,
            max_attempts=CORRECTOR_MAX_AUTO_ATTEMPTS,
            operator_confirmed_success=operator_confirmed_success,
            operator_uncertain_exclude=operator_uncertain_exclude,
            operator_chose_record_correction=operator_chose_record_correction,
        )

        had_human_correction = False
        skip_human_correction = (
            unsafe
            or auto_success
            or operator_confirmed_success
            or operator_uncertain_exclude
        )
        if not skip_human_correction:
            human_start = len(combined_frames)
            human_frames = self._record_human_joystick_segment(
                trial,
                phase="human_corrector_demo",
                control_source="human_corrector_demo_from_temporary_home",
            )
            if human_frames:
                combined_frames.extend(human_frames)
                human_end = len(combined_frames) - 1
                _stamp_segment_frames(
                    combined_frames,
                    start=human_start,
                    end=human_end,
                    segment_id=next_segment_id,
                    source_block="corrector_only",
                    attempt_idx=attempt_idx,
                )
                control_segments.append(
                    _make_control_segment(
                        phase="human_corrector_demo",
                        control_source="human_corrector_demo_from_temporary_home",
                        source_block="corrector_only",
                        segment_id=next_segment_id,
                        start=human_start,
                        end=human_end,
                        attempt_idx=attempt_idx,
                    )
                )
                next_segment_id += 1
                had_human_correction = True

        if unsafe:
            feedback = Feedback(label="unsafe_abort", operator=self.args.operator)
        elif auto_success:
            feedback = Feedback(label="corrector_success", operator=self.args.operator)
        elif operator_confirmed_success:
            feedback = Feedback(label="corrector_success", operator=self.args.operator)
        elif operator_uncertain_exclude:
            note = ""
            if manual_stop_decisions:
                note = str(manual_stop_decisions[-1].get("note") or "")
            elif model_stop_token_decisions:
                note = str(model_stop_token_decisions[-1].get("note") or "")
            feedback = Feedback(
                label="operator_uncertain_exclude",
                operator=self.args.operator,
                note=note,
            )
        else:
            feedback = Feedback(label="human_corrected_then_save", operator=self.args.operator)

        # V1.0 (2026-05-21) release gate: any corrector_success path (auto
        # verifier, manual_stop choice 1, or model_stop_token borderline
        # choice 1) must prompt the operator to confirm release of the held
        # component BEFORE any scripted post-trial motion. Fail-closed: if
        # the operator declines, the label downgrades to
        # operator_uncertain_exclude and no scripted return is issued.
        release_declined = False
        if feedback.label == "corrector_success":
            if auto_success:
                release_success_kind = "auto_success"
            elif operator_accepted_model_stop_token:
                release_success_kind = "model_stop_token_accepted"
            else:
                release_success_kind = "operator_confirmed_success"
            release_decision = self._corrector_success_release_gate(
                trial, success_kind=release_success_kind
            )
            timeline_events.append({
                "event_type": "corrector_release_after_success",
                **release_decision,
            })
            if not release_decision.get("released", False):
                release_declined = True
                # Fail-closed: downgrade label, no scripted post-trial motion
                # with a possibly-held component.
                decline_note = (
                    f"release_declined_after_success:{release_success_kind}"
                )
                feedback = Feedback(
                    label="operator_uncertain_exclude",
                    operator=self.args.operator,
                    note=decline_note,
                )
                operator_uncertain_exclude = True

        # Post-trial reset: only ``corrector_success`` returns to the cached
        # corrector failure_start. ``human_corrected_then_save`` deliberately
        # SKIPS the scripted return so the operator avoids the observed
        # extra lift+return after a manual correction — the next trial's
        # corrector-start cache prompt handles the next start pose.
        # ``unsafe_abort``, ``operator_uncertain_exclude``, and a
        # release-declined success all avoid scripted motion.
        if feedback.label == "corrector_success" and not release_declined:
            post_trial_move_start = time.time()
            timeline_events.append(_make_move_home_event(
                event_type="return_to_failure_start_post_trial",
                inserted_after_frame=len(combined_frames) - 1,
                reason="post_trial_return_to_corrector_failure_start",
                control_source="scripted_return_to_failure_start",
                start_timestamp=post_trial_move_start,
                end_timestamp=None,
            ))
            post_trial_skip_event = self._return_to_corrector_failure_start(failure_point)
            if (
                timeline_events
                and timeline_events[-1].get("event_type") == "return_to_failure_start_post_trial"
            ):
                timeline_events[-1]["end_timestamp"] = time.time()
            if post_trial_skip_event is not None:
                timeline_events.append(post_trial_skip_event)

        metadata = self._base_metadata(
            trial, stop_source=terminal_stop_source, control_segments=control_segments
        )
        metadata.update({
            "corrector_protocol_version": CORRECTOR_PROTOCOL_VERSION,
            "max_corrector_rollouts_per_trial": CORRECTOR_ROLLOUTS_PER_TRIAL,
            "corrector_rollouts": corrector_attempts,
            "corrector_retry_protocol_version": CORRECTOR_RETRY_PROTOCOL_VERSION,
            "max_corrector_attempts": CORRECTOR_MAX_AUTO_ATTEMPTS,
            "corrector_attempts": corrector_attempts,
            "failure_start_point": failure_point.to_dict(),
            "failure_start_source": failure_start_source,
            "had_human_correction": had_human_correction,
            "auto_success": auto_success,
            "recording_timeline_version": RECORDING_TIMELINE_VERSION,
            "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
            "timeline_events": timeline_events,
            "manual_stop_decisions": manual_stop_decisions,
            "manual_stop_semantics": manual_stop_semantics,
            "model_stop_token_decisions": model_stop_token_decisions,
            "model_stop_token_semantics": model_stop_token_semantics,
            "operator_confirmed_success": operator_confirmed_success,
            "operator_uncertain_exclude": operator_uncertain_exclude,
            "operator_confirmed_unsafe": operator_confirmed_unsafe,
            "operator_chose_record_correction": operator_chose_record_correction,
            "operator_accepted_model_stop_token": operator_accepted_model_stop_token,
            "stop_token_supervision": stop_token_supervision,
            "corrector_uncalibrated_component": corrector_uncalibrated_component,
        })

        self.writer.save_trial(
            trial,
            combined_frames,
            metadata,
            feedback=feedback,
            action_trace=combined_action_trace,
        )

    def _verify_corrector_attempt(
        self,
        trial: TrialSpec,
        result: Any,
        attempt_idx: int,
    ) -> GripperVerification:
        """Deterministic post-attempt verification for V0.9 corrector_only.

        Extracts best-effort gripper telemetry from the attempt result and
        delegates to `verify_gripper`. The physical observation is the
        authoritative signal for component contact: corrector/planner OpenPI values
        near action[6]≈1.0 / 255 may be synthesized stop-token intent, not a
        requirement that the model match the CSV skill's calibrated cmd≈231.
        Tests monkey-patch this method to inject deterministic outcomes;
        production paths get the helper's component-specific strict physical
        observation rule:
        ram obs<227; connector/cpu_fan/graphic_card obs<220; cpu obs<165.
        Values at or above each threshold are empty/full close failures.
        """
        final_obs = getattr(result, "final_obs", None)
        cmd: int | None = None
        obs_val: int | None = None

        def _normalize_gripper(value: Any) -> int | None:
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
                if arr.size == 0:
                    return None
                v = float(arr[0])
            except Exception:
                return None
            # Normalize 0..1 → 0..255 only when the value clearly is a
            # normalized fraction. Raw 0..255 values pass through.
            if 0.0 <= v <= 1.0:
                return int(round(v * 255))
            return int(round(v))

        if isinstance(final_obs, dict):
            for key in ("gripper_command", "gripper_target"):
                if key in final_obs and final_obs[key] is not None:
                    cmd = _normalize_gripper(final_obs[key])
                    if cmd is not None:
                        break
            for key in ("gripper_position", "gripper_obs", "gripper_pos"):
                if key in final_obs and final_obs[key] is not None:
                    obs_val = _normalize_gripper(final_obs[key])
                    if obs_val is not None:
                        break
        # Prefer the freshest post-command telemetry from the last executed
        # action trace row. `result.final_obs` is captured at the START of a
        # runner step, before that step's action is applied, so after a final
        # grasp command it can be stale. The apply_action path stores the
        # immediate post-command gripper read in extra.gripper_after_cmd.
        trace = getattr(result, "action_trace", None) or []
        for row in reversed(trace):
            if not (isinstance(row, dict) and row.get("type") == "executed_action"):
                continue
            post_action = row.get("post_safety_clamp_action")
            action_list = post_action if isinstance(post_action, list) else row.get("executed_action")
            if cmd is None and isinstance(action_list, list) and action_list:
                cmd = _normalize_gripper(action_list[-1])
            extra = row.get("extra")
            if isinstance(extra, dict) and extra.get("gripper_after_cmd") is not None:
                post_obs = _normalize_gripper(extra.get("gripper_after_cmd"))
                if post_obs is not None:
                    obs_val = post_obs
            break

        return verify_gripper(
            trial.component, cmd, obs_val, event=f"corrector_attempt_{attempt_idx}"
        )

    def _run_planner_skill_no_corrector(self, trial: TrialSpec) -> None:
        startup_timing: dict[str, float | None] = {
            "trial_start_ts": time.time(),
            "setup_start_ts": time.time(),
            "setup_home_done_ts": None,
            "fresh_obs_wait_start_ts": None,
            "fresh_obs_wait_end_ts": None,
            "adapter_init_start_ts": None,
            "adapter_init_end_ts": None,
        }
        # Stage-B0 V0.1: planner_skill inherits the target_region from the
        # planner_only cache. Fail BEFORE robot motion if no matching capture
        # exists yet — the operator should run planner_only first.
        inherited_target_region, planner_reference_binding = (
            self._resolve_planner_skill_target_region(trial)
        )
        if getattr(self.args, "planner_skill_setup", "auto_home") == "manual":
            self._run_planner_skill_setup_gate(trial)
        else:
            self._run_planner_skill_auto_home_setup(trial)
        startup_timing["setup_home_done_ts"] = time.time()
        startup_timing["fresh_obs_wait_start_ts"] = time.time()
        self._wait_for_fresh_start_observation()
        startup_timing["fresh_obs_wait_end_ts"] = time.time()
        result, final_tcp = self._run_model_segment(trial, phase="teleop", server_port=self.args.planner_server_port)
        skill_frames: list[dict[str, Any]] = []
        skill_info: dict[str, Any] = {}
        skill_completed = False
        skill_outcome = "skill_not_run"
        if result.stop_source == "model_stop_token":
            # V0.3 RTDE patch (2026-05-20): model_stop_token is a clean,
            # non-emergency handoff into CSV skill execution. The rollout
            # was running servoJ (OpenPI) or synchronous moveL (OpenVLA),
            # so there is no speedL window to clear. V0.2 used
            # _recoverable_speed_stop() (speedStop) here, but speedStop
            # against an active servoJ session is the toxic primitive
            # that kills the RTDE control script. The right exit is to
            # stop issuing commands and passively settle before the
            # upcoming moveL. _passive_policy_stop_settle() already
            # sleeps 0.2 s, so no extra time.sleep is needed.
            self._passive_policy_stop_settle()
            # V0.4 RTDE patch (2026-05-20): for OpenPI, additionally
            # release the still-active servoJ mode via servoStop() before
            # the moveL inside _snap_to_vertical() can take over. Without
            # this, the upcoming moveL is silently dropped by the
            # still-in-servoJ script (same root cause as the V0.3
            # recovery_correction "joystick gripper works but TCP does
            # not move" symptom). For OpenVLA / OFT this helper is a
            # no-op (their rollout uses synchronous moveL).
            self._release_servo_mode_for_handoff()
            self._snap_to_vertical()
            trigger_tcp = list(self.robot.get_tcp_pose_raw())
            skill_frames, skill_completed, skill_info = self._execute_skill_recorded(trial.component, trigger_tcp)
            if skill_completed:
                skill_outcome = "completed"
            else:
                skill_outcome = "skill_failed_no_corrector"
        else:
            skill_outcome = f"planner_{result.stop_source}"
        frames = result.frames + skill_frames
        # Stage-B0 V0.1: score against inherited reference (was None).
        pose_score_obj = score_planner_final_pose(
            final_tcp, inherited_target_region, result.stop_source, result.steps
        )
        pose_score = pose_score_obj if isinstance(pose_score_obj, dict) else pose_score_obj.to_dict()
        movement_score = score_model_rollout_performance(result.frames, result.action_trace)
        score = {**pose_score, **movement_score}
        # V0.7 blocker-fix #3: skill verification failure must HARD-PIN the
        # feedback label so the operator cannot save a contradicting
        # success_skill_completed. Codex review §3: the physical verifier
        # outcome must be authoritative for VLM/RL training value.
        auto_skill_verification_failure = (
            result.stop_source == "model_stop_token"
            and not skill_completed
        )
        if auto_skill_verification_failure:
            # Constrained menu: only the auto-label, plus the universal
            # escape hatches. Operator cannot pick success or near-miss.
            constrained_labels = [
                "skill_verification_failed_planner_slightly_inaccurate",
                "operator_uncertain_exclude",
                "unsafe_abort",
            ]
            print(
                "\n[AUTO-LABEL] Skill verification failed after model_stop_token.\n"
                "Choose label (constrained menu — physical verifier outcome is authoritative):"
            )
            feedback = prompt_label(
                constrained_labels,
                f"Planner-skill auto verification-failure feedback "
                f"(stop_source={result.stop_source}, skill_outcome={skill_outcome})",
                # Skip the compatibility-table filter so the constrained
                # menu shows the auto label even if the table-derived
                # label set would not have included it for this stop_source.
            )
            # Defense-in-depth: if anything (a patched prompt, a stale
            # cache, an operator typing the wrong string) returns a label
            # outside the constrained set, HARD-PIN to the auto label.
            # Physical verifier outcome must be authoritative.
            if feedback.label not in constrained_labels:
                print(
                    f"[AUTO-LABEL] Returned label {feedback.label!r} not in "
                    f"constrained set; pinning to auto label "
                    "skill_verification_failed_planner_slightly_inaccurate."
                )
                feedback = Feedback(
                    label="skill_verification_failed_planner_slightly_inaccurate",
                    note=feedback.note,
                    operator=feedback.operator,
                )
        else:
            feedback = prompt_label(
                _planner_skill_feedback_labels_for_prompt(result.stop_source, skill_completed),
                f"Planner-skill no-corrector feedback (stop_source={result.stop_source}, skill_outcome={skill_outcome})",
                block=trial.block,
                stop_source=result.stop_source,
            )
        control_segments: list[dict[str, Any]] = []
        timeline_events: list[dict[str, Any]] = []
        human_segment_decisions: list[dict[str, Any]] = []
        had_correction = False
        next_segment_id = 0
        # V0.8 schema-quality: stamp the model rollout frames and emit a
        # model_teleop control_segments entry covering [0, len(result.frames)-1].
        # Then stamp the CSV-skill frames (auto-motion) and emit their range
        # right after the model segment.
        num_model_frames = len(result.frames)
        if num_model_frames > 0:
            _stamp_segment_frames(
                frames,
                start=0,
                end=num_model_frames - 1,
                segment_id=next_segment_id,
                source_block="planner_skill",
                attempt_idx=None,
            )
            pq_by_frame, ea_by_frame = _build_action_trace_linkage(
                getattr(result, "action_trace", None), num_model_frames=num_model_frames
            )
            _attach_action_trace_indices(
                frames, start=0, end=num_model_frames - 1,
                pq_by_frame=pq_by_frame, ea_by_frame=ea_by_frame,
            )
            model_phase = frames[0].get("phase", "teleop")
            model_control_source = frames[0].get("control_source", f"model_{model_phase}")
            control_segments.append(
                _make_control_segment(
                    phase=model_phase,
                    control_source=model_control_source,
                    source_block="planner_skill",
                    segment_id=next_segment_id,
                    start=0,
                    end=num_model_frames - 1,
                )
            )
            next_segment_id += 1
        if skill_frames:
            skill_start = num_model_frames
            skill_end = num_model_frames + len(skill_frames) - 1
            _stamp_segment_frames(
                frames,
                start=skill_start,
                end=skill_end,
                segment_id=next_segment_id,
                source_block="planner_skill",
                attempt_idx=None,
            )
            skill_phase = frames[skill_start].get("phase", "skill")
            skill_control_source = frames[skill_start].get(
                "control_source", "csv_skill_executor"
            )
            control_segments.append(
                _make_control_segment(
                    phase=skill_phase,
                    control_source=skill_control_source,
                    source_block="planner_skill",
                    segment_id=next_segment_id,
                    start=skill_start,
                    end=skill_end,
                )
            )
            next_segment_id += 1
        # V0.7 mandatory-correction gate.
        correction_required_v07 = _v07_correction_required("planner_skill", feedback.label)
        legacy_correction_mode = getattr(self.args, "planner_skill_correction", "ask")
        legacy_full_demo_mode = getattr(self.args, "post_failure_full_demo", "ask")
        eligible_for_human_segments = (
            (correction_required_v07
             or feedback.label in PLANNER_SKILL_FAILURE_LABELS
             or auto_skill_verification_failure)
            and feedback.label not in RECOVERY_INELIGIBLE_LABELS
            and legacy_correction_mode != "never"
        )
        # Stage-B0 V0.1 / Fix 1: route planner_skill correction through
        # `_decide_human_segment`. V0.7 / auto-verification-failure cases
        # are ``forced`` (default-yes-but-skippable [Y/n]); the legacy mode
        # otherwise governs the prompt style.
        if eligible_for_human_segments:
            ps_forced = correction_required_v07 or auto_skill_verification_failure
            correction_decision = _decide_human_segment(
                self,
                segment="recovery_correction",
                label=feedback.label,
                mode_attr="planner_skill_correction",
                forced=ps_forced,
            )
            human_segment_decisions.append(correction_decision)
            record_correction_now = correction_decision["decision"] == "recorded"
        else:
            human_segment_decisions.append(
                {
                    "phase": "recovery_correction",
                    "source_feedback_label": feedback.label,
                    "mode": legacy_correction_mode,
                    "operator": getattr(self.args, "operator", "operator"),
                    "decision": "ineligible",
                    "reason": (
                        "label_in_recovery_ineligible"
                        if feedback.label in RECOVERY_INELIGIBLE_LABELS
                        else "label_not_in_failure_set"
                    ),
                    "note": "",
                }
            )
            record_correction_now = False
        if record_correction_now:
            correction_start = len(frames)
            # V0.1 §4: human-segment aborts propagate and fail the trial.
            # The outer run() loop turns the exception into a trial-failed entry
            # rather than silently saving partial data.
            correction_frames = self._record_human_joystick_segment(
                trial,
                phase="recovery_correction",
                control_source="human_correction_from_failure_pose",
            )
            if correction_frames:
                frames.extend(correction_frames)
                correction_end = len(frames) - 1
                _stamp_segment_frames(
                    frames,
                    start=correction_start,
                    end=correction_end,
                    segment_id=next_segment_id,
                    source_block="planner_skill_failure",
                    attempt_idx=None,
                )
                control_segments.append(
                    _make_control_segment(
                        phase="recovery_correction",
                        control_source="human_correction_from_failure_pose",
                        source_block="planner_skill_failure",
                        segment_id=next_segment_id,
                        start=correction_start,
                        end=correction_end,
                        source_feedback_label=feedback.label,
                    )
                )
                next_segment_id += 1
                had_correction = True
        # Stage-B0 V0.1 / Fix 1: clean_full_demo also routes through the
        # skippable decision helper.
        if eligible_for_human_segments and legacy_full_demo_mode != "never":
            ps_demo_forced = correction_required_v07 or auto_skill_verification_failure
            full_demo_decision = _decide_human_segment(
                self,
                segment="clean_full_demo",
                label=feedback.label,
                mode_attr="post_failure_full_demo",
                forced=ps_demo_forced,
            )
            human_segment_decisions.append(full_demo_decision)
            record_full_demo_now = (
                full_demo_decision["decision"] == "recorded"
                and self._pre_clean_demo_safety_checkpoint(feedback.label)
            )
        else:
            human_segment_decisions.append(
                {
                    "phase": "clean_full_demo",
                    "source_feedback_label": feedback.label,
                    "mode": legacy_full_demo_mode,
                    "operator": getattr(self.args, "operator", "operator"),
                    "decision": "ineligible" if not eligible_for_human_segments else "skipped_by_mode",
                    "reason": (
                        "mode_never" if legacy_full_demo_mode == "never"
                        else "label_in_recovery_ineligible"
                        if feedback.label in RECOVERY_INELIGIBLE_LABELS
                        else "label_not_in_failure_set"
                    ),
                    "note": "",
                }
            )
            record_full_demo_now = False
        record_full_demo_now = (
            record_full_demo_now
        )
        if record_full_demo_now:
            print("Moving home before clean full demonstration from home...")
            move_home_start = time.time()
            ps_between_skip = self._move_home()
            timeline_events.append(_make_move_home_event(
                event_type="move_home_between_segments",
                inserted_after_frame=len(frames) - 1,
                reason="recovery_to_clean_demo",
                start_timestamp=move_home_start,
                end_timestamp=time.time(),
            ))
            if ps_between_skip is not None:
                timeline_events.append(ps_between_skip)
            demo_start = len(frames)
            demo_frames = self._record_human_joystick_segment(
                trial,
                phase="clean_full_demo",
                control_source="human_demo_from_home",
            )
            if demo_frames:
                frames.extend(demo_frames)
                demo_end = len(frames) - 1
                _stamp_segment_frames(
                    frames,
                    start=demo_start,
                    end=demo_end,
                    segment_id=next_segment_id,
                    source_block="planner_skill_failure",
                    attempt_idx=None,
                )
                control_segments.append(
                    _make_control_segment(
                        phase="clean_full_demo",
                        control_source="human_demo_from_home",
                        source_block="planner_skill_failure",
                        segment_id=next_segment_id,
                        start=demo_start,
                        end=demo_end,
                        source_feedback_label=feedback.label,
                    )
                )
                next_segment_id += 1
                metadata_full_demo = True
            else:
                metadata_full_demo = False
        else:
            metadata_full_demo = False
        # V0.7: post-trial scripted move-home as no-frame timeline event.
        post_trial_move_start = time.time()
        timeline_events.append(_make_move_home_event(
            event_type="move_home_post_trial",
            inserted_after_frame=len(frames) - 1,
            reason="post_trial_return_home",
            start_timestamp=post_trial_move_start,
            end_timestamp=None,
        ))
        metadata = self._base_metadata(
            trial, stop_source=result.stop_source, control_segments=control_segments
        )
        metadata.update(
            {
                "skill_execution_enabled": True,
                "skill_completed": skill_completed,
                "skill_outcome": skill_outcome,
                "skill_info": skill_info,
                "skill_verification_failure_auto_labeled": auto_skill_verification_failure,
                "recording_timeline_version": RECORDING_TIMELINE_VERSION,
                "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
                "timeline_events": timeline_events,
                "startup_timing": startup_timing,
                "had_correction": had_correction,
                "had_full_demo": metadata_full_demo,
                "planner_reference": planner_reference_binding,
                "human_segment_decisions": list(human_segment_decisions),
            }
        )
        # V0.7 blocker-fix #2: complete post-trial move-home BEFORE save so
        # the persisted move_home_post_trial.end_timestamp is non-null.
        ps_post_trial_skip = self._move_home()
        if timeline_events and timeline_events[-1].get("event_type") == "move_home_post_trial":
            timeline_events[-1]["end_timestamp"] = time.time()
        if ps_post_trial_skip is not None:
            timeline_events.append(ps_post_trial_skip)
        # Stage-B0 V0.1: planner_skill writes the inherited target_region as
        # a sidecar so downstream loaders see one consistent shape across
        # planner_only and planner_skill. `TargetRegion.source` is preserved
        # as ``planner_reference_cache`` in the binding metadata above.
        self.writer.save_trial(
            trial,
            frames,
            metadata,
            target_region=inherited_target_region,
            feedback=feedback,
            score=score,
            action_trace=result.action_trace,
        )

    def _should_record_planner_skill_correction(self, label: str) -> bool:
        # V0.1 §3: missing namespace attr defaults to "never", not "ask", so
        # existing test fixtures that construct partial argparse.Namespaces do
        # not accidentally read from stdin during pytest stdin capture.
        mode = getattr(self.args, "planner_skill_correction", "never")
        if mode == "never":
            return False
        if mode == "always":
            return True
        # mode == "ask"
        try:
            ans = input(f"Record human correction for label '{label}'? [y/N]: ").strip().lower()
        except EOFError:
            return False
        return ans in {"y", "yes"}

    def _should_record_planner_only_correction(self, label: str) -> bool:
        mode = getattr(self.args, "planner_only_correction", "never")
        if mode == "never":
            return False
        if mode == "always":
            return True
        try:
            ans = input(f"Record human recovery correction for label '{label}'? [y/N]: ").strip().lower()
        except EOFError:
            return False
        return ans in {"y", "yes"}

    def _should_record_post_failure_full_demo(self, label: str, had_correction: bool) -> bool:
        mode = getattr(self.args, "post_failure_full_demo", "never")
        if mode == "never":
            return False
        if mode == "always":
            return True
        if not had_correction:
            print(
                f"Record clean full demonstration from home for label '{label}'? "
                "No recovery correction was recorded."
            )
        try:
            ans = input(
                f"Move home and record clean full demo from home for label '{label}'? [y/N]: "
            ).strip().lower()
        except EOFError:
            return False
        return ans in {"y", "yes"}

    def _pre_clean_demo_safety_checkpoint(self, label: str) -> bool:
        """Operator safety checkpoint required before _move_home() for clean_full_demo.

        V0.1 §10: the operator must visually clear any held part, bad gripper
        state, or unsafe orientation before the robot snaps to home joints,
        because move_joints(home_joints) after a failure can swing a held part
        through the workspace.

        Returns True only on an explicit typed 'y' or 'yes' confirmation. An
        empty/N response, EOFError, or any other input skips the clean_full_demo
        for this trial. This is intentionally fail-safe: ambiguous confirmation
        is treated as "do not move home blindly".
        """
        print(
            f"\n[SAFETY CHECKPOINT] About to move robot to home joints for clean_full_demo "
            f"(label={label}).\n"
            "  - Visually clear any held part from the gripper.\n"
            "  - Verify the gripper is not jammed or in a stuck state.\n"
            "  - Verify the wrist orientation will not swing through the workspace.\n"
            "Type 'y' or 'yes' to confirm and continue, anything else skips this clean_full_demo."
        )
        try:
            ans = input("Confirm pre-clean-demo safety (y/yes/N): ").strip().lower()
        except EOFError:
            print("[SAFETY CHECKPOINT] No interactive stdin; skipping clean_full_demo.")
            return False
        if ans not in {"y", "yes"}:
            print("[SAFETY CHECKPOINT] Not confirmed; skipping clean_full_demo.")
            return False
        return True

    def _run_planner_skill_auto_home_setup(self, trial: TrialSpec) -> None:
        """Start planner_skill like the working quick inference script.

        The user's validated `scripts/run_planner_skill.py` always moves to the
        configured home pose, settles, then starts OpenPI inference from the
        trained start distribution. The older Stage-B-pre manual setup gate made
        arbitrary pre-start poses the default, which can create a distribution
        mismatch and make planner-skill results incomparable.
        """
        print(
            "PLANNER-SKILL AUTO-HOME SETUP: moving to configured home before planner inference "
            "(matches scripts/run_planner_skill.py). Use --planner-skill-setup manual for joystick setup."
        )
        # V0.2 RTDE patch (2026-05-20): pre-inference setup is non-emergency.
        # stopL/stopJ against an idle queue can kill the RTDE control script;
        # speed_stop alone (plus 50 ms settle) is sufficient to clear any
        # lingering speedL before moveJ.
        self.teleop.set_active(False)
        self._recoverable_speed_stop()
        self._move_home()
        self._recoverable_speed_stop()
        time.sleep(0.3)

    def _run_planner_skill_setup_gate(self, trial: TrialSpec) -> None:
        """Allow manual joystick positioning before planner-skill inference.

        This pre-start gate is intentionally out-of-band: it records no frames,
        saves no trial data, and returns only after joystick teleop is disabled
        so the VLA segment has sole ownership of robot motion.
        """
        self._assert_teleop_alive("planner_skill_setup_gate")
        print(
            "PLANNER-SKILL SETUP: joystick enabled for manual positioning.\n"
            "Move/home the robot and reset the scene as needed.\n"
            "L34 = move to configured home only in this setup gate.\n"
            "planner_skill does NOT use L25 target-point recording; no target input is required.\n"
            "Enter in terminal = start planner inference. Ctrl+C = quit without saving this trial."
        )
        self.teleop.set_active(True)
        try:
            while True:
                self._assert_teleop_alive("planner_skill_setup_gate_loop")
                if _stdin_ready():
                    _ = sys.stdin.readline()
                    break
                event = self.agent._poll_once()  # noqa: SLF001
                if event and event[0] == "home_and_save":
                    print("Moving to configured home before planner inference...")
                    # V0.2 RTDE patch (2026-05-20): home in the setup gate is
                    # a non-emergency handoff; use recoverable stop only so
                    # the RTDE control script survives across the moveJ.
                    self.teleop.set_active(False)
                    self._recoverable_speed_stop()
                    self._move_home()
                    self._recoverable_speed_stop()
                    print(
                        "Home complete. Joystick is now disabled to avoid drift/stale velocity. "
                        "Press Enter to start planner inference, or Ctrl+C to abort."
                    )
                time.sleep(0.01)
        finally:
            # V0.2 RTDE patch (2026-05-20): final cleanup leaves the robot
            # idle for the upcoming VLA inference. No emergency, no in-flight
            # moveL/moveJ — recoverable stop is the right primitive.
            self.teleop.set_active(False)
            self._recoverable_speed_stop()
            time.sleep(0.5)

    def _wait_for_fresh_start_observation(self, min_wait_s: float = 0.5, timeout_s: float = 3.0) -> None:
        """Wait until cameras/robot state are fresh after manual setup/home.

        This prevents the first VLA query from pairing post-home robot motion
        with stale camera timestamps or stale robot history from before the
        setup gate, which can make OpenPI produce a first absolute joint action
        toward the old start pose.
        """
        cutoff = time.time()
        deadline = cutoff + timeout_s
        latest_wrist = 0.0
        latest_base = 0.0
        print("Waiting for fresh post-setup camera/robot observation before planner inference...")
        while time.time() < deadline:
            latest_wrist = float(self.cameras["wrist"].read()[0])
            latest_base = float(self.cameras["base"].read()[0])
            if latest_wrist >= cutoff and latest_base >= cutoff and time.time() >= cutoff + min_wait_s:
                break
            time.sleep(0.02)
        else:
            print(
                "[WARN] Timed out waiting for fresh post-setup camera timestamps; "
                f"wrist_age={time.time() - latest_wrist:.2f}s base_age={time.time() - latest_base:.2f}s. "
                "Proceeding with query-observation safety clamp (legacy path)."
            )
        # Force one live robot read after the camera freshness wait so the
        # observation server history includes the home/setup pose before the
        # first action is applied.
        _ = self.obs_client.get_observations()
        time.sleep(0.1)

    def _execute_skill_recorded(self, component: str, trigger_tcp: list[float]):
        if not self.skill_executor.has_skill(component):
            return [], False, {"error": f"missing_skill_{component}"}
        frames: list[dict[str, Any]] = []
        stop_event = threading.Event()

        def record_loop() -> None:
            build_frame, _create_adapter, get_obs = _load_inference_utils()
            dt = 1.0 / float(self.args.fps)
            while not stop_event.is_set():
                t0 = time.time()
                try:
                    obs = get_obs(
                        self.obs_client,
                        self.cameras,
                        obs_mode=self.args.inference_obs_mode,
                        max_wait_s=self.args.inference_obs_max_wait_ms / 1000.0,
                    )
                    frame = build_frame(obs, self.cameras, self.args.image_size)
                    frame["phase"] = "skill"
                    frame["control_source"] = "csv_skill_executor"
                    frames.append(frame)
                except Exception as exc:
                    print(f"[skill-record] frame capture failed: {exc}")
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        thread = threading.Thread(target=record_loop, daemon=True)
        thread.start()
        interrupt = threading.Event()
        try:
            completed, info = self.skill_executor.execute(
                component,
                trigger_tcp_raw=np.asarray(trigger_tcp, dtype=float),
                interrupt_event=interrupt,
                on_grasp_failed=lambda: interrupt.set(),
            )
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
        return frames, bool(completed), dict(info or {})

    def _run_model_segment(self, trial: TrialSpec, phase: str, server_port: int):
        _print_model_segment_banner(
            block=trial.block, component=trial.component, phase=phase
        )
        build_frame, create_adapter, get_obs = _load_inference_utils()
        prompt = TASK_INSTRUCTIONS[trial.component]
        adapter_args = argparse.Namespace(**vars(self.args))
        adapter_args.server_port = server_port
        adapter_args.server_host = self.args.server_host
        adapter = create_adapter(adapter_args)  # type: ignore[arg-type]
        vla_agent = VLAAgent(adapter, self.args.fps, prompt, task=trial.component, safety_monitor=self.safety)
        runner = TracedChunkRunner(
            fps=self.args.fps,
            max_steps=self.args.max_steps,
            open_loop_horizon=self.args.open_loop_horizon,
            model_type=self.args.model_type,
            checkpoint_name=f"{self.args.model_type}:{server_port}",
            adapter_action_format="openpi_absolute_joint" if self.args.model_type == "openpi" else "eef_delta",
            gripper_cap=GripperCapConfig(
                enabled=bool(getattr(self.args, "planner_gripper_cap", False)),
                value_norm=float(getattr(self.args, "planner_gripper_cap_value", 180.0 / 255.0)),
                components=tuple(
                    c.strip()
                    for c in str(getattr(self.args, "planner_gripper_cap_components", "cpu_fan,ram,connector,graphic_card")).split(",")
                    if c.strip()
                ),
            ),
        )

        def obs_fn():
            return get_obs(
                self.obs_client,
                self.cameras,
                obs_mode=self.args.inference_obs_mode,
                max_wait_s=self.args.inference_obs_max_wait_ms / 1000.0,
            )

        def infer_fn(obs, text_prompt):
            return adapter.infer(obs, text_prompt)

        def apply_fn(action, obs):
            # Match the legacy run_inference.py execution path: clamp against
            # the query observation, not a second live read. The previous
            # live-obs swap is removed because it can collapse OpenPI absolute
            # targets to ~current joints when the live read returns stale or
            # cached state, producing zero arm motion.
            #
            # The adapter return value (OpenPI: clamped 7-D target; OpenVLA*:
            # None) plus a non-fatal post-command obs read flow into the
            # runner's diagnostic dict so the trace records what was actually
            # sent and what the robot reported after.
            pre_joints = None
            try:
                pre = obs.get("joint_positions") if isinstance(obs, dict) else None
                if pre is not None:
                    pre_joints = _float_list(pre, limit=6)
            except Exception:
                pre_joints = None

            live_obs_joints_pre = None
            live_obs_pre_error = None
            try:
                live_pre = self.obs_client.get_observations()
                live_obs_joints_pre = _float_list(live_pre["joint_positions"], limit=6)
            except Exception as exc:
                live_obs_pre_error = repr(exc)
            clamped = adapter.apply_action(action, obs, self.robot, self.safety)

            joints_after = None
            gripper_after = None
            post_err = None
            try:
                obs_after = self.obs_client.get_observations()
                joints_after = _float_list(obs_after["joint_positions"], limit=6)
                grip_raw = obs_after.get("gripper_position")
                if grip_raw is not None:
                    try:
                        grip_arr = np.asarray(grip_raw, dtype=float).reshape(-1)
                        gripper_after = float(grip_arr[0]) if grip_arr.size else None
                    except Exception:
                        gripper_after = None
            except Exception as exc:
                post_err = repr(exc)

            clamp_list = None
            if clamped is not None:
                try:
                    clamp_list = _float_list(clamped)
                except Exception:
                    clamp_list = None

            return {
                "post_safety_clamp_action": clamp_list,
                "extra": {
                    "pre_action_obs_joints": pre_joints,
                    "live_obs_joints_pre": live_obs_joints_pre,
                    "live_obs_pre_error": live_obs_pre_error,
                    "joints_after_cmd": joints_after,
                    "gripper_after_cmd": gripper_after,
                    "post_command_read_error": post_err,
                },
            }

        query_diag_fn = _build_query_diag_fn_for_test()

        def frame_fn(obs):
            return build_frame(obs, self.cameras, self.args.image_size)

        def stop_fn(chunk, obs):
            current_state = adapter.get_current_state(obs)
            return vla_agent._check_chunk_stop(chunk, current_state)  # noqa: SLF001

        def manual_stop():
            if _stdin_ready():
                ch = sys.stdin.read(1)
                if ch == "s":
                    return "manual_stop"
                if ch == " ":
                    return "operator_space_skill_trigger"
                if ch == "q":
                    return "unsafe_abort"
            return None

        with _raw_terminal_if_possible():
            result = runner.run(
                component=trial.component,
                prompt=prompt,
                phase=phase,
                get_obs=obs_fn,
                infer=infer_fn,
                apply_action=apply_fn,
                build_frame=frame_fn,
                stop_check=stop_fn,
                manual_stop=manual_stop,
                query_diag_fn=query_diag_fn,
            )
        self._stop_after_model_rollout(result.stop_source)
        final_tcp = None
        try:
            final_tcp = list(self.robot.get_tcp_pose_raw())
        except Exception:
            pass
        return result, final_tcp

    # ------------------------------------------------------------------
    # Stage-B0 V0.1 (2026-05-20) reference-cache resolvers
    # ------------------------------------------------------------------

    def _resolve_config_dir(self, trial: TrialSpec) -> Path | None:
        """Best-effort config-dir lookup for cache modules.

        Real CLI runs use ``StageBPreWriter.config_root(config_id)``; test
        stubs may provide a writer without that method, in which case the
        callers fall back to legacy behavior so V0.7 unit tests stay green.
        """
        writer = getattr(self, "writer", None)
        if writer is None:
            return None
        cfg_root = getattr(writer, "config_root", None)
        if callable(cfg_root):
            try:
                return Path(cfg_root(trial.config_id))
            except Exception:
                return None
        output_root = getattr(writer, "output_root", None)
        if output_root is not None:
            return Path(output_root) / trial.config_id
        return None

    def _planner_reference_cache(self, trial: TrialSpec) -> PlannerReferenceCache | None:
        config_dir = self._resolve_config_dir(trial)
        if config_dir is None:
            return None
        return PlannerReferenceCache(config_dir)

    def _corrector_start_cache(self, trial: TrialSpec) -> CorrectorStartPoseCache | None:
        config_dir = self._resolve_config_dir(trial)
        if config_dir is None:
            return None
        return CorrectorStartPoseCache(config_dir)

    def _resolve_planner_only_target_region(
        self, trial: TrialSpec
    ) -> tuple[TargetRegion, dict[str, Any]]:
        """Capture-or-reuse the planner_only target_region for ``trial``.

        Returns ``(target_region, binding_metadata)``. The first trial in
        each group captures a fresh region and writes it into the cache;
        later trials in the same group reuse the cache without prompting.

        For writer stubs that don't expose ``config_root`` / ``output_root``
        (used by V0.7 unit tests), this falls back to the legacy
        per-trial capture without touching disk.
        """
        cache = self._planner_reference_cache(trial)
        if cache is None:
            region = self._capture_target_region(trial)
            source = getattr(region, "source", "stub")
            return region, {
                "reference_mode": "captured_now_no_cache",
                "target_region_source": source,
                "note": "no writer.config_root / output_root; cache bypassed",
            }
        try:
            group_id = resolve_target_group(trial.component, trial.trial_index)
        except ValueError as exc:
            # Component not in schedule (or trial_index out of range);
            # fall back to legacy per-trial capture and warn.
            print(f"[PLANNER REFERENCE] {exc}; falling back to legacy per-trial capture.")
            region = self._capture_target_region(trial)
            return region, {
                "reference_mode": "captured_now_no_schedule",
                "target_region_source": region.source,
            }
        cached = cache.get(trial.component, group_id)
        if cached is None:
            # First trial in this group — operator captures fresh reference.
            print(
                f"[PLANNER REFERENCE] capturing target_region for "
                f"component={trial.component} group_id={group_id} "
                f"(trial_index={trial.trial_index})."
            )
            region = self._capture_target_region(trial)
            if not isinstance(region, TargetRegion):
                # Test stubs may return a placeholder; do not pollute the
                # cache with non-TargetRegion data.
                return region, {
                    "reference_mode": "captured_now_test_stub",
                    "target_region_source": "stub",
                    "target_group_id": group_id,
                }
            cache.put(
                component=trial.component,
                group_id=group_id,
                target_region=region,
                captured_in_trial_id=trial.trial_id,
            )
            binding = build_planner_reference_binding(
                cache,
                component=trial.component,
                group_id=group_id,
                reference_mode="captured_now",
                target_region_source=region.source,
            )
            return region, binding

        print(
            f"[PLANNER REFERENCE] reusing cached target_region for "
            f"component={trial.component} group_id={group_id} "
            f"(captured_in={cached.captured_in_trial_id})."
        )
        binding = build_planner_reference_binding(
            cache,
            component=trial.component,
            group_id=group_id,
            reference_mode="reused_from_planner_only_cache",
            target_region_source="planner_reference_cache",
        )
        return cached.target_region, binding

    def _resolve_planner_skill_target_region(
        self, trial: TrialSpec
    ) -> tuple[TargetRegion | None, dict[str, Any]]:
        """Inherit the planner_only target_region for a planner_skill trial.

        Fails closed (raises ``MissingPlannerReferenceError``) before any
        robot motion if no matching planner_only capture exists yet **and**
        the writer is the production ``StageBPreWriter``.  V0.7 test stubs
        that supply a minimal writer get a soft fallback (target_region =
        None) so they continue to exercise the segment-recording logic
        without needing a planner_only cache.
        """
        from pacer.stage_b_pre.reference_cache import MissingPlannerReferenceError

        cache = self._planner_reference_cache(trial)
        if cache is None:
            return None, {
                "reference_mode": "no_cache_available",
                "target_region_source": None,
                "note": "writer is a test stub without config_root/output_root",
            }
        try:
            group_id = resolve_planner_skill_target_group(
                trial.component, trial.trial_index
            )
        except ValueError as exc:
            print(f"[PLANNER REFERENCE] {exc}; planner_skill cannot inherit target_region.")
            return None, {
                "reference_mode": "no_schedule_for_component",
                "target_region_source": None,
                "note": str(exc),
            }
        cached = cache.get(trial.component, group_id)
        if cached is None:
            # Fix 1 (2026-05-20): fail-closed is the default under the new
            # protocol (component_major or planner_side / all). The soft
            # fallback only applies when both:
            #   * the run is in legacy_block_major mode (typically a resume
            #     of a pre-V0.1 config or a one-off debug trial), and
            #   * the operator did NOT pass --require-planner-reference.
            require_flag = bool(getattr(self.args, "require_planner_reference", False))
            collection_order = canonical_collection_order(
                getattr(self.args, "collection_order", LEGACY_BLOCK_MAJOR)
            )
            new_protocol = collection_order == COMPONENT_MAJOR_PLANNER_THEN_CORRECTOR
            if require_flag or new_protocol:
                raise MissingPlannerReferenceError(
                    f"planner_skill {trial.trial_id} cannot inherit target_region "
                    f"for component={trial.component} group_id={group_id}: "
                    f"no planner_only capture in cache. "
                    f"Run planner_only for this component/group first "
                    f"(--collection-order={collection_order!r}, "
                    f"--require-planner-reference={require_flag})."
                )
            fallback_binding = {
                "reference_mode": "missing_planner_only_capture",
                "target_region_source": None,
                "target_group_id": group_id,
                "note": (
                    "no planner_only capture; legacy_block_major debug fallback. "
                    "Pass --require-planner-reference to enforce strict inheritance."
                ),
            }
            fallback_binding.update(
                build_planner_skill_coverage_metadata(
                    component=trial.component,
                    planner_skill_count=int(
                        getattr(self.args, "planner_skill_count", 0) or 0
                    ),
                    policy="representative_groups",
                )
            )
            return None, fallback_binding
        binding = build_planner_reference_binding(
            cache,
            component=trial.component,
            group_id=group_id,
            reference_mode="inherited_from_planner_only",
            target_region_source="planner_reference_cache",
        )
        binding.update(
            build_planner_skill_coverage_metadata(
                component=trial.component,
                planner_skill_count=int(
                    getattr(self.args, "planner_skill_count", 0) or 0
                ),
                policy="representative_groups",
            )
        )
        print(
            f"[PLANNER REFERENCE] inheriting target_region for "
            f"component={trial.component} group_id={group_id} "
            f"(captured_in={cached.captured_in_trial_id})."
        )
        return cached.target_region, binding

    def _resolve_corrector_failure_start(
        self, trial: TrialSpec
    ) -> tuple[TargetPoint, dict[str, Any]]:
        """Capture or reuse the corrector failure_start pose for ``trial``."""
        from pacer.stage_b_pre.reference_cache import decide_corrector_start_action

        cache = self._corrector_start_cache(trial)
        if cache is None:
            # V0.7 test-stub writer with no config_root: fall back to legacy
            # per-trial capture; cache binding still records the absence.
            point = self._capture_single_pose(trial, "failure_start")
            return point, {
                "mode": "no_cache_available",
                "action": "capture",
                "operator_confirmed_for_trial": True,
                "adjusted_for_trial": False,
            }
        cached = cache.get(trial.component)
        if cached is None:
            print(
                f"[CORRECTOR START CACHE] no cached failure_start for "
                f"component={trial.component}; capturing now."
            )
            point = self._capture_single_pose(trial, "failure_start")
            cached = cache.put(
                component=trial.component,
                pose=point,
                captured_in_trial_id=trial.trial_id,
            )
            source = {
                "mode": "corrector_start_pose_cache",
                "cache_revision": cached.revision,
                "captured_in_trial_id": cached.captured_in_trial_id,
                "operator_confirmed_for_trial": True,
                "adjusted_for_trial": False,
                "action": "capture",
            }
            return point, source

        action = decide_corrector_start_action(cache, trial.component)
        if action == "accept":
            # Fix 1 (2026-05-20): accepting the cached pose only counts as
            # physical reuse when the robot is actually at the cached joints.
            # If not, refuse and force the operator to either move via
            # adjust/recapture or abort. We do NOT silently start a rollout
            # from home while saving the cached pose as failure_start.
            try:
                obs = self.obs_client.get_observations()
                cur_joints = list(obs["joint_positions"])[: len(cached.pose.joint_positions)]
                errors = [
                    abs(float(cur_joints[i]) - float(cached.pose.joint_positions[i]))
                    for i in range(len(cached.pose.joint_positions))
                ]
                max_err = max(errors) if errors else float("inf")
            except Exception as exc:
                raise RuntimeError(
                    f"corrector_only {trial.trial_id}: could not read current "
                    f"joints to verify cached failure_start ({exc!r}); aborting "
                    f"fail-closed."
                ) from exc
            if max_err > CORRECTOR_START_JOINT_TOL_RAD:
                raise RuntimeError(
                    f"corrector_only {trial.trial_id}: operator accepted cached "
                    f"failure_start for {trial.component} but robot is not at "
                    f"cached failure_start pose (max joint error "
                    f"{max_err:.4f} rad > {CORRECTOR_START_JOINT_TOL_RAD:.4f} "
                    f"rad). Move to the cached pose via 'j'/'r' (joystick "
                    f"adjust/recapture) or abort the trial; do not start a "
                    f"corrector rollout from an unverified pose."
                )
            cache.confirm_for_trial(trial.component, trial.trial_id)
            source = {
                "mode": "corrector_start_pose_cache",
                "cache_revision": cached.revision,
                "captured_in_trial_id": cached.captured_in_trial_id,
                "operator_confirmed_for_trial": True,
                "adjusted_for_trial": False,
                "action": "accept",
                "joint_tolerance_rad": float(CORRECTOR_START_JOINT_TOL_RAD),
                "joint_error_max_rad": float(max_err),
            }
            return cached.pose, source
        if action in ("adjust", "recapture"):
            print(
                f"[CORRECTOR START CACHE] operator chose {action}; "
                "joystick-capturing a new failure_start."
            )
            point = self._capture_single_pose(trial, "failure_start")
            cached = cache.put(
                component=trial.component,
                pose=point,
                captured_in_trial_id=trial.trial_id,
                is_adjustment=(action == "adjust"),
            )
            source = {
                "mode": "corrector_start_pose_cache",
                "cache_revision": cached.revision,
                "captured_in_trial_id": cached.captured_in_trial_id,
                "operator_confirmed_for_trial": True,
                "adjusted_for_trial": True,
                "action": action,
            }
            return point, source
        # action == "exclude" or unknown → propagate as RuntimeError so the
        # run() loop records the trial as failed without robot motion.
        raise RuntimeError(
            f"corrector_only {trial.trial_id} excluded by operator at "
            f"corrector_start cache prompt (action={action!r})."
        )

    def _capture_target_region(self, trial: TrialSpec) -> TargetRegion:
        self._assert_teleop_alive("capture_target_region")
        print("TARGET RECORDING: joystick enabled. Press L25 to capture target points; Enter to confirm.")
        points: list[TargetPoint] = []
        self.teleop.set_active(True)
        try:
            while True:
                self._assert_teleop_alive("capture_target_region_loop")
                if _stdin_ready():
                    _ = sys.stdin.readline()
                    if points:
                        break
                    print("Need at least one target point before confirming.")
                event = self.agent._poll_once()  # noqa: SLF001
                if event and event[0] == "start_recording":
                    points.append(self._current_target_point(tag=f"target_{len(points)}"))
                    print(f"Captured target point {len(points)}")
                elif event and event[0] in {"home_and_save", "interrupt"}:
                    raise RuntimeError("target capture aborted by operator")
                time.sleep(0.01)
        finally:
            self.teleop.set_active(False)
        return TargetRegion(config_id=trial.config_id, component=trial.component, points=points)

    def _capture_single_pose(self, trial: TrialSpec, tag: str) -> TargetPoint:
        self._assert_teleop_alive(f"capture_single_pose:{tag}")
        print(f"{tag.upper()} SETUP: joystick enabled. Press L25 to capture pose; Enter confirms after capture.")
        point: TargetPoint | None = None
        self.teleop.set_active(True)
        try:
            while True:
                self._assert_teleop_alive(f"capture_single_pose_loop:{tag}")
                if _stdin_ready():
                    _ = sys.stdin.readline()
                    if point is not None:
                        break
                    print("Press L25 to capture pose before confirming.")
                event = self.agent._poll_once()  # noqa: SLF001
                if event and event[0] == "start_recording":
                    point = self._current_target_point(tag=tag)
                    print(f"Captured {tag} pose")
                elif event and event[0] in {"home_and_save", "interrupt"}:
                    raise RuntimeError(f"{tag} capture aborted by operator")
                time.sleep(0.01)
        finally:
            self.teleop.set_active(False)
        assert point is not None
        return point

    def _record_human_correction(self, trial: TrialSpec) -> list[dict[str, Any]]:
        return self._record_human_joystick_segment(
            trial,
            phase="correction",
            control_source="human_correction",
        )

    def _record_human_joystick_segment(
        self,
        trial: TrialSpec,
        *,
        phase: str,
        control_source: str,
        finish_component: str = CPU_FINISH_COMPONENT,
    ) -> list[dict[str, Any]]:
        """Record a joystick segment bounded by motion-start and CPU-button finish.

        The operator has already confirmed the segment in the terminal (or via
        an always mode). To avoid redundant pre-motion frames and terminal
        finish latency, recording does not append frames until the first
        joystick_motion event. The segment ends when the right-stick CPU skill
        button emits `skill_button:{component: cpu}`.
        """
        self._assert_teleop_alive(f"record_human_joystick_segment:{phase}")
        # V0.1 §5: JoystickAgent emits `joystick_motion` only once after
        # reset_motion(). Without resetting the latch here, any prior joystick
        # motion in setup/target capture leaves the latch set and recording
        # would never start. This is the live-operation regression Codex and
        # Claude both flagged.
        self.agent.reset_motion()
        # V0.4 RTDE patch (2026-05-20): if the preceding model rollout used
        # OpenPI/servoJ, the control script is still latched in servoJ mode.
        # The first speedL from the joystick teleop driver is silently dropped
        # by an in-servoJ script (live-observed under V0.3 — gripper moved on
        # the separate Robotiq socket but TCP did not). Release servo mode
        # via servoStop() (the documented partner of servoJ, NOT speedStop)
        # before re-enabling teleop. For OpenVLA / OFT this is a no-op.
        self._release_servo_mode_for_handoff()
        # V0.7: swap to the correction-only button map (no skills, no zones,
        # R34 -> finish_recording) for the duration of this human segment so
        # an accidental skills-section button press cannot dispatch a CSV
        # skill mid-correction. Restored in the finally block.
        button_map_swapped = False
        if hasattr(self.agent, "reload_button_map"):
            try:
                self.agent.reload_button_map(CORRECTION_BUTTON_MAP_PATH)
                button_map_swapped = True
            except Exception as exc:
                print(f"[WARN] could not swap to correction button map: {exc!r}; "
                      f"using legacy CPU-skill finish trigger as fallback.")
        # Match run_collection.py teleop feel before enabling joystick control.
        # Skills or prior code paths may have changed the Robotiq SPE register.
        try:
            self.robot.set_gripper_speed(TELEOP_GRIPPER_SPEED)
        except Exception:
            pass
        build_frame, _, get_obs = _load_inference_utils()
        print(
            f"{phase.upper()}: joystick enabled. Move the robot to start recording; "
            f"press R34 (finish_recording) to finish "
            f"[legacy fallback: CPU skill button '{finish_component}']."
        )
        frames: list[dict[str, Any]] = []
        recording = False
        self.teleop.set_active(True)
        dt = 1.0 / float(self.args.fps)
        try:
            while True:
                self._assert_teleop_alive(f"record_human_joystick_segment_loop:{phase}")
                t0 = time.time()
                event = self.agent._poll_once()  # noqa: SLF001
                if event:
                    name, payload = event
                    if name == "joystick_motion" and not recording:
                        recording = True
                        print(f"{phase}: recording started on first joystick motion.")
                    elif name == "reorient":
                        # V0.7 blocker-fix #1: L38 reorient must actually snap
                        # the gripper vertical mid-segment. We pause teleop,
                        # clear the speedL window with the same recoverable
                        # stop the script uses for non-emergency cleanup,
                        # call _snap_to_vertical() (synchronous moveL), then
                        # re-enable teleop. The segment does NOT end here —
                        # operator must still press R34 / finish_recording.
                        print(f"{phase}: reorient received; snapping gripper vertical.")
                        try:
                            self.teleop.set_active(False)
                            self._recoverable_speed_stop()
                            self._snap_to_vertical()
                        finally:
                            self.teleop.set_active(True)
                        # Continue the same recording segment from here.
                    elif name == "finish_recording":
                        # V0.7: correction-only button map sets R34 ->
                        # finish_recording. Preferred end-of-segment event.
                        if recording:
                            print(f"{phase}: finish_recording received; ending segment.")
                            break
                        print(f"{phase}: finish_recording ignored until recording starts; move joystick first.")
                    elif name == "skill_button" and isinstance(payload, dict):
                        # Legacy V0.6 finish trigger (CPU skill button). Kept so
                        # that operator sessions using the default map without a
                        # correction-map swap still terminate; deprecated in V0.7.
                        component = payload.get("component")
                        if component == finish_component:
                            if recording:
                                print(f"{phase}: finish button received; ending segment.")
                                break
                            print(f"{phase}: finish button ignored until recording starts; move joystick first.")
                        else:
                            print(
                                f"{phase}: ignoring skill button {component!r}; "
                                f"press {finish_component!r} or R34 (finish_recording) to finish."
                            )
                    elif name in {"interrupt", "home_and_save"}:
                        raise RuntimeError(f"{phase} aborted by operator event {name}")
                if recording:
                    obs = get_obs(
                        self.obs_client,
                        self.cameras,
                        self.args.inference_obs_mode,
                        self.args.inference_obs_max_wait_ms / 1000.0,
                    )
                    frame = build_frame(obs, self.cameras, self.args.image_size)
                    frame["phase"] = phase
                    frame["control_source"] = control_source
                    frames.append(frame)
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        finally:
            self.teleop.set_active(False)
            self._recoverable_speed_stop()
            # V0.7: always restore the default button map so subsequent
            # trials see the normal skills/zones layout, even on exception.
            if button_map_swapped and hasattr(self.agent, "reload_button_map"):
                try:
                    self.agent.reload_button_map(DEFAULT_BUTTON_MAP_PATH)
                except Exception as exc:
                    print(f"[WARN] failed to restore default button map: {exc!r}")
        return frames

    def _current_target_point(self, tag: str) -> TargetPoint:
        tcp = list(self.robot.get_tcp_pose_raw())
        obs = self.obs_client.get_observations()
        joints = list(obs["joint_positions"][:6])
        grip_raw = obs.get("gripper_position", 0.0)
        try:
            import numpy as _np
            grip_arr = _np.asarray(grip_raw, dtype=float).reshape(-1)
            grip_norm = float(grip_arr[0]) if grip_arr.size else 0.0
        except Exception:
            grip_norm = float(grip_raw) if isinstance(grip_raw, (int, float)) else 0.0
        grip = int(round(grip_norm * 255))
        return TargetPoint(tcp_pose=[float(x) for x in tcp[:6]], joint_positions=[float(x) for x in joints], gripper_obs=grip, timestamp=time.time(), tag=tag)

    def _assert_teleop_alive(self, where: str) -> None:
        """Fail fast if the teleop background thread is dead.

        Without this guard the script will happily print "joystick enabled"
        even when _TeleopThread has exited from an exception — leaving the
        operator believing teleop works while no velocity command is sent.
        """
        if not self.teleop_thread.is_alive():
            err = getattr(self.teleop_thread, "last_error", None)
            raise RuntimeError(
                f"teleop background thread is not alive ({where}); last_error={err!r}"
            )

    def _move_home(self) -> dict[str, Any] | None:
        # Stage-B0 V0.1 (2026-05-20) / Fix 1: return the skip event so that
        # callers can persist tolerance evidence into ``timeline_events``.
        # Returns ``None`` when the legacy lift + joint-home actually ran.
        return _move_home_with_event(self)

    def _corrector_success_release_gate(
        self,
        trial: TrialSpec,
        *,
        success_kind: str,
        input_fn: Any = None,
    ) -> dict[str, Any]:
        """V1.0 release gate fired after any corrector_success.

        The operator must explicitly confirm it is safe to open the gripper
        before any scripted post-trial motion.  The release uses
        ``self.home_gripper`` as the target (configured open value at home;
        defaults to 0 when unset).

        Parameters
        ----------
        trial:
            The completed corrector trial spec (for logging/audit).
        success_kind:
            One of ``"auto_success"``, ``"operator_confirmed_success"``,
            ``"model_stop_token_accepted"``.  Recorded into the returned
            decision dict and the upstream timeline event so loaders can
            audit which success path requested the release.
        input_fn:
            Optional callable replacing ``input`` for tests / scripted runs.

        Returns
        -------
        dict
            Decision dict with the V1.0 release schema.  Always includes a
            ``"decision"`` key (``"released" | "declined" |
            "noninteractive_eof_declined" | "release_command_failed"``) and a
            boolean ``"released"`` flag.  Fail-closed semantics: any path
            other than a clean ``y`` / ``yes`` confirmation leaves the
            gripper untouched, and the upstream caller MUST avoid scripted
            post-trial motion with a possibly-held component.
        """
        fn = input_fn if input_fn is not None else input
        start_ts = time.time()
        component = getattr(trial, "component", "?")
        print(
            f"\n[CORRECTOR RELEASE] {success_kind} for component={component}.\n"
            f"  Is it safe to release/open the gripper now? (y/N)"
        )
        try:
            raw = fn("Release? [y/N]: ").strip().lower()
        except EOFError:
            raw = ""
            decision = "noninteractive_eof_declined"
        else:
            decision = "released" if raw in ("y", "yes") else "declined"

        released = False
        gripper_target: int | None = None
        if decision == "released":
            home_gripper_raw = getattr(self, "home_gripper", 0)
            try:
                gripper_target = int(home_gripper_raw) if home_gripper_raw is not None else 0
            except (TypeError, ValueError):
                gripper_target = 0
            try:
                self.robot.set_gripper(gripper_target)
                released = True
            except Exception as exc:
                print(
                    f"[CORRECTOR RELEASE] gripper open failed ({exc!r}); "
                    f"treating release as declined (fail closed)."
                )
                decision = "release_command_failed"
                released = False

        end_ts = time.time()
        if released:
            print(
                f"[CORRECTOR RELEASE] released gripper to "
                f"target={gripper_target} ({success_kind})."
            )
        else:
            print(
                f"[CORRECTOR RELEASE] declined ({decision}); "
                f"no scripted motion will be issued with a held component."
            )
        return {
            "schema_version": CORRECTOR_RELEASE_GATE_SCHEMA,
            "trial_id": trial.trial_id,
            "component": component,
            "success_kind": success_kind,
            "decision": decision,
            "released": released,
            "gripper_target": gripper_target,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "note": "" if released else "operator_declined_release_after_success",
        }

    def _return_to_corrector_failure_start(
        self, failure_point: TargetPoint
    ) -> dict[str, Any] | None:
        """V1.0 corrector_only post-trial: return to the per-component
        temporary failure_start pose — NOT the configured global home.

        Mirrors ``_move_home_with_event``'s already-there short-circuit
        and fail-closed pre-lift, but homes to ``failure_point`` joints.
        Used to keep consecutive corrector_only trials starting from the
        same physical pose so the cached failure_start + confirm-start
        gate semantics hold across the whole block.

        Returns a ``return_to_failure_start_skipped`` event when current
        joints are already within ``CORRECTOR_START_JOINT_TOL_RAD`` of
        ``failure_point.joint_positions`` (no motion). Returns ``None``
        after running the lift + joint-move sequence.
        """
        target_joints = list(failure_point.joint_positions)
        if not target_joints:
            ts = time.time()
            print(
                "[RETURN TO FAILURE_START] no target joints in failure_point; "
                "skipping motion."
            )
            return {
                "event_type": "return_to_failure_start_skipped",
                "control_source": "scripted_return_to_failure_start",
                "reason": "no_target_joints",
                "start_timestamp": ts,
                "end_timestamp": ts,
            }
        try:
            obs = self.obs_client.get_observations()
            cur_joints = list(obs["joint_positions"])[: len(target_joints)]
            errors = [
                abs(float(cur_joints[i]) - float(target_joints[i]))
                for i in range(len(target_joints))
            ]
            max_err = max(errors) if errors else float("inf")
            if max_err <= CORRECTOR_START_JOINT_TOL_RAD:
                # Verify TCP readability too before declaring already-at.
                _ = self.robot.get_tcp_pose_raw()
                ts = time.time()
                print(
                    "[RETURN TO FAILURE_START] already at failure_start within "
                    "tolerance; skipping lift + joint-move."
                )
                return {
                    "event_type": "return_to_failure_start_skipped",
                    "control_source": "scripted_return_to_failure_start",
                    "reason": "already_at_failure_start_within_tolerance",
                    "failure_start_joint_error_max_rad": float(max_err),
                    "joint_tolerance_rad": float(CORRECTOR_START_JOINT_TOL_RAD),
                    "start_timestamp": ts,
                    "end_timestamp": ts,
                }
        except Exception as exc:
            print(
                f"[RETURN TO FAILURE_START] could not probe current state "
                f"({exc!r}); falling back to lift + joint-move."
            )

        print("Returning to corrector failure_start pose (post-trial)...")
        try:
            current_tcp = np.asarray(self.robot.get_tcp_pose_raw(), dtype=float).copy()
            lift_tcp = current_tcp.copy()
            lift_tcp[2] += POST_TRIAL_LIFT_Z_M
            print(
                f"Lifting TCP +Z by {POST_TRIAL_LIFT_Z_M:.2f} m before "
                f"failure_start..."
            )
            self.robot.move_linear(lift_tcp, speed=0.05, accel=0.1, asynchronous=False)
        except Exception as exc:
            raise RuntimeError(
                f"pre-failure_start vertical lift failed ({exc!r}); automatic "
                f"joint-return not attempted — operator must inspect the cell "
                f"before continuing."
            ) from exc
        self.robot.move_joints(target_joints, speed=0.5, accel=0.3)
        target_gripper = int(failure_point.gripper_obs)
        try:
            self.robot.set_gripper(target_gripper)
        except Exception:
            pass
        return None

    def _snap_to_vertical(self) -> None:
        """Match run_planner_skill.py vertical snap before CSV skill execution."""
        current_tcp = np.asarray(self.robot.get_tcp_pose_raw(), dtype=float).copy()
        R_cur = Rotation.from_rotvec(current_tcp[3:6]).as_matrix()
        tool_x = R_cur[:, 0].copy()
        tool_x[2] = 0.0
        norm = float(np.linalg.norm(tool_x))
        if norm < 1e-9:
            # Degenerate yaw projection: keep a deterministic horizontal x-axis
            # rather than risking NaNs in a hardware motion command.
            tool_x = np.array([1.0, 0.0, 0.0])
        else:
            tool_x /= norm
        new_z = np.array([0.0, 0.0, -1.0])
        new_y = np.cross(new_z, tool_x)
        new_y /= np.linalg.norm(new_y)
        R_vert = np.column_stack([tool_x, new_y, new_z])
        current_tcp[3:6] = Rotation.from_matrix(R_vert).as_rotvec()
        print("Reorienting gripper to vertical before CSV skill execution...")
        self.robot.move_linear(current_tcp, speed=0.05, accel=0.1, asynchronous=False)

    def _full_stop(self) -> None:
        for method_name in ("speed_stop", "stop_linear", "stop_joints"):
            method = getattr(self.robot, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    def _release_servo_mode_for_handoff(self) -> None:
        """OpenPI/servoJ mode-matched release before speedL teleop or moveL handoff.

        V0.4 RTDE patch (2026-05-20): OpenPI's apply_action sends actions
        via servoJ (control_period 1/500 s). servoJ does NOT self-clear —
        the script holds its last joint target until either a new servoJ
        arrives or `servoStop()` is called. Without an explicit servoStop
        the next `speedL` (joystick teleop) or `moveL` (`_snap_to_vertical`,
        CSV skill executor) against the still-in-servoJ script is silently
        dropped (live-observed under V0.3 as "joystick gripper works but
        TCP does not move" during recovery_correction).

        `servoStop()` is the documented, idempotent partner of `servoJ`.
        It is NOT the V0.2-toxic `speedStop()` — the two are different
        ur_rtde primitives. `URRobot.servo_stop()` already wraps it and
        `ZMQClientRobot.servo_stop()` already exposes it.

        Gating:
        - `args.model_type == "openpi"` → call `self.robot.servo_stop()`
          and short settle (0.05 s).
        - OpenVLA / OpenVLA-OFT → no-op (those adapters use synchronous
          moveL; there is no servoJ session to release).
        - Missing / unknown `model_type` → conservative no-op (protects
          existing test fixtures with partial argparse.Namespaces).

        servoStop is treated as forgiving: if ur_rtde raises (some
        versions error when called against a non-servoJ context), the
        exception is swallowed because the failure is benign in that
        case and we do not want to abort the calling workflow.

        This helper MUST NOT call speed_stop, stop_linear,
        _recoverable_speed_stop, _full_stop, or _passive_policy_stop_settle.
        It is a single, narrow mode-transition primitive.
        """
        model_type = getattr(self.args, "model_type", None)
        if model_type != "openpi":
            return
        servo_stop = getattr(getattr(self, "robot", None), "servo_stop", None)
        if callable(servo_stop):
            try:
                servo_stop()
            except Exception as exc:
                print(f"[V0.4] servo_stop on handoff raised (treated as benign): {exc!r}")
        time.sleep(0.05)

    def _passive_policy_stop_settle(self) -> None:
        """Passive settle after a non-emergency policy-loop termination.

        V0.3 RTDE patch (2026-05-20): the OpenPI adapter sends actions via
        servoJ (control_period = 1/500 s); OpenVLA / OpenVLA-OFT send
        synchronous moveL. Neither has a speedL window in flight when the
        TracedChunkRunner exits, so calling speedStop() — which is what
        _recoverable_speed_stop() does — has been observed to kill the UR
        RTDE control script (it is the wrong stop primitive for the active
        control mode).

        For manual_stop (`s`), timeout, and the model_stop_token handoff,
        the first step is to simply STOP issuing new commands. The last
        servoJ window (~2 ms) self-expires; the last synchronous moveL has
        already returned. A short passive settle (0.2 s) is enough to avoid
        racing the last in-flight policy command. Callers that need a
        downstream speedL/moveL/moveJ handoff still release OpenPI servoJ
        afterward with `_release_servo_mode_for_handoff()`; that helper uses
        servoStop, not the toxic speedStop/stopL/full-stop primitives.

        This method MUST NOT touch self.robot. The emergency `q`
        (unsafe_abort) path is unaffected — it still uses _full_stop().
        """
        time.sleep(0.2)

    def _recoverable_speed_stop(self) -> None:
        """Clear an active speedL window without invoking the full stop stack.

        V0.3 RTDE patch (2026-05-20) scope note: this helper is NOT used on
        the model-rollout exit path anymore (manual_stop / timeout /
        model_stop_token now use `_passive_policy_stop_settle()` because the
        active control mode there is servoJ, not speedL — see that helper's
        docstring and `_stop_after_model_rollout` for rationale).

        Remaining callers in Stage-B0, all of which have a genuine speedL
        teleop window in flight that needs clearing:

          - `_run_planner_skill_auto_home_setup()` — around the pre-inference
            `_move_home()` (teleop has just been disabled; lingering speedL
            command may still be running its timed window).
          - `_run_planner_skill_setup_gate()` — `home_and_save` event and
            `finally` cleanup (operator was driving the joystick).
          - `_record_human_joystick_segment()` `finally` — end of a
            recovery_correction or clean_full_demo segment (operator was
            driving the joystick).

        Calling the full stop stack at any of these sites includes stopL/stopJ
        on UR, which can leave RTDE reporting "control script is not running"
        and make the next joystick speedL phase appear enabled while no motion
        is accepted. For these recoverable speedL transitions, only clear
        speedL and let the short timed speed command expire.
        """
        method = getattr(getattr(self, "robot", None), "speed_stop", None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
        time.sleep(0.05)

    def _stop_after_model_rollout(self, stop_source: str) -> None:
        """Use stop strength appropriate to the next operator workflow.

        V0.3 RTDE patch (2026-05-20): the model rollout's active control
        mode is servoJ (OpenPI) or synchronous moveL (OpenVLA / OFT) —
        there is no speedL queue to stop. Calling speedStop here was the
        immediate cause of "RTDE control script is not running!" after
        every `s` press, because speedStop against an active servoJ
        session is undefined / toxic in ur_rtde. The right exit for a
        non-emergency policy-loop termination is to STOP issuing new
        commands and passively settle.

        V0.5/V0.6 RTDE patch (2026-05-21): `model_stop_token` and
        `manual_stop` also need an OpenPI servoJ-mode release before any
        downstream scripted motion or human speedL teleop handoff can take
        effect. Without an explicit `servoStop()`, the next moveL/moveJ is
        silently dropped by the still-in-servoJ control script
        (live-observed on 2026-05-21:
        planner_only graphic_card 001 "success" → post-trial home returned
        in ~15 ms instead of the ~4.7 s a real lift+moveJ takes; trial 002
        then started from the previous setpoint instead of home and
        produced an 11-frame stuck-pose rollout). The release uses the
        existing `_release_servo_mode_for_handoff()` helper which is
        OpenPI-gated and idempotent, so the explicit pre-_snap_to_vertical
        call in `_run_planner_skill_no_corrector` becomes a benign
        redundancy (servoStop is documented forgiving) and is left in
        place for code clarity.

        - manual_stop (`s`)    -> passive settle + servoStop (OpenPI) or
          passive settle only (OpenVLA / OFT, gating inside the helper).
          No speedStop/stopL/full-stop.
        - timeout              -> passive settle, NO RTDE stop primitive.
        - model_stop_token     -> passive settle + servoStop (OpenPI) or
          passive settle only (OpenVLA / OFT, gating inside the helper).
          Same primitives used by the planner_skill model_stop_token
          handoff before _snap_to_vertical.
        - unsafe_abort (`q`)   -> full stop (speedStop + stopL). Emergency
          contract preserved; RTDE script death is the documented price.
        """
        if stop_source == "manual_stop":
            self._passive_policy_stop_settle()
            self._release_servo_mode_for_handoff()
        elif stop_source == "timeout":
            self._passive_policy_stop_settle()
        elif stop_source == "model_stop_token":
            self._passive_policy_stop_settle()
            self._release_servo_mode_for_handoff()
        elif stop_source == "unsafe_abort":
            self._full_stop()

    def _base_metadata(self, trial: TrialSpec, stop_source: str, control_segments: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "trial_mode": trial.block,
            "language_instruction": TASK_INSTRUCTIONS[trial.component],
            "camera_mode": self.args.inference_obs_mode,
            "fps": self.args.fps,
            "record_hz": self.args.fps,
            "stop_source": stop_source,
            "control_segments": control_segments,
            "direction_bin": self.args.direction_bin,
            "location_bin": self.args.location_bin,
            "operator": self.args.operator,
        }


def _stdin_ready() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


@contextlib.contextmanager
def _raw_terminal_if_possible():
    """Temporarily make stdin character-buffered for immediate s/Space/q stops.

    During VLA inference the operator has three live keys:
      s     -> manual_stop (any non-emergency reason; qualified by the post-rollout
               feedback label, e.g. manual_stop_bad / success_manual_stop_near_target)
      Space -> operator_space_skill_trigger (manually trigger skill execution)
      q     -> unsafe_abort (safety/collision/drop/emergency)
    Otherwise the rollout waits for the model stop token or timeout.

    If stdin is not a TTY, this is a no-op so tests/subprocess dry-runs remain
    portable.
    """
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _load_inference_utils():
    # Keep dry-run usable on machines without the live inference dependency
    # stack. Hardware paths still reuse the existing implementation.
    from scripts.run_inference import build_frame, create_adapter, get_obs

    return build_frame, create_adapter, get_obs


if __name__ == "__main__":
    main()
