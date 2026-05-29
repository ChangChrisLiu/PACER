"""Configuration constants for TRACE-VLA collection rollout collection."""
from __future__ import annotations

from pathlib import Path
from typing import Final

COMPONENT_SEQUENCE: Final[list[str]] = [
    "cpu_fan",
    "ram",
    "connector",
    "graphic_card",
    "cpu",
]

DEFAULT_TRIAL_COUNTS: Final[dict[str, int]] = {
    "planner_skill": 2,
    "planner_only": 10,
    "corrector_only": 5,
}

VALID_BLOCKS: Final[set[str]] = set(DEFAULT_TRIAL_COUNTS) | {"all"}

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("data/tracevla_correction_rollouts")
DEFAULT_FPS: Final[int] = 10
DEFAULT_MAX_STEPS: Final[int] = 600
DEFAULT_IMAGE_SIZE: Final[int] = 256

TASK_INSTRUCTIONS: Final[dict[str, str]] = {
    "cpu": (
        "Extract the CPU by unlocking the bracket first, then pick up "
        "the CPU and place it inside the blue area as a high value "
        "component."
    ),
    "ram": (
        "Extract the closest visible RAM stick and place it inside the "
        "blue area as a high value component."
    ),
    "cpu_fan": (
        "Extract the CPU fan and place it inside the yellow area as a "
        "low value component."
    ),
    "connector": (
        "Extract the closest visible connector and place it inside the "
        "yellow area as a low value component."
    ),
    "graphic_card": (
        "Extract the graphics card and place it inside the green area, "
        "as it needs subsequent disassembly."
    ),
}

PLANNER_FEEDBACK_LABELS: Final[list[str]] = [
    # V0.7 semantic labels (preferred for new trials).
    "success",
    "stop_token_should_emit_here",
    "near_but_not_accurate",
    "wrong_orientation_or_wrong_location",
    "totally_off",
    "totally_off_wrong_region_or_target",
    # Legacy V0.6 labels (kept for back-compat with already-saved trials).
    "success_stop_token",
    "success_manual_stop_near_target",
    "near_miss_stop_token",
    "near_miss_manual_stop",
    "near_miss_timeout",
    "wrong_target",
    "bad_orientation",
    "unsafe_abort",
    "no_stop_timeout",
    "manual_stop_bad",
    "operator_uncertain_exclude",
]

PLANNER_SKILL_FEEDBACK_LABELS: Final[list[str]] = [
    # V0.7 semantic labels (preferred for new trials).
    "success_skill_completed",
    "stop_token_should_emit_here",
    "near_but_not_accurate",
    "wrong_orientation_or_wrong_location",
    "totally_off_wrong_region_or_target",
    "skill_verification_failed_planner_slightly_inaccurate",
    # Legacy V0.6 labels.
    "success_manual_stop_near_target",
    "planner_bad_position",
    "planner_near_miss_position",
    "planner_bad_orientation",
    "near_miss_timeout",
    "no_stop_timeout",
    "wrong_target",
    "skill_grasp_failed",
    "skill_execution_failed",
    "manual_stop_bad",
    "unsafe_abort",
    "operator_uncertain_exclude",
]

# V0.7 mandatory-correction exemption sets. A failure label NOT in these
# sets triggers BOTH recovery_correction and clean_full_demo without any
# operator y/N prompt. The exemption sets cover operator successes, the
# stop-token positive supervision label, the universal escape hatch, and
# unsafe_abort (recovery-ineligible by safety policy).
PLANNER_ONLY_CORRECTION_EXEMPT_LABELS: Final[frozenset[str]] = frozenset({
    "success",
    "stop_token_should_emit_here",
    "operator_uncertain_exclude",
    "unsafe_abort",
    "success_stop_token",
    "success_manual_stop_near_target",
})
PLANNER_SKILL_CORRECTION_EXEMPT_LABELS: Final[frozenset[str]] = frozenset({
    "success_skill_completed",
    "stop_token_should_emit_here",
    "operator_uncertain_exclude",
    "unsafe_abort",
    "success_manual_stop_near_target",
})

# Labels eligible to trigger optional human-correction recording after a
# planner_skill failure. unsafe_abort is intentionally included but requires
# explicit operator confirmation at the use-site before joystick is enabled.
# operator_uncertain_exclude is intentionally NOT here: an uncertain trial is
# not a clean failure pose to record a human-correction segment from.
PLANNER_SKILL_FAILURE_LABELS: Final[list[str]] = [
    "planner_bad_position",
    "planner_near_miss_position",
    "planner_bad_orientation",
    "near_miss_timeout",
    "no_stop_timeout",
    "wrong_target",
    "skill_grasp_failed",
    "skill_execution_failed",
    "manual_stop_bad",
    "unsafe_abort",
]

# Split into two parts so corrector_only V0.9 (programmatic labels, no
# operator menu) can print the live-key controls without misleading the
# operator with a "choose the feedback label" suffix. Planner_only and
# planner_skill still need the suffix because they DO have a post-rollout
# operator menu. MODEL_SEGMENT_OPERATOR_CONTROLS is preserved as the
# concatenation so any external importer keeps the existing string.
MODEL_SEGMENT_LIVE_CONTROLS: Final[str] = (
    "During model rollout:\n"
    "  press 's' = NON-EMERGENCY stop (manual_stop). Stops issuing new "
    "model commands and passively settles for ~0.2 s — no RTDE stop "
    "primitive is called, because the rollout's active control mode is "
    "servoJ (OpenPI) or synchronous moveL (OpenVLA), neither of which "
    "needs a speedStop. This avoids the speedStop-vs-servoJ mismatch that "
    "killed the RTDE control script in earlier versions, so "
    "recovery_correction / clean_full_demo and the next trial are intended "
    "to work without restarting launch_robot.py. Use 's' for any "
    "non-safety reason (wrong target, bad orientation, near-miss, model "
    "went too far).\n"
    "  press 'q' = EMERGENCY abort (unsafe_abort). Calls the full stop stack "
    "(speedStop + stopL). This is safety-correct but MAY KILL THE RTDE "
    "CONTROL SCRIPT — you may need to restart launch_robot.py before the next "
    "trial. Use 'q' ONLY for genuine danger: imminent collision, dropped part, "
    "runaway motion, operator-safety issue.\n"
    "  (no key) = wait for the model stop token or the timeout."
)
MODEL_SEGMENT_FEEDBACK_LABEL_SUFFIX: Final[str] = (
    "After the rollout stops, choose the feedback label that explains "
    "why/how it stopped."
)
MODEL_SEGMENT_OPERATOR_CONTROLS: Final[str] = (
    MODEL_SEGMENT_LIVE_CONTROLS + "\n" + MODEL_SEGMENT_FEEDBACK_LABEL_SUFFIX
)

CORRECTOR_FEEDBACK_LABELS: Final[list[str]] = [
    "corrector_success",
    "near_miss",
    "wrong_recovery_pose",
    "human_corrected_then_save",
    "retry_corrector",
    "manual_drop_or_abort",
    "unsafe_abort",
    "operator_uncertain_exclude",
]

# Labels that require a non-empty operator note before the trial may be
# treated as clean accepted feedback. Empty-note trials must be rejected at
# collection prompt time and re-validated by the writer.
LABELS_REQUIRING_NOTE: Final[frozenset[str]] = frozenset({
    "operator_uncertain_exclude",
})

# P0 (pipeline/control diagnostic) labels are not ordinary operator labels.
# Detection lives in tracevla_auto_score / future Phase-A2 sidecar diagnostics,
# not in the operator menu. Listed here so tests can assert that none of these
# tokens accidentally leaks into any operator-facing label set.
P0_DIAGNOSTIC_LABELS: Final[frozenset[str]] = frozenset({
    "pipeline_suspect_no_motion_with_commands",
    "stale_observation_suspect",
    "robot_command_error",
    "pipeline_suspect_override",
    "missing_motion_sanity_score",
    "missing_required_sidecar",
})
