"""Stop-source x feedback-label compatibility table for PACER collection.

This module is the single source of truth for which `(stop_source, label)`
pairs may be saved as clean accepted feedback. Invalid pairs encode operator
mistakes (e.g. labeling a model-stop-token trial as `manual_stop_bad`) or
physical impossibilities (e.g. `skill_grasp_failed` after `manual_stop`, where
the skill never runs). The collection prompt soft-filters the menu against
this table, and the writer hard-rejects incompatible pairs.

Binding requirement: impossible `(stop_source, label)` pairs cannot be saved
as clean accepted feedback.
"""
from __future__ import annotations

from typing import Final

from .config import (
    CORRECTOR_FEEDBACK_LABELS,
    PLANNER_FEEDBACK_LABELS,
    PLANNER_SKILL_FEEDBACK_LABELS,
)

# Block names mirror src.pacer_pre.schemas.BlockName
BlockName = str
StopSource = str

# Always-allowed labels per block: operator escape hatches that can apply to
# any stop source. `operator_uncertain_exclude` is the canonical example and
# always carries a non-empty note (enforced in feedback.py / writer.py).
_ALWAYS_ALLOWED_BY_BLOCK: Final[dict[BlockName, frozenset[str]]] = {
    "planner_only": frozenset({"operator_uncertain_exclude"}),
    "planner_skill": frozenset({"operator_uncertain_exclude"}),
    "corrector_only": frozenset({"operator_uncertain_exclude"}),
}


def _planner_only_table() -> dict[StopSource, frozenset[str]]:
    return {
        "model_stop_token": frozenset({
            # V0.7
            "success",
            "near_but_not_accurate",
            "wrong_orientation_or_wrong_location",
            "totally_off",
            # Legacy V0.6
            "success_stop_token",
            "near_miss_stop_token",
            "wrong_target",
            "bad_orientation",
        }),
        "manual_stop": frozenset({
            # V0.7
            "stop_token_should_emit_here",
            "near_but_not_accurate",
            "wrong_orientation_or_wrong_location",
            "totally_off_wrong_region_or_target",
            # Legacy V0.6
            "success_manual_stop_near_target",
            "near_miss_manual_stop",
            "wrong_target",
            "bad_orientation",
            "manual_stop_bad",
        }),
        "timeout": frozenset({
            # V0.7
            "near_but_not_accurate",
            "wrong_orientation_or_wrong_location",
            "totally_off_wrong_region_or_target",
            # Legacy V0.6
            "near_miss_timeout",
            "no_stop_timeout",
            "wrong_target",
            "bad_orientation",
        }),
        "unsafe_abort": frozenset({"unsafe_abort"}),
        # `not_run` exists for diagnostic save-without-rollout flows. We do
        # not allow any policy-quality label here; only the uncertainty
        # escape hatch (added via _ALWAYS_ALLOWED_BY_BLOCK).
        "not_run": frozenset(),
    }


def _planner_skill_table() -> dict[StopSource, frozenset[str]]:
    # The fixed skill only runs after `model_stop_token`. So skill-failure
    # labels are reachable only on `model_stop_token`.
    return {
        "model_stop_token": frozenset({
            # V0.7
            "success_skill_completed",
            "skill_verification_failed_planner_slightly_inaccurate",
            "near_but_not_accurate",
            "wrong_orientation_or_wrong_location",
            "totally_off_wrong_region_or_target",
            # Legacy V0.6
            "planner_bad_position",
            "planner_near_miss_position",
            "planner_bad_orientation",
            "wrong_target",
            "skill_grasp_failed",
            "skill_execution_failed",
        }),
        "manual_stop": frozenset({
            # V0.7
            "stop_token_should_emit_here",
            "near_but_not_accurate",
            "wrong_orientation_or_wrong_location",
            "totally_off_wrong_region_or_target",
            # Legacy V0.6
            "success_manual_stop_near_target",
            "planner_bad_position",
            "planner_near_miss_position",
            "planner_bad_orientation",
            "wrong_target",
            "manual_stop_bad",
        }),
        "timeout": frozenset({
            # V0.7
            "near_but_not_accurate",
            "wrong_orientation_or_wrong_location",
            "totally_off_wrong_region_or_target",
            # Legacy V0.6
            "near_miss_timeout",
            "no_stop_timeout",
            "planner_bad_position",
            "planner_near_miss_position",
            "planner_bad_orientation",
            "wrong_target",
        }),
        "unsafe_abort": frozenset({"unsafe_abort"}),
        "not_run": frozenset(),
    }


def _corrector_table() -> dict[StopSource, frozenset[str]]:
    # V0.9 corrector_only auto-retry: corrector_success and
    # human_corrected_then_save are programmatic labels written by the
    # 3-attempt loop, not operator-picked. Both are valid for any
    # non-unsafe terminal stop_source the last attempt may have produced.
    # The legacy quick-menu labels (near_miss, wrong_recovery_pose,
    # retry_corrector, manual_drop_or_abort) are kept for back-compat with
    # previously-saved V0.6/V0.7 corrector trials.
    return {
        "model_stop_token": frozenset({
            "corrector_success",
            "human_corrected_then_save",
            "near_miss",
            "wrong_recovery_pose",
            "retry_corrector",
        }),
        "manual_stop": frozenset({
            "corrector_success",
            "human_corrected_then_save",
            "near_miss",
            "wrong_recovery_pose",
            "retry_corrector",
            "manual_drop_or_abort",
        }),
        "timeout": frozenset({
            "corrector_success",
            "human_corrected_then_save",
            "near_miss",
            "wrong_recovery_pose",
            "retry_corrector",
            "manual_drop_or_abort",
        }),
        "unsafe_abort": frozenset({"unsafe_abort"}),
        "not_run": frozenset(),
    }


# Top-level compatibility table: block -> stop_source -> set of valid labels.
# Read it as: "if collection block is X and the recorded stop_source is Y,
# then the operator may choose any label in the set; everything else is
# rejected as an impossible pair."
STOP_SOURCE_TO_VALID_LABELS: Final[dict[BlockName, dict[StopSource, frozenset[str]]]] = {
    "planner_only": _planner_only_table(),
    "planner_skill": _planner_skill_table(),
    "corrector_only": _corrector_table(),
}

_BLOCK_TO_LABEL_SET: Final[dict[BlockName, frozenset[str]]] = {
    "planner_only": frozenset(PLANNER_FEEDBACK_LABELS),
    "planner_skill": frozenset(PLANNER_SKILL_FEEDBACK_LABELS),
    "corrector_only": frozenset(CORRECTOR_FEEDBACK_LABELS),
}


class IncompatibleFeedbackError(ValueError):
    """Raised when a `(block, stop_source, label)` triple is not allowed."""


def valid_labels_for(block: BlockName, stop_source: StopSource) -> frozenset[str]:
    """Return the set of labels that are valid for `(block, stop_source)`.

    Unknown blocks or stop_sources return an empty set; the caller should
    treat that as "no labels allowed except the escape hatch".
    """
    block_table = STOP_SOURCE_TO_VALID_LABELS.get(block)
    if block_table is None:
        return frozenset()
    base = block_table.get(stop_source, frozenset())
    extras = _ALWAYS_ALLOWED_BY_BLOCK.get(block, frozenset())
    return base | extras


def is_compatible(block: BlockName, stop_source: StopSource, label: str) -> bool:
    """Return True if the triple may be saved as clean accepted feedback."""
    return label in valid_labels_for(block, stop_source)


def assert_compatible(
    block: BlockName,
    stop_source: StopSource,
    label: str,
) -> None:
    """Hard-reject incompatible triples.

    Raises IncompatibleFeedbackError if the triple is not in the table. The
    error message lists the legal labels so the caller / operator can recover.
    """
    if block not in STOP_SOURCE_TO_VALID_LABELS:
        raise IncompatibleFeedbackError(
            f"unknown block {block!r}; expected one of "
            f"{sorted(STOP_SOURCE_TO_VALID_LABELS)}"
        )
    allowed = _BLOCK_TO_LABEL_SET.get(block, frozenset())
    if label not in allowed:
        raise IncompatibleFeedbackError(
            f"label {label!r} is not in the {block} label set; "
            f"valid labels: {sorted(allowed)}"
        )
    if is_compatible(block, stop_source, label):
        return
    legal = sorted(valid_labels_for(block, stop_source))
    raise IncompatibleFeedbackError(
        f"incompatible (block={block!r}, stop_source={stop_source!r}, "
        f"label={label!r}); valid labels for this stop_source: {legal}"
    )
