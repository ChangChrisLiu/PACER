"""Operator score menu + EvalFeedback for rollout-only model comparison.

The menu is intentionally compact (5 items, V0.7-style semantics). Strict
success is derived as `label == "success" AND stop_source != "unsafe_abort"`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any


SUCCESS_LABEL = "success"
NEAR_LABEL = "near_but_not_accurate"
WRONG_OR_LOC_LABEL = "wrong_orientation_or_wrong_location"
TOTALLY_OFF_LABEL = "totally_off_wrong_region_or_target"
UNCERTAIN_LABEL = "operator_uncertain_exclude"
UNSAFE_ABORT_AUTO_LABEL = "unsafe_abort"

ROLLOUT_EVAL_LABELS: tuple[str, ...] = (
    SUCCESS_LABEL,
    NEAR_LABEL,
    WRONG_OR_LOC_LABEL,
    TOTALLY_OFF_LABEL,
    UNCERTAIN_LABEL,
)

ROLLOUT_EVAL_STOP_SOURCES: tuple[str, ...] = (
    "model_stop_token",
    "manual_stop",
    "timeout",
    "unsafe_abort",
)

LABELS_REQUIRING_NOTE: frozenset[str] = frozenset({UNCERTAIN_LABEL})

InputFn = Callable[[str], str]


@dataclass(frozen=True)
class EvalFeedback:
    """Operator's per-trial review.

    `auto_assigned` is True only on the `unsafe_abort` shortcut: the menu is
    skipped, the label is forced, and the operator can still leave an optional
    note explaining the cause.
    """

    label: str
    note: str = ""
    operator: str = "operator"
    auto_assigned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_strict_success(*, label: str, stop_source: str) -> bool:
    """Headline metric: `success` label AND not unsafe_abort."""
    return label == SUCCESS_LABEL and stop_source != "unsafe_abort"


def is_excluded(*, label: str, stop_source: str) -> bool:
    """`operator_uncertain_exclude` OR unsafe_abort drops the trial from the
    strict-success denominator."""
    return label == UNCERTAIN_LABEL or stop_source == "unsafe_abort"


def prompt_eval_label(
    *,
    stop_source: str,
    input_fn: InputFn = input,
    operator: str = "operator",
    print_fn: Callable[[str], None] = print,
) -> EvalFeedback:
    """Prompt the operator for a rollout-only review.

    Routing:
      - `unsafe_abort`: skip the menu; auto-assign the `unsafe_abort` label.
        Operator may still enter an optional note (default empty).
      - other stop sources (`manual_stop`, `model_stop_token`, `timeout`):
        present the 5-item menu; require a non-empty note for
        `operator_uncertain_exclude`.

    Caller is responsible for printing pre-prompt context (component, model,
    final pose) before invoking this function; the prompt itself is compact.
    """
    if stop_source == "unsafe_abort":
        print_fn(
            "[UNSAFE ABORT] auto-labeled as 'unsafe_abort'; menu skipped. "
            "Operator must restart the robot server (T1) and confirm before "
            "the next trial."
        )
        note = input_fn(
            "Optional note describing the cause (Enter to skip): "
        ).strip()
        return EvalFeedback(
            label=UNSAFE_ABORT_AUTO_LABEL,
            note=note,
            operator=operator,
            auto_assigned=True,
        )

    print_fn(f"\nRollout review (stop_source={stop_source}). Pick a label:")
    for idx, label in enumerate(ROLLOUT_EVAL_LABELS, start=1):
        print_fn(f"  [{idx}] {label}")

    while True:
        raw = input_fn("Choice: ").strip().lower()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(ROLLOUT_EVAL_LABELS):
                label = ROLLOUT_EVAL_LABELS[idx - 1]
                break
        if raw in ROLLOUT_EVAL_LABELS:
            label = raw
            break
        print_fn("Invalid choice. Enter number or exact label.")

    if label in LABELS_REQUIRING_NOTE:
        note_prompt = f"Note REQUIRED for label {label!r} (cannot be empty): "
    else:
        note_prompt = "Optional note (Enter to skip): "
    note = input_fn(note_prompt).strip()
    while label in LABELS_REQUIRING_NOTE and not note:
        print_fn(
            f"Label {label!r} requires a non-empty note explaining why the "
            f"trial is excluded from clean success counts."
        )
        note = input_fn(note_prompt).strip()

    return EvalFeedback(label=label, note=note, operator=operator, auto_assigned=False)


def wait_for_operator_ready(
    *,
    prompt: str,
    input_fn: InputFn = input,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Operator-attended readiness gate.

    Used after `unsafe_abort` (RTDE script may be dead) and between models
    (T3 swap). Requires a literal `ok`/`yes`/`y`/`ready` response so a stray
    Enter cannot accidentally hand the robot off into the next trial.
    """
    print_fn(prompt)
    while True:
        raw = input_fn(
            "Type 'ok' (or 'ready'/'yes'/'y') to continue, or Ctrl+C to abort: "
        ).strip().lower()
        if raw in {"ok", "ready", "yes", "y"}:
            return
        print_fn("Operator readiness not confirmed; re-prompting.")
