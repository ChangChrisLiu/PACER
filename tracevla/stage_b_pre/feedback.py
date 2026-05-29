"""Terminal feedback prompts for Stage-B-pre collection."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from .config import LABELS_REQUIRING_NOTE
from .feedback_compatibility import valid_labels_for
from .schemas import Feedback

InputFn = Callable[[str], str]


def prompt_label(
    labels: Sequence[str],
    title: str,
    input_fn: InputFn = input,
    *,
    block: str | None = None,
    stop_source: str | None = None,
) -> Feedback:
    """Prompt the operator for a feedback label and optional note.

    When `block` and `stop_source` are both provided, the menu is filtered
    against `feedback_compatibility.valid_labels_for(block, stop_source)`
    so the operator cannot pick an impossible pair. Labels in
    `LABELS_REQUIRING_NOTE` (currently `operator_uncertain_exclude`)
    re-prompt until a non-empty note is supplied.
    """
    if not labels:
        raise ValueError("labels must be non-empty")
    if block is not None and stop_source is not None:
        allowed = valid_labels_for(block, stop_source)
        filtered = [lbl for lbl in labels if lbl in allowed]
        if not filtered:
            raise ValueError(
                f"no labels valid for block={block!r}, stop_source={stop_source!r}; "
                f"caller should not invoke prompt_label with this triple"
            )
        labels = filtered
    print(f"\n{title}")
    for idx, label in enumerate(labels, start=1):
        print(f"  [{idx}] {label}")
    while True:
        raw = input_fn("Choice: ").strip().lower()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(labels):
                label = labels[idx - 1]
                break
        if raw in labels:
            label = raw
            break
        print("Invalid choice. Enter number or exact label.")
    note_prompt = "Optional note (Enter to skip): "
    if label in LABELS_REQUIRING_NOTE:
        note_prompt = (
            f"Note REQUIRED for label {label!r} (cannot be empty): "
        )
    note = input_fn(note_prompt).strip()
    while label in LABELS_REQUIRING_NOTE and not note:
        print(
            f"Label {label!r} requires a non-empty note explaining the "
            f"uncertainty; trial would otherwise be excluded from clean data."
        )
        note = input_fn(note_prompt).strip()
    return Feedback(label=label, note=note)


def wait_for_enter(prompt: str = "Press Enter to continue...", input_fn: InputFn = input) -> None:
    input_fn(prompt)
