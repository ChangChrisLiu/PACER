"""Trial-plan generation and progress tracking for PACER collection collection."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .config import COMPONENT_SEQUENCE, DEFAULT_TRIAL_COUNTS
from .schemas import BlockName, TrialSpec

BLOCK_ORDER: tuple[BlockName, ...] = ("planner_skill", "planner_only", "corrector_only")

# PACER V0.1 (2026-05-20) collection-order modes.
LEGACY_BLOCK_MAJOR = "legacy_block_major"
COMPONENT_MAJOR_PLANNER_THEN_CORRECTOR = "component_major_planner_then_corrector"
# Fix 1 (2026-05-20): accept ``block_major`` as a short alias for the legacy
# order; canonical token remains ``legacy_block_major`` in saved metadata.
_COLLECTION_ORDER_ALIASES: dict[str, str] = {
    "block_major": LEGACY_BLOCK_MAJOR,
}
COLLECTION_ORDERS: tuple[str, ...] = (
    LEGACY_BLOCK_MAJOR,
    COMPONENT_MAJOR_PLANNER_THEN_CORRECTOR,
)
COLLECTION_ORDER_CHOICES: tuple[str, ...] = tuple(
    list(COLLECTION_ORDERS) + list(_COLLECTION_ORDER_ALIASES)
)
DEFAULT_COLLECTION_ORDER = LEGACY_BLOCK_MAJOR


def canonical_collection_order(value: str) -> str:
    """Resolve ``block_major`` alias to ``legacy_block_major``."""
    return _COLLECTION_ORDER_ALIASES.get(value, value)

# `planner_side` covers planner_only + planner_skill for each component without
# touching corrector_only. Only meaningful under component-major ordering.
PHASE_BLOCK_ALIASES: tuple[str, ...] = ("all", "planner_side")
VALID_PHASE_BLOCKS: tuple[str, ...] = BLOCK_ORDER + PHASE_BLOCK_ALIASES


@dataclass
class SessionMeta:
    config_id: str
    component_sequence: list[str] = field(default_factory=lambda: list(COMPONENT_SEQUENCE))
    active_components: list[str] = field(default_factory=lambda: list(COMPONENT_SEQUENCE))
    trial_counts: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TRIAL_COUNTS))
    fps: int = 10
    robot_port: int = 6000
    obs_port: int = 6002
    planner_server_port: int = 8000
    corrector_server_port: int = 8001
    operator: str = "operator"
    direction_bin: str = "unspecified"
    location_bin: str = "unspecified"
    collection_order: str = DEFAULT_COLLECTION_ORDER
    gripper_cap_config: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_config_id(config_id: str | int) -> str:
    text = str(config_id).strip()
    if text.isdigit():
        return f"config_{int(text):03d}"
    if text.startswith("config_"):
        return text
    return text


def _resolve_blocks_for_phase(phase_block: str) -> tuple[tuple[BlockName, ...], tuple[BlockName, ...]]:
    """Split blocks into planner-side and corrector-side groups."""
    planner_side: tuple[BlockName, ...] = ("planner_only", "planner_skill")
    if phase_block == "all":
        return planner_side, ("corrector_only",)
    if phase_block == "planner_side":
        return planner_side, ()
    if phase_block in BLOCK_ORDER:
        if phase_block == "corrector_only":
            return (), (phase_block,)  # type: ignore[return-value]
        return (phase_block,), ()  # type: ignore[return-value]
    raise ValueError(
        f"Unknown phase_block {phase_block!r}; expected one of {VALID_PHASE_BLOCKS}"
    )


def build_trial_plan(
    config_id: str,
    phase_block: str = "all",
    components: Iterable[str] | None = None,
    counts: dict[str, int] | None = None,
    collection_order: str = DEFAULT_COLLECTION_ORDER,
) -> list[TrialSpec]:
    """Build the deterministic PACER collection trial plan.

    Two collection orders are supported:

    * `legacy_block_major` (default for back-compat with already-saved data):
      iterate blocks in `BLOCK_ORDER` (planner_skill → planner_only →
      corrector_only); within each block, iterate components in
      `COMPONENT_SEQUENCE`. This preserves the pre-V0.1 trial-index sequence
      that `progress.json` files on disk depend on.

    * `component_major_planner_then_corrector` (PACER V0.1, 2026-05-20):
      for each component, collect planner_only then planner_skill; once every
      component has finished planner-side trials, sweep corrector_only across
      components. This matches the package's default operator sequence
      and is the recommended live order going forward.

    `trial_id = config_id__block__component__trial_index` is stable across
    both orders, so resume / progress tracking does not break.
    """
    collection_order = canonical_collection_order(collection_order)
    if collection_order not in COLLECTION_ORDERS:
        raise ValueError(
            f"Unknown collection_order {collection_order!r}; expected one of {COLLECTION_ORDER_CHOICES}"
        )

    config_id = normalize_config_id(config_id)
    component_list = list(components) if components is not None else list(COMPONENT_SEQUENCE)
    trial_counts = dict(DEFAULT_TRIAL_COUNTS)
    if counts:
        trial_counts.update({k: int(v) for k, v in counts.items()})

    for component in component_list:
        if component not in COMPONENT_SEQUENCE:
            raise ValueError(f"Unknown component {component!r}; expected one of {COMPONENT_SEQUENCE}")

    planner_blocks, corrector_blocks = _resolve_blocks_for_phase(phase_block)

    plan: list[TrialSpec] = []
    global_index = 0

    def _append(block: BlockName, component: str, trial_index: int) -> None:
        nonlocal global_index
        component_index = COMPONENT_SEQUENCE.index(component)
        plan.append(
            TrialSpec(
                config_id=config_id,
                block=block,
                component=component,
                component_index=component_index,
                trial_index=trial_index,
                global_index=global_index,
            )
        )
        global_index += 1

    if collection_order == LEGACY_BLOCK_MAJOR:
        # Preserve the pre-V0.1 block sweep order: planner_skill →
        # planner_only → corrector_only. This is what existing
        # session_meta.json / progress.json files expect, so the legacy
        # default does not silently re-shuffle resumed configs.
        legacy_blocks = tuple(b for b in BLOCK_ORDER if b in planner_blocks + corrector_blocks)
        for block in legacy_blocks:
            for component in component_list:
                for trial_index in range(int(trial_counts[block])):
                    _append(block, component, trial_index)
        return plan

    # component_major_planner_then_corrector: planner_only then planner_skill
    # per component, then corrector_only swept across components.
    component_major_planner_order: tuple[BlockName, ...] = ()
    if "planner_only" in planner_blocks:
        component_major_planner_order += ("planner_only",)
    if "planner_skill" in planner_blocks:
        component_major_planner_order += ("planner_skill",)

    for component in component_list:
        for block in component_major_planner_order:
            for trial_index in range(int(trial_counts[block])):
                _append(block, component, trial_index)

    for block in corrector_blocks:
        for component in component_list:
            for trial_index in range(int(trial_counts[block])):
                _append(block, component, trial_index)

    return plan


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


class ProgressTracker:
    def __init__(self, path: Path):
        self.path = path
        self.completed: set[str] = set()
        self.failed: dict[str, str] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self.completed = set(raw.get("completed", []))
            self.failed = dict(raw.get("failed", {}))

    def is_completed(self, trial: TrialSpec) -> bool:
        # Failed trials are still terminal for resume: they were saved with a
        # failure label and the collection plan says to move on.
        return trial.trial_id in self.completed or trial.trial_id in self.failed

    def mark_completed(self, trial: TrialSpec) -> None:
        self.completed.add(trial.trial_id)
        self._save()

    def mark_failed(self, trial: TrialSpec, reason: str) -> None:
        self.failed[trial.trial_id] = reason
        self._save()

    def _save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "completed": sorted(self.completed),
                "failed": self.failed,
                "num_completed": len(self.completed),
                "num_failed": len(self.failed),
            },
        )
