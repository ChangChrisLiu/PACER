"""Configuration-disjoint case-group split (paper app:split).

For each component, the collection configurations are split into training
configurations and validation configuration(s) under a fixed seed that selects
the validation configuration(s). Every trace recorded under a configuration —
autonomous rollouts, corrections, supplementary clean demonstrations — is
assigned as a whole to the corresponding side, so no trace contributes rows to
both training and validation. Held-out hardware scene configurations never
enter this split; they are physical evaluation setups, not logged rows.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

PACER_FRAMEWORK_SPLIT_SCHEMA = "pacer_framework_split.v1"


def assign_case_group_split(
    rows: list[dict[str, Any]],
    *,
    seed: int = 20260610,
    val_configs_per_component: int = 1,
    config_field: str = "collection_config",
    frozen_configs: Mapping[str, Iterable[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign row["split"] in {train, val, heldout} configuration-disjointly.

    - seed: fixed protocol seed; the validation configuration of each component
      is a deterministic function of (seed, component), independent of row order.
    - frozen_configs: optional per-component configurations whose rows are
      marked "heldout" (frozen rows retained for audit, excluded from training
      loss and validation scoring — paper app:split).

    Raises ValueError when a row lacks the configuration field or a component
    has too few configurations to leave at least one training configuration.
    """
    configs_by_component: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        config = row.get(config_field)
        if config in (None, ""):
            raise ValueError(
                f"row {row.get('row_id')!r} missing {config_field!r}; "
                "case-group splitting requires a collection configuration per trace"
            )
        configs_by_component[str(row.get("component", "unknown"))].add(str(config))

    frozen_by_component = {
        str(component): {str(c) for c in configs}
        for component, configs in (frozen_configs or {}).items()
    }

    plan: dict[str, dict[str, list[str]]] = {}
    for component in sorted(configs_by_component):
        configs = sorted(configs_by_component[component])
        frozen = frozen_by_component.get(component, set())
        unknown_frozen = frozen - set(configs)
        if unknown_frozen:
            raise ValueError(f"frozen configs {sorted(unknown_frozen)} not present for component {component!r}")
        selectable = [c for c in configs if c not in frozen]
        if len(selectable) <= val_configs_per_component:
            raise ValueError(
                f"component {component!r} has {len(selectable)} selectable configs; "
                f"need more than val_configs_per_component={val_configs_per_component}"
            )
        rng = random.Random(f"{seed}|{component}")
        val = sorted(rng.sample(selectable, val_configs_per_component))
        train = [c for c in selectable if c not in val]
        plan[component] = {
            "configs": configs,
            "train_configs": train,
            "val_configs": val,
            "heldout_configs": sorted(frozen),
        }

    out_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    component_split_counts: Counter[str] = Counter()
    for row in rows:
        component = str(row.get("component", "unknown"))
        config = str(row.get(config_field))
        entry = plan[component]
        if config in entry["heldout_configs"]:
            split = "heldout"
        elif config in entry["val_configs"]:
            split = "val"
        else:
            split = "train"
        new_row = json.loads(json.dumps(row))
        new_row["split"] = split
        out_rows.append(new_row)
        split_counts[split] += 1
        component_split_counts[f"{component}|{split}"] += 1

    manifest = {
        "schema": PACER_FRAMEWORK_SPLIT_SCHEMA,
        "seed": seed,
        "val_configs_per_component": val_configs_per_component,
        "config_field": config_field,
        "plan": plan,
        "split_counts": dict(split_counts),
        "component_split_counts": dict(component_split_counts),
    }
    return out_rows, manifest
