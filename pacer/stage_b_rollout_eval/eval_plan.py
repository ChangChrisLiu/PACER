"""Eval plan, model registry, trial spec, and progress tracker for rollout-eval.

The plan is the cartesian product of:
    models  x  configs  x  components  x  trials_per_component
with `cpu_fan` first in the canonical component order. `eval_trial_id` is the
stable join key across `progress.json` / `summary.csv` / per-trial folders.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from pacer.stage_b_pre.config import COMPONENT_SEQUENCE
from pacer.stage_b_pre.trial_plan import normalize_config_id


DEFAULT_TRIALS_PER_COMPONENT = 10
DEFAULT_CONFIGS = ("config_001", "config_002", "config_003", "config_004", "config_005")
# Canonical component order: cpu_fan first, then the remaining COMPONENT_SEQUENCE
# in its declared order. COMPONENT_SEQUENCE already begins with cpu_fan, so the
# default is just COMPONENT_SEQUENCE as-is.
DEFAULT_COMPONENTS: tuple[str, ...] = tuple(COMPONENT_SEQUENCE)


@dataclass(frozen=True)
class ModelSpec:
    """One row in the model registry (models.json).

    `model_id` is the join key everywhere (progress, summary, per-model folder).
    Path/host/port belong to the operator's T3 launch; the runner does not
    spawn or own those servers.
    """

    model_id: str
    model_type: str
    checkpoint_path: str
    unnorm_key: str = "ur5e_vla_planner_10hz"
    openpi_base: str = "droid"
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalTrialSpec:
    """One trial in the eval plan."""

    run_id: str
    model_id: str
    config_id: str
    component: str
    trial_index: int
    component_index: int
    global_index: int

    @property
    def eval_trial_id(self) -> str:
        return (
            f"{self.run_id}__{self.model_id}__{self.config_id}__"
            f"{self.component}__{self.trial_index:03d}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["eval_trial_id"] = self.eval_trial_id
        return d


def _validate_components(components: Sequence[str]) -> list[str]:
    out: list[str] = []
    for c in components:
        if c not in COMPONENT_SEQUENCE:
            raise ValueError(
                f"Unknown component {c!r}; expected one of {COMPONENT_SEQUENCE}"
            )
        out.append(c)
    if not out:
        raise ValueError("components must be non-empty")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate component in {out!r}")
    return out


def _validate_configs(configs: Sequence[str]) -> list[str]:
    if not configs:
        raise ValueError("configs must be non-empty")
    out = [normalize_config_id(c) for c in configs]
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate config in {out!r}")
    return out


def _validate_models(models: Sequence[ModelSpec]) -> list[ModelSpec]:
    if not models:
        raise ValueError("models must be non-empty")
    ids = [m.model_id for m in models]
    dupes = {mid for mid in ids if ids.count(mid) > 1}
    if dupes:
        raise ValueError(f"duplicate model_id(s) in registry: {sorted(dupes)}")
    return list(models)


def load_model_registry(path: str | Path) -> list[ModelSpec]:
    """Load `models.json` into a list of `ModelSpec`. Validates uniqueness."""
    raw = json.loads(Path(path).read_text())
    rows = raw.get("models") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(
            f"{path}: expected dict with 'models' list or top-level list, got "
            f"{type(rows).__name__}"
        )
    specs: list[ModelSpec] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: model row must be dict, got {type(row).__name__}")
        # Defensive: filter to ModelSpec-known fields only so unrelated keys in
        # the JSON (e.g. operator notes) don't crash the loader.
        known = {f for f in ModelSpec.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in row.items() if k in known}
        for required in ("model_id", "model_type", "checkpoint_path"):
            if required not in kwargs or not kwargs[required]:
                raise ValueError(
                    f"{path}: model row missing required field {required!r}: {row}"
                )
        specs.append(ModelSpec(**kwargs))
    return _validate_models(specs)


def build_eval_plan(
    *,
    run_id: str,
    models: Sequence[ModelSpec],
    configs: Sequence[str] = DEFAULT_CONFIGS,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    trials_per_component: int = DEFAULT_TRIALS_PER_COMPONENT,
) -> list[EvalTrialSpec]:
    """Build the deterministic rollout-eval plan.

    Order: model -> config -> component -> trial_index.

    The model loop is outermost because the operator must swap T3 between
    models; finishing one model entirely before moving to the next minimizes
    server restarts and lets the operator resume mid-model after a crash.
    """
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if trials_per_component <= 0:
        raise ValueError(f"trials_per_component must be > 0, got {trials_per_component}")

    models = _validate_models(list(models))
    configs = _validate_configs(list(configs))
    components = _validate_components(list(components))

    plan: list[EvalTrialSpec] = []
    global_index = 0
    for model in models:
        for config_id in configs:
            for component in components:
                component_index = COMPONENT_SEQUENCE.index(component)
                for trial_index in range(int(trials_per_component)):
                    plan.append(
                        EvalTrialSpec(
                            run_id=run_id,
                            model_id=model.model_id,
                            config_id=config_id,
                            component=component,
                            trial_index=trial_index,
                            component_index=component_index,
                            global_index=global_index,
                        )
                    )
                    global_index += 1

    # Defensive: all eval_trial_ids must be unique (no two equal compositions
    # of run_id/model_id/config_id/component/trial_index).
    seen: set[str] = set()
    for trial in plan:
        if trial.eval_trial_id in seen:
            raise RuntimeError(
                f"duplicate eval_trial_id generated: {trial.eval_trial_id}"
            )
        seen.add(trial.eval_trial_id)
    return plan


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


class EvalProgressTracker:
    """Persistent progress tracker for rollout-eval.

    A trial is `is_completed` iff its `eval_trial_id` is in `completed` OR
    `failed`. Failed trials are not retried automatically (mirrors Stage-B-pre
    semantics: failures are recorded and the operator moves on, so the same
    trial does not run twice). `--restart` deletes the progress file
    explicitly and is destructive.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.completed: set[str] = set()
        self.failed: dict[str, str] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.completed = set(raw.get("completed", []))
            self.failed = dict(raw.get("failed", {}))

    def is_completed(self, trial: EvalTrialSpec) -> bool:
        return trial.eval_trial_id in self.completed or trial.eval_trial_id in self.failed

    def mark_completed(self, trial: EvalTrialSpec) -> None:
        self.completed.add(trial.eval_trial_id)
        self._save()

    def mark_failed(self, trial: EvalTrialSpec, reason: str) -> None:
        self.failed[trial.eval_trial_id] = reason
        self._save()

    def remaining(self, plan: Iterable[EvalTrialSpec]) -> list[EvalTrialSpec]:
        return [t for t in plan if not self.is_completed(t)]

    def _save(self) -> None:
        _atomic_write_json(
            self.path,
            {
                "completed": sorted(self.completed),
                "failed": self.failed,
                "num_completed": len(self.completed),
                "num_failed": len(self.failed),
            },
        )


def plan_summary(plan: Sequence[EvalTrialSpec]) -> dict[str, Any]:
    """Compact summary used by `print_plan` and dry-run output."""
    by_model: dict[str, int] = {}
    by_config: dict[str, int] = {}
    by_component: dict[str, int] = {}
    for trial in plan:
        by_model[trial.model_id] = by_model.get(trial.model_id, 0) + 1
        by_config[trial.config_id] = by_config.get(trial.config_id, 0) + 1
        by_component[trial.component] = by_component.get(trial.component, 0) + 1
    return {
        "total_trials": len(plan),
        "by_model": by_model,
        "by_config": by_config,
        "by_component": by_component,
    }
