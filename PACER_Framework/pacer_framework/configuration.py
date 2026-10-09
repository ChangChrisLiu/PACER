"""Portable configuration for the shared offline validation protocol."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


VALIDATION_SUBMETRICS = (
    "progress", "proximity", "terminal", "direction", "stop", "align", "no_regression",
)
GEOMETRIC_SUBMETRICS = ("progress", "proximity", "terminal", "direction")
SUBMETRIC_ALIASES = {"reg": "no_regression"}
DEFAULT_SUBMETRIC_WEIGHTS: dict[str, float] = {
    "progress": 0.25,
    "proximity": 0.25,
    "terminal": 0.20,
    "direction": 0.15,
    "stop": 0.15,
    "align": 0.15,
    "no_regression": 0.10,
}
CONFIG_SCHEMA = "pacer.scoring_config.v1"
DEFAULT_REFERENCE_ALIGNMENT_MODE = "trajectory"


def validate_submetric_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Copy a nonnegative, finite coefficient map without rescaling its values."""
    if not isinstance(weights, Mapping):
        raise ValueError("submetric_weights must be an object mapping metric names to numbers")
    out: dict[str, float] = {}
    for raw_name, value in weights.items():
        name = SUBMETRIC_ALIASES.get(str(raw_name), str(raw_name))
        if name not in VALIDATION_SUBMETRICS:
            raise ValueError(f"unknown validation submetric {raw_name!r}")
        if name in out:
            raise ValueError(f"duplicate validation submetric or alias {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"coefficient for {name!r} must be a finite nonnegative number")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"coefficient for {name!r} must be a finite nonnegative number")
        out[name] = float(value)
    if not out or not any(value > 0 for value in out.values()):
        raise ValueError("submetric_weights must contain at least one positive coefficient")
    if not math.isfinite(sum(out.values())):
        raise ValueError("the sum of submetric coefficients must be finite")
    return out


@dataclass(frozen=True)
class ScoringConfig:
    """One configuration shared by construction, scoring and candidate audits.

    Defaults are generic starting values, not task-specific optimal settings.
    A supplied coefficient map replaces the default map; absent row metrics are
    omitted by the scorer. Create a new configuration to change a protocol.
    """

    submetric_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SUBMETRIC_WEIGHTS)
    )
    reference_alignment_mode: str = DEFAULT_REFERENCE_ALIGNMENT_MODE
    delta_reg: float = 0.0
    require_reference_component_scores: bool = True

    def __post_init__(self) -> None:
        weights = validate_submetric_weights(self.submetric_weights)
        if weights.get("no_regression", 0.0) > 0 and not any(weights.get(key, 0.0) > 0 for key in GEOMETRIC_SUBMETRICS):
            raise ValueError("active no_regression requires at least one active geometric coefficient")
        if not isinstance(self.reference_alignment_mode, str):
            raise ValueError("reference_alignment_mode must be 'trajectory' or 'endpoint'")
        mode_aliases = {
            "whole_chunk": "trajectory", "whole_chunk_trajectory": "trajectory",
            "endpoint_net_direction": "endpoint", "net_direction": "endpoint",
        }
        mode = mode_aliases.get(self.reference_alignment_mode, self.reference_alignment_mode)
        if mode not in {"trajectory", "endpoint"}:
            raise ValueError("reference_alignment_mode must be 'trajectory' or 'endpoint'")
        margin = self.delta_reg
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise ValueError("delta_reg must be a finite nonnegative number")
        if not math.isfinite(float(margin)) or float(margin) < 0:
            raise ValueError("delta_reg must be a finite nonnegative number")
        if not isinstance(self.require_reference_component_scores, bool):
            raise ValueError("require_reference_component_scores must be a boolean")
        object.__setattr__(self, "submetric_weights", MappingProxyType(weights))
        object.__setattr__(self, "reference_alignment_mode", mode)
        object.__setattr__(self, "delta_reg", float(margin))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA,
            "aggregation": "component_mean",
            "submetric_weights": dict(self.submetric_weights),
            "reference_alignment_mode": self.reference_alignment_mode,
            "delta_reg": self.delta_reg,
            "require_reference_component_scores": self.require_reference_component_scores,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScoringConfig:
        if not isinstance(payload, Mapping):
            raise ValueError("scoring configuration must be a JSON object")
        allowed = {
            "schema", "aggregation", "submetric_weights", "reference_alignment_mode",
            "delta_reg", "require_reference_component_scores",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown scoring configuration fields: {sorted(unknown)}")
        if payload.get("schema", CONFIG_SCHEMA) != CONFIG_SCHEMA:
            raise ValueError("unsupported scoring configuration schema")
        if payload.get("aggregation", "component_mean") != "component_mean":
            raise ValueError("ScoringConfig uses component_mean; diagnostic profiles are explicit opt-ins")
        kwargs = {key: value for key, value in payload.items() if key not in {"schema", "aggregation"}}
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> ScoringConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, allow_nan=False)
            stream.write("\n")


def resolve_scoring_config(
    scoring_config: ScoringConfig | None = None,
    *,
    submetric_weights: Mapping[str, float] | None = None,
    reference_alignment_mode: str | None = None,
    delta_reg: float | None = None,
    require_reference_component_scores: bool | None = None,
) -> ScoringConfig:
    """Use one complete config or explicit overrides, never ambiguous mixtures."""
    overrides = {
        "submetric_weights": submetric_weights,
        "reference_alignment_mode": reference_alignment_mode,
        "delta_reg": delta_reg,
        "require_reference_component_scores": require_reference_component_scores,
    }
    provided: dict[str, Any] = {key: value for key, value in overrides.items() if value is not None}
    if scoring_config is not None:
        if not isinstance(scoring_config, ScoringConfig):
            raise TypeError("scoring_config must be a ScoringConfig instance")
        if provided:
            raise ValueError("pass scoring_config or individual scoring overrides, not both")
        return scoring_config
    return ScoringConfig(**provided)
