"""PACER parameterized evidence reweighting utilities.

PACER = Process-Aware Correction and Evidence Reweighting.

This module is deliberately offline-only. It reads compiled action-chunk rows and
produces candidate BO/structured-search weights under non-overridable safety
and eligibility gates. It does not import robot, OpenPI, VLM, or hardware code.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

PACER_WEIGHT_SCHEMA_VERSION = "tracevla_bo_lc_fma_weights.v0.1"
PACER_ABLATION_WEIGHT_SCHEMA_VERSION = "tracevla_ablation_weight_view.v0.1"

ABLATION_WEIGHT_MODES = frozenset(
    {
        "clean_demo_only",
        "uniform_replay",
        "correction_only",
        "outcome_only",
        "fixed_rw_fma",
        "random_weight",
        "tracevla_eta",
    }
)


@dataclass(frozen=True)
class PacerEta:
    """Candidate evidence-to-weight parameters.

    Base weights are the paper's declared evidence coefficients. Multipliers
    adjust role-specific credit in the final row-weight equation.
    """

    progress: float = 0.30
    proximity: float = 0.45
    terminal: float = 0.15
    direction: float = 0.00
    stop: float = 0.05
    operator: float = 0.05
    provenance: float = 0.00
    correction_multiplier: float = 1.20
    demo_multiplier: float = 1.00
    partial_multiplier: float = 0.60
    success_multiplier: float = 1.00
    failure_multiplier: float = 0.00
    excluded_multiplier: float = 0.00
    tau: float = 1.00
    w_min: float = 0.00
    w_max: float = 4.50
    clean_demo_floor: float = 0.80
    human_correction_floor: float = 0.65

    FREE_DIMS = (
        "progress", "proximity", "terminal", "stop",
        "correction_multiplier", "partial_multiplier", "tau", "w_max",
    )
    FIXED_DIMS = (
        "direction", "operator", "provenance", "demo_multiplier", "success_multiplier",
        "failure_multiplier", "excluded_multiplier", "clean_demo_floor", "human_correction_floor",
    )
    SEARCH_RANGES = {
        "progress": (0.10, 0.60),
        "proximity": (0.10, 0.60),
        "terminal": (0.05, 0.40),
        "stop": (0.0, 0.20),
        "correction_multiplier": (0.80, 2.00),
        "partial_multiplier": (0.10, 1.20),
        "tau": (0.50, 2.00),
        "w_max": (2.00, 5.00),
    }

    def components(self) -> dict[str, float]:
        raw = {
            "progress": max(0.0, float(self.progress)),
            "proximity": max(0.0, float(self.proximity)),
            "terminal": max(0.0, float(self.terminal)),
            "direction": max(0.0, float(self.direction)),
            "stop": max(0.0, float(self.stop)),
            "operator": max(0.0, float(self.operator)),
            "provenance": max(0.0, float(self.provenance)),
        }
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("At least one base evidence weight must be positive")
        return raw

    def normalized_components(self) -> dict[str, float]:
        """Deprecated compatibility view; PACER scoring uses raw components."""
        raw = self.components()
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PacerEta":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def free_vector(self) -> dict[str, float]:
        return {dim: float(getattr(self, dim)) for dim in self.FREE_DIMS}

    def to_normalized(self) -> list[float]:
        result: list[float] = []
        for dim in self.FREE_DIMS:
            lo, hi = self.SEARCH_RANGES[dim]
            val = float(getattr(self, dim))
            result.append((val - lo) / (hi - lo) if hi > lo else 0.5)
        return result

    @classmethod
    def from_normalized(cls, x: list[float]) -> "PacerEta":
        if len(x) != len(cls.FREE_DIMS):
            raise ValueError(f"expected {len(cls.FREE_DIMS)} values, got {len(x)}")
        kwargs: dict[str, float] = {}
        for dim, xi in zip(cls.FREE_DIMS, x):
            lo, hi = cls.SEARCH_RANGES[dim]
            kwargs[dim] = lo + xi * (hi - lo)
        return cls(**kwargs)

    def eta_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


DEFAULT_ETA_POOL: dict[str, PacerEta] = {
    "rw_like": PacerEta(),
    "progress_heavy": PacerEta(progress=0.55, proximity=0.20, terminal=0.15, stop=0.05, operator=0.05),
    "hover_proximity_heavy": PacerEta(progress=0.20, proximity=0.60, terminal=0.10, stop=0.05, operator=0.05),
    "terminal_stop_heavy": PacerEta(progress=0.20, proximity=0.25, terminal=0.35, stop=0.15, operator=0.05),
    "correction_heavy": PacerEta(correction_multiplier=1.80, partial_multiplier=0.70, w_max=4.5),
    "conservative": PacerEta(tau=1.50, w_max=2.5, partial_multiplier=0.40),
    "ram_connector_recovery": PacerEta(progress=0.45, proximity=0.30, terminal=0.10, stop=0.05, correction_multiplier=1.70, partial_multiplier=0.90, w_max=5.0),
    "balanced_low_clip": PacerEta(tau=0.85, w_max=3.5),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def nested_get(row: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _valid_mask_mass(row: Mapping[str, Any]) -> bool:
    mask = nested_get(row, "masks.policy_loss_mask", []) or []
    if not isinstance(mask, list) or len(mask) == 0:
        return False
    values = [as_float(v, -1.0) for v in mask]
    return all(v >= 0.0 for v in values) and any(v > 0.0 for v in values)


def _provenance_ok(row: Mapping[str, Any]) -> bool:
    if nested_get(row, "returns.aw_fma_loss_eligible", None) is False:
        return False
    if bool(nested_get(row, "eligibility.quarantined", False)):
        return False
    if nested_get(row, "returns.aw_fma_vlm_authority_used", False) is not False:
        return False
    if nested_get(row, "semantic_sidecar.vlm_authority_used", False) is not False:
        return False
    return _valid_mask_mass(row)


def _chunk_local_evidence_components(row: Mapping[str, Any]) -> dict[str, float] | None:
    """Derive PACER evidence directly from per-chunk trajectory geometry.

    Human correction/demo anchors can carry useful `chunk_local_to_active_target`
    evidence even when the older AW-FMA return block used a constant human anchor.
    PACER should rank those correction chunks by measured progress/proximity rather
    than giving every correction in the same stratum an identical score.
    """
    chunk = nested_get(row, "distance_features.chunk_local_to_active_target", {}) or {}
    if not isinstance(chunk, Mapping) or not chunk.get("present"):
        return None

    e_end = as_float(chunk.get("end_combined_pose_distance_m"), default=float("nan"))
    if not math.isfinite(e_end):
        e_end = as_float(chunk.get("end_pos_error_m"), default=float("nan"))
    delta = as_float(chunk.get("delta_combined_pose_distance_m"), default=float("nan"))
    if not math.isfinite(delta):
        delta = as_float(chunk.get("delta_pos_error_m"), default=float("nan"))

    proximity = math.exp(-e_end / 0.05) if math.isfinite(e_end) else 0.0
    # delta is start distance - end distance, so positive means moved closer.
    progress = 0.5 + 0.5 * _clamp01(delta / 0.05) if math.isfinite(delta) else 0.5
    terminal = 1.0 if (
        chunk.get("inside_region_at_end") is True
        or chunk.get("hover_05cm_at_end") is True
        or chunk.get("inside_region_first_t") is not None
        or chunk.get("hover_05cm_first_t") is not None
    ) else 0.0
    stop = 1.0 if nested_get(row, "reward.stop_source") in {"model_stop_token", "manual_stop_near_target"} else 0.0
    operator = max(0.0, as_float(nested_get(row, "returns.rank_weight_unvalidated"), 0.0))
    direction = as_float(chunk.get("direction") or chunk.get("target_direction_cosine"), default=0.0)
    provenance = 1.0 if _provenance_ok(row) else 0.0
    target_ok, _target_reasons = target_match(row)
    target_factor = 1.0 if target_ok else 0.0

    return {
        "progress": target_factor * _clamp01(progress),
        "proximity": target_factor * _clamp01(proximity),
        "terminal": target_factor * _clamp01(terminal),
        "direction": target_factor * _clamp01(direction),
        "stop": _clamp01(stop),
        "operator": _clamp01(operator),
        "provenance": _clamp01(provenance),
    }


def evidence_components(row: Mapping[str, Any]) -> dict[str, float]:
    comps = nested_get(row, "returns.aw_fma_reward_components", {}) or {}
    chunk_evidence = _chunk_local_evidence_components(row)

    # Human correction rows should be internally ranked by measured trajectory
    # evidence when available. Older compiler views store human anchors with
    # constant/null AW-FMA components; using those constants would make all
    # corrections receive identical PACER scores regardless of recovery quality.
    role = str(row.get("sample_role", ""))
    has_anchor_override = isinstance(comps, Mapping) and comps.get("anchor_override") is not None
    missing_normalized_aw = not isinstance(comps, Mapping) or any(
        comps.get(k) is None for k in ("r_progress_signed01", "r_proximity", "r_terminal")
    )
    if chunk_evidence is not None and role in {"human_correction", "clean_demo"} and (has_anchor_override or missing_normalized_aw):
        return chunk_evidence

    # Existing AW-FMA fields are already normalized to useful ranges. Fall back
    # to per-chunk geometry when the compiler did not emit normalized fields,
    # then to legacy route/terminal proxies.
    progress_default = chunk_evidence["progress"] if chunk_evidence is not None else as_float(nested_get(row, "reward.route_reward_components.progress_signed01"), 0.5)
    proximity_default = chunk_evidence["proximity"] if chunk_evidence is not None else max(0.0, 1.0 - as_float(nested_get(row, "reward.pos_error_min_m"), 1.0))
    terminal_default = chunk_evidence["terminal"] if chunk_evidence is not None else (1.0 if nested_get(row, "reward.inside_region", False) else 0.0)
    stop_default = chunk_evidence["stop"] if chunk_evidence is not None else (1.0 if nested_get(row, "reward.stop_source") in {"model_stop_token", "manual_stop_near_target"} else 0.0)
    operator_default = chunk_evidence["operator"] if chunk_evidence is not None else max(0.0, as_float(nested_get(row, "returns.rank_weight_unvalidated"), 0.0))
    direction_default = chunk_evidence["direction"] if chunk_evidence is not None else 0.0
    provenance_default = chunk_evidence["provenance"] if chunk_evidence is not None else (1.0 if _provenance_ok(row) else 0.0)

    progress = as_float(comps.get("r_progress_signed01") if isinstance(comps, Mapping) else None, default=progress_default)
    proximity = as_float(comps.get("r_proximity") if isinstance(comps, Mapping) else None, default=proximity_default)
    terminal = as_float(comps.get("r_terminal") if isinstance(comps, Mapping) else None, default=terminal_default)
    stop = as_float(comps.get("r_stop_handoff") if isinstance(comps, Mapping) else None, default=stop_default)
    operator = as_float(comps.get("r_operator_rank") if isinstance(comps, Mapping) else None, default=operator_default)
    direction = as_float(comps.get("r_direction") if isinstance(comps, Mapping) else None, default=direction_default)
    provenance = as_float(comps.get("r_provenance") if isinstance(comps, Mapping) else None, default=provenance_default)

    # Paper eq. app_geom: the target-consistency indicator T_i multiplies the
    # geometric evidence channels only. Operator/stop/provenance evidence stay
    # readable for audit, while wrong-target outcome labels are additionally
    # role-gated to zero weight upstream of this function.
    target_ok, _target_reasons = target_match(row)
    target_factor = 1.0 if target_ok else 0.0
    return {
        "progress": target_factor * _clamp01(progress),
        "proximity": target_factor * _clamp01(proximity),
        "terminal": target_factor * _clamp01(terminal),
        "direction": target_factor * _clamp01(direction),
        "stop": _clamp01(stop),
        "operator": _clamp01(operator),
        "provenance": _clamp01(provenance),
    }


def role_multiplier(row: Mapping[str, Any], eta: PacerEta) -> float:
    role = str(row.get("sample_role", ""))
    if role == "human_correction":
        return eta.correction_multiplier
    if role == "clean_demo":
        return eta.demo_multiplier
    if role == "model_partial":
        return eta.partial_multiplier
    if role == "model_success":
        return eta.success_multiplier
    return 1.0


def _first_present(row: Mapping[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value = nested_get(row, path, None)
        if value is not None:
            return value
    return None


def target_match(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return whether explicit target metadata is internally consistent.

    Legacy rows without target identity metadata remain valid. Once a row carries
    target identity or wrong-target labels, PACER treats inconsistency as a hard
    gate so distance-based evidence cannot credit the wrong component.
    """
    reasons: list[str] = []
    consistency = str(
        _first_present(
            row,
            [
                "target.target_consistency",
                "semantic_sidecar.target_consistency",
                "assistant_target.target_consistency",
                "reward.target_consistency",
                "returns.target_consistency",
            ],
        )
        or ""
    )
    if consistency in {"wrong_target", "wrong_region"}:
        reasons.append("target_consistency_wrong")

    stop_label = str(
        _first_present(row, ["operator_stop_label", "reward.stop_source", "returns.stop_source"])
        or ""
    )
    if stop_label == "wrong_target_abort":
        reasons.append("wrong_target_abort_label")

    active = _first_present(
        row,
        [
            "target.active_target_id",
            "active_target_id",
            "target_region.active_target_id",
            "semantic_sidecar.active_target_id",
        ],
    )
    geometry = _first_present(
        row,
        [
            "target.geometry_target_id",
            "geometry_target_id",
            "target_region.target_id",
            "target_region_ref_target_id",
        ],
    )
    observed = _first_present(
        row,
        [
            "target.observed_target_id",
            "observed_target_id",
            "contact.target_id",
            "semantic_sidecar.observed_target_id",
        ],
    )
    component = row.get("component")

    if active is not None and geometry is not None and str(active) != str(geometry):
        reasons.append("active_geometry_target_id_mismatch")
    if active is not None and observed is not None and str(active) != str(observed):
        reasons.append("observed_target_id_mismatch")
    if active is not None and component is not None and str(active) != str(component):
        aliases = nested_get(row, "target.component_aliases", []) or []
        allowed = {str(component), *(str(a) for a in aliases)}
        if str(active) not in allowed:
            reasons.append("component_target_id_mismatch")

    active_idx = _first_present(row, ["target.active_target_idx", "active_target_idx"])
    closest_idx = _first_present(
        row,
        [
            "target.terminal_closest_target_idx",
            "returns.terminal_closest_target_idx",
            "reward.closest_target_idx",
            "tracevla_auto_score.closest_target_idx",
        ],
    )
    if active_idx is not None and closest_idx is not None and str(active_idx) != str(closest_idx):
        reasons.append("closest_target_idx_mismatch")

    return len(reasons) == 0, reasons


def hard_gate(row: Mapping[str, Any], *, require_train_split: bool = True) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if require_train_split and row.get("split") != "train":
        reasons.append("non_train_split")
    loss_eligible = nested_get(row, "returns.aw_fma_loss_eligible", None)
    if loss_eligible is False:
        reasons.append("not_aw_fma_loss_eligible")
    if row.get("sample_role") == "model_failure":
        reasons.append("model_failure_no_positive_imitation")
    if row.get("sample_role") == "excluded":
        reasons.append("excluded_role_no_positive_imitation")
    if bool(nested_get(row, "eligibility.quarantined", False)):
        reasons.append("quarantined")
    if nested_get(row, "returns.aw_fma_vlm_authority_used", False) is not False:
        reasons.append("returns_vlm_authority_used")
    if nested_get(row, "semantic_sidecar.vlm_authority_used", False) is not False:
        reasons.append("semantic_vlm_authority_used")
    # Target identity is enforced inside evidence construction (target-metadata
    # inconsistency zeroes the geometric evidence channels via the T_i factor)
    # and by validation-time wrong-target hard failures. The training gate does
    # not add an extra target-mismatch factor, so changing the validation target
    # rule does not silently change an in-flight training run.
    # G_mask requires nonzero valid mask mass: an all-zero policy_loss_mask row
    # can contribute no loss horizon and must not count as trainable.
    if not _valid_mask_mass(row):
        reasons.append("invalid_policy_loss_mask")
    return len(reasons) == 0, reasons


def raw_score(row: Mapping[str, Any], eta: PacerEta) -> float:
    """Process score s_eta(i) = sum_k beta_k * e_k(i) over the paper evidence vector.

    Coefficients are the raw declared beta values (no unit-sum renormalization),
    matching the paper's process-score definition. The role multiplier is applied
    here so role-stratified centering sees the role-adjusted score.
    """
    weights = eta.components()
    comps = evidence_components(row)
    base = sum(weights[k] * comps[k] for k in weights)
    return max(0.0, min(1.0, base * role_multiplier(row, eta)))


def stratum_key(row: Mapping[str, Any]) -> str:
    return "|".join(str(row.get(k, "unknown")) for k in ("component", "block", "sample_role"))


def _robust_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    med = median(values)
    absdev = [abs(v - med) for v in values]
    mad = median(absdev) if absdev else 0.0
    scale = 1.4826 * mad
    if scale < 1e-6:
        # Fallback to standard deviation-like scale.
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
        scale = math.sqrt(var) or 1.0
    return med, max(scale, 1e-6)


def compute_pacer_weights(rows: list[dict[str, Any]], eta: PacerEta, *, eta_id: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eta_id = eta_id or f"eta_{eta.eta_hash()}"
    scored: list[tuple[dict[str, Any], float, str, bool, list[str]]] = []
    by_stratum: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = raw_score(row, eta)
        key = stratum_key(row)
        ok, reasons = hard_gate(row)
        scored.append((row, score, key, ok, reasons))
        if ok:
            by_stratum[key].append(score)

    stats = {k: _robust_stats(v) for k, v in by_stratum.items()}
    all_train_scores = [score for _, score, _, ok, _ in scored if ok]
    global_center, global_scale = _robust_stats(all_train_scores)

    out_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    weights: list[float] = []
    role_weight_sum: defaultdict[str, float] = defaultdict(float)
    component_weight_sum: defaultdict[str, float] = defaultdict(float)

    for row, score, key, ok, reasons in scored:
        center, scale = stats.get(key, (global_center, global_scale))
        advantage = score - center
        norm_adv = advantage / scale
        if ok:
            unclipped = math.exp(norm_adv / max(eta.tau, 1e-6))
            clipped = max(eta.w_min, min(eta.w_max, unclipped))
            role = row.get("sample_role")
            if role == "clean_demo":
                clipped = max(clipped, eta.clean_demo_floor)
            elif role == "human_correction":
                clipped = max(clipped, eta.human_correction_floor)
            weight = float(clipped)
        else:
            unclipped = 0.0
            weight = 0.0
        new_row = json.loads(json.dumps(row))  # deep copy via JSON-safe data
        returns = new_row.setdefault("returns", {})
        returns.update(
            {
                "tracevla_schema": PACER_WEIGHT_SCHEMA_VERSION,
                "tracevla_eta_id": eta_id,
                "tracevla_eta_hash": eta.eta_hash(),
                "tracevla_raw_evidence_score": score,
                "tracevla_stratum_key": key,
                "tracevla_train_baseline": center,
                "tracevla_advantage_like_score": advantage,
                "tracevla_advantage_normalized": norm_adv,
                "tracevla_weight_unclipped": unclipped,
                "tracevla_weight": weight,
                "tracevla_loss_eligible": bool(ok),
                "tracevla_loss_ineligible_reasons": reasons,
            }
        )
        out_rows.append(new_row)
        counters["rows"] += 1
        counters[f"role:{row.get('sample_role')}"] += 1
        counters[f"component:{row.get('component')}"] += 1
        if ok:
            counters["eligible"] += 1
            weights.append(weight)
            role_weight_sum[str(row.get("sample_role"))] += weight
            component_weight_sum[str(row.get("component"))] += weight
        else:
            for reason in reasons:
                counters[f"ineligible:{reason}"] += 1

    positive = [w for w in weights if w > 0]
    eff_count = (sum(positive) ** 2 / sum(w * w for w in positive)) if positive else 0.0
    manifest = {
        "schema": PACER_WEIGHT_SCHEMA_VERSION,
        "eta_id": eta_id,
        "eta": eta.to_dict(),
        "eta_hash": eta.eta_hash(),
        "counts": dict(counters),
        "num_strata": len(by_stratum),
        "weight_min": min(positive) if positive else 0.0,
        "weight_max": max(positive) if positive else 0.0,
        "weight_sum": sum(positive),
        "effective_sample_count": eff_count,
        "role_weight_sum": dict(role_weight_sum),
        "component_weight_sum": dict(component_weight_sum),
        "safety_leakage": safety_leakage_report(out_rows),
    }
    return out_rows, manifest


def safety_leakage_report(rows: Iterable[Mapping[str, Any]], *, weight_path: str = "returns.tracevla_weight") -> dict[str, int]:
    report: Counter[str] = Counter()
    for row in rows:
        w = as_float(nested_get(row, weight_path), 0.0)
        if w <= 0:
            continue
        if row.get("sample_role") == "model_failure":
            report["model_failure_positive_weight"] += 1
        if bool(nested_get(row, "eligibility.quarantined", False)):
            report["quarantined_positive_weight"] += 1
        if nested_get(row, "returns.aw_fma_vlm_authority_used", False) is not False:
            report["returns_vlm_authority_positive_weight"] += 1
        if nested_get(row, "semantic_sidecar.vlm_authority_used", False) is not False:
            report["semantic_vlm_authority_positive_weight"] += 1
        if row.get("split", "train") != "train":
            report["non_train_positive_weight"] += 1
    return dict(report)


def _stable_random_weight(row: Mapping[str, Any], random_seed: int) -> float:
    key = f"{random_seed}:{row.get('row_id', '')}:{row.get('config_id', '')}:{row.get('sample_role', '')}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    unit = int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)
    return round(0.05 + 0.95 * unit, 9)


def _mode_weight(row: Mapping[str, Any], mode: str, *, random_seed: int) -> tuple[float, str | None, list[str]]:
    ok, reasons = hard_gate(row)
    if not ok:
        return 0.0, None, reasons

    role = str(row.get("sample_role", ""))
    if mode == "clean_demo_only":
        if role != "clean_demo":
            return 0.0, None, [f"mode_excludes_role:{role}"]
        return 1.0, "constant_clean_demo", []
    if mode == "uniform_replay":
        return 1.0, "constant_uniform", []
    if mode == "correction_only":
        if role != "human_correction":
            return 0.0, None, [f"mode_excludes_role:{role}"]
        return 1.0, "constant_human_correction", []
    if mode == "outcome_only":
        if role not in {"clean_demo", "model_success"}:
            return 0.0, None, [f"mode_excludes_role:{role}"]
        return 1.0, "constant_success_or_demo", []
    if mode == "fixed_rw_fma":
        weight = as_float(nested_get(row, "returns.rank_weight_unvalidated"), 0.0)
        if weight <= 0:
            return 0.0, "returns.rank_weight_unvalidated", ["non_positive_fixed_rw_fma_weight"]
        return weight, "returns.rank_weight_unvalidated", []
    if mode == "random_weight":
        return _stable_random_weight(row, random_seed), "stable_row_hash", []
    raise ValueError(f"unknown ablation mode: {mode}")


def _attach_ablation_fields(
    row: Mapping[str, Any],
    *,
    mode: str,
    weight: float,
    source: str | None,
    reasons: list[str],
    eta_id: str | None,
    eta_hash: str | None,
) -> dict[str, Any]:
    new_row = json.loads(json.dumps(row))
    returns = new_row.setdefault("returns", {})
    returns.update(
        {
            "tracevla_ablation_schema": PACER_ABLATION_WEIGHT_SCHEMA_VERSION,
            "tracevla_ablation_mode": mode,
            "tracevla_ablation_eta_id": eta_id,
            "tracevla_ablation_eta_hash": eta_hash,
            "tracevla_ablation_weight": float(weight),
            "tracevla_ablation_weight_source": source,
            "tracevla_ablation_loss_eligible": bool(weight > 0),
            "tracevla_ablation_ineligible_reasons": reasons,
            "loss_weight": float(weight),
        }
    )
    return new_row


def _ablation_manifest(
    rows: list[Mapping[str, Any]],
    *,
    mode: str,
    eta: PacerEta | None,
    eta_id: str | None,
    random_seed: int,
    pacer_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    weights = [as_float(nested_get(r, "returns.loss_weight"), 0.0) for r in rows]
    positive = [w for w in weights if w > 0]
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    role_weight_sum: defaultdict[str, float] = defaultdict(float)
    component_weight_sum: defaultdict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        counts["rows"] += 1
        split_counts[str(row.get("split"))] += 1
        if weight > 0:
            counts["positive_weight_rows"] += 1
            role_weight_sum[str(row.get("sample_role"))] += weight
            component_weight_sum[str(row.get("component"))] += weight
        else:
            for reason in nested_get(row, "returns.tracevla_ablation_ineligible_reasons", []) or []:
                counts[f"ineligible:{reason}"] += 1
    eff_count = (sum(positive) ** 2 / sum(w * w for w in positive)) if positive else 0.0
    manifest = {
        "schema": PACER_ABLATION_WEIGHT_SCHEMA_VERSION,
        "ablation_mode": mode,
        "weight_field": "returns.loss_weight",
        "method_weight_field": "returns.tracevla_ablation_weight",
        "source_row_count": len(rows),
        "counts": dict(counts),
        "split_counts": dict(split_counts),
        "weight_min": min(positive) if positive else 0.0,
        "weight_max": max(positive) if positive else 0.0,
        "weight_sum": sum(positive),
        "effective_sample_count": eff_count,
        "role_weight_sum": dict(role_weight_sum),
        "component_weight_sum": dict(component_weight_sum),
        "random_seed": random_seed if mode == "random_weight" else None,
        "eta_id": eta_id if mode == "tracevla_eta" else None,
        "eta_hash": eta.eta_hash() if mode == "tracevla_eta" and eta is not None else None,
        "safety_leakage": safety_leakage_report(rows, weight_path="returns.loss_weight"),
    }
    if pacer_manifest is not None:
        manifest["tracevla_manifest"] = dict(pacer_manifest)
    return manifest


def compute_ablation_weights(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    eta: PacerEta | None = None,
    eta_id: str | None = None,
    random_seed: int = 20260526,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a method-specific weighted view over the same canonical rows.

    The returned rows preserve row order and split membership. Every mode writes
    a generic train-consumable ``returns.loss_weight`` plus an audit-specific
    ``returns.tracevla_ablation_weight``. Non-train or hard-gated rows always get
    zero weight but remain present for validation/heldout context.
    """
    if mode not in ABLATION_WEIGHT_MODES:
        raise ValueError(f"unknown ablation mode {mode!r}; choices={sorted(ABLATION_WEIGHT_MODES)}")

    eta = eta or PacerEta()
    eta_id = eta_id or f"eta_{eta.eta_hash()}"
    if mode == "tracevla_eta":
        pacer_rows, pacer_manifest = compute_pacer_weights(rows, eta, eta_id=eta_id)
        out_rows = [
            _attach_ablation_fields(
                row,
                mode=mode,
                weight=as_float(nested_get(row, "returns.tracevla_weight"), 0.0),
                source="returns.tracevla_weight",
                reasons=list(nested_get(row, "returns.tracevla_loss_ineligible_reasons", []) or []),
                eta_id=eta_id,
                eta_hash=eta.eta_hash(),
            )
            for row in pacer_rows
        ]
        return out_rows, _ablation_manifest(out_rows, mode=mode, eta=eta, eta_id=eta_id, random_seed=random_seed, pacer_manifest=pacer_manifest)

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        weight, source, reasons = _mode_weight(row, mode, random_seed=random_seed)
        out_rows.append(
            _attach_ablation_fields(
                row,
                mode=mode,
                weight=weight,
                source=source,
                reasons=reasons,
                eta_id=None,
                eta_hash=None,
            )
        )
    return out_rows, _ablation_manifest(out_rows, mode=mode, eta=None, eta_id=None, random_seed=random_seed)


def freeze_config_splits(rows: list[dict[str, Any]], *, val_configs: set[str] | None = None, heldout_configs: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign deterministic train/val/heldout splits.

    Prefer config-level splits when at least three configs are available. For
    small smoke subsets with fewer configs, fall back to trial-level groups so
    train rows remain available and validation/heldout leakage is still grouped.
    """
    configs = sorted({str(r.get("config_id", "unknown")) for r in rows})
    explicit = val_configs is not None or heldout_configs is not None
    strategy = "config_id_holdout"

    if explicit or len(configs) >= 3:
        if heldout_configs is None:
            heldout_configs = {configs[-1]} if configs else set()
        if val_configs is None:
            remaining = [c for c in configs if c not in heldout_configs]
            val_configs = {remaining[-1]} if remaining else set()
        def split_for(row: Mapping[str, Any]) -> str:
            cfg = str(row.get("config_id", "unknown"))
            if cfg in heldout_configs:
                return "heldout"
            if cfg in val_configs:
                return "val"
            return "train"
        manifest_extra = {
            "configs": configs,
            "val_configs": sorted(val_configs),
            "heldout_configs": sorted(heldout_configs),
        }
    else:
        strategy = "trial_id_holdout_fallback"
        groups = sorted({str(r.get("trial_id") or r.get("episode_dir") or r.get("row_id") or "unknown") for r in rows})
        heldout_n = 1 if len(groups) >= 3 else 0
        val_n = 1 if len(groups) >= 2 else 0
        heldout_groups = set(groups[-heldout_n:]) if heldout_n else set()
        remaining = [g for g in groups if g not in heldout_groups]
        val_groups = set(remaining[-val_n:]) if val_n else set()
        def split_for(row: Mapping[str, Any]) -> str:
            group = str(row.get("trial_id") or row.get("episode_dir") or row.get("row_id") or "unknown")
            if group in heldout_groups:
                return "heldout"
            if group in val_groups:
                return "val"
            return "train"
        manifest_extra = {
            "configs": configs,
            "group_key": "trial_id|episode_dir|row_id",
            "groups": groups,
            "val_groups": sorted(val_groups),
            "heldout_groups": sorted(heldout_groups),
        }

    out = []
    split_counts: Counter[str] = Counter()
    by_component_split: Counter[str] = Counter()
    for row in rows:
        r = json.loads(json.dumps(row))
        split = split_for(r)
        r["split"] = split
        out.append(r)
        split_counts[split] += 1
        by_component_split[f"{r.get('component')}|{split}"] += 1
    manifest = {
        "schema": "tracevla_split_manifest.v0.1",
        "strategy": strategy,
        **manifest_extra,
        "split_counts": dict(split_counts),
        "component_split_counts": dict(by_component_split),
    }
    return out, manifest
