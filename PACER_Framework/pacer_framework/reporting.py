"""Held-out comparison and candidate-ranking reports.

Utilities for user-provided held-out results:
- overall held-out success with binomial **Wilson** confidence intervals;
- paired success-rate differences with descriptive **component/task
  cluster-bootstrap** percentile intervals (groups resampled with replacement,
  both methods sharing each draw);
- candidate ranking by component-balanced J_val with the selected flag.

All functions are pure and deterministic given the seed.
"""
from __future__ import annotations

import math
import random
from typing import Any, Mapping

Counts = Mapping[str, tuple[int, int]]  # component -> (successes, trials)


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as fractions in [0, 1]."""
    if trials <= 0:
        raise ValueError("wilson_interval requires trials > 0")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _overall(counts: Counts) -> tuple[int, int]:
    successes = sum(int(s) for s, _t in counts.values())
    trials = sum(int(t) for _s, t in counts.values())
    return successes, trials


def success_table(
    results: Mapping[str, Counts],
    *,
    z: float = 1.96,
) -> list[dict[str, Any]]:
    """Per-method held-out summary: per-component counts, overall rate, Wilson CI."""
    table: list[dict[str, Any]] = []
    for method, counts in results.items():
        successes, trials = _overall(counts)
        lo, hi = wilson_interval(successes, trials, z=z)
        table.append(
            {
                "method": method,
                "per_component": {c: f"{s}/{t}" for c, (s, t) in sorted(counts.items())},
                "successes": successes,
                "trials": trials,
                "rate": successes / trials,
                "ci95_pct": (round(lo * 100.0, 1), round(hi * 100.0, 1)),
            }
        )
    return table


def paired_component_bootstrap(
    method_a: Counts,
    method_b: Counts,
    *,
    n_boot: int = 10000,
    seed: int = 20260610,
) -> dict[str, Any]:
    """Paired component cluster-bootstrap difference (method_a − method_b).

    Components are the resampling clusters: each draw samples components with
    replacement, the same draw applies to both methods, and the pooled success
    rates are differenced. Returns the point difference and the percentile
    95 percent interval, in percentage points.
    """
    components = sorted(method_a)
    if sorted(method_b) != components:
        raise ValueError("both methods must report the same component set")
    if not components:
        raise ValueError("paired_component_bootstrap requires at least one component")

    def pooled_diff(sample: list[str]) -> float:
        sa = sum(method_a[c][0] for c in sample)
        ta = sum(method_a[c][1] for c in sample)
        sb = sum(method_b[c][0] for c in sample)
        tb = sum(method_b[c][1] for c in sample)
        return (sa / ta - sb / tb) * 100.0

    point = pooled_diff(components)
    rng = random.Random(seed)
    draws = sorted(
        pooled_diff([components[rng.randrange(len(components))] for _ in components])
        for _ in range(int(n_boot))
    )
    lo = draws[max(0, math.ceil(0.025 * n_boot) - 1)]
    hi = draws[min(n_boot - 1, math.floor(0.975 * n_boot))]
    return {
        "difference_pp": round(point, 6),
        "ci95_pp": (round(lo, 1), round(hi, 1)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "components": components,
    }


def candidate_ranking_table(
    j_vals: Mapping[str, float],
    feasible: Mapping[str, bool],
    selected: str | None,
) -> list[dict[str, Any]]:
    """Table app_ranking — candidates sorted by J_val with feasibility/selection."""
    rows = [
        {
            "candidate": eta_id,
            "j_val": float(score),
            "feasible": bool(feasible.get(eta_id, False)),
            "selected": eta_id == selected,
        }
        for eta_id, score in j_vals.items()
    ]
    rows.sort(key=lambda r: (-r["j_val"], r["candidate"]))
    return rows
