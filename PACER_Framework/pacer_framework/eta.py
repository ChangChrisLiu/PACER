"""Candidate configuration eta (paper eq:eta_free, eq:eta_fixed, app:candidates).

Free fields (8, searched): beta_prog, beta_prox, beta_term, beta_stop,
gamma_correction, gamma_partial, tau, w_max — bounds per Table app_eta_free.
Fixed protocol fields (9): beta_dir, beta_op, beta_prov, gamma_clean,
gamma_auto_success, gamma_failure, gamma_excluded, f_clean, f_corr.

The process score is the RAW weighted sum s_eta(i) = sum_k beta_k e_k(i) over
the seven evidence keys — no unit-sum renormalization of coefficients.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

# Paper evidence index set K (app:features). Order matters for reporting only.
EVIDENCE_KEYS = ("prog", "prox", "term", "dir", "stop", "op", "prov")

# Table app_eta_free bounds.
FREE_BOUNDS = {
    "beta_prog": (0.10, 0.60),
    "beta_prox": (0.10, 0.60),
    "beta_term": (0.05, 0.40),
    "beta_stop": (0.00, 0.20),
    "gamma_correction": (0.80, 2.00),
    "gamma_partial": (0.10, 1.20),
    "tau": (0.50, 2.00),
    "w_max": (2.00, 5.00),
}


@dataclass(frozen=True)
class PaperEta:
    # Free fields (eq:eta_free), defaults = rw_like candidate.
    beta_prog: float = 0.30
    beta_prox: float = 0.45
    beta_term: float = 0.15
    beta_stop: float = 0.05
    gamma_correction: float = 1.20
    gamma_partial: float = 0.60
    tau: float = 1.00
    w_max: float = 4.50
    # Fixed protocol fields (eq:eta_fixed; values from app:candidates text).
    beta_dir: float = 0.00
    beta_op: float = 0.05
    beta_prov: float = 0.00
    gamma_clean: float = 1.00
    gamma_auto_success: float = 1.00
    gamma_failure: float = 0.00
    gamma_excluded: float = 0.00
    f_clean: float = 0.80
    f_corr: float = 0.65

    FREE_FIELDS = (
        "beta_prog", "beta_prox", "beta_term", "beta_stop",
        "gamma_correction", "gamma_partial", "tau", "w_max",
    )
    FIXED_FIELDS = (
        "beta_dir", "beta_op", "beta_prov",
        "gamma_clean", "gamma_auto_success", "gamma_failure", "gamma_excluded",
        "f_clean", "f_corr",
    )

    def betas(self) -> dict[str, float]:
        """Raw evidence coefficients beta_k over K (no renormalization)."""
        return {
            "prog": float(self.beta_prog),
            "prox": float(self.beta_prox),
            "term": float(self.beta_term),
            "dir": float(self.beta_dir),
            "stop": float(self.beta_stop),
            "op": float(self.beta_op),
            "prov": float(self.beta_prov),
        }

    def role_multiplier(self, role: str) -> float:
        """alpha_eta(rho) in eq:weight — gamma for each paper role."""
        return {
            "clean": float(self.gamma_clean),
            "correction": float(self.gamma_correction),
            "auto_success": float(self.gamma_auto_success),
            "partial": float(self.gamma_partial),
            "failure": float(self.gamma_failure),
            "excluded": float(self.gamma_excluded),
        }.get(role, 0.0)

    def floor(self, role: str) -> float:
        """Protocol floor f_rho: f_clean / f_corr for clean / correction, else 0."""
        if role == "clean":
            return float(self.f_clean)
        if role == "correction":
            return float(self.f_corr)
        return 0.0

    def validate_bounds(self) -> list[str]:
        """Check free fields against Table app_eta_free and floor sanity."""
        errors: list[str] = []
        for field, (lo, hi) in FREE_BOUNDS.items():
            value = float(getattr(self, field))
            if not (lo <= value <= hi):
                errors.append(f"{field}={value} outside [{lo}, {hi}]")
        for floor_field in ("f_clean", "f_corr"):
            value = float(getattr(self, floor_field))
            if not (0.0 <= value <= float(self.w_max)):
                errors.append(f"{floor_field}={value} outside [0, w_max={self.w_max}]")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PaperEta":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: float(v) for k, v in d.items() if k in allowed})

    def eta_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def process_score(eta: PaperEta, evidence: Mapping[str, float]) -> float:
    """s_eta(i) = sum_k beta_k e_k(i) over K (raw coefficients, bounded evidence)."""
    betas = eta.betas()
    total = 0.0
    for key in EVIDENCE_KEYS:
        e_k = float(evidence.get(key, 0.0))
        e_k = max(0.0, min(1.0, e_k))
        total += betas[key] * e_k
    return total


# Table app_candidates — the M = 8 predeclared dimension-emphasized candidates.
CANDIDATE_POOL: dict[str, PaperEta] = {
    "rw_like": PaperEta(),
    "progress_heavy": PaperEta(beta_prog=0.55, beta_prox=0.20, beta_term=0.15, beta_stop=0.05),
    "hover_proximity_heavy": PaperEta(beta_prog=0.20, beta_prox=0.60, beta_term=0.10, beta_stop=0.05),
    "terminal_stop_heavy": PaperEta(beta_prog=0.20, beta_prox=0.25, beta_term=0.35, beta_stop=0.15),
    "correction_heavy": PaperEta(gamma_correction=1.80, gamma_partial=0.70, w_max=4.50),
    "conservative": PaperEta(tau=1.50, w_max=2.50, gamma_partial=0.40),
    "ram_connector_recovery": PaperEta(
        beta_prog=0.45, beta_prox=0.30, beta_term=0.10, beta_stop=0.05,
        gamma_correction=1.70, gamma_partial=0.90, w_max=5.00,
    ),
    "balanced_low_clip": PaperEta(tau=0.85, w_max=3.50),
}
