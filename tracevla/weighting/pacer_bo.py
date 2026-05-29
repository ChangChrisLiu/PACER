"""Bayesian optimization helpers for PACER eta search.

This module turns the existing PACER structured eta pool into an explicit
Bayesian-optimization loop: observed (eta, J_B_val) pairs are mapped to the
normalized 8D :class:`PacerEta` search space, a Gaussian-process surrogate is
fit, and Expected Improvement proposes the next eta candidates.

It is intentionally offline-only. It does not train policies or touch hardware.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from tracevla.weighting.pacer_bo_weights import DEFAULT_ETA_POOL, PacerEta
from tracevla.weighting.pacer_eval_contract import validate_pacer_bo_eval_report

PACER_BO_PROPOSAL_SCHEMA_VERSION = "tracevla_bo_proposals.v0.1"


class NoCandidateProposalsError(RuntimeError):
    """Raised when acquisition search has no candidates after filtering."""


@dataclass(frozen=True)
class PacerBoObservation:
    eta_id: str
    eta: PacerEta
    j_b_val: float
    score_path: Path | None = None
    eta_config_path: Path | None = None
    training_run_id: str | None = None

    @property
    def x_normalized(self) -> list[float]:
        return self.eta.to_normalized()


@dataclass(frozen=True)
class PacerBoProposal:
    rank: int
    eta_id: str
    eta: PacerEta
    acquisition_value: float
    predicted_mean: float
    predicted_std: float


@dataclass(frozen=True)
class BayesianOptimizationResult:
    observations: list[PacerBoObservation]
    proposals: list[PacerBoProposal]
    surrogate_model: str
    acquisition_rule: str
    random_state: int
    n_search_samples: int
    xi: float
    y_best: float
    min_distance: float
    kernel_fitted: str
    fit_warnings: list[str]


class PacerGaussianProcessSurrogate:
    """Thin wrapper around sklearn GP with stable metadata."""

    model_name = "GaussianProcessRegressor"

    def __init__(self, model: GaussianProcessRegressor, fit_warnings: list[str] | None = None):
        self.model = model
        self.fit_warnings = fit_warnings or []

    @property
    def kernel_fitted(self) -> str:
        return str(getattr(self.model, "kernel_", self.model.kernel))

    def predict(self, x: np.ndarray, return_std: bool = False):
        return self.model.predict(x, return_std=return_std)


def _coerce_observations(observations: Iterable[PacerBoObservation | tuple[PacerEta, float]]) -> list[PacerBoObservation]:
    out: list[PacerBoObservation] = []
    for idx, item in enumerate(observations):
        if isinstance(item, PacerBoObservation):
            out.append(item)
            continue
        eta, score = item
        out.append(PacerBoObservation(eta_id=f"obs_{idx:03d}", eta=eta, j_b_val=float(score)))
    return out


def _eta_id_from_score_path(score_path: Path) -> str:
    # Typical path: validation_scores_full_val/<method>/validation/scores.json.
    if score_path.parent.name == "validation":
        return score_path.parent.parent.name
    return score_path.parent.name


def _canonical_pool_key(eta_id: str) -> str | None:
    key = eta_id
    for prefix in ("tracevla_pacer_eta_", "tracevla_eta_", "eta_", "tracevla_pacer_", "tracevla_"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    # Full checkpoint dirs also carry job ids: pacer_pacer_eta_rw_like_18634958.
    parts = key.split("_")
    if parts and parts[-1].isdigit():
        key = "_".join(parts[:-1])
    return key if key in DEFAULT_ETA_POOL else None


def _find_eta_config_for_score(score_path: Path) -> Path | None:
    for parent in [score_path.parent, *score_path.parents]:
        candidate = parent / "eta_config.json"
        if candidate.exists():
            return candidate
    return None


def _load_score(score_path: Path) -> tuple[float, str | None]:
    report = json.loads(score_path.read_text())
    errors = validate_pacer_bo_eval_report(report)
    if errors:
        raise ValueError(f"invalid PACER score report {score_path}: {errors}")
    score = report.get("scores", {}).get("J_B_val")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError(f"missing finite scores.J_B_val in {score_path}")
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    return float(score), candidate.get("training_run_id")


def load_bo_observations(root: str | Path, *, include_non_eta: bool = False) -> list[PacerBoObservation]:
    """Load PACER BO observations from eta configs and validation scores.

    The loader supports two layouts:
    1. candidate_dir/eta_config.json + candidate_dir/validation/scores.json;
    2. validation_scores_full_val/<method>/validation/scores.json, with eta
       recovered from DEFAULT_ETA_POOL by method name.

    Non-eta ablations are skipped by default because they are not points in the
    PacerEta 8D search space and should not fit the BO surrogate.
    """
    root = Path(root)
    observations: list[PacerBoObservation] = []
    for score_path in sorted(root.rglob("scores.json")):
        eta_id = _eta_id_from_score_path(score_path)
        eta_config_path = _find_eta_config_for_score(score_path)
        eta: PacerEta | None = None
        if eta_config_path is not None:
            eta = PacerEta.from_dict(json.loads(eta_config_path.read_text()))
        else:
            pool_key = _canonical_pool_key(eta_id)
            if pool_key is not None:
                eta = DEFAULT_ETA_POOL[pool_key]
        if eta is None:
            if include_non_eta:
                continue
            continue
        j_b_val, training_run_id = _load_score(score_path)
        observations.append(
            PacerBoObservation(
                eta_id=eta_id,
                eta=eta,
                j_b_val=j_b_val,
                score_path=score_path,
                eta_config_path=eta_config_path,
                training_run_id=training_run_id,
            )
        )
    return observations


def fit_gp_surrogate(
    observations: Iterable[PacerBoObservation | tuple[PacerEta, float]],
    *,
    random_state: int = 20260528,
) -> PacerGaussianProcessSurrogate:
    """Fit a Gaussian-process surrogate over normalized PacerEta vectors."""
    obs = _coerce_observations(observations)
    if len(obs) < 2:
        raise ValueError("Need at least two eta observations to fit a BO surrogate")
    x = np.asarray([o.x_normalized for o in obs], dtype=np.float64)
    y = np.asarray([o.j_b_val for o in obs], dtype=np.float64)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(x.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=random_state,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    fit_warnings = [str(w.message) for w in caught]
    return PacerGaussianProcessSurrogate(model, fit_warnings=fit_warnings)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, *, y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement for maximization."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    imp = mu - float(y_best) - float(xi)
    out = np.zeros_like(mu, dtype=np.float64)
    mask = sigma > 1e-12
    z = np.zeros_like(mu, dtype=np.float64)
    z[mask] = imp[mask] / sigma[mask]
    out[mask] = imp[mask] * norm.cdf(z[mask]) + sigma[mask] * norm.pdf(z[mask])
    out[~mask] = np.maximum(imp[~mask], 0.0)
    return np.maximum(out, 0.0)


def _candidate_pool(random_state: int, n_search_samples: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    # Include corners and random samples. With only 8D and few observations this
    # is more robust than over-optimizing the acquisition surface.
    random = rng.random((max(0, n_search_samples), len(PacerEta.FREE_DIMS)))
    anchors = np.asarray([eta.to_normalized() for eta in DEFAULT_ETA_POOL.values()], dtype=np.float64)
    center = np.full((1, len(PacerEta.FREE_DIMS)), 0.5, dtype=np.float64)
    return np.vstack([anchors, center, random])


def propose_next_etas(
    observations: Iterable[PacerBoObservation | tuple[PacerEta, float]],
    *,
    n_candidates: int = 4,
    random_state: int = 20260528,
    n_search_samples: int = 4096,
    xi: float = 0.01,
    min_distance: float = 0.025,
) -> BayesianOptimizationResult:
    """Fit GP+EI and propose the next PACER eta candidates."""
    obs = _coerce_observations(observations)
    surrogate = fit_gp_surrogate(obs, random_state=random_state)
    observed_x = np.asarray([o.x_normalized for o in obs], dtype=np.float64)
    y_best = max(o.j_b_val for o in obs)
    candidates = _candidate_pool(random_state=random_state, n_search_samples=n_search_samples)
    if observed_x.size:
        distances = np.linalg.norm(candidates[:, None, :] - observed_x[None, :, :], axis=2).min(axis=1)
        candidates = candidates[distances >= min_distance]
    if len(candidates) == 0:
        raise NoCandidateProposalsError(
            "No candidate proposals remain after observed-point filtering; "
            f"min_distance={min_distance}, n_search_samples={n_search_samples}. "
            "Increase --n-search-samples or lower --min-distance."
        )
    mu, sigma = surrogate.predict(candidates, return_std=True)
    acq = expected_improvement(mu, sigma, y_best=y_best, xi=xi)
    order = np.argsort(-acq)
    proposals: list[PacerBoProposal] = []
    seen: set[tuple[float, ...]] = set()
    for idx in order:
        x = np.clip(candidates[idx], 0.0, 1.0)
        key = tuple(np.round(x, 6))
        if key in seen:
            continue
        seen.add(key)
        eta = PacerEta.from_normalized(x.tolist())
        rank = len(proposals) + 1
        proposals.append(
            PacerBoProposal(
                rank=rank,
                eta_id=f"bo_ei_{rank:02d}_{eta.eta_hash()}",
                eta=eta,
                acquisition_value=float(acq[idx]),
                predicted_mean=float(mu[idx]),
                predicted_std=float(sigma[idx]),
            )
        )
        if len(proposals) >= n_candidates:
            break
    return BayesianOptimizationResult(
        observations=obs,
        proposals=proposals,
        surrogate_model=surrogate.model_name,
        acquisition_rule="expected_improvement",
        random_state=random_state,
        n_search_samples=n_search_samples,
        xi=xi,
        y_best=float(y_best),
        min_distance=float(min_distance),
        kernel_fitted=surrogate.kernel_fitted,
        fit_warnings=list(surrogate.fit_warnings),
    )


def write_bo_proposal_artifacts(result: BayesianOptimizationResult, output_dir: str | Path) -> Path:
    """Write proposal metadata and one eta_config.json per proposed eta."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    proposal_items: list[dict[str, Any]] = []
    observed_vectors = np.asarray([o.x_normalized for o in result.observations], dtype=np.float64)
    for proposal in result.proposals:
        proposal_x = np.asarray(proposal.eta.to_normalized(), dtype=np.float64)
        if len(result.observations) and observed_vectors.size:
            dists = np.linalg.norm(observed_vectors - proposal_x[None, :], axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_observed_eta_id = result.observations[nearest_idx].eta_id
            nearest_observed_distance = float(dists[nearest_idx])
        else:
            nearest_observed_eta_id = None
            nearest_observed_distance = None
        rel_dir = Path("candidates") / proposal.eta_id
        abs_dir = out / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        eta_path = abs_dir / "eta_config.json"
        eta_path.write_text(json.dumps(proposal.eta.to_dict(), indent=2, sort_keys=True))
        proposal_items.append(
            {
                "rank": proposal.rank,
                "eta_id": proposal.eta_id,
                "eta_hash": proposal.eta.eta_hash(),
                "eta": proposal.eta.to_dict(),
                "eta_config_path": str(eta_path.relative_to(out)),
                "normalized_vector": proposal.eta.to_normalized(),
                "acquisition_value": proposal.acquisition_value,
                "predicted_mean": proposal.predicted_mean,
                "predicted_std": proposal.predicted_std,
                "nearest_observed_eta_id": nearest_observed_eta_id,
                "nearest_observed_distance": nearest_observed_distance,
            }
        )
    payload = {
        "schema": PACER_BO_PROPOSAL_SCHEMA_VERSION,
        "surrogate_model": result.surrogate_model,
        "acquisition_rule": result.acquisition_rule,
        "random_state": result.random_state,
        "n_search_samples": result.n_search_samples,
        "xi": result.xi,
        "y_best": result.y_best,
        "min_distance": result.min_distance,
        "kernel_fitted": result.kernel_fitted,
        "fit_warnings": result.fit_warnings,
        "search_dims": list(PacerEta.FREE_DIMS),
        "search_ranges": PacerEta.SEARCH_RANGES,
        "observations": [
            {
                "eta_id": o.eta_id,
                "eta_hash": o.eta.eta_hash(),
                "eta": o.eta.to_dict(),
                "normalized_vector": o.x_normalized,
                "J_B_val": o.j_b_val,
                "score_path": str(o.score_path) if o.score_path else None,
                "eta_config_path": str(o.eta_config_path) if o.eta_config_path else None,
                "training_run_id": o.training_run_id,
            }
            for o in result.observations
        ],
        "proposals": proposal_items,
        "notes": [
            "GP+EI is used to propose next PACER eta candidates from full-val J_B_val observations.",
            "Heldout split must remain untouched until final selected-candidate reporting.",
            "With few initial observations, proposals should be presented as BO-guided candidates, not a converged optimum.",
        ],
    }
    (out / "next_eta_candidates.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out
