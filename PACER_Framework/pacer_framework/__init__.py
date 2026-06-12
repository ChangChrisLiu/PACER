"""PACER framework — robot/VLA-agnostic process-aware post-training.

Reference implementation of the PACER paper main text and appendix:
targets -> rollout/correction/clean-demo records -> standard rows -> evidence,
gates, and weights -> eight training views -> external VLA training -> offline
open-loop evaluation (EEF cosine, v_j, J_val, audits) -> hardware candidate
selection. See README.md for the pipeline and the claim-to-code map. The
package is offline-only: no robot, model, or hardware imports.
"""
from pacer_framework.roles import (
    AUDIT_ONLY_ROLES,
    COMPILER_TO_PAPER,
    OPERATOR_LABELS,
    PAPER_ROLES,
    POSITIVE_IMITATION_ROLES,
    normalize_role,
    role_from_source_and_label,
    strict_success,
)
from pacer_framework.eta import CANDIDATE_POOL, EVIDENCE_KEYS, PaperEta, process_score
from pacer_framework.evidence import (
    compute_evidence,
    d_ref_for,
    provenance_check,
    psi_op,
    r_target_for,
    stop_evidence,
    target_consistency,
)
from pacer_framework.gate import eligibility_gate, mask_has_mass
from pacer_framework.schema import (
    training_row_template,
    training_row_warnings,
    validate_training_row,
    validate_validation_row,
    validation_row_template,
)
from pacer_framework.split import assign_case_group_split
from pacer_framework.weights import compile_weights, robust_center_scale, row_weight
from pacer_framework.alignment import (
    blocker_flags_from_open_loop,
    chunk_direction,
    chunk_trajectory_alignment,
    direction_cosine01,
    open_loop_submetrics,
    reference_alignment,
    target_direction_cosine,
    with_no_regression,
)
from pacer_framework.validation import (
    DEFAULT_SUBMETRIC_WEIGHTS,
    SUBMETRIC_ALIASES,
    VALIDATION_SUBMETRICS,
    candidate_feasibility,
    component_balanced_j_val,
    evaluate_candidate,
    geometric_row_score,
    no_regression,
    row_score,
    select_candidate,
)
from pacer_framework.baselines import BASELINE_MODES, compile_baseline
from pacer_framework.reporting import (
    candidate_ranking_table,
    paired_component_bootstrap,
    success_table,
    wilson_interval,
)
from pacer_framework.export import export_training_view, read_jsonl, write_jsonl
from pacer_framework.hooks import RobotHooks, TargetRegion, TrainerHooks, VLAHooks, validate_target_region

__all__ = [
    "AUDIT_ONLY_ROLES",
    "BASELINE_MODES",
    "CANDIDATE_POOL",
    "COMPILER_TO_PAPER",
    "DEFAULT_SUBMETRIC_WEIGHTS",
    "EVIDENCE_KEYS",
    "OPERATOR_LABELS",
    "PAPER_ROLES",
    "POSITIVE_IMITATION_ROLES",
    "RobotHooks",
    "SUBMETRIC_ALIASES",
    "TargetRegion",
    "TrainerHooks",
    "VALIDATION_SUBMETRICS",
    "VLAHooks",
    "PaperEta",
    "assign_case_group_split",
    "blocker_flags_from_open_loop",
    "candidate_feasibility",
    "candidate_ranking_table",
    "chunk_direction",
    "chunk_trajectory_alignment",
    "compile_baseline",
    "compile_weights",
    "component_balanced_j_val",
    "compute_evidence",
    "d_ref_for",
    "direction_cosine01",
    "eligibility_gate",
    "evaluate_candidate",
    "export_training_view",
    "geometric_row_score",
    "mask_has_mass",
    "no_regression",
    "normalize_role",
    "open_loop_submetrics",
    "paired_component_bootstrap",
    "process_score",
    "provenance_check",
    "psi_op",
    "r_target_for",
    "read_jsonl",
    "reference_alignment",
    "robust_center_scale",
    "role_from_source_and_label",
    "row_score",
    "row_weight",
    "select_candidate",
    "stop_evidence",
    "strict_success",
    "success_table",
    "target_consistency",
    "target_direction_cosine",
    "training_row_template",
    "training_row_warnings",
    "validate_training_row",
    "validate_validation_row",
    "validate_target_region",
    "validation_row_template",
    "wilson_interval",
    "with_no_regression",
    "write_jsonl",
]
