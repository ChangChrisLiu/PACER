# Trajectory-robust PACER scoring

This document defines the current default PACER trajectory-robust diagnostic profile.

The original endpoint/outcome-weighted strict `J_val` remains available through `component_balanced_j_val` and `DEFAULT_SUBMETRIC_WEIGHTS`. The current process-aware PACER calculation uses `robust_profile_j_val` with the default robust profile:

```text
trajectory_robust_component_role_q25_A_plus
```

## Default robust profile: A+

A+ is the current default for trajectory-primary PACER calculation:

```text
family_outcome = 0.20
family_traj    = 0.70
family_reg     = 0.10

outcome    = mean(progress, proximity, terminal)
trajectory = 0.15 * direction + 0.85 * align
row_score  = 0.20 * outcome + 0.70 * trajectory + 0.10 * no_regression
aggregation = lower quartile over component × role cells
```

Hard gates and role handling:

```text
blocked rows score 0:
  wrong_target
  non_target_exclusion
  invalid_orientation
  unsafe

positive scoring roles:
  clean
  correction
  auto_success
  partial

audit-only roles:
  failure
  excluded
```

Why this is the default:

1. It makes whole-chunk EEF/TCP trajectory alignment the primary process-fidelity term.
2. It keeps outcome and no-regression terms nonzero.
3. It requires cross-regime robustness through component×role lower-tail aggregation.
4. It does not hide or manually penalize random-weight controls; every method uses the same rule.

## API

```python
from pacer_framework.validation import robust_profile_j_val

score, rows, manifest = robust_profile_j_val(validation_rows)
assert manifest["profile"] == "trajectory_robust_component_role_q25_A_plus"
```

Equivalent explicit call:

```python
score, rows, manifest = robust_profile_j_val(
    validation_rows,
    profile="trajectory_robust_component_role_q25_A_plus",
)
```

The previous robust profile remains available:

```text
trajectory_robust_component_role_q25_A
```

The stronger sensitivity profile remains available:

```text
trajectory_robust_component_role_q25_B
```

## Computing scores

GitHub tracks the scoring method, not experiment-specific result tables. To compute scores for a candidate set, collect one validation-row list per method and call `robust_profile_j_val` on each list:

```python
from pacer_framework.validation import robust_profile_j_val

scores = {}
manifests = {}
for method_name, validation_rows in rows_by_method.items():
    score, row_scores, manifest = robust_profile_j_val(validation_rows)
    scores[method_name] = score
    manifests[method_name] = manifest

ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
```

For an auditable table, save both the scalar score and the manifest for each method. The manifest records the profile name, family weights, aggregation rule, component scores, component×role cell scores, and audit counts.

```python
assert all(
    manifest["profile"] == "trajectory_robust_component_role_q25_A_plus"
    for manifest in manifests.values()
)
```

## Reporting guidance

Report A+ as a trajectory-robust PACER diagnostic unless it has been frozen before a held-out/protected evaluation. The diagnostic is intended to test whether a method is process-consistent across correction regimes; it is not a substitute for robot success rates.

Use wording like:

```text
We score candidates with the default trajectory-robust PACER profile A+, which keeps outcome and no-regression terms nonzero, makes whole-chunk EEF/TCP trajectory alignment the primary process-fidelity family, scores blocked rows as zero, and aggregates by the lower quartile over component×role cells.
```

Do not treat framework documentation as if it contained a protected result table. Experiment-specific numbers should live in generated evaluation outputs or project-specific reports, not in the framework method definition.
