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

## Current config_001 all10 diagnostic numbers

These numbers are from the saved strict config_001 all10 artifacts and should be reported as trajectory-robust sensitivity diagnostics unless the profile is frozen before a held-out/protected evaluation.

A+ ranking summary:

```text
rank 1   pacer_pacer_eta_progress_heavy_18634956_fixed_norm_serving             0.519729
rank 2   pacer_pacer_eta_correction_heavy_18634954_fixed_norm_serving           0.518210
rank 3   pacer_pacer_eta_rw_like_18634958_fixed_norm_serving                    0.518210
rank 4   pacer_pacer_eta_hover_proximity_heavy_18634955_fixed_norm_serving      0.511436
rank 5   pacer_pacer_eta_conservative_18634953_fixed_norm_serving               0.511063
rank 6   pacer_pacer_eta_terminal_stop_heavy_18634959_fixed_norm_serving        0.510535
rank 7   pacer_pacer_eta_ram_connector_recovery_18634957_fixed_norm_serving     0.509901
rank 8   pacer_pacer_eta_balanced_low_clip_18634952_fixed_norm_serving          0.509848
rank 9   pacer_random_weight_18634960_fixed_norm_serving                        0.504061
rank 10  pacer_fixed_rw_fma_18634951_fixed_norm_serving                         0.495619
rank 11  pacer_uniform_replay_18634961_fixed_norm_serving                       0.470490
rank 12  pacer_sftpp_clean_correction_demo_18634962_fixed_norm_serving          0.401049
rank 13  pacer_outcome_only_18632502_fixed_norm_serving                         0.381767
rank 14  pacer_clean_demo_only_18634949_fixed_norm_serving                      0.357204
rank 15  pacer_correction_only_18634950_fixed_norm_serving                      0.213213
```

Summary:

```text
8 PACER η variants outperform fixed RW-FMA, SFT++, and random-weight under A+.
random_weight remains visible in the same ranking and is not penalized by name.
```

## Caveat

A+ was chosen after inspecting config_001 all10 artifacts. Unless it is frozen and evaluated on a held-out/protected split, call these numbers:

```text
trajectory-robust sensitivity diagnostics
```

Do not call them protected validation or robot-success proof.
