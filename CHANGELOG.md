# Release notes

## Unreleased / next minor release

- Rollout-only evaluation gains an optional `--auto-judge` mode
  ([docs/AUTO_TRIAL_JUDGE.md](docs/AUTO_TRIAL_JUDGE.md)). Each predicted chunk is
  scored with PACER open-loop geometry before execution and saved to
  `chunk_scores.jsonl`. Trials stop as `auto_success_stop` or `auto_blocker_stop`
  and are labelled automatically, with `needs_review` notes near thresholds.
  Targets per board position live in a JSON file
  (`python -m pacer.rollout_eval.auto_judge_targets`); obstacle regions are
  opt-in per target, and each trial records `auto_judge_context.json`. The mode
  accepts OpenPI absolute-joint models only. UR5e forward kinematics
  and an episode check are in `pacer.rollout_eval.ur5e_kinematics`. Manual mode
  is unchanged.
- `TracedChunkRunner` `stop_check` may return a stop-source string; `True` still
  means `model_stop_token`. Query traces record `extra.stop_check_source`.

- The shared default validation route is direct component-balanced mean.
  `robust_profile_j_val` without a profile no longer selects a factor-family
  lower-tail diagnostic. Explicit named diagnostics remain replayable.
- Full candidate audits require explicit boolean blocker inputs; missing safety
  observations no longer pass as clear flags. Overflowing coefficient sums are
  rejected before they can create nonfinite intermediate scores. Nonfinite or
  malformed EEF/TCP geometry is rejected instead of becoming a clipped score.
- `ScoringConfig` provides validated JSON/API configuration, an immutable copy of
  coefficients and a reproducible output fingerprint. The same configuration
  flows through geometry, reference comparison, scoring and candidate audits.
- Whole-chunk trajectory alignment is the construction default. Endpoint mode is
  explicit. Short matched trajectories raise rather than changing the metric set.
- Matched `reference_validation_rows` rebuild both candidate and reference scores
  under one configuration. Missing reference component scores fail closed by
  default; a diagnostic opt-out must be explicit and is marked as skipped.
  Conflicting embedded and separately supplied reference geometry is rejected.
- Rows may retain `reference_submetrics` for consistent no-regression rescoring.
  Cached, recomputed and disagreeing indicators are reported. Missing expected
  geometry cannot produce a winning default score; malformed inputs fail loudly.
- The runtime row scorer delegates to the canonical framework, including its
  normalization epsilon, aliases and conservative blocker interpretation.
  Integer unsafe flags no longer bypass the canonical blocker. Recompute tables
  after changing protocol/version rather than mixing old and new scalars.
- The full runtime distribution includes the same framework source as the
  minimal distribution. Install exactly one per environment. Both expose
  installed `pacer-score` and `pacer-demo` commands.

Training eta compilation, policy losses, model integration and hardware-control
behavior are unchanged. Bring application-specific geometry, validation data
and robot/model hooks; the bundled demonstration is synthetic.
