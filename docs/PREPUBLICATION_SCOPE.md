# Pre-publication scope

This `TRACE-VLA` package is a GitHub pre-publication extraction from SEA-VLA. It is not the full SEA-VLA repository and is not yet a public release.

Included:

- TRACE-VLA process-aware chunk compilation.
- Eta-weighted `returns.loss_weight` materialization logic.
- Eval-report schema and validation contract.
- Bayesian-optimization proposal logic over eta candidates.
- Rollout/correction/demo schemas and writer/evaluation helpers.
- SEA-VLA hardware integration scripts as reference adapters.

Excluded on purpose:

- Initial SFT collection/training pipeline.
- Raw robot data, videos, checkpoints, LoRA weights, logs, private config, API tokens.
- Final paper results or overclaimed robot-success tables.

Starting assumption:

- You already have a base policy checkpoint, e.g. Pi0.5/OpenPI, plus the runtime needed to serve it. TRACE-VLA begins at rollout, correction evidence collection, and post-rollout weighted improvement.
