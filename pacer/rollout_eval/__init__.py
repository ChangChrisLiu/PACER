"""PACER rollout-only evaluation package.

This package provides a narrow, planner-only live-eval pipeline that runs trained
PACER model checkpoints against real robot hardware for comparison. It is
NOT a data-collection pipeline: no target_region capture, no human correction
recording, no clean demo, no CSV skill, no corrector_only.

Stop semantics inherit from the PACER collection collection script:
  - manual 's'         -> passive settle + servoStop release   (manual_stop)
  - model stop token   -> passive settle + servoStop release   (model_stop_token)
  - timeout            -> passive settle only                  (timeout)
  - 'q' unsafe abort   -> full stop (speedStop + stopL)        (unsafe_abort)

The operator scores each rollout with a compact 5-label menu; strict_success is
derived as ``label == "success" AND stop_source != "unsafe_abort"``.
"""

ROLLOUT_EVAL_SCHEMA_VERSION = "pacer_rollout_eval.v0.1"

__all__ = ["ROLLOUT_EVAL_SCHEMA_VERSION"]
