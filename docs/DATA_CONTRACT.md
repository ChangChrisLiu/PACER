# Data contract

This repo keeps schemas and lightweight utilities, not raw robot data.

## Rollout evaluation

A rollout trial should provide:

- model id and checkpoint metadata
- task/config/component identifiers
- per-step policy queries and executed action traces
- operator/verifier outcome feedback
- optional target/EEF diagnostic features

## Correction / clean demonstration

Correction evidence should distinguish:

- model-produced failure or weak segment
- human or corrector intervention segment
- clean demonstration segment, if collected separately
- stop-token emission versus stop-token acceptance

## TRACE-VLA weighted view

A materialized weighted training view should expose:

```json
{
  "returns": {
    "loss_weight": 1.0
  },
  "tracevla": {
    "eta_id": "tracevla_eta_rw_like",
    "stratum": "planner_only/cpu_fan",
    "evidence": {}
  }
}
```

The exact dataset storage backend can be LeRobot/OpenPI or another policy-training stack, but the train/validation/heldout split discipline must be explicit.
