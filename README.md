# TRACE-VLA

TRACE-VLA = Trace-based Rollout Adaptation with Correction Evidence for VLA policies.

This repository is an early, private, pre-publication GitHub package extracted from the SEA-VLA work. It intentionally starts from the point where a base robot policy already exists (for example a Pi0.5/OpenPI checkpoint). It does **not** include the initial SFT/training-data pipeline. The intended workflow is:

1. Serve or load an existing base policy.
2. Run rollout evaluation on real or replay tasks.
3. Collect correction and/or clean-demonstration evidence after failed or weak rollouts.
4. Compile the resulting evidence into process-aware action chunks.
5. Materialize TRACE-VLA `returns.loss_weight` views.
6. Fine-tune/evaluate candidate eta settings with fixed norm-stat discipline.
7. Use validation scores to propose the next eta setting with BO.

This repo is deliberately private for pre-publication use. Do not treat it as a clean public release yet.

## Repository structure

```text
tracevla/
  stage_b_rl/              # TRACE-VLA core: chunk compiler, eta weights, eval contract, BO
  stage_b_pre/             # rollout/correction/demo schemas and writer utilities
  stage_b_rollout_eval/    # fixed-model rollout evaluation plan/writer utilities
scripts/sea_vla_integration/
  run_stage_b_rollout_eval.py     # SEA-VLA rollout evaluation reference
  run_stage_b_pre_collection.py   # correction/demo collection reference
scripts/hardware/
  run_policy_inference.py         # single policy inference smoke runner
  zmq_agent_probe.py              # JSON ZMQ REQ/REP liveness/status probe
configs/
  pi05_rollout_then_correction.example.yaml
  tracevla_hardware.example.yaml
docs/
  QUICKSTART_PI05.md
  TRACEVLA_ALGORITHM.md          # process-aware eta/weighting details
  DATA_CONTRACT.md
  HARDWARE_SOFTWARE_SETUP.md
  ZMQ_AGENT_CONTRACT.md
  PREPUBLICATION_SCOPE.md
  INTERNAL_HANDOFFS_NOT_INCLUDED.md
examples/
  eta_candidates.json
  minimal_scores.json
tests/
  stage_b_rl/              # lightweight core tests
```

## Quick install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

## Important scope notes

- No raw robot videos, logs, checkpoints, token files, or lab-specific secrets are tracked.
- The SEA-VLA integration scripts are included as reference adapters; they expect the robot/camera/OpenPI runtime from the SEA-VLA stack.
- Validation scores may guide eta selection, but heldout/final test sets must remain untouched until final reporting.
- Method-specific norm stats must be injected at serving/evaluation time for fair fixed-norm comparisons.
- TCP/EEF cosine diagnostics are diagnostic only, not robot success metrics.

## Example workflow

See `docs/QUICKSTART_PI05.md` and `configs/pi05_rollout_then_correction.example.yaml`.
For complete live bring-up, read `docs/HARDWARE_SOFTWARE_SETUP.md`, copy
`configs/tracevla_hardware.example.yaml` to a local untracked config, then probe
policy/ZMQ services before using `--confirm-hardware`.

Useful smoke commands:

```bash
python scripts/hardware/run_policy_inference.py --backend mock --mock-observation --prompt "smoke" --output outputs/inference_smoke.json
python scripts/hardware/run_policy_inference.py --backend openpi --host 127.0.0.1 --port 8000 --prompt "smoke" --mock-observation --dry-run
python scripts/hardware/zmq_agent_probe.py --endpoint tcp://127.0.0.1:5555 --request ping --dry-run
```
