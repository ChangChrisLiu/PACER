# TRACE-VLA

TRACE-VLA = Trace-based Rollout Adaptation with Correction Evidence for VLA policies.

This repository is an early, private, pre-publication GitHub package for TRACE-VLA. It intentionally starts from the point where a base robot policy already exists (for example a Pi0.5/OpenPI checkpoint). It does **not** include the initial SFT/training-data pipeline. The intended package workflow is:

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
  weighting/              # TRACE-VLA core: chunk compiler, eta weights, eval contract, BO
  collection/             # rollout/correction/demo schemas and writer utilities
  rollout_eval/    # fixed-model rollout evaluation plan/writer utilities
scripts/robot_runtime/
  run_rollout_eval.py     # TRACE-VLA rollout evaluation reference
  run_correction_collection.py   # correction/demo collection reference
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
  weighting/              # lightweight core tests
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
- The robot-runtime integration scripts are included as reference adapters; live mode expects a compatible robot/camera/OpenPI runtime supplied outside this package.
- Validation scores may guide eta selection, but heldout/final test sets must remain untouched until final reporting.
- Method-specific norm stats must be injected at serving/evaluation time for fair fixed-norm comparisons.
- TCP/EEF cosine diagnostics are diagnostic only, not robot success metrics.

## Example package workflow

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
