# Repository scope

This repository is a reusable PACER framework and runtime-adapter reference. It is intended for users who want to inspect, install, adapt, and test PACER on their own robot, simulator, VLA model, or offline logs.

## Included

- Generic PACER framework code under `PACER_Framework/`.
- Runtime-adapter utilities under `pacer/` and `scripts/`.
- Schemas, validation contracts, and adapter hook definitions.
- Synthetic examples and no-hardware smoke tests.
- Documentation for setup, data contracts, training-view export, evaluation scoring, and runtime integration.

## Excluded

- Raw robot data, videos, checkpoints, LoRA weights, logs, private config, API tokens, and lab-specific paths.
- Initial SFT collection/training infrastructure that belongs to a separate model stack.
- Private handoff notes, scratch reviews, and unpublished result tables.
- Robot drivers, camera drivers, policy-server implementations, or hard real-time control code.

## Starting assumption

You already have a base policy checkpoint and the runtime needed to serve it. PACER begins at rollout/correction/demo recording, process-evidence compilation, weighted-view export, and validation scoring.
