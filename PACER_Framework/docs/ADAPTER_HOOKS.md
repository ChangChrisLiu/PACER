# Adapter hooks — connect PACER to your VLA and robot

PACER_Framework is intentionally offline and framework-agnostic. It does not
import ROS, robot SDKs, PyTorch, JAX, OpenPI, LeRobot, or model code. You keep
those dependencies in your project and implement three small hook boundaries
from `pacer_framework/hooks.py`.

## 1. RobotHooks: robot-specific geometry and safety

Implement this for your robot type:

```python
from pacer_framework import RobotHooks, TargetRegion

class MyRobotHooks:
    def target_region(self, component: str, collection_config: str) -> TargetRegion:
        # Load the fixed target you captured before scoring.
        return TargetRegion(
            target_id=f"{collection_config}:{component}",
            component=component,
            point=(x, y, z),              # base-frame TCP target
            radius=r_target,              # your task tolerance
            orientation_tolerance_rad=orientation_tol,
        )

    def integrate_action_chunk(self, *, start_tcp, action_chunk, observation):
        # Convert your VLA action convention to TCP/EEF positions:
        # - joint deltas -> FK per horizon;
        # - EEF deltas -> cumulative base-frame TCP;
        # - absolute waypoints -> validate frame + append;
        # - bimanual/mobile robots -> choose the task TCP used for scoring.
        return [start_tcp, ...]

    def safety_flags(self, trace_or_prediction):
        return {"unsafe": False, "manual_safety_stop": False, "quarantined": False}
```

Common robot types:

| Robot/action type | Hook responsibility |
|---|---|
| EEF-delta arm policy | inverse-normalize deltas and cumulatively add them to the logged start TCP |
| Joint-space arm policy | integrate joint commands, run FK, emit the task TCP in the base frame |
| Absolute waypoint policy | verify frame/units, clip/flag invalid jumps, emit waypoints as TCP positions |
| Bimanual policy | pick the task-relevant TCP per row or emit a derived task TCP before PACER scoring |
| Mobile manipulator | express target and TCP in a common base/world frame before calling PACER |
| Simulator-only setup | use the simulator FK/state API, but still output plain `[x,y,z]` positions |

## 2. VLAHooks: model-specific inference

Implement this for your model API:

```python
from pacer_framework import VLAHooks

class MyVLAHooks:
    def predict_action_chunk(self, *, model_id, observation, prompt=None, deterministic=True):
        # Call OpenVLA/OpenPI/RT-style/diffusion/custom model once.
        # Return the raw or inverse-normalized action chunk expected by RobotHooks.
        return action_chunk

    def stop_emitted(self, prediction):
        # Return whether the candidate emitted the stop/handoff token.
        return bool_stop
```

VLA families this covers:

- action-token models: decode tokens into your action representation;
- diffusion/flow policies: fix the sampling seed/noise for deterministic validation;
- autoregressive chunk policies: return the emitted action horizon and stop token;
- custom PyTorch/JAX policies: return whatever your `RobotHooks` can integrate.

## 3. TrainerHooks: model-specific training

PACER exports plain files:

- `rows.jsonl`: original row payload plus top-level `loss_weight`;
- `manifest.json`: view accounting and loss-normalizer metadata.

Your trainer only needs to implement:

```python
loss = sum_i sum_h loss_weight_i * mask_ih * native_action_loss(i, h) \
       / max(sum_i sum_h loss_weight_i * mask_ih, eps)
```

Optionally wrap it:

```python
from pacer_framework import TrainerHooks

class MyTrainerHooks:
    def train_weighted_view(self, *, view_dir, starting_checkpoint, output_model_id, recipe):
        # Load rows.jsonl + manifest.json, train one adapter/checkpoint, return path/id.
        return output_model_id
```

Keep the starting checkpoint, optimizer, update budget, normalization policy,
and sampling recipe identical across PACER candidates and baselines.

## 4. Where hooks fit in the PACER pipeline

1. `RobotHooks.target_region` defines fixed target regions before data scoring.
2. Your logger records rollout/correction/clean-demo traces into standard rows.
3. PACER computes evidence, gates, scores, weights, and exports training views.
4. `TrainerHooks.train_weighted_view` trains one model per view/candidate.
5. `VLAHooks.predict_action_chunk` + `RobotHooks.integrate_action_chunk` produce
   validation TCP/EEF traces for trained candidates.
6. PACER computes EEF cosine/reference alignment, v_j, J_val, audits, and the
   selected model for hardware evaluation.
