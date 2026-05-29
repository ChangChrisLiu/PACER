#!/usr/bin/env python3
# TRACE-VLA pre-publication integration script.
# This script is copied from the SEA-VLA robotics stack and expects the robot/camera/OpenPI
# adapters from SEA-VLA to be importable. It is intentionally included as an integration
# reference, not as a standalone hardware driver.

"""Stage-B rollout-only model evaluation entrypoint (T4).

Pairs with three operator-managed terminals:
    T1: launch_robot.py
    T2: launch_cameras.py
    T3: one OpenPI/OpenVLA model server (swapped per model_id)
    T4: this script

This is NOT a data-collection pipeline. There is no target_region capture, no
recovery_correction, no clean_full_demo, and no CSV skill. The operator scores
each rollout with a 5-item label menu; strict_success is derived as
`label == "success" AND stop_source != "unsafe_abort"`.

Dry-run (`--dry-run`) does NOT initialize hardware/model clients. Live runs
require `--confirm-hardware` (mirrors the Stage-B-pre collection gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracevla.stage_b_pre.config import COMPONENT_SEQUENCE, DEFAULT_FPS, DEFAULT_IMAGE_SIZE, DEFAULT_MAX_STEPS
from tracevla.stage_b_rollout_eval.eval_plan import (
    DEFAULT_CONFIGS,
    DEFAULT_TRIALS_PER_COMPONENT,
    EvalProgressTracker,
    ModelSpec,
    build_eval_plan,
    load_model_registry,
    plan_summary,
)
from tracevla.stage_b_rollout_eval.eval_writer import RolloutEvalWriter


DEFAULT_OUTPUT_ROOT = "data/stage_b_rollout_eval"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SEA-VLA Stage-B rollout-only model evaluation"
    )
    ap.add_argument(
        "--run-id",
        required=True,
        help="Operator-supplied run identifier (e.g. 20260601_operator_v1). Stable "
        "across resume; output lives under run_<run_id>/.",
    )

    # Two ways to specify models:
    #   1) --models-file path/to/models.json (preferred for >1 model)
    #   2) --model-id + --model-type + --checkpoint-path (single-model smoke)
    ap.add_argument(
        "--models-file",
        type=Path,
        default=None,
        help="Path to models.json registry. Required for multi-model sweeps.",
    )
    ap.add_argument(
        "--model-id",
        default=None,
        help="Single-model shortcut: model_id for the registry row. Requires "
        "--model-type and --checkpoint-path.",
    )
    ap.add_argument(
        "--model-type",
        default=None,
        choices=["openpi", "openvla", "openvla_oft"],
        help="Single-model shortcut: model type.",
    )
    ap.add_argument(
        "--checkpoint-path",
        default=None,
        help="Single-model shortcut: filesystem path to the checkpoint.",
    )
    ap.add_argument(
        "--unnorm-key",
        default="ur5e_vla_planner_10hz",
        help="Single-model shortcut: OpenPI unnorm key.",
    )
    ap.add_argument(
        "--openpi-base",
        default="droid",
        help="Single-model shortcut: OpenPI base.",
    )

    ap.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated config ids (default config_001..config_005).",
    )
    ap.add_argument(
        "--components",
        default=",".join(COMPONENT_SEQUENCE),
        help="Comma-separated component subset; default cpu_fan,ram,connector,graphic_card,cpu.",
    )
    ap.add_argument(
        "--trials-per-component",
        type=int,
        default=DEFAULT_TRIALS_PER_COMPONENT,
        help="Trials per (model, config, component) cell. Default 10.",
    )
    ap.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for run_<run_id>/. Default data/stage_b_rollout_eval.",
    )

    # Robot / camera / inference args (same names/defaults as collection).
    ap.add_argument("--robot-host", default="127.0.0.1")
    ap.add_argument("--camera-host", default="127.0.0.1")
    ap.add_argument("--robot-port", type=int, default=6000)
    ap.add_argument("--obs-port", type=int, default=6002)
    ap.add_argument("--wrist-camera-port", type=int, default=5000)
    ap.add_argument("--base-camera-port", type=int, default=5001)
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--server-port-default", type=int, default=8000,
                    help="Fallback model-server port when a registry row omits server_port.")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--open-loop-horizon", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    ap.add_argument("--inference-obs-mode", default="and", choices=["and", "parallel"])
    ap.add_argument("--inference-obs-max-wait-ms", type=float, default=50.0)

    # OpenPI execution gripper cap/intercept (defaults match collection script).
    # Default ON for live rollout eval: non-CPU components cap physical gripper
    # commands to 160/255 first, while stop-token-band values are held centrally.
    ap.add_argument("--planner-gripper-cap", dest="planner_gripper_cap", action="store_true", default=True)
    ap.add_argument("--no-planner-gripper-cap", dest="planner_gripper_cap", action="store_false")
    ap.add_argument("--planner-gripper-cap-value", type=float, default=160.0 / 255.0)
    ap.add_argument(
        "--planner-gripper-cap-components",
        default="cpu_fan,ram,connector,graphic_card",
    )

    ap.add_argument("--operator", default="operator")
    ap.add_argument(
        "--mandatory-break-every",
        type=int,
        default=30,
        help="Operator readiness-gate cadence within a model. 0 to disable.",
    )
    ap.add_argument("--disable-safety", action="store_true")

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan + write eval_session_meta.json + models.json. NO hardware.",
    )
    ap.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="Required for live robot/model execution.",
    )
    ap.add_argument(
        "--restart",
        action="store_true",
        help="Destructive: delete progress.json for this run_id before iterating.",
    )

    return ap.parse_args(argv)


def _resolve_models(args: argparse.Namespace) -> list[ModelSpec]:
    if args.models_file:
        return load_model_registry(args.models_file)
    # Single-model shortcut.
    missing = [
        name
        for name, val in (
            ("--model-id", args.model_id),
            ("--model-type", args.model_type),
            ("--checkpoint-path", args.checkpoint_path),
        )
        if not val
    ]
    if missing:
        raise SystemExit(
            f"--models-file is required unless all of these are given: {missing}"
        )
    return [
        ModelSpec(
            model_id=args.model_id,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint_path,
            unnorm_key=args.unnorm_key,
            openpi_base=args.openpi_base,
            server_host=args.server_host,
            server_port=args.server_port_default,
            notes="single-model shortcut",
        )
    ]


def _build_session_meta(
    *,
    args: argparse.Namespace,
    plan_count: int,
    summary: dict[str, Any],
    models: list[ModelSpec],
) -> dict[str, Any]:
    return {
        "schema_version": "stage_b_rollout_eval.v0.1",
        "run_id": args.run_id,
        "operator": args.operator,
        "components": [c.strip() for c in args.components.split(",") if c.strip()],
        "configs": [c.strip() for c in args.configs.split(",") if c.strip()],
        "trials_per_component": int(args.trials_per_component),
        "total_planned_trials": plan_count,
        "plan_summary": summary,
        "models": [m.to_dict() for m in models],
        "runner": {
            "robot_host": args.robot_host,
            "camera_host": args.camera_host,
            "robot_port": args.robot_port,
            "obs_port": args.obs_port,
            "wrist_camera_port": args.wrist_camera_port,
            "base_camera_port": args.base_camera_port,
            "server_host": args.server_host,
            "fps": args.fps,
            "max_steps": args.max_steps,
            "open_loop_horizon": args.open_loop_horizon,
            "image_size": args.image_size,
            "inference_obs_mode": args.inference_obs_mode,
            "inference_obs_max_wait_ms": args.inference_obs_max_wait_ms,
            "gripper_cap_enabled": bool(args.planner_gripper_cap),
            "gripper_cap_value_norm": float(args.planner_gripper_cap_value),
            "gripper_cap_components": [
                c.strip()
                for c in args.planner_gripper_cap_components.split(",")
                if c.strip()
            ],
            "mandatory_break_every": int(args.mandatory_break_every),
            "disable_safety": bool(args.disable_safety),
        },
    }


def print_plan(plan: list, args: argparse.Namespace) -> None:
    print("=" * 72)
    print("SEA-VLA Stage-B rollout-only model evaluation")
    print(f"run_id: {args.run_id}")
    print(f"components: {args.components}")
    print(f"configs: {args.configs}")
    print(f"trials/component: {args.trials_per_component}")
    print(f"output_root: {args.output_root}")
    print("=" * 72)
    summary = plan_summary(plan)
    print(f"total_trials: {summary['total_trials']}")
    print(f"by_model: {summary['by_model']}")
    print(f"by_config: {summary['by_config']}")
    print(f"by_component: {summary['by_component']}")
    print("=" * 72)
    for trial in plan[:20]:
        print(f"{trial.global_index:04d}: {trial.eval_trial_id}")
    if len(plan) > 20:
        print(f"... {len(plan) - 20} more trials")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    models = _resolve_models(args)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    components = [c.strip() for c in args.components.split(",") if c.strip()]

    plan = build_eval_plan(
        run_id=args.run_id,
        models=models,
        configs=configs,
        components=components,
        trials_per_component=args.trials_per_component,
    )

    writer = RolloutEvalWriter(args.output_root, args.run_id)
    summary = plan_summary(plan)
    session_meta = _build_session_meta(
        args=args, plan_count=len(plan), summary=summary, models=models,
    )
    writer.save_session_meta(session_meta)
    writer.save_models_registry(models)

    progress_path = writer.progress_path()
    if args.restart and progress_path.exists():
        progress_path.unlink()
    progress = EvalProgressTracker(progress_path)

    print_plan(plan, args)
    pending = progress.remaining(plan)
    if len(pending) != len(plan):
        print(f"[RESUME] {len(plan) - len(pending)} trials already completed; "
              f"{len(pending)} pending.")

    if args.dry_run:
        print("[DRY-RUN] no hardware initialized. eval_session_meta.json + "
              "models.json written. Exiting.")
        return

    if not args.confirm_hardware:
        raise SystemExit(
            "Refusing live robot control without --confirm-hardware. "
            "Run with --dry-run first to verify the plan."
        )

    # --- LIVE PATH ---------------------------------------------------------
    # Only import hardware/inference modules under the live path so dry-run
    # remains usable on machines without OpenPI/ZMQ installed.
    from scripts.run_inference import build_frame, create_adapter, get_obs  # noqa: E402
    from src.agents.joystick_agent import load_home_pose  # noqa: E402
    from src.agents.safety import SafetyMonitor  # noqa: E402
    from src.agents.vla_agent import VLAAgent  # noqa: E402
    from src.comms.camera_node import ZMQClientCamera  # noqa: E402
    from src.comms.robot_node import ZMQClientRobot  # noqa: E402
    from tracevla.stage_b_rollout_eval.eval_runner import (  # noqa: E402
        HardwareHandles,
        RolloutEvalConfig,
        RolloutEvalRunner,
    )

    # Reuse the collection script's safety / settle / handoff helpers via a
    # tiny adapter so we don't duplicate stop_after_model_rollout semantics.
    # We import LiveCollector lazily because instantiating it would pull in
    # teleop + CSV skill executor. Instead we import the module and call the
    # module-level helpers _move_home_with_event and the per-stop primitives
    # off a thin shim object that exposes the minimal attributes those
    # helpers read (robot, obs_client, args, home_joints, home_gripper).
    import scripts.run_stage_b_pre_collection as collection  # noqa: E402

    robot = ZMQClientRobot(port=args.robot_port, host=args.robot_host)
    obs_client = ZMQClientRobot(port=args.obs_port, host=args.robot_host)
    cameras = {
        "wrist": ZMQClientCamera(
            port=args.wrist_camera_port,
            host=args.camera_host,
            camera_name="wrist",
        ),
        "base": ZMQClientCamera(
            port=args.base_camera_port,
            host=args.camera_host,
            camera_name="base",
        ),
    }
    safety_monitor = None if args.disable_safety else SafetyMonitor()
    home_joints, home_gripper = load_home_pose(Path("configs"))

    eval_config = RolloutEvalConfig(
        server_host=args.server_host,
        model_type=models[0].model_type,  # operator swaps T3 between models
        openpi_base=models[0].openpi_base,
        unnorm_key=models[0].unnorm_key,
        fps=args.fps,
        max_steps=args.max_steps,
        open_loop_horizon=args.open_loop_horizon,
        image_size=args.image_size,
        inference_obs_mode=args.inference_obs_mode,
        inference_obs_max_wait_ms=args.inference_obs_max_wait_ms,
        gripper_cap_enabled=bool(args.planner_gripper_cap),
        gripper_cap_value_norm=float(args.planner_gripper_cap_value),
        gripper_cap_components=tuple(
            c.strip()
            for c in args.planner_gripper_cap_components.split(",")
            if c.strip()
        ),
        disable_safety=bool(args.disable_safety),
    )
    hardware = HardwareHandles(
        robot=robot,
        obs_client=obs_client,
        cameras=cameras,
        safety_monitor=safety_monitor,
        home_joints=home_joints,
        home_gripper=home_gripper,
    )

    # Reuse collection's _move_home_with_event by handing it a shim that
    # carries the attributes the helper reads. The shim is the runner itself
    # plus a `.args` namespace mirroring the model_type so the OpenPI-only
    # servo_stop branch in _release_servo_mode_for_handoff is honored.
    class _CollectionShim:
        """Minimal stand-in for LiveCollector exposing only the attributes
        read by the collection script's stop / move-home helpers. Holds a
        per-call `args.model_type` so the OpenPI-only servoStop release gate
        applies to mixed-type sweeps (e.g. an openvla model_id followed by
        an openpi one)."""

        def __init__(self, runner_self: RolloutEvalRunner, model_type: str) -> None:
            self.robot = runner_self.hardware.robot
            self.obs_client = runner_self.hardware.obs_client
            self.cameras = runner_self.hardware.cameras
            self.home_joints = runner_self.hardware.home_joints
            self.home_gripper = runner_self.hardware.home_gripper
            self.safety = runner_self.hardware.safety_monitor
            self.args = argparse.Namespace(model_type=model_type)

        # Rebind the collection script's bound methods onto the shim.
        def _release_servo_mode_for_handoff(self) -> None:
            collection.LiveCollector._release_servo_mode_for_handoff(self)  # type: ignore[arg-type]

        def _passive_policy_stop_settle(self) -> None:
            collection.LiveCollector._passive_policy_stop_settle(self)  # type: ignore[arg-type]

        def _full_stop(self) -> None:
            collection.LiveCollector._full_stop(self)  # type: ignore[arg-type]

        def _stop_after_model_rollout(self, stop_source: str) -> None:
            collection.LiveCollector._stop_after_model_rollout(self, stop_source)  # type: ignore[arg-type]

    runner = RolloutEvalRunner(
        config=eval_config,
        hardware=hardware,
        writer=writer,
        progress=progress,
        build_frame=build_frame,
        create_adapter=create_adapter,
        get_obs=get_obs,
        vla_agent_factory=lambda adapter, fps, prompt, component, safety_monitor: VLAAgent(
            adapter, fps, prompt, task=component, safety_monitor=safety_monitor,
        ),
        stop_after_rollout=lambda src, model: _CollectionShim(
            runner, model.model_type,
        )._stop_after_model_rollout(src),
        # Pre-trial move_home uses the *previous* model's model_type for the
        # servoStop branch, which is fine because home is preceded by a real
        # stop_after_rollout call from the prior trial that already released
        # servoJ. For the very first trial there is no prior servoJ session.
        move_home=lambda: collection._move_home_with_event(
            _CollectionShim(runner, eval_config.model_type),
        ),
        query_diag_fn=collection._build_query_diag_fn_for_test(),
        operator=args.operator,
        mandatory_break_every=int(args.mandatory_break_every),
    )

    models_by_id = {m.model_id: m for m in models}
    runner.run(plan=pending, models_by_id=models_by_id)


if __name__ == "__main__":
    main()
