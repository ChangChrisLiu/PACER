"""RolloutEvalRunner — planner-side only, no correction/demo/target-region.

This class is structured so the unit tests can drive it without any hardware
import. The hardware-touching pieces (robot client, camera clients, adapter,
obs/build_frame functions) are constructor inputs supplied either by the
production wiring in ``scripts/run_rollout_eval.py`` or by fakes in
``tests/rollout_eval/``.

The rollout body mirrors the safe primitives used by
``LiveCollector._run_model_segment``:
  - TracedChunkRunner produces frames + action_trace
  - manual_stop() reads `s` -> manual_stop / `q` -> unsafe_abort
  - stop-after-rollout uses the same passive-settle / servo-release branches
    that the collection script proved out in V0.3/V0.5/V0.6
  - operator scoring is the only post-rollout step (no correction, no demo)

Critical guard rails:
  - `q` (unsafe_abort) does NOT auto-continue. The runner asks the operator to
    restart the robot server, then for an explicit "ok" before the next trial.
  - Between models, the operator confirms the T3 model server swap with the
    same "ok" gate so we never silently fire the next model's first trial.
"""
from __future__ import annotations

import argparse
import contextlib
import select
import sys
import termios
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pacer.collection.config import TASK_INSTRUCTIONS
from pacer.collection.inference_runner import (
    GripperCapConfig,
    RolloutResult,
    TracedChunkRunner,
)

from .eval_feedback import (
    EvalFeedback,
    UNSAFE_ABORT_AUTO_LABEL,
    prompt_eval_label,
    wait_for_operator_ready,
)
from .eval_plan import EvalProgressTracker, EvalTrialSpec, ModelSpec
from .eval_writer import RolloutEvalWriter


def _stdin_ready() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


@contextlib.contextmanager
def _raw_terminal_if_possible():
    """Make stdin character-buffered so `s` / `q` are read without Enter.

    Mirrors the helper in `run_correction_collection.py`. Non-TTY callers
    (tests, subprocesses) get a no-op context.
    """
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


@dataclass
class RolloutEvalConfig:
    """All non-model knobs the rollout body needs.

    Mirrors the subset of `LiveCollector.args` that `_run_model_segment` reads:
    server_host (model-server host), model_type, fps, max_steps,
    open_loop_horizon, image_size, inference obs mode + max wait,
    gripper-cap configuration.
    """

    server_host: str = "127.0.0.1"
    model_type: str = "openpi"
    openpi_base: str = "droid"
    unnorm_key: str = "ur5e_vla_planner_10hz"
    fps: int = 10
    max_steps: int = 600
    open_loop_horizon: int = 10
    image_size: int = 256
    inference_obs_mode: str = "and"
    inference_obs_max_wait_ms: float = 50.0
    gripper_cap_enabled: bool = False
    gripper_cap_value_norm: float = 160.0 / 255.0
    gripper_cap_components: tuple[str, ...] = (
        "cpu_fan",
        "ram",
        "connector",
        "graphic_card",
    )
    disable_safety: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_host": self.server_host,
            "model_type": self.model_type,
            "openpi_base": self.openpi_base,
            "unnorm_key": self.unnorm_key,
            "fps": self.fps,
            "max_steps": self.max_steps,
            "open_loop_horizon": self.open_loop_horizon,
            "image_size": self.image_size,
            "inference_obs_mode": self.inference_obs_mode,
            "inference_obs_max_wait_ms": self.inference_obs_max_wait_ms,
            "gripper_cap_enabled": bool(self.gripper_cap_enabled),
            "gripper_cap_value_norm": float(self.gripper_cap_value_norm),
            "gripper_cap_components": list(self.gripper_cap_components),
            "disable_safety": bool(self.disable_safety),
        }


@dataclass
class HardwareHandles:
    """Hardware/client handles the runner needs. Tests pass fakes."""

    robot: Any
    obs_client: Any
    cameras: dict[str, Any]
    safety_monitor: Any | None
    home_joints: Any
    home_gripper: float


# Type aliases mirroring LiveCollector._run_model_segment hooks.
BuildFrameFn = Callable[[dict[str, Any], dict[str, Any], int], dict[str, Any]]
GetObsFn = Callable[..., dict[str, Any]]
CreateAdapterFn = Callable[[argparse.Namespace], Any]
# Stop-after needs the active model so the OpenPI-gated servoStop release
# can be applied per-model (mixed-type sweeps are not blocked).
MoveHomeFn = Callable[[], dict[str, Any] | None]
StopAfterFn = Callable[[str, ModelSpec], None]
PreRolloutGateFn = Callable[..., None]


def wait_for_pre_rollout_enter(
    *,
    trial: EvalTrialSpec,
    model: ModelSpec,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Pause after home/reset and before autonomous model rollout.

    This is an operator visual-check gate only: it does not touch T1 / robot
    hardware. Unlike the post-unsafe-abort and model-swap readiness gates,
    the operator requested a simple Enter press to start the next trial.
    """
    print_fn(
        "[PRE-ROLLOUT CHECK] Home/reset finished. Inspect home pose, gripper, "
        f"and scene for {trial.eval_trial_id} (model={model.model_id})."
    )
    input_fn("Press Enter to start this rollout, or Ctrl+C to abort: ")


@dataclass
class RolloutEvalRunner:
    """Drives one trial-by-trial pass over an eval plan.

    `inference_utils` callables and `move_home` / `stop_after_rollout` /
    `vla_agent_factory` are injected; production wiring lives in
    ``scripts/run_rollout_eval.py``. Tests supply fakes so the
    runner exercises plan iteration / feedback routing / writer output
    without hardware.
    """

    config: RolloutEvalConfig
    hardware: HardwareHandles
    writer: RolloutEvalWriter
    progress: EvalProgressTracker
    build_frame: BuildFrameFn
    create_adapter: CreateAdapterFn
    get_obs: GetObsFn
    vla_agent_factory: Callable[..., Any]
    stop_after_rollout: StopAfterFn
    move_home: MoveHomeFn
    prompt_label: Callable[..., EvalFeedback] = field(default=prompt_eval_label)
    readiness_gate: Callable[..., None] = field(default=wait_for_operator_ready)
    pre_rollout_gate: PreRolloutGateFn = field(default=wait_for_pre_rollout_enter)
    query_diag_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    chunk_runner_factory: Callable[..., TracedChunkRunner] | None = None
    raw_terminal: Callable[[], Any] = field(default=_raw_terminal_if_possible)
    manual_stop_reader: Callable[[], str | None] | None = None
    operator: str = "operator"
    clock: Callable[[], float] = field(default=time.time)
    mandatory_break_every: int = 30
    print_fn: Callable[[str], None] = field(default=print)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        plan: list[EvalTrialSpec],
        models_by_id: dict[str, ModelSpec],
    ) -> None:
        """Iterate the eval plan, running each pending trial.

        Iteration order is the plan's natural model -> config -> component
        -> trial sequence. Resume skips trial ids already in progress.json.
        At each model boundary the operator is prompted to confirm the T3
        swap before continuing.
        """
        completed_this_session = 0
        last_model_id: str | None = None
        for trial in plan:
            if self.progress.is_completed(trial):
                continue
            if trial.model_id != last_model_id:
                if last_model_id is not None:
                    self.readiness_gate(
                        prompt=(
                            f"[MODEL SWAP] previous model={last_model_id!r}. "
                            f"Switch T3 to {trial.model_id!r} "
                            f"(checkpoint: {models_by_id[trial.model_id].checkpoint_path}) "
                            f"and confirm before continuing."
                        ),
                        print_fn=self.print_fn,
                    )
                last_model_id = trial.model_id

            if (
                completed_this_session
                and self.mandatory_break_every
                and completed_this_session % self.mandatory_break_every == 0
            ):
                self.readiness_gate(
                    prompt="Mandatory break: inspect robot/cables/scene, then confirm to continue.",
                    print_fn=self.print_fn,
                )

            self.print_fn("\n" + "=" * 72)
            self.print_fn(
                f"TRIAL {trial.global_index}: {trial.eval_trial_id}"
            )
            self.print_fn("=" * 72)

            model = models_by_id[trial.model_id]
            try:
                stop_source = self._run_one_trial(trial, model)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.progress.mark_failed(trial, repr(exc))
                self.print_fn(f"[ERROR] trial failed: {exc!r}")
                self.readiness_gate(
                    prompt=(
                        "Trial errored. Inspect robot/cell. Confirm to "
                        "continue with the next trial."
                    ),
                    print_fn=self.print_fn,
                )
                continue

            # Per the spec, an unsafe_abort never auto-continues: the
            # robot-server may be dead and the move_home that opens the
            # next trial would crash. Prompt for an explicit readiness gate.
            if stop_source == "unsafe_abort":
                self.readiness_gate(
                    prompt=(
                        "[UNSAFE ABORT] launch_robot.py (T1) may have stopped. "
                        "If needed: restart T1, re-launch cameras (T2), restart "
                        "the model server (T3), and only then confirm readiness."
                    ),
                    print_fn=self.print_fn,
                )

            self.progress.mark_completed(trial)
            completed_this_session += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_runner(self, model: ModelSpec) -> TracedChunkRunner:
        if self.chunk_runner_factory is not None:
            return self.chunk_runner_factory(self.config, model)
        return TracedChunkRunner(
            fps=self.config.fps,
            max_steps=self.config.max_steps,
            open_loop_horizon=self.config.open_loop_horizon,
            model_type=model.model_type,
            checkpoint_name=f"{model.model_id}:{model.server_port}",
            adapter_action_format=(
                "openpi_absolute_joint"
                if model.model_type == "openpi"
                else "eef_delta"
            ),
            gripper_cap=GripperCapConfig(
                enabled=bool(self.config.gripper_cap_enabled),
                value_norm=float(self.config.gripper_cap_value_norm),
                components=tuple(self.config.gripper_cap_components),
            ),
        )

    def _default_manual_stop(self) -> str | None:
        if not _stdin_ready():
            return None
        try:
            ch = sys.stdin.read(1)
        except Exception:
            return None
        if ch == "s":
            return "manual_stop"
        if ch == "q":
            return "unsafe_abort"
        return None

    def _run_one_trial(self, trial: EvalTrialSpec, model: ModelSpec) -> str:
        """Run one trial: pre-trial home, model rollout, stop, score, save.

        Returns the rollout `stop_source` so the caller can route post-trial
        flow (e.g. unsafe_abort readiness gate).
        """
        self.move_home()
        self.pre_rollout_gate(
            trial=trial,
            model=model,
            print_fn=self.print_fn,
        )

        prompt = TASK_INSTRUCTIONS[trial.component]
        # Build per-trial adapter (mirrors collection script line 2716, which
        # also rebuilds per `_run_model_segment` call). The adapter args
        # carry server_host/server_port for the model-server this model runs
        # against; everything else is inherited from the runner config.
        adapter_args = argparse.Namespace(
            server_host=self.config.server_host,
            server_port=model.server_port,
            model_type=model.model_type,
            openpi_base=model.openpi_base or self.config.openpi_base,
            unnorm_key=model.unnorm_key,
            image_size=self.config.image_size,
            fps=self.config.fps,
            max_steps=self.config.max_steps,
            open_loop_horizon=self.config.open_loop_horizon,
            inference_obs_mode=self.config.inference_obs_mode,
            inference_obs_max_wait_ms=self.config.inference_obs_max_wait_ms,
        )
        adapter = self.create_adapter(adapter_args)
        vla_agent = self.vla_agent_factory(
            adapter=adapter,
            fps=self.config.fps,
            prompt=prompt,
            component=trial.component,
            safety_monitor=self.hardware.safety_monitor,
        )

        runner = self._build_runner(model)

        def obs_fn() -> dict[str, Any]:
            return self.get_obs(
                self.hardware.obs_client,
                self.hardware.cameras,
                obs_mode=self.config.inference_obs_mode,
                max_wait_s=self.config.inference_obs_max_wait_ms / 1000.0,
            )

        def infer_fn(obs: dict[str, Any], text_prompt: str) -> list[Any]:
            return adapter.infer(obs, text_prompt)

        def apply_fn(action: Any, obs: dict[str, Any]) -> Any:
            return adapter.apply_action(
                action, obs, self.hardware.robot, self.hardware.safety_monitor
            )

        def frame_fn(obs: dict[str, Any]) -> dict[str, Any]:
            return self.build_frame(obs, self.hardware.cameras, self.config.image_size)

        def stop_fn(chunk: list[Any], obs: dict[str, Any]) -> bool:
            current_state = adapter.get_current_state(obs)
            return bool(vla_agent._check_chunk_stop(chunk, current_state))  # noqa: SLF001

        manual_stop_fn = self.manual_stop_reader or self._default_manual_stop

        started_at = float(self.clock())
        with self.raw_terminal():
            result: RolloutResult = runner.run(
                component=trial.component,
                prompt=prompt,
                phase="teleop",
                get_obs=obs_fn,
                infer=infer_fn,
                apply_action=apply_fn,
                build_frame=frame_fn,
                stop_check=stop_fn,
                manual_stop=manual_stop_fn,
                query_diag_fn=self.query_diag_fn,
            )
        self.stop_after_rollout(result.stop_source, model)
        ended_at = float(self.clock())

        final_tcp: list[float] | None = None
        if result.stop_source != "unsafe_abort":
            # After unsafe_abort the RTDE script may be dead — avoid touching it.
            try:
                pose = self.hardware.robot.get_tcp_pose_raw()
                final_tcp = list(np.asarray(pose, dtype=float).reshape(-1))
            except Exception:
                final_tcp = None

        feedback = self.prompt_label(
            stop_source=result.stop_source,
            operator=self.operator,
        )

        self.writer.save_trial(
            trial=trial,
            model=model,
            frames=result.frames,
            action_trace=result.action_trace,
            feedback=feedback,
            stop_source=result.stop_source,
            steps=result.steps,
            started_at_ts=started_at,
            ended_at_ts=ended_at,
            wall_seconds=max(ended_at - started_at, 0.0),
            final_tcp=final_tcp,
            runner_args={
                **self.config.to_dict(),
                "language_instruction": prompt,
                "gripper_cap_components": list(self.config.gripper_cap_components),
            },
        )
        # Defensive: if the auto-assigned label disagrees with the actual stop
        # source, we still trust the operator menu output. But the unsafe_abort
        # auto-route must never produce a non-unsafe label.
        if result.stop_source == "unsafe_abort" and feedback.label != UNSAFE_ABORT_AUTO_LABEL:
            raise RuntimeError(
                "unsafe_abort rollout produced non-unsafe label; "
                f"got {feedback.label!r} (expected {UNSAFE_ABORT_AUTO_LABEL!r})"
            )

        return result.stop_source
