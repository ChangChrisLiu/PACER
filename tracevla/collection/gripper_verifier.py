"""Component/event-specific gripper verification helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

GripperRuleName = Literal["obs_lt_threshold", "obs_le_threshold", "gap_ge_threshold", "unknown"]

# Corrector-only physical-contact validation points (raw Robotiq observation).
# The user-facing rule is strict: success iff obs < component threshold.
# - RAM keeps the previous validation point: obs < 227.
# - Connector / CPU fan / graphic card: obs < 220.
# - CPU: obs < 165.
RAM_CONTACT_OBS_THRESHOLD = 227
DEFAULT_NON_CPU_CONTACT_OBS_THRESHOLD = 220
CPU_CONTACT_OBS_THRESHOLD = 165


def _obs_threshold_description(threshold: int) -> str:
    return (
        "Corrector verification is observation-threshold based: "
        f"obs<{threshold} indicates blocked closure/contact; "
        f"obs>={threshold} means empty/full close for this component. "
        "Model action values near 255 may encode synthesized stop-token intent, "
        "so cmd and gap are telemetry and must not gate obs-threshold success."
    )


@dataclass(frozen=True)
class GripperRule:
    component: str
    rule: GripperRuleName
    desired_cmd: int | None = None
    cmd_tolerance: int | None = None
    obs_threshold: int | None = None
    gap_threshold: int | None = None
    description: str = ""


@dataclass(frozen=True)
class GripperVerification:
    component: str
    event: str
    cmd: int | None
    obs: int | None
    gap: int | None
    rule: str
    success: bool | None
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


RULES: dict[str, GripperRule] = {
    "ram": GripperRule(
        component="ram",
        rule="obs_lt_threshold",
        obs_threshold=RAM_CONTACT_OBS_THRESHOLD,
        description=_obs_threshold_description(RAM_CONTACT_OBS_THRESHOLD),
    ),
    "connector": GripperRule(
        component="connector",
        rule="obs_lt_threshold",
        obs_threshold=DEFAULT_NON_CPU_CONTACT_OBS_THRESHOLD,
        description=_obs_threshold_description(DEFAULT_NON_CPU_CONTACT_OBS_THRESHOLD),
    ),
    "cpu_fan": GripperRule(
        component="cpu_fan",
        rule="obs_lt_threshold",
        obs_threshold=DEFAULT_NON_CPU_CONTACT_OBS_THRESHOLD,
        description=_obs_threshold_description(DEFAULT_NON_CPU_CONTACT_OBS_THRESHOLD),
    ),
    "graphic_card": GripperRule(
        component="graphic_card",
        rule="obs_lt_threshold",
        obs_threshold=DEFAULT_NON_CPU_CONTACT_OBS_THRESHOLD,
        description=_obs_threshold_description(DEFAULT_NON_CPU_CONTACT_OBS_THRESHOLD),
    ),
    "cpu": GripperRule(
        component="cpu",
        rule="obs_lt_threshold",
        obs_threshold=CPU_CONTACT_OBS_THRESHOLD,
        description=_obs_threshold_description(CPU_CONTACT_OBS_THRESHOLD),
    ),
}


def verify_gripper(component: str, cmd: int | None, obs: int | None, event: str = "verify") -> GripperVerification:
    rule = RULES.get(component, GripperRule(component=component, rule="unknown", description="No rule configured."))
    gap = None if cmd is None or obs is None else int(cmd) - int(obs)
    success: bool | None
    cmd_ok: bool | None
    if rule.desired_cmd is None:
        cmd_ok = True
    elif cmd is None:
        cmd_ok = None
    else:
        tol = 0 if rule.cmd_tolerance is None else int(rule.cmd_tolerance)
        cmd_ok = abs(int(cmd) - int(rule.desired_cmd)) <= tol

    if rule.rule == "obs_lt_threshold":
        obs_ok = None if obs is None or rule.obs_threshold is None else int(obs) < rule.obs_threshold
        # Observation-threshold rules are calibrated on the recorded physical
        # observation. Keep cmd/gap in the verification payload for telemetry,
        # but do not let the model action channel gate success: in planner and
        # corrector data, values near 1.0/255 can encode synthesized stop-token
        # intent rather than the skill executor's calibrated Robotiq command.
        success = None if obs_ok is None else bool(obs_ok)
    elif rule.rule == "obs_le_threshold":
        # Historical compatibility for any legacy rule object injected by a
        # test or downstream caller. Current TRACE-VLA rules use strict <.
        obs_ok = None if obs is None or rule.obs_threshold is None else int(obs) <= rule.obs_threshold
        success = None if obs_ok is None else bool(obs_ok)
    elif rule.rule == "gap_ge_threshold":
        gap_ok = None if gap is None or rule.gap_threshold is None else gap >= rule.gap_threshold
        success = None if gap_ok is None or cmd_ok is None else bool(gap_ok and cmd_ok)
    else:
        success = None
    return GripperVerification(
        component=component,
        event=event,
        cmd=cmd,
        obs=obs,
        gap=gap,
        rule=rule.rule,
        success=success,
        description=rule.description,
    )
