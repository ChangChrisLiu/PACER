"""Environment checker for new PACER_Framework users.

Run from the repo or an installed environment:

    python -m pacer_framework.check_setup

The checker verifies the core package and reports optional robot/VLA/training
packages if they happen to be installed. Missing optional packages are not
errors: users install the ones required by their own robot/model stack.
"""
from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class OptionalPackage:
    import_name: str
    purpose: str
    install_hint: str


OPTIONAL_PACKAGES = [
    OptionalPackage("pytest", "run PACER tests", "pip install -e '.[dev]'"),
    OptionalPackage("build", "build wheel/sdist", "pip install -e '.[dev]'"),
    OptionalPackage("zmq", "ZeroMQ data collection from robot/simulator processes", "pip install -e '.[zmq]'"),
    OptionalPackage("torch", "PyTorch VLA training/inference stacks", "install from pytorch.org for your CUDA/CPU setup"),
    OptionalPackage("jax", "JAX/OpenPI-style VLA stacks", "install JAX/OpenPI per their official instructions"),
    OptionalPackage("transformers", "OpenVLA/transformer-style model wrappers", "pip install transformers accelerate"),
    OptionalPackage("lerobot", "LeRobot dataset/training integrations", "install LeRobot per its official instructions"),
    OptionalPackage("rclpy", "ROS 2 robot communication", "install ROS 2; source its setup.bash"),
]


def _available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print("PACER_Framework setup check")
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")
    if sys.version_info < (3, 10):
        print("ERROR: Python >= 3.10 is required")
        return 1

    try:
        import pacer_framework as pf
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(f"ERROR: could not import pacer_framework: {exc}")
        return 1

    print(f"pacer_framework import: OK ({pf.__file__})")
    print("core dependencies: OK (stdlib-only)")
    print("optional packages:")
    for pkg in OPTIONAL_PACKAGES:
        status = "found" if _available(pkg.import_name) else "not installed"
        print(f"  {pkg.import_name:12s} {status:13s} - {pkg.purpose}")
        if status == "not installed":
            print(f"    hint: {pkg.install_hint}")

    print("\nNext steps:")
    print("  1. Read docs/SETUP.md and docs/ADAPTER_HOOKS.md")
    print("  2. Run: python -m pytest tests -q")
    print("  3. Run: python -m pacer_framework.demo --out-dir .local/demo")
    print("  4. Export config: python -m pacer_framework.scoring --write-config .local/scoring.json")
    print("  5. Implement RobotHooks / VLAHooks / TrainerHooks in your own project")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
