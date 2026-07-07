import os
import subprocess
import sys
from pathlib import Path


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(part for part in (package_root, existing) if part)
    return env


def test_check_setup_module_runs_successfully():
    result = subprocess.run(
        [sys.executable, "-m", "pacer_framework.check_setup"],
        cwd=".",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        env=_subprocess_env(),
    )
    assert "PACER_Framework setup check" in result.stdout
    assert "pacer_framework import: OK" in result.stdout
    assert "core dependencies: OK" in result.stdout
    assert "optional packages:" in result.stdout
