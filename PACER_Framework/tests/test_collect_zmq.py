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


def test_collect_zmq_help_does_not_require_pyzmq():
    result = subprocess.run(
        [sys.executable, "-m", "pacer_framework.collect_zmq", "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        env=_subprocess_env(),
    )
    assert "Collect PACER JSON messages over ZeroMQ" in result.stdout
    assert "--bind" in result.stdout
    assert "--connect" in result.stdout
    assert "--max-messages" in result.stdout
