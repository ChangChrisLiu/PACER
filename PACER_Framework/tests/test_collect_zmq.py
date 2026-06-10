import subprocess
import sys


def test_collect_zmq_help_does_not_require_pyzmq():
    result = subprocess.run(
        [sys.executable, "-m", "pacer_framework.collect_zmq", "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "Collect PACER JSON messages over ZeroMQ" in result.stdout
    assert "--bind" in result.stdout
    assert "--connect" in result.stdout
    assert "--max-messages" in result.stdout
