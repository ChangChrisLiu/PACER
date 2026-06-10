import subprocess
import sys


def test_check_setup_module_runs_successfully():
    result = subprocess.run(
        [sys.executable, "-m", "pacer_framework.check_setup"],
        cwd=".",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "PACER_Framework setup check" in result.stdout
    assert "pacer_framework import: OK" in result.stdout
    assert "core dependencies: OK" in result.stdout
    assert "optional packages:" in result.stdout
