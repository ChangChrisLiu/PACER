import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def test_policy_inference_mock_smoke(tmp_path):
    out = tmp_path / "inference.json"
    proc = run_cmd(
        "scripts/hardware/run_policy_inference.py",
        "--backend",
        "mock",
        "--mock-observation",
        "--prompt",
        "smoke",
        "--output",
        str(out),
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["backend"] == "mock"
    assert len(payload["result"]["actions"]) == 10
    assert out.exists()


def test_policy_inference_openpi_dry_run():
    proc = run_cmd(
        "scripts/hardware/run_policy_inference.py",
        "--backend",
        "openpi",
        "--mock-observation",
        "--prompt",
        "smoke",
        "--dry-run",
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["result"]["dry_run"] is True
    assert payload["result"]["backend"] == "openpi"


def test_zmq_probe_dry_run():
    proc = run_cmd(
        "scripts/hardware/zmq_agent_probe.py",
        "--endpoint",
        "tcp://127.0.0.1:5555",
        "--request",
        "status",
        "--dry-run",
    )
    payload = json.loads(proc.stdout)
    assert payload["dry_run"] is True
    assert payload["request"]["type"] == "status"
