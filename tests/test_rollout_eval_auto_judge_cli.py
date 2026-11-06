import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/robot_runtime/run_rollout_eval.py"
BASE = ["--run-id", "aj", "--model-id", "m1", "--model-type", "openpi", "--checkpoint-path", "ck",
        "--configs", "config_001", "--components", "ram,cpu", "--trials-per-component", "2"]


def run_script(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_auto_judge_requires_targets_and_r_near(tmp_path):
    proc = run_script(*BASE, "--output-root", str(tmp_path), "--dry-run", "--auto-judge",
                      "--auto-judge-r-near-m", "0.05")
    assert proc.returncode != 0
    assert "--auto-judge-targets" in proc.stderr

    proc = run_script(*BASE, "--output-root", str(tmp_path), "--dry-run", "--auto-judge",
                      "--auto-judge-targets", str(tmp_path / "targets.json"))
    assert proc.returncode != 0
    assert "--auto-judge-r-near-m" in proc.stderr


def test_dry_run_records_auto_judge_meta_and_missing_cells(tmp_path):
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({
        "schema": "pacer_auto_judge_targets.v1",
        "targets": {"config_001/cpu": {"points": [[0.4, -0.1, 0.05, 3.14, 0.0, 0.0]], "source": "tcp_pose"}},
    }))
    proc = run_script(*BASE, "--output-root", str(tmp_path / "out"), "--dry-run", "--auto-judge",
                      "--auto-judge-targets", str(targets), "--auto-judge-r-near-m", "0.05")
    assert proc.returncode == 0, proc.stderr
    meta = json.loads((tmp_path / "out" / "run_aj" / "eval_session_meta.json").read_text())
    assert meta["auto_judge"]["config"]["r_near_m"] == 0.05
    assert meta["auto_judge"]["targets_path"] == str(targets)
    missing_lines = [line for line in proc.stdout.splitlines() if "missing" in line.lower()]
    assert any("config_001/ram" in line for line in missing_lines)
    assert not any("config_001/cpu" in line for line in missing_lines)


def test_auto_judge_refuses_non_openpi_models(tmp_path):
    args = [a if a != "openpi" else "openvla" for a in BASE]
    proc = run_script(*args, "--output-root", str(tmp_path), "--dry-run", "--auto-judge",
                      "--auto-judge-targets", str(tmp_path / "targets.json"), "--auto-judge-r-near-m", "0.05")
    assert proc.returncode != 0
    assert "openpi" in proc.stderr.lower()


def test_dry_run_without_auto_judge_records_null(tmp_path):
    proc = run_script(*BASE, "--output-root", str(tmp_path), "--dry-run")
    assert proc.returncode == 0, proc.stderr
    meta = json.loads((tmp_path / "run_aj" / "eval_session_meta.json").read_text())
    assert meta["auto_judge"] is None
