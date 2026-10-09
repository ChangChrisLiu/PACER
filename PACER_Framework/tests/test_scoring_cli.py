"""Installed command-line configuration and scoring are public user seams."""
import json
import os
import subprocess
import sys
from pathlib import Path

from pacer_framework import ScoringConfig


def test_cli_exports_editable_configuration_and_uses_it_for_component_mean(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    config_path = tmp_path / "scoring.json"
    command = [sys.executable, "-m", "pacer_framework.scoring"]
    created = subprocess.run(command + ["--write-config", str(config_path)], env=env, capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    payload = json.loads(config_path.read_text())
    payload["submetric_weights"] = {"align": 1.0}
    config_path.write_text(json.dumps(payload))
    rows = [
        {"row_id": "one", "component": "widget", "role": "clean", "submetrics": {"align": 1.0}},
        {"row_id": "two", "component": "widget", "role": "clean", "submetrics": {"align": 1.0}, "unsafe": True},
    ]
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output_path = tmp_path / "score.json"
    result = subprocess.run(command + ["--rows", str(input_path), "--config", str(config_path), "--output", str(output_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    scored = json.loads(output_path.read_text())
    assert abs(scored["j_val"] - 0.5) < 1e-5
    assert scored["manifest"]["aggregation"] == "component_mean"
    assert scored["manifest"]["candidate_feasibility_checked"] is False
    assert scored["manifest"]["scoring_config_hash"] == ScoringConfig.load(config_path).fingerprint
    repeated = subprocess.run(command + ["--write-config", str(config_path)], env=env, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert json.loads(config_path.read_text())["submetric_weights"] == {"align": 1.0}
