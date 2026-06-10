import importlib.util
import json
from pathlib import Path

from pacer_framework.eta import CANDIDATE_POOL, PaperEta
from pacer_framework.export import read_jsonl
from pacer_framework.schema import validate_training_row, validate_validation_row

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_training_rows_satisfy_the_contract():
    rows = read_jsonl(EXAMPLES / "training_rows.jsonl")
    assert len(rows) >= 6
    roles = set()
    for row in rows:
        assert validate_training_row(row) == [], row["row_id"]
        roles.add(row.get("role"))
    assert {"clean", "correction", "auto_success", "partial", "failure", "excluded"} <= roles


def test_example_validation_rows_satisfy_the_contract():
    rows = read_jsonl(EXAMPLES / "validation_rows.jsonl")
    assert len(rows) >= 4
    for row in rows:
        assert validate_validation_row(row) == [], row["row_id"]


def test_example_candidate_config_matches_the_pool():
    config = json.loads((EXAMPLES / "candidate_terminal_stop_heavy.json").read_text())
    assert PaperEta.from_dict(config) == CANDIDATE_POOL["terminal_stop_heavy"]


def test_end_to_end_demo_runs_and_selects_the_good_candidate(tmp_path):
    spec = importlib.util.spec_from_file_location("pacer_demo", EXAMPLES / "end_to_end_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    result = demo.main(out_dir=tmp_path)
    assert result["contract_errors"] == []
    assert result["selected"] == "terminal_stop_heavy"
    assert result["ranking"][0]["candidate"] == "terminal_stop_heavy"
    assert set(result["baselines"]) == set(
        ("no_post_train", "vanilla_post_sft", "uniform_all_eligible", "outcome_only",
         "clean_only", "correction_only", "fixed_geometry", "pacer_selected")
    )
    assert (tmp_path / "terminal_stop_heavy" / "rows.jsonl").exists()
    assert result["bootstrap"]["difference_pp"] == 37.5
