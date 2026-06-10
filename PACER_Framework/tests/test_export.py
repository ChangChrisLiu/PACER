import json

import pytest

from pacer_framework.baselines import compile_baseline
from pacer_framework.eta import CANDIDATE_POOL
from pacer_framework.export import export_training_view, read_jsonl
from pacer_framework.weights import compile_weights


def rows():
    out = []
    for role, prog in (("clean", 1.0), ("correction", 0.8), ("failure", 0.5)):
        out.append(
            {
                "row_id": f"r:{role}",
                "component": "widget",
                "phase": "approach",
                "split": "train",
                "role": role,
                "mask": [1] * 4,
                "target_meta": {"target_id": "widget_0", "geometry_record_present": True},
                "evidence": {"prog": prog, "prox": 0.0, "term": 0.0, "dir": 0.0, "stop": 0.0, "op": 0.0, "prov": 1.0},
            }
        )
    return out


def test_export_training_view_round_trips_weights(tmp_path):
    weighted, manifest = compile_weights(rows(), CANDIDATE_POOL["rw_like"], eta_id="rw_like")
    paths = export_training_view(weighted, manifest, tmp_path / "view")
    loaded = read_jsonl(paths["rows"])
    assert len(loaded) == 3
    for row in loaded:
        assert row["loss_weight"] == row["pacer_framework"]["weight"]
    view_manifest = json.loads(paths["manifest"].read_text())
    assert view_manifest["weight_field"] == "loss_weight"
    assert view_manifest["num_rows"] == 3
    assert view_manifest["num_positive_weight_rows"] == 2  # failure row is zero
    assert view_manifest["source_manifest"]["eta_id"] == "rw_like"


def test_export_handles_baseline_views_and_missing_blocks(tmp_path):
    baseline_rows, manifest = compile_baseline(rows(), "uniform_all_eligible")
    paths = export_training_view(baseline_rows, manifest, tmp_path / "uniform")
    loaded = read_jsonl(paths["rows"])
    assert {r["loss_weight"] for r in loaded} == {0.0, 1.0}
    with pytest.raises(ValueError):
        export_training_view(rows(), {}, tmp_path / "bad")  # no weight block yet
