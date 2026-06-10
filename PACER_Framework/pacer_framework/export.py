"""Training-view export for the external VLA trainer.

PACER touches the trainer through exactly one channel (eq:loss): the per-row
weight multiplying the native per-horizon action loss, normalized by
sum(w_i * m_{i,h}). ``export_training_view`` materializes a weighted view as
``rows.jsonl`` (each row carrying a top-level ``loss_weight``) plus
``manifest.json``, so any training stack — OpenPI, custom JAX/PyTorch — can
consume the view without importing this package.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

PACER_FRAMEWORK_VIEW_SCHEMA = "pacer_framework_training_view.v1"
LOSS_WEIGHT_FIELD = "loss_weight"


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def export_training_view(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    out_dir: str | Path,
    *,
    source_block: str | None = None,
) -> dict[str, Path]:
    """Write rows.jsonl + manifest.json for one weighted view.

    The weight source is row["pacer_framework"]["weight"] (PACER candidates)
    or row["pacer_framework_baseline"]["weight"] (baseline modes); when
    source_block is not given it is detected per row, preferring the baseline
    block when present. Every exported row gets a top-level ``loss_weight``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for row in rows:
        block_name = source_block
        if block_name is None:
            block_name = "pacer_framework_baseline" if "pacer_framework_baseline" in row else "pacer_framework"
        block = row.get(block_name)
        if not isinstance(block, Mapping) or "weight" not in block:
            raise ValueError(
                f"row {row.get('row_id')!r} has no {block_name!r} weight block; "
                "run compile_weights/compile_baseline first"
            )
        new_row = json.loads(json.dumps(row))
        new_row[LOSS_WEIGHT_FIELD] = float(block["weight"])
        exported.append(new_row)

    rows_path = write_jsonl(out / "rows.jsonl", exported)
    view_manifest = {
        "schema": PACER_FRAMEWORK_VIEW_SCHEMA,
        "weight_field": LOSS_WEIGHT_FIELD,
        "num_rows": len(exported),
        "num_positive_weight_rows": sum(1 for r in exported if r[LOSS_WEIGHT_FIELD] > 0.0),
        "source_manifest": dict(manifest),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(view_manifest, indent=2, sort_keys=True))
    return {"rows": rows_path, "manifest": manifest_path}
