"""Command-line access to configurable component-balanced validation scoring.

This command scores validation rows. Full candidate feasibility and selection
use evaluate_candidate/select_candidate; a score alone is not a deployment
approval.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from pacer_framework.configuration import ScoringConfig
from pacer_framework.export import read_jsonl
from pacer_framework.schema import validate_validation_row
from pacer_framework.validation import component_balanced_j_val


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, help="Validation rows in the framework JSONL contract")
    parser.add_argument("--config", type=Path, help="Editable JSON scoring configuration; defaults are used if omitted")
    parser.add_argument("--output", type=Path, help="Write the score and protocol manifest instead of printing JSON")
    parser.add_argument("--write-config", type=Path, help="Export the default or loaded configuration for editing")
    parser.add_argument("--force", action="store_true", help="Explicitly allow replacing an existing output file")
    args = parser.parse_args(argv)
    if args.rows is None and args.write_config is None:
        parser.error("provide --rows or --write-config")
    if args.output is not None and args.rows is None:
        parser.error("--output requires --rows")
    try:
        config = ScoringConfig.load(args.config) if args.config is not None else ScoringConfig()
        if args.write_config is not None:
            config.save(args.write_config, overwrite=args.force)
        if args.rows is not None:
            rows = read_jsonl(args.rows)
            if not rows or not all(isinstance(row, dict) for row in rows):
                raise ValueError("validation rows must be a nonempty collection of JSON objects")
            errors = [f"row {index}: {error}" for index, row in enumerate(rows)
                      for error in validate_validation_row(row)]
            if errors:
                raise ValueError("; ".join(errors[:5]))
            score, row_scores, manifest = component_balanced_j_val(rows, scoring_config=config)
            payload = {
                "schema": "pacer.validation_score.v1", "j_val": score,
                "row_scores": [asdict(row) for row in row_scores], "manifest": manifest,
            }
            encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
            if args.output is None:
                print(encoded, end="")
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w" if args.force else "x", encoding="utf-8") as stream:
                    stream.write(encoded)
                print(f"Wrote validation score: {args.output}")
        elif args.write_config is not None:
            print(f"Wrote scoring configuration: {args.write_config}")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
