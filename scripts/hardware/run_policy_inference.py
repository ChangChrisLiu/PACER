#!/usr/bin/env python3
"""Single-step PACER policy inference smoke runner.

The script is intentionally useful in three modes:
1. --backend mock: no external dependencies, verifies observation/action JSON.
2. --backend openpi --dry-run: checks arguments without connecting.
3. --backend openpi: queries an OpenPI websocket policy server if openpi-client
   is installed in the active environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _mock_observation() -> dict[str, Any]:
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.zeros((256, 256, 3), dtype=np.uint8)
    return {
        "observation/image": base,
        "observation/wrist_image": wrist,
        "observation/state": np.zeros(7, dtype=np.float32),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_npz_observation(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    required = ["observation/image", "observation/wrist_image", "observation/state"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"missing required observation arrays in {path}: {missing}")
    return {k: data[k] for k in required}


def _infer_openpi(host: str, port: int, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except Exception as exc:  # pragma: no cover - optional hardware dependency
        raise SystemExit(
            "openpi-client is not installed in this environment. Install the "
            "client package from your OpenPI checkout, then rerun. Original "
            f"import error: {exc}"
        )
    policy = WebsocketClientPolicy(host=host, port=port)
    request = dict(obs)
    request["prompt"] = prompt
    return policy.infer(request)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["mock", "openpi"], default="mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--observation-npz", type=Path)
    ap.add_argument("--mock-observation", action="store_true", help="Use a zero-valued synthetic observation.")
    ap.add_argument("--dry-run", action="store_true", help="Validate setup without contacting a policy server.")
    ap.add_argument("--output", type=Path, help="Write JSON result to this path.")
    args = ap.parse_args(argv)

    if args.observation_npz:
        obs = _load_npz_observation(args.observation_npz)
    elif args.mock_observation or args.backend == "mock":
        obs = _mock_observation()
    else:
        raise SystemExit("provide --observation-npz or --mock-observation")

    started = time.time()
    if args.dry_run:
        result = {"dry_run": True, "backend": args.backend, "host": args.host, "port": args.port, "prompt": args.prompt}
    elif args.backend == "mock":
        result = {"actions": np.zeros((10, 7), dtype=np.float32), "backend": "mock"}
    else:
        result = _infer_openpi(args.host, args.port, obs, args.prompt)

    payload = {
        "ok": True,
        "backend": args.backend,
        "latency_s": time.time() - started,
        "result": _json_safe(result),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
