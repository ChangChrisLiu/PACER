#!/usr/bin/env python3
"""Probe a TRACE-VLA-compatible ZMQ JSON REQ/REP agent."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path


def _request_from_args(args: argparse.Namespace) -> dict:
    if args.json_file:
        return json.loads(args.json_file.read_text())
    return {"type": args.request, "client": "tracevla", "request_id": str(uuid.uuid4())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True, help="Example: tcp://127.0.0.1:5555")
    ap.add_argument("--request", default="ping", choices=["ping", "status", "get_observation"])
    ap.add_argument("--json-file", type=Path, help="Send an explicit JSON request object instead of --request.")
    ap.add_argument("--timeout-ms", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    request = _request_from_args(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "endpoint": args.endpoint, "request": request}, indent=2, sort_keys=True))
        return 0

    try:
        import zmq
    except Exception as exc:  # pragma: no cover - optional hardware dependency
        raise SystemExit(f"pyzmq is required for live probing: {exc}")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    sock.connect(args.endpoint)
    started = time.time()
    try:
        sock.send_json(request)
        response = sock.recv_json()
    except Exception as exc:
        raise SystemExit(f"ZMQ probe failed for {args.endpoint}: {exc}")
    finally:
        sock.close(0)
    print(json.dumps({"ok": True, "latency_s": time.time() - started, "response": response}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
