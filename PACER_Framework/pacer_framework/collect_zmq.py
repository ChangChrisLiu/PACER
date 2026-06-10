"""Collect PACER JSON messages over ZeroMQ and append them to JSONL.

This is an optional utility for labs that stream rollout/correction/demo logs
from a robot process, simulator, or policy server. It intentionally accepts
plain JSON dictionaries so it is independent of ROS/vendor SDK/model code.

Install the optional dependency first:

    python -m pip install -e '.[zmq]'

Examples:

    # Receiver binds and waits for PUSH clients:
    python -m pacer_framework.collect_zmq --bind tcp://*:5557 --out data/raw/pacer_stream.jsonl

    # Receiver connects to a PUB endpoint and subscribes to all topics:
    python -m pacer_framework.collect_zmq --connect tcp://127.0.0.1:5557 --socket SUB --out data/raw/pacer_stream.jsonl

Each message can be either:
- a complete PACER standard row (`row_id`, `component`, `role`, ...), or
- an event/trace dictionary that your offline converter later chunks into rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_zmq():
    try:
        import zmq  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit(
            "pyzmq is required for ZMQ collection. Install with: "
            "python -m pip install -e '.[zmq]'"
        ) from exc
    return zmq


def _parse_json_message(raw: bytes) -> dict[str, Any]:
    try:
        msg = raw.decode("utf-8")
        return json.loads(msg)
    except Exception as exc:
        return {
            "message_type": "collector_error",
            "error": f"invalid_json: {exc}",
            "raw_utf8_lossy": raw.decode("utf-8", errors="replace"),
            "collector_time": time.time(),
        }


def _stamp(message: dict[str, Any], *, source: str) -> dict[str, Any]:
    out = dict(message)
    out.setdefault("collector", {})
    if isinstance(out["collector"], dict):
        out["collector"].setdefault("source", source)
        out["collector"].setdefault("received_unix_time", time.time())
    return out


def collect(
    *,
    endpoint: str,
    bind: bool,
    socket_type: str,
    out_path: Path,
    topic: str = "",
    max_messages: int | None = None,
) -> int:
    zmq = _load_zmq()
    ctx = zmq.Context.instance()
    sock_type = getattr(zmq, socket_type)
    sock = ctx.socket(sock_type)

    if socket_type == "SUB":
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)

    if bind:
        sock.bind(endpoint)
        source = f"bind:{endpoint}"
    else:
        sock.connect(endpoint)
        source = f"connect:{endpoint}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("a", encoding="utf-8") as f:
        while max_messages is None or count < max_messages:
            parts = sock.recv_multipart()
            raw = parts[-1] if parts else b"{}"
            message = _stamp(_parse_json_message(raw), source=source)
            f.write(json.dumps(message, sort_keys=True) + "\n")
            f.flush()
            count += 1
            print(f"collected {count}: {message.get('row_id') or message.get('event') or message.get('message_type', 'message')}")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect PACER JSON messages over ZeroMQ into JSONL.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bind", help="Endpoint to bind, e.g. tcp://*:5557")
    mode.add_argument("--connect", help="Endpoint to connect, e.g. tcp://127.0.0.1:5557")
    parser.add_argument("--socket", choices=["PULL", "SUB"], default="PULL", help="ZMQ socket type for receiver")
    parser.add_argument("--topic", default="", help="SUB topic prefix; empty subscribes to all")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--max-messages", type=int, default=None, help="Stop after N messages; useful for tests/smoke")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    endpoint = args.bind or args.connect
    count = collect(
        endpoint=endpoint,
        bind=bool(args.bind),
        socket_type=args.socket,
        out_path=Path(args.out),
        topic=args.topic,
        max_messages=args.max_messages,
    )
    print(f"done: wrote {count} message(s) to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
