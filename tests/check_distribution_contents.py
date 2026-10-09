"""Verify that either installation option ships the identical canonical scorer."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile


def framework_sources(directory: str) -> dict[str, bytes]:
    wheels = list(Path(directory).glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected one wheel in {directory}, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as archive:
        result = {name: archive.read(name) for name in archive.namelist()
                  if name.startswith("pacer_framework/") and name.endswith(".py")}
    for module in ("configuration.py", "validation.py", "alignment.py", "scoring.py", "demo.py"):
        if "pacer_framework/" + module not in result:
            raise ValueError(f"wheel does not contain canonical module {module}")
    return result


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_distribution_contents.py RUNTIME_WHEEL_DIR FRAMEWORK_WHEEL_DIR")
    runtime = framework_sources(sys.argv[1])
    minimal = framework_sources(sys.argv[2])
    if runtime != minimal:
        raise AssertionError("installation options do not ship identical framework sources")
    source_root = Path(__file__).resolve().parents[1] / "PACER_Framework" / "pacer_framework"
    current = {"pacer_framework/" + path.relative_to(source_root).as_posix(): path.read_bytes()
               for path in source_root.rglob("*.py") if "__pycache__" not in path.parts}
    if runtime != current:
        raise AssertionError("wheel sources do not match the current checkout; rebuild both wheels")
    print(json.dumps({"status": "PASS_IDENTICAL_FRAMEWORK_SOURCES", "modules": len(runtime),
                      "matches_current_checkout": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
