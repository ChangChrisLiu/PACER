"""Compatibility launcher for the installed, configurable synthetic demo."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacer_framework.demo import main, cli  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(cli())
