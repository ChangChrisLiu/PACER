import sys
from pathlib import Path

# Make the package importable without installation: tests live in
# PACER_Framework/tests and the package in PACER_Framework/pacer_framework.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
