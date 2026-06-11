from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Terms that should not appear in publication-facing source/docs because they
# identify local/private artifacts, old internal project names, or unpublished
# result-specific checkpoints. Keep this list narrow so it catches real leaks
# without blocking generic safety warnings such as "do not commit tokens".
FORBIDDEN_SUBSTRINGS = (
    "/home/chris",
    "10.125.145.49",
    "TRACE-VLA",
    "TRACEVLA",
    "tracevla",
    "RW-FMA 8499",
    "8499 checkpoint",
    "legacy 8499",
    "SEA-VLA reference",
    "current SEA-VLA",
    "HPRC job",
    "pi05_droid_ur5e_stage_b",
)

SKIP_PARTS = {
    ".git",
    ".claude",
    ".claude-flow",
    ".pytest_cache",
    "__pycache__",
    "agent_reviews",
}

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".sh",
}


def iter_publication_text_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.name == "test_publication_hygiene.py":
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        yield path


def test_publication_repo_has_no_private_or_old_project_identifiers():
    hits: list[str] = []
    for path in iter_publication_text_files():
        text = path.read_text(errors="ignore")
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in text:
                hits.append(f"{path.relative_to(REPO_ROOT)} contains {needle!r}")
    assert hits == []
