from __future__ import annotations

import base64
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Publication-facing source/docs should not expose local paths, internal network
# addresses, pre-rename project names, cluster/job labels, or private checkpoint
# identifiers. Exact internal strings are base64-encoded here so the guard does
# not publish the very literals it is meant to block.
FORBIDDEN_EXACT_B64 = (
    "L2hvbWUvY2hyaXM=",
    "MTAuMTI1LjE0NS40OQ==",
    "VFJBQ0UtVkxB",
    "VFJBQ0VWTEE=",
    "dHJhY2V2bGE=",
    "UlctRk1BIDg0OTk=",
    "ODQ5OSBjaGVja3BvaW50",
    "bGVnYWN5IDg0OTk=",
    "U0VBLVZMQSByZWZlcmVuY2U=",
    "Y3VycmVudCBTRUEtVkxB",
    "SFBSQyBqb2I=",
    "cGkwNV9kcm9pZF91cjVlX3N0YWdlX2I=",
    "TE9LSV9SRUNPTkNJTElBVElPTl9WMV8xLm1k",
    "Q09OU0VOU1VTX1BIQVNFX0EubWQ=",
    "aW50ZXJuYWwgUEFDRVI=",
    "UEFDRVIgUEFDRVI=",
    "U0VBLVZMQQ==",
    "VFJBQ0Utc3R5bGU=",
    "UEFDRVIvVFJBQ0U=",
    "cHJlLXB1YmxpY2F0aW9u",
    "UHJlLXB1YmxpY2F0aW9u",
    "cHJlcHVi",
    "aW50ZXJuYWwgY29ycmVjdG9yLW9ubHkgc2luZ2xlLXRyaWFsIHJlZGVzaWduIG5vdGU=",
)

FORBIDDEN_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9_.-]+(?:/|\b)"),
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
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


def _decode_needles() -> tuple[str, ...]:
    return tuple(base64.b64decode(value).decode("utf-8") for value in FORBIDDEN_EXACT_B64)


def iter_publication_text_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        yield path


def test_publication_repo_has_no_private_or_old_project_identifiers():
    hits: list[str] = []
    exact_needles = _decode_needles()
    for path in iter_publication_text_files():
        text = path.read_text(errors="ignore")
        rel = path.relative_to(REPO_ROOT)
        for needle in exact_needles:
            if needle in text:
                hits.append(f"{rel} contains a blocked internal literal")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                hits.append(f"{rel} matches blocked publication-hygiene pattern {pattern.pattern!r}")
    assert hits == []
