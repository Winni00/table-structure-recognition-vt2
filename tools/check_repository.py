#!/usr/bin/env python3
"""Reject common private or oversized artifacts before publication."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_SIZE = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".7z", ".ckpt", ".key", ".npy", ".npz", ".onnx", ".pdf", ".pickle",
    ".pkl", ".ppt", ".pptx", ".pt", ".pth", ".safetensors", ".tar", ".tgz",
    ".zip",
}
PRIVATE_NAMES = re.compile(
    "|".join(("ana" + "is", "ana" + "\u00efs", "thamma" + "vongsa", r"\bth" + r"mv\b")),
    re.IGNORECASE,
)
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
TEXT_SUFFIXES = {
    "", ".cff", ".cfg", ".csv", ".env", ".html", ".ini", ".json", ".jsonl",
    ".md", ".patch", ".py", ".sh", ".tex", ".toml", ".tsv", ".txt", ".xml",
    ".yaml", ".yml",
}


def files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    )


def main() -> int:
    errors: list[str] = []
    path_warnings: list[str] = []
    for path in files():
        relative = path.relative_to(ROOT)
        suffix = path.suffix.lower()
        if PRIVATE_NAMES.search(relative.as_posix()):
            errors.append(f"private name in path: {relative}")
        if suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact: {relative}")
        if path.stat().st_size > MAX_SIZE:
            errors.append(f"file exceeds 10 MiB: {relative}")
        if suffix not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PRIVATE_NAMES.search(content):
            errors.append(f"private name in content: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"possible {label}: {relative}")
        if "/cluster/home/" in content or "/Users/" in content:
            path_warnings.append(str(relative))

    for item in sorted(set(errors)):
        print(f"ERROR: {item}")
    if path_warnings:
        print(
            "WARNING: historical absolute paths remain in "
            f"{len(set(path_warnings))} files; adapt them before execution."
        )
    if errors:
        return 1
    print("Repository publication checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
