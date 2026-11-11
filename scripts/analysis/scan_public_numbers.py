#!/usr/bin/env python3
"""Flag numeric literals in public prose outside generated or cited code blocks."""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NUMBER = re.compile(r"(?<![A-Za-z])(?:\d[\d,]*(?:\.\d+)?)(?![A-Za-z])")


def main():
    path = REPO / "README.md"
    exceptions = json.loads((REPO / "release/numeric_literal_exceptions.json").read_text(encoding="utf-8"))
    allowed = [(re.compile(item["pattern"]), item["reason"]) for item in exceptions]
    generated = False
    fenced = False
    candidates = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("<!-- BEGIN GENERATED:"):
            generated = True
        if line.startswith("<!-- END GENERATED:"):
            generated = False
            continue
        if line.startswith("```"):
            fenced = not fenced
            continue
        if generated or fenced or not NUMBER.search(line):
            continue
        if any(pattern.search(line) for pattern, _ in allowed):
            continue
        candidates.append(f"{path}:{line_number}: {line.strip()}")
    if candidates:
        raise SystemExit("Unreviewed numeric literals:\n" + "\n".join(candidates))
    print("README numeric-literal candidates: 0")


if __name__ == "__main__":
    main()
