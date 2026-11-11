#!/usr/bin/env python3
"""Flag unreviewed numeric literals in poster prose.

Experimental quantities must appear as generated macros. The exclusions below
cover layout dimensions, model/product names, citation years, and URLs.
"""

import re
from pathlib import Path

TEX = Path(__file__).with_name("poster.tex")
NUMBER = re.compile(r"(?<![A-Za-z\\])\d[\d,.]*(?![A-Za-z])")
EXCEPTIONS = [
    re.compile(r"(^|\\)(usepackage|definecolor|newtcolorbox|multicolumn|vspace|hspace|fontsize|setlength|addtolength|includegraphics|begin\{(?:column|minipage)|node|fill|draw)"),
    re.compile(r"(?:cm|pt|em|paperwidth|linewidth|arraystretch|tabcolsep|text width|minimum height|minimum width)"),
    re.compile(r"(?:NLLB-200|600M|1\.3B|3\.3B|8B|Joint-5K|5K merges|95\\%|ISO 639-3)"),
    re.compile(r"(?:Virpioja|Akiba|Banerjee|Duan|Kudo|Gowda)[^\d]*20\d{2}"),
    re.compile(r"(?:AmericasNLP|ACL) 2026"),
    re.compile(r"(?:https?://|aclanthology|github|hf\.co)"),
]


def main():
    candidates = []
    for number, line in enumerate(TEX.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or not NUMBER.search(line):
            continue
        if any(pattern.search(line) for pattern in EXCEPTIONS):
            continue
        candidates.append(f"{TEX}:{number}: {stripped}")
    if candidates:
        raise SystemExit("Unreviewed poster numeric literals:\n" + "\n".join(candidates))
    print("Poster numeric-literal candidates: 0")


if __name__ == "__main__":
    main()
