#!/usr/bin/env python3
"""Render README result and release blocks from their JSON sources of truth."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
STATS = REPO / "results/paper_stats.json"
RELEASE = REPO / "results/release_status.json"
SIZES = ("600M", "1.3B", "3.3B")
CONDITIONS = ("Standard BPE", "Joint-5K BPE", "Mono BPE", "Optuna BPE",
              "Morfessor", "Morfessor-VC", "Morfessor-BPE", "UnigramLM")


def replace_block(text, name, body):
    start = f"<!-- BEGIN GENERATED: {name} -->"
    end = f"<!-- END GENERATED: {name} -->"
    before, separator, remainder = text.partition(start)
    if not separator:
        raise ValueError(f"missing marker {start}")
    _, separator, after = remainder.partition(end)
    if not separator:
        raise ValueError(f"missing marker {end}")
    return before + start + "\n" + body.rstrip() + "\n" + end + after


def result_table(cells, direction):
    arrow = "arn→es" if direction == "arn-es" else "es→arn"
    lines = [f"### {arrow}", "", "| Condition | " + " | ".join(SIZES) + " |",
             "|---|---:|---:|---:|"]
    rounded = {(condition, size): round(cells[f"{condition}|{size}|{direction}"]["chrfpp_mean"], 2)
               for condition in CONDITIONS for size in SIZES}
    best = {size: max(rounded[(condition, size)] for condition in CONDITIONS) for size in SIZES}
    for condition in CONDITIONS:
        values = []
        for size in SIZES:
            value = f"{rounded[(condition, size)]:.2f}"
            values.append(f"**{value}**" if rounded[(condition, size)] == best[size] else value)
        lines.append(f"| {condition} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    stats = json.loads(STATS.read_text(encoding="utf-8"))["paper_v2"]
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    counts = [stats["data"][f"{split}|arn"]["sents"] for split in ("train", "dev", "test")]
    text = replace_block(text, "data-summary",
                         f"The corrected block-aligned dataset contains **{counts[0]:,} training**, "
                         f"**{counts[1]:,} development**, and **{counts[2]:,} test** sentence pairs.")
    released, expected = len(release["released"]), len(release["expected"])
    url = "https://huggingface.co/collections/" + release["collection"]
    if released == expected:
        body = f"All **{expected}** fine-tuned checkpoints are available in the [Hugging Face collection]({url})."
    else:
        body = (f"The [Hugging Face collection]({url}) currently contains **{released} of {expected}** "
                "planned checkpoints. See [`results/release_status.json`](results/release_status.json) "
                "for the machine-checked list.")
    text = replace_block(text, "release-status", body)
    text = replace_block(text, "results", result_table(stats["cells"], "arn-es") + "\n\n" +
                         result_table(stats["cells"], "es-arn"))
    README.write_text(text, encoding="utf-8")
    print(f"Rendered {README}")


if __name__ == "__main__":
    main()
