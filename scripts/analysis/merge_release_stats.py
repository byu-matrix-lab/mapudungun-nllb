"""Merge the reconstruction checks into results/paper_stats.json["paper_v2"]["release"].

Five released checkpoints were retrained after their originals were lost. A rebuild
counts as identical when its test predictions match the original's archived
predictions line for line on the GPU type the archived predictions were made on:
release_reconstruction_check.json (A100) for three models, and
release_reconstruction_check_h200.json (H200) for the two whose archived predictions
came from an H200. Merges into the JSON; never overwrites other keys.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATS = REPO / "results" / "paper_stats.json"


def main():
    a100 = json.loads((REPO / "results/release_reconstruction_check.json").read_text())
    h200 = json.loads((REPO / "results/release_reconstruction_check_h200.json").read_text())["models"]
    models = {}
    for name, check in a100.items():
        identical = check["all_identical"] or h200.get(name, {}).get("all_identical", False)
        models[name] = {"identical": identical, "checked_on": "H200" if name in h200 else "A100"}
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    stats.setdefault("paper_v2", {})["release"] = {
        "n_reconstructed": len(models),
        "n_identical": sum(m["identical"] for m in models.values()),
        "models": models,
    }
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats["paper_v2"]["release"], indent=1))


if __name__ == "__main__":
    main()
