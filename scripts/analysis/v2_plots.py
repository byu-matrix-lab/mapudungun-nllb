#!/usr/bin/env python3
"""
v2_plots.py: the paper's three figures for main_v2.tex, from results/paper_stats.json.

Generated from the committed scripts/analysis/plot_results.py (HEAD), with the plotting
code unchanged and its hand-typed data replaced by values read from the JSON:
  1. latex/figures/v2/results_chrf.png       overview; fine-tuned bars = best condition at each size
  2. latex/figures/v2/results_heatmap.png    conditions x sizes, mean sentence-level chrF++
  3. latex/figures/v2/results_fertility.png  fertility on the original training data
Usage: .venv/bin/python scripts/analysis/v2_plots.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np


# ── shared data ────────────────────────────────────────────────────────────────

# All 8 tokenization conditions, (arn→es, es→arn) per size
import json
STATS = json.loads((Path(__file__).resolve().parents[2] / "results/paper_stats.json").read_text(encoding="utf-8"))["paper_v2"]
CELLS = STATS["cells"]
FINETUNED = ["Standard BPE", "Joint-5K BPE", "Mono BPE", "Optuna BPE", "Morfessor",
             "Morfessor-VC", "Morfessor-BPE", "UnigramLM"]


def mean(cond, size, d):
    return CELLS[f"{cond}|{size}|{d}"]["chrfpp_mean"]


ALL_CONDITIONS = [(c, [(mean(c, s, "arn-es"), mean(c, s, "es-arn")) for s in ("600M", "1.3B", "3.3B")])
                  for c in FINETUNED]
SIZES = ["600M", "1.3B", "3.3B"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",           default="latex/figures/v2/results_chrf.png")
    parser.add_argument("--output-tok",       default="latex/figures/v2/results_heatmap.png")
    parser.add_argument("--output-fertility", default="latex/figures/v2/results_fertility.png")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW BAR CHART  (prior work vs our systems)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_group_bracket(ax, x_indices, label, y_line=-0.13, y_text=-0.17):
    """Draw a bracket + label below x-axis."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    left  = min(x_indices) - 0.35
    right = max(x_indices) + 0.35
    cx    = np.mean(x_indices)
    line = Line2D([left, right], [y_line, y_line],
                  transform=trans, color="#444444", lw=0.9, clip_on=False)
    ax.add_line(line)
    tick_h = 0.016
    for xp in [left, right]:
        t = Line2D([xp, xp], [y_line, y_line + tick_h],
                   transform=trans, color="#444444", lw=0.9, clip_on=False)
        ax.add_line(t)
    ax.text(cx, y_text, label, transform=trans,
            ha="center", va="top", fontsize=8.5, fontweight="bold", clip_on=False)


def plot_overview(args):
    # Lira et al. (2025) omitted: their test set differs from ours (1,250 vs 9,382 pairs)
    # and is not publicly available for direct comparison. See Related Work.
    # Fine-tuned bars: the best tokenization condition at each size and direction.
    best = {(sz, d): max(mean(c, sz, d) for c in FINETUNED) for sz in SIZES for d in ("arn-es", "es-arn")}
    systems = [(sz, mean("Zero-shot NLLB", sz, "arn-es"), mean("Zero-shot NLLB", sz, "es-arn"), "zero") for sz in SIZES]
    systems += [("Llama\n3.1 8B", mean("Llama 3.1 8B", "8B", "arn-es"), mean("Llama 3.1 8B", "8B", "es-arn"), "llm"),
                ("Aya\nExp. 8B", mean("Aya Expanse 8B", "8B", "arn-es"), mean("Aya Expanse 8B", "8B", "es-arn"), "llm")]
    systems += [(sz, best[(sz, "arn-es")], best[(sz, "es-arn")], "ft") for sz in SIZES]

    n_zero = 3; n_llm = 2; n_ft = 3
    n_total = n_zero + n_llm + n_ft

    idx_zero  = list(range(0, n_zero))
    idx_llm   = list(range(n_zero, n_zero + n_llm))
    idx_ft    = list(range(n_zero + n_llm, n_total))

    colors = {"zero": "#90caf9", "llm": "#ffb74d", "ft": "#1565c0"}

    all_values = [ae for _, ae, _, _ in systems] + [ea for _, _, ea, _ in systems]
    global_ymax = max(all_values) * 1.18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.08, bottom=0.22, top=0.88)

    for ax, is_arn_es, dir_label in [
        (ax1, True,  "Mapudungun → Spanish"),
        (ax2, False, "Spanish → Mapudungun"),
    ]:
        labels, values, bar_colors = [], [], []
        for lbl, ae, ea, ck in systems:
            labels.append(lbl)
            values.append(ae if is_arn_es else ea)
            bar_colors.append(colors[ck])

        x = np.arange(n_total)
        bars = ax.bar(x, values, color=bar_colors, width=0.6,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, ha="center")

        # Dashed divider between baselines and fine-tuned systems
        div_x = (n_zero + n_llm) - 0.5
        ax.axvline(x=div_x, color="#666666", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.text(div_x - 0.12, global_ymax * 0.97, "Baselines", fontsize=9, color="#666666",
                va="top", ha="right", style="italic")
        ax.text(div_x + 0.12, global_ymax * 0.97, "Fine-tuned →", fontsize=9, color="#666666",
                va="top", ha="left", style="italic")

        draw_group_bracket(ax, idx_zero,  "NLLB\n(zero-shot)")
        draw_group_bracket(ax, idx_llm,   "Prompted\nLLM")
        draw_group_bracket(ax, idx_ft,    "Fine-tuned NLLB\n(best tokenization)")

        ax.set_ylabel("chrF++", fontsize=12)
        ax.set_title(dir_label, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(0, global_ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelleft=True)

    legend_patches = [
        mpatches.Patch(color=colors["zero"],  label="Zero-shot NLLB (lower bound)"),
        mpatches.Patch(color=colors["llm"],   label="5-shot LLM (zero-shot multilingual frontier)"),
        mpatches.Patch(color=colors["ft"],    label="Fine-tuned NLLB"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HEATMAP  (conditions × model sizes, one panel per direction)
# ═══════════════════════════════════════════════════════════════════════════════

C_MVC = "#E07020"

def plot_heatmap(args):
    cond_display = ["Standard BPE", "Joint-5K BPE", "Mono BPE", "Optuna BPE",
                    "Morfessor", "Morfessor-VC*", "Morfessor-BPE", "UnigramLM"]
    mvc_row = 5

    data_ae = np.array([[d[i][0] for i in range(3)] for _, d in ALL_CONDITIONS])
    data_ea = np.array([[d[i][1] for i in range(3)] for _, d in ALL_CONDITIONS])

    vlo = min(data_ae.min(), data_ea.min()) - 1.0
    vhi = max(data_ae.max(), data_ea.max()) + 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(left=0.16, right=0.93, wspace=0.55, top=0.88, bottom=0.10)

    for ax, data, title in [
        (ax1, data_ae, "Mapudungun → Spanish"),
        (ax2, data_ea, "Spanish → Mapudungun"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=vlo, vmax=vhi)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(SIZES, fontsize=10)
        ax.set_yticks(range(8))
        ax.set_yticklabels(cond_display, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        for j in range(3):
            col_vals = data[:, j]
            best = col_vals.max()
            for i in range(8):
                v = data[i, j]
                weight = "bold" if v == best else "normal"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, color="black", fontweight=weight)

        for j in range(3):
            ax.add_patch(plt.Rectangle((j - 0.5, mvc_row - 0.5), 1, 1,
                fill=False, edgecolor=C_MVC, lw=2.5, zorder=5))

        cb = plt.colorbar(im, ax=ax, label="chrF++", fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9)

    out = Path(args.output_tok)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════

def plot_fertility(args):
    C_MVC  = "#E07020"
    C_MORF = "#1565C0"
    C_OTHER= "#AAAAAA"
    fert = {c: v["fertility_train"] for c, v in STATS["segmentation"]["conditions"].items()}
    order = sorted(fert, key=lambda c: -fert[c])
    conds       = [c + " (ours)" if c == "Morfessor-VC" else c for c in order]
    fertilities = [fert[c] for c in order]
    bar_colors  = [C_MVC if c == "Morfessor-VC" else C_MORF if c == "Morfessor" else C_OTHER for c in order]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.subplots_adjust(left=0.27, right=0.94, top=0.93, bottom=0.13)
    y = np.arange(len(conds))
    bars = ax.barh(y, fertilities, color=bar_colors, height=0.55, zorder=3, left=0.01)
    for bar, v in zip(bars, fertilities):
        ax.text(v + 0.02 + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9.5, fontweight="bold", zorder=4)
    ax.axvline(1.0, color="#888", ls="--", lw=1.0, alpha=0.7, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(conds, fontsize=10)
    ax.set_xlabel("Tokens per whitespace word (fertility)", fontsize=10)
    ax.set_xlim(0, 2.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(1.02, len(conds) - 0.6, "ideal (1.0)", fontsize=8, color="#888", style="italic", va="bottom")
    ax.xaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    ax.set_axisbelow(True)

    out = Path(args.output_fertility)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    args = parse_args()
    plot_overview(args)
    plot_heatmap(args)
    plot_fertility(args)


if __name__ == "__main__":
    main()
