#!/usr/bin/env python3
"""
plot_results.py

Generate dual bar chart comparing all systems for both translation directions
(arn→es and es→arn) using blocks approach (best performing).

Prior work:
- Duan et al. (2020): arn→es 0.5 chrF, es→arn 0.4 chrF (vanilla Transformer, 220K pairs,
  official conversation splits — same test conversations as ours)
- Lira et al. (2025): arn→es 30.30 chrF, es→arn 31.20 chrF (transfer learning, Spanish-Finnish;
  evaluated on a different random 1,250-pair test set from 10K sample)

Usage:
  uv run python scripts/analysis/plot_results.py --output notes/results_chrf.png
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="notes/results_chrf.png")
    return parser.parse_args()


def draw_group_bracket(ax, x_indices, label, y_line=-0.13, y_text=-0.17):
    """Draw a bracket + label below x-axis. Ticks point up, label below the line."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    left  = min(x_indices) - 0.35
    right = max(x_indices) + 0.35
    cx    = np.mean(x_indices)

    # Horizontal line
    line = Line2D([left, right], [y_line, y_line],
                  transform=trans, color="#444444", lw=0.9, clip_on=False)
    ax.add_line(line)

    # Short vertical tick marks pointing UP (toward bars)
    tick_h = 0.016
    for xp in [left, right]:
        t = Line2D([xp, xp], [y_line, y_line + tick_h],
                   transform=trans, color="#444444", lw=0.9, clip_on=False)
        ax.add_line(t)

    # Label below the line
    ax.text(cx, y_text, label, transform=trans,
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            clip_on=False)


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # Data — blocks approach, both directions
    # -------------------------------------------------------------------------

    # (short_label, arn_es_chrf, es_arn_chrf, color_key)
    prior_work = [
        ("Duan*\n(2020)",  0.50,  0.40, "prior"),
        ("Lira†\n(2025)", 30.30, 31.20, "prior"),
    ]

    systems = [
        # NLLB zero-shot
        ("600M",    16.25, 10.96, "zero"),
        ("1.3B",    17.22, 12.44, "zero"),
        ("3.3B‡",    5.94,  3.43, "zero"),
        # 5-shot LLM
        ("Llama 3.1", 16.54, 16.20, "llm"),
        ("Aya Exp.",  20.05, 16.11, "llm"),
        # Fine-tuned NLLB (arn→es: corrected test-set; es→arn: pending retrain fix)
        ("600M",  35.71, 14.40, "ft"),
        ("1.3B",  42.24, 14.25, "ft"),
        ("3.3B",  43.62, 14.22, "ft"),
    ]

    n_prior   = len(prior_work)   # 2
    n_zero    = 3
    n_llm     = 2
    n_ft      = 3
    n_total   = n_prior + n_zero + n_llm + n_ft  # 10

    # x index ranges for group brackets
    idx_prior = list(range(0, n_prior))                    # [0, 1]
    idx_zero  = list(range(n_prior, n_prior + n_zero))     # [2, 3, 4]
    idx_llm   = list(range(n_prior + n_zero,
                           n_prior + n_zero + n_llm))      # [5, 6]
    idx_ft    = list(range(n_prior + n_zero + n_llm,
                           n_total))                       # [7, 8, 9]

    colors = {
        "prior": "#9e9e9e",
        "zero":  "#90caf9",
        "llm":   "#ffb74d",
        "ft":    "#1565c0",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=False)
    fig.subplots_adjust(left=0.05, right=0.98, wspace=0.15, bottom=0.25)

    for ax, direction, dir_label in [
        (ax1, "arn_es", "Mapudungun → Spanish"),
        (ax2, "es_arn", "Spanish → Mapudungun"),
    ]:
        is_arn_es = (direction == "arn_es")

        labels = []
        values = []
        bar_colors = []

        for short_label, chrf_arn_es, chrf_es_arn, ck in prior_work:
            labels.append(short_label)
            values.append(chrf_arn_es if is_arn_es else chrf_es_arn)
            bar_colors.append(colors[ck])

        for short_label, chrf_arn_es, chrf_es_arn, ck in systems:
            labels.append(short_label)
            values.append(chrf_arn_es if is_arn_es else chrf_es_arn)
            bar_colors.append(colors[ck])

        x = np.arange(n_total)
        bars = ax.bar(x, values, color=bar_colors, width=0.6,
                      edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
            )

        # Short bar-level tick labels (model size only)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, ha="center")

        # Divider between prior work and our systems
        div_x = n_prior - 0.5
        ax.axvline(x=div_x, color="#666666", linestyle="--",
                   linewidth=1.1, alpha=0.7)
        ymax = max(values) * 1.18
        ax.text(div_x - 0.12, ymax * 0.97,
                "Prior work", fontsize=7.5, color="#666666",
                va="top", ha="right", style="italic")
        ax.text(div_x + 0.12, ymax * 0.97,
                "This work →", fontsize=7.5, color="#666666",
                va="top", ha="left", style="italic")

        # Group brackets below x-axis
        draw_group_bracket(ax, idx_prior, "Prior work")
        draw_group_bracket(ax, idx_zero,  "NLLB\n(zero-shot)")
        draw_group_bracket(ax, idx_llm,   "5-shot\nLLM")
        draw_group_bracket(ax, idx_ft,    "Fine-tuned\nNLLB")

        ax.set_ylabel("chrF++", fontsize=11)
        ax.set_title(dir_label, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(0, ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    # Footnotes
    footnotes = (
        "* Duan et al. used same test conversations but pre-cleaning (36.3K pairs vs. our 9.4K)    "
        "† Lira et al. evaluated on a different 1,250-pair test set (random sample from 10K)    "
        "‡ Non-distilled model; proxy-token initialization less effective without distillation"
    )
    fig.text(0.02, 0.005, footnotes, fontsize=7, color="#555555", style="italic",
             wrap=True)

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors["prior"], label="Prior work"),
        mpatches.Patch(color=colors["zero"],  label="Zero-shot NLLB"),
        mpatches.Patch(color=colors["llm"],   label="5-shot LLM"),
        mpatches.Patch(color=colors["ft"],    label="Fine-tuned NLLB"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "chrF++ by System and Translation Direction  (blocks segmentation)",
        fontsize=13, fontweight="bold",
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
