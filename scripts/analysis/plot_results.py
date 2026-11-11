#!/usr/bin/env python3
"""
plot_results.py

Generates publication-ready figures for the Mapudungun MT paper.

Usage:
  uv run python scripts/analysis/plot_results.py               # PNG to notes/
  uv run python scripts/analysis/plot_results.py --poster      # PDF to poster/figures/
  uv run python scripts/analysis/plot_results.py --format pdf --output poster/figures/results_chrf.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np


# ── BYU brand colors ───────────────────────────────────────────────────────────
NAVY       = "#002E5D"
ROYAL      = "#0047BA"
LIGHT_BLUE = "#BDD6E6"
ORANGE     = "#D14124"
SLATE      = "#7C878E"
SAND       = "#D1CCBD"

# ── shared data from the paper SSOT ───────────────────────────────────────────
STATS = json.loads(
    (Path(__file__).resolve().parents[2] / "results/paper_stats.json").read_text(encoding="utf-8")
)["paper_v2"]
CELLS = STATS["cells"]
SIZES = ["600M", "1.3B", "3.3B"]
CONDITIONS = ["Standard BPE", "Morfessor-VC", "Morfessor", "Optuna BPE",
              "Morfessor-BPE", "Joint-5K BPE", "UnigramLM", "Mono BPE"]


def mean(condition, size, direction):
    return CELLS[f"{condition}|{size}|{direction}"]["chrfpp_mean"]


ALL_CONDITIONS = [
    (condition, [(mean(condition, size, "arn-es"), mean(condition, size, "es-arn"))
                 for size in SIZES])
    for condition in CONDITIONS
]

BYU_CMAP = LinearSegmentedColormap.from_list("byu", [LIGHT_BLUE, ROYAL])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="png", choices=["png", "pdf"],
                        help="Output format (png or pdf)")
    parser.add_argument("--poster", action="store_true",
                        help="Shortcut: --format pdf with output to poster/figures/")
    parser.add_argument("--output",           default=None)
    parser.add_argument("--output-tok",       default=None)
    parser.add_argument("--output-table",     default=None)
    parser.add_argument("--output-fertility", default=None)
    args = parser.parse_args()

    if args.poster:
        args.format = "pdf"
        fmt = "pdf"
        if args.output is None:
            args.output = f"poster/figures/results_chrf_final.{fmt}"
        if args.output_tok is None:
            args.output_tok = f"poster/figures/results_heatmap_final.{fmt}"
        if args.output_table is None:
            args.output_table = f"notes/results_table.{fmt}"
        if args.output_fertility is None:
            args.output_fertility = f"poster/figures/results_fertility_final.{fmt}"
    else:
        fmt = args.format
        if args.output is None:
            args.output = f"notes/results_chrf.{fmt}"
        if args.output_tok is None:
            args.output_tok = f"notes/results_tokenization.{fmt}"
        if args.output_table is None:
            args.output_table = f"notes/results_table.{fmt}"
        if args.output_fertility is None:
            args.output_fertility = f"notes/results_fertility.{fmt}"

    return args


def savefig(fig, path, fmt):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format=fmt, bbox_inches="tight",
                dpi=300 if fmt == "png" else None)
    plt.close()
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════

def draw_group_bracket(ax, x_indices, label, FS=1.0, y_line=-0.26, y_text=-0.29):
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    left  = min(x_indices) - 0.35
    right = max(x_indices) + 0.35
    cx    = np.mean(x_indices)
    line = Line2D([left, right], [y_line, y_line],
                  transform=trans, color=SLATE, lw=0.9, clip_on=False)
    ax.add_line(line)
    tick_h = 0.016
    for xp in [left, right]:
        t = Line2D([xp, xp], [y_line, y_line + tick_h],
                   transform=trans, color=SLATE, lw=0.9, clip_on=False)
        ax.add_line(t)
    ax.text(cx, y_text, label, transform=trans,
            ha="center", va="top", fontsize=9*FS, fontweight="bold",
            color=NAVY, clip_on=False)


def plot_overview(args, FS=1.0):
    systems = [(size, mean("Zero-shot NLLB", size, "arn-es"),
                mean("Zero-shot NLLB", size, "es-arn"), "zero") for size in SIZES]
    systems += [
        ("Llama\n3.1 8B", mean("Llama 3.1 8B", "8B", "arn-es"),
         mean("Llama 3.1 8B", "8B", "es-arn"), "llm"),
        ("Aya\nExp. 8B", mean("Aya Expanse 8B", "8B", "arn-es"),
         mean("Aya Expanse 8B", "8B", "es-arn"), "llm"),
    ]
    systems += [(size, max(mean(c, size, "arn-es") for c in CONDITIONS),
                 max(mean(c, size, "es-arn") for c in CONDITIONS), "ft") for size in SIZES]

    n_zero = 3; n_llm = 2; n_ft = 3
    n_total = n_zero + n_llm + n_ft

    idx_zero = list(range(0, n_zero))
    idx_llm  = list(range(n_zero, n_zero + n_llm))
    idx_ft   = list(range(n_zero + n_llm, n_total))

    colors = {"zero": SLATE, "llm": SLATE, "ft": ORANGE}

    all_values = [ae for _, ae, _, _ in systems] + [ea for _, _, ea, _ in systems]
    global_ymax = max(all_values) * 1.35

    fw, fh = (13, 4.5) if args.poster else (13, 5.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fw, fh), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.08, bottom=0.48, top=0.93)

    for ax, is_arn_es, dir_label in [
        (ax1, True,  "Mapudungun → Spanish"),
        (ax2, False, "Spanish → Mapudungun"),
    ]:
        labels, values, bar_colors, bar_kinds = [], [], [], []
        for lbl, ae, ea, ck in systems:
            labels.append(lbl)
            values.append(ae if is_arn_es else ea)
            bar_colors.append(colors[ck])
            bar_kinds.append(ck)

        x = np.arange(n_total)
        bars = ax.bar(x, values, color=bar_colors, width=0.6,
                      edgecolor="white", linewidth=0.5)
        # All bars get value labels; fine-tuned bars get bold navy, baselines get smaller slate
        for bar, val, ck in zip(bars, values, bar_kinds):
            if ck == "ft":
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=10*FS,
                        fontweight="bold", color=NAVY)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8*FS,
                        color=NAVY)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8*FS, ha="center")

        div_x = (n_zero + n_llm) - 0.5
        ax.axvline(x=div_x, color=SLATE, linestyle="--", linewidth=1.1, alpha=0.7)
        _blend = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(div_x - 0.12, 0.94, "← Baselines", fontsize=9*FS,
                color=NAVY, va="top", ha="right", style="italic", transform=_blend)
        ax.text(div_x + 0.12, 0.94, "Fine-tuned →", fontsize=9*FS,
                color=NAVY, va="top", ha="left", style="italic", transform=_blend)

        draw_group_bracket(ax, idx_zero, "NLLB\n(zero-shot)", FS=FS)
        draw_group_bracket(ax, idx_llm,  "Prompted\nLLM", FS=FS)
        draw_group_bracket(ax, idx_ft,   "Fine-tuned NLLB\n(best condition)", FS=FS)

        ax.set_ylabel("chrF++", fontsize=12*FS)
        ax.set_title(dir_label, fontsize=13*FS, fontweight="bold", pad=10, color=NAVY)
        ax.set_ylim(0, global_ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SLATE)
        ax.spines["bottom"].set_color(SLATE)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--", color=SLATE)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelleft=True, labelsize=10*FS)

    # No legend — group bracket labels explain the groups

    savefig(fig, args.output, args.format)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(args, FS=1.0):
    cond_display = ["Standard BPE", "Morfessor-VC", "Morfessor", "Optuna BPE",
                    "Morfessor-BPE", "Joint-5K BPE", "UnigramLM", "Mono BPE"]
    mvc_row = 1

    data_ae = np.array([[d[i][0] for i in range(3)] for _, d in ALL_CONDITIONS])
    data_ea = np.array([[d[i][1] for i in range(3)] for _, d in ALL_CONDITIONS])

    vlo = min(data_ae.min(), data_ea.min()) - 1.0
    vhi = max(data_ae.max(), data_ea.max()) + 1.0

    if args.poster:
        # Two-panel: arn→es (with condition labels) + es→arn (no y-labels), shared colorbar
        fw, fh = (12, 4.0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fw, fh))
        fig.subplots_adjust(left=0.20, right=0.86, wspace=0.06, top=0.88, bottom=0.12)

        for ax, data, title, show_ylabels in [
            (ax1, data_ae, "Mapudungun → Spanish", True),
            (ax2, data_ea, "Spanish → Mapudungun", False),
        ]:
            im = ax.imshow(data, aspect="auto", cmap=BYU_CMAP, vmin=vlo, vmax=vhi)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(SIZES, fontsize=10*FS)
            ax.set_yticks(range(8))
            if show_ylabels:
                ax.set_yticklabels(cond_display, fontsize=10*FS)
            else:
                ax.set_yticklabels([])
            ax.set_title(title, fontsize=11*FS, fontweight="bold", pad=8, color=NAVY)

            for j in range(3):
                col_vals = data[:, j]
                best = col_vals.max()
                for i in range(8):
                    v = data[i, j]
                    weight = "bold" if v == best else "normal"
                    bg_norm = (v - vlo) / (vhi - vlo)
                    txt_color = "white" if bg_norm > 0.5 else NAVY
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=9*FS, color=txt_color, fontweight=weight)

            for j in range(3):
                ax.add_patch(plt.Rectangle((j - 0.5, mvc_row - 0.5), 1, 1,
                    fill=False, edgecolor=ORANGE, lw=2.5, zorder=5))

        # Single shared colorbar on the far right
        cbar_ax = fig.add_axes([0.88, 0.12, 0.018, 0.76])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.ax.tick_params(labelsize=9*FS)
        cb.set_label("chrF++", fontsize=10*FS)

    else:
        fw, fh = (11, 4.8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fw, fh))
        fig.subplots_adjust(left=0.18, right=0.93, wspace=0.55, top=0.88, bottom=0.10)

        for ax, data, title in [
            (ax1, data_ae, "Mapudungun → Spanish"),
            (ax2, data_ea, "Spanish → Mapudungun"),
        ]:
            im = ax.imshow(data, aspect="auto", cmap=BYU_CMAP, vmin=vlo, vmax=vhi)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(SIZES, fontsize=10*FS)
            ax.set_yticks(range(8))
            ax.set_yticklabels(cond_display, fontsize=10*FS)
            ax.set_title(title, fontsize=11*FS, fontweight="bold", pad=8, color=NAVY)

            for j in range(3):
                col_vals = data[:, j]
                best = col_vals.max()
                for i in range(8):
                    v = data[i, j]
                    weight = "bold" if v == best else "normal"
                    bg_norm = (v - vlo) / (vhi - vlo)
                    txt_color = "white" if bg_norm > 0.5 else NAVY
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=9*FS, color=txt_color, fontweight=weight)

            for j in range(3):
                ax.add_patch(plt.Rectangle((j - 0.5, mvc_row - 0.5), 1, 1,
                    fill=False, edgecolor=ORANGE, lw=2.5, zorder=5))

            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=9*FS)
            cb.set_label("chrF++", fontsize=10*FS)

    savefig(fig, args.output_tok, args.format)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FERTILITY CHART
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fertility(args, FS=1.0):
    fertility = {condition: values["fertility_train"]
                 for condition, values in STATS["segmentation"]["conditions"].items()}
    ordered = sorted(fertility, key=fertility.get, reverse=True)
    conds = [condition + " (ours)" if condition == "Morfessor-VC" else condition
             for condition in ordered]
    fertilities = [fertility[condition] for condition in ordered]
    bar_colors = [ORANGE if condition == "Morfessor-VC" else
                  ROYAL if condition == "Morfessor" else SLATE for condition in ordered]

    fw, fh = (6.5, 2.2) if args.poster else (6.5, 3.8)
    left_margin = 0.35 if args.poster else 0.28
    fig, ax = plt.subplots(figsize=(fw, fh))
    fig.subplots_adjust(left=left_margin, right=0.97, top=0.93, bottom=0.15)
    y = np.arange(len(conds))
    bars = ax.barh(y, fertilities, color=bar_colors, height=0.55, zorder=3, left=0.01)
    for bar, v in zip(bars, fertilities):
        ax.text(v + 0.03, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=10*FS, fontweight="bold", color=NAVY, zorder=4)
    ax.axvline(1.0, color=SLATE, ls="--", lw=1.0, alpha=0.7, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(conds, fontsize=10*FS)
    ax.set_xlabel("Tokens per whitespace word (fertility)", fontsize=10*FS)
    ax.tick_params(axis="x", labelsize=9*FS)
    ax.set_xlim(0, 2.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)
    ax.text(1.02, len(conds) - 0.6, "ideal (1.0)", fontsize=8*FS,
            color=NAVY, style="italic", va="bottom")
    ax.xaxis.grid(True, alpha=0.25, ls="--", zorder=0, color=SLATE)
    ax.set_axisbelow(True)

    savefig(fig, args.output_fertility, args.format)


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    FS = 1.4 if args.poster else 1.0
    plot_overview(args, FS)
    plot_heatmap(args, FS)
    plot_fertility(args, FS)


if __name__ == "__main__":
    main()
