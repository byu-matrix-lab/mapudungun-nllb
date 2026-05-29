#!/usr/bin/env python3
"""
plot_results.py

Generates three publication-ready figures:
  1. notes/results_chrf.png         — overview bar chart (prior work vs our systems)
  2. notes/results_tokenization.png — grouped bar chart: all 8 tokenization conditions × 3 sizes
  3. notes/results_table.png        — formatted table image of all chrF++ results

Usage:
  uv run python scripts/analysis/plot_results.py
  uv run python scripts/analysis/plot_results.py --output notes/results_chrf.png \
      --output-tok notes/results_tokenization.png --output-table notes/results_table.png
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
ALL_CONDITIONS = [
    ("Standard BPE",  [(34.90, 39.87), (40.64, 41.94), (42.85, 42.89)]),
    ("Joint-5K BPE",  [(42.36, 38.17), (44.68, 41.06), (45.25, 42.30)]),
    ("Mono BPE",      [(41.72, 38.04), (43.99, 40.62), (44.85, 42.02)]),
    ("Optuna BPE",    [(42.41, 38.79), (44.67, 41.34), (45.21, 42.33)]),
    ("Morfessor",     [(43.09, 39.90), (45.40, 42.00), (45.51, 42.76)]),
    ("Morfessor-VC",  [(43.16, 39.89), (45.37, 42.04), (45.84, 42.77)]),
    ("Morfessor-BPE", [(42.80, 38.74), (44.97, 41.25), (45.56, 42.40)]),
    ("UnigramLM",     [(42.18, 38.20), (44.64, 41.05), (45.07, 42.30)]),
]
SIZES = ["600M", "1.3B", "3.3B"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",           default="notes/results_chrf.png")
    parser.add_argument("--output-tok",       default="notes/results_tokenization.png")
    parser.add_argument("--output-table",     default="notes/results_table.png")
    parser.add_argument("--output-fertility", default="notes/results_fertility.png")
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
    systems = [
        ("600M",         12.86,  7.37, "zero"),
        ("1.3B",         13.43,  9.91, "zero"),
        ("3.3B",         13.19,  9.42, "zero"),
        ("Llama\n3.1 8B", 13.96, 11.32, "llm"),
        ("Aya\nExp. 8B",  15.92, 12.36, "llm"),
        # Best tokenization per direction: Morfessor-VC (arn→es) / Standard BPE (es→arn)
        ("600M",         43.16, 39.87, "ft"),
        ("1.3B",         45.37, 41.94, "ft"),
        ("3.3B",         45.84, 42.89, "ft"),
    ]

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
# 3. TABLE IMAGE  (all conditions × sizes × directions)
# ═══════════════════════════════════════════════════════════════════════════════

# Baselines (shown at bottom of table)
BASELINES = [
    # (label, arn_es, es_arn, size_label)
    ("Zero-shot NLLB 600M",  12.86,  7.37, "600M"),
    ("Zero-shot NLLB 1.3B",  13.43,  9.91, "1.3B"),
    ("Zero-shot NLLB 3.3B",  13.19,  9.42, "3.3B"),
    ("Llama 3.1 8B",         13.96, 11.32, "8B"),
    ("Aya Expanse 8B",        15.92, 12.36, "8B"),
]


def plot_table(args):
    col_headers = [
        "Condition",
        "600M\narn→es", "600M\nes→arn",
        "1.3B\narn→es", "1.3B\nes→arn",
        "3.3B\narn→es", "3.3B\nes→arn",
    ]

    # Build rows for fine-tuned conditions
    rows = []
    for cond, data in ALL_CONDITIONS:
        row = [cond]
        for si in range(3):
            row.append(f"{data[si][0]:.2f}")
            row.append(f"{data[si][1]:.2f}")
        rows.append(row)

    # Find column-wise best (finetuned only)
    col_best = {}
    for ci in range(1, 7):
        vals = []
        for row in rows:
            try:
                vals.append(float(row[ci]))
            except ValueError:
                pass
        col_best[ci] = max(vals) if vals else None

    # Baseline rows
    baseline_rows = []
    for lbl, ae, ea, size in BASELINES:
        row = [lbl, "—", "—", "—", "—", "—", "—"]
        if size == "600M": row[1], row[2] = f"{ae:.2f}", f"{ea:.2f}"
        elif size == "1.3B": row[3], row[4] = f"{ae:.2f}", f"{ea:.2f}"
        elif size == "3.3B": row[5], row[6] = f"{ae:.2f}", f"{ea:.2f}"
        elif size == "8B":
            # show in last two columns (label them 8B elsewhere — just place values)
            row[5], row[6] = f"{ae:.2f}", f"{ea:.2f}"
        baseline_rows.append(row)

    all_rows = rows + [None] + baseline_rows   # None = separator row

    n_display = len([r for r in all_rows if r is not None])
    fig_h = 0.40 * (n_display + 3)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")

    display_rows = [r for r in all_rows if r is not None]
    is_baseline  = [False] * len(rows) + [True] * len(baseline_rows)

    t = ax.table(
        cellText=display_rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.55)

    # Header
    for j in range(len(col_headers)):
        cell = t[0, j]
        cell.set_facecolor("#1A3A5C")
        cell.set_text_props(color="white", fontweight="bold")

    # Row styling
    MVC_IDX = next(i for i, (c, _) in enumerate(ALL_CONDITIONS) if c == "Morfessor-VC")
    SEP_IDX = len(rows)   # where separator would be — displayed as first baseline row

    for i, (row, bl) in enumerate(zip(display_rows, is_baseline)):
        alt = "#F7F7F7" if i % 2 == 0 else "#FFFFFF"
        if bl:
            alt = "#EAF0FB"
        if i == MVC_IDX:
            alt = "#FFF3E0"

        # Draw a light top border on first baseline row
        for j in range(len(col_headers)):
            cell = t[i + 1, j]
            cell.set_facecolor(alt)

            if bl and i == SEP_IDX:
                # top separator line via edgecolor hack — matplotlib doesn't do per-edge easily
                pass  # rely on the color change as visual separator

            val_str = row[j]
            if j > 0 and val_str != "—" and not bl:
                try:
                    if float(val_str) == col_best.get(j):
                        cell.set_text_props(fontweight="bold")
                except ValueError:
                    pass
            if i == MVC_IDX:
                cell.set_text_props(color="#B35000")

    ax.set_title(
        "chrF++ scores — all tokenization conditions, model sizes, and translation directions\n"
        "(bold = column-best; highlighted row = Morfessor-VC; shaded rows = baselines)",
        fontsize=10, fontweight="bold", pad=14, loc="left"
    )

    out = Path(args.output_table)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════

def plot_fertility(args):
    C_MVC  = "#E07020"
    C_MORF = "#1565C0"
    C_OTHER= "#AAAAAA"
    conds       = ["Mono BPE", "UnigramLM", "Joint-5K BPE", "Morfessor-BPE",
                   "Optuna BPE", "Morfessor", "Morfessor-VC (ours)"]
    fertilities = [2.037, 1.982, 1.665, 1.628, 1.552, 1.133, 1.130]
    bar_colors  = [C_OTHER, C_OTHER, C_OTHER, C_OTHER, C_OTHER, C_MORF, C_MVC]

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
