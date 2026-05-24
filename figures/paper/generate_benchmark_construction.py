#!/usr/bin/env python3
"""Generate a benchmark construction figure for LocalSearch-R1."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


BLUE = "#4E79A7"
GREEN = "#59A14F"
ORANGE = "#F28E2B"
RED = "#D62728"
GRAY = "#555555"
LIGHT_GRAY = "#F7F7F7"
INK = "#222222"


def text(ax, x, y, s, **kwargs):
    defaults = dict(ha="left", va="top", fontsize=7, color=INK,
                    transform=ax.transAxes, zorder=5)
    defaults.update(kwargs)
    ax.text(x, y, s, **defaults)


def round_box(ax, xy, wh, edge="#BBBBBB", face="#FFFFFF", lw=0.6,
              radius=0.012, z=2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def stage(ax, x, title):
    w = 0.205
    round_box(ax, (x, 0.08), (w, 0.78), edge="#333333", face="#FFFFFF",
              lw=0.85, radius=0.014, z=1)
    ax.add_patch(Rectangle((x, 0.79), w, 0.07, transform=ax.transAxes,
                           facecolor=LIGHT_GRAY, edgecolor="#333333",
                           linewidth=0.6, zorder=2))
    text(ax, x + w / 2, 0.825, title, ha="center", va="center",
         fontsize=7.8, fontweight="bold")
    return w


def chip(ax, xy, wh, label, color, fontsize=5.8):
    round_box(ax, xy, wh, edge=color, face=color, lw=0.55,
              radius=0.009, z=2)
    ax.patches[-1].set_alpha(0.16)
    text(ax, xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, label,
         ha="center", va="center", fontsize=fontsize, color=color,
         fontweight="bold", linespacing=0.95)


def small_chip(ax, xy, wh, label, color, fontsize=4.0):
    x, y = xy
    w, h = wh
    bg_colors = {
        BLUE: "#EAF1F8",
        GREEN: "#EAF5E8",
        ORANGE: "#FFF0DE",
        RED: "#FDE8E8",
    }
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.001,rounding_size=0.0045",
        linewidth=0.55, edgecolor=color, facecolor=bg_colors.get(color, "#F7F7F7"),
        alpha=0.95,
        transform=ax.transAxes, zorder=2,
    ))
    text(ax, x + w / 2, y + h / 2, label, ha="center", va="center",
         fontsize=fontsize, color=color, fontweight="bold")


def arrow(ax, start, end, color="#333333", scale=10, lw=0.8):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
        color=color, shrinkA=2.0, shrinkB=2.0, zorder=4,
    ))


def item_card(ax, xy, wh, title):
    x, y = xy
    w, h = wh
    round_box(ax, (x, y), (w, h), edge="#AAAAAA", face="#FCFCFC",
              lw=0.55, radius=0.010, z=2)
    text(ax, x + 0.010, y + h - 0.018, title, fontsize=5.5,
         fontweight="bold", color=GRAY)
    rows = [
        ("brand: Nike", BLUE),
        ("price: $89", BLUE),
        ("rating: 4.5", BLUE),
        ("text description", GREEN),
        ("image", ORANGE),
    ]
    row_h = 0.026
    for i, (label, color) in enumerate(rows):
        yy = y + h - 0.052 - i * row_h
        chip(ax, (x + 0.010, yy), (w - 0.020, row_h * 0.70), label, color,
             fontsize=4.35)


def generate():
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.025, 0.275, 0.525, 0.775]
    titles = [
        "1. Local Corpus",
        "2. Query Intents",
        "3. Executable Targets",
        "4. Evaluation Split",
    ]
    for x, title in zip(xs, titles):
        stage(ax, x, title)

    for i in range(3):
        arrow(ax, (xs[i] + 0.205, 0.47), (xs[i + 1], 0.47))

    # Stage 1: local item collection.
    text(ax, xs[0] + 0.020, 0.750, "Local items share a\nsearchable schema",
         fontsize=5.4, color=GRAY, fontweight="bold", linespacing=1.05)
    round_box(ax, (xs[0] + 0.026, 0.385), (0.154, 0.230),
              edge="#999999", face="#FCFCFC", lw=0.55, radius=0.010)
    text(ax, xs[0] + 0.045, 0.600, "item record", fontsize=5.35,
         color=INK, fontweight="bold")
    chip(ax, (xs[0] + 0.045, 0.500), (0.116, 0.034), "metadata fields",
         BLUE, fontsize=4.8)
    chip(ax, (xs[0] + 0.045, 0.452), (0.116, 0.034), "text content",
         GREEN, fontsize=4.8)
    chip(ax, (xs[0] + 0.045, 0.404), (0.116, 0.034), "image / file",
         ORANGE, fontsize=4.8)
    text(ax, xs[0] + 0.103, 0.333, "local domains",
         ha="center", fontsize=5.0, color=GRAY, fontweight="bold")
    domain_w = 0.052
    domain_h = 0.034
    domain_gap = 0.008
    domain_start = xs[0] + 0.103 - (3 * domain_w + 2 * domain_gap) / 2
    domain_offsets = [0.0, -0.008, 0.0]
    for i, (label, color) in enumerate([
        ("product", BLUE), ("email", GREEN), ("document", ORANGE),
    ]):
        small_chip(ax, (domain_start + i * (domain_w + domain_gap),
                        0.249 + domain_offsets[i]),
                   (domain_w, domain_h), label, color, fontsize=3.7)
    text(ax, xs[0] + 0.103, 0.195, "closed-world local index",
         ha="center", fontsize=4.9, color=GRAY)

    # Stage 2: intent composition.
    text(ax, xs[1] + 0.020, 0.750, "Compose user queries from\northogonal intent types",
         fontsize=5.4, color=GRAY, fontweight="bold", linespacing=1.05)
    chip(ax, (xs[1] + 0.026, 0.608), (0.150, 0.046), "metadata filter", BLUE)
    chip(ax, (xs[1] + 0.026, 0.520), (0.150, 0.046), "text intent", GREEN)
    chip(ax, (xs[1] + 0.026, 0.432), (0.150, 0.046), "image intent", ORANGE)
    text(ax, xs[1] + 0.101, 0.586, "+", ha="center", va="center",
         fontsize=7.0, color=GRAY, fontweight="bold")
    text(ax, xs[1] + 0.101, 0.498, "+", ha="center", va="center",
         fontsize=7.0, color=GRAY, fontweight="bold")
    round_box(ax, (xs[1] + 0.052, 0.300), (0.098, 0.056),
              edge="#777777", face="#F5F5F5", lw=0.55, radius=0.010)
    text(ax, xs[1] + 0.101, 0.328, "AND / OR", ha="center", va="center",
         fontsize=5.6, color=GRAY, fontweight="bold")
    arrow(ax, (xs[1] + 0.101, 0.420), (xs[1] + 0.101, 0.356),
          color="#777777", scale=7, lw=0.55)

    # Stage 3: executable target supervision.
    text(ax, xs[2] + 0.020, 0.750, "Generate executable labels\nfor supervised evaluation",
         fontsize=5.4, color=GRAY, fontweight="bold", linespacing=1.05)
    round_box(ax, (xs[2] + 0.022, 0.270), (0.162, 0.330),
              edge="#888888", face="#FCFCFC", lw=0.55, radius=0.010)
    text(ax, xs[2] + 0.036, 0.585, "SQL target", fontsize=5.4,
         fontweight="bold")
    chip(ax, (xs[2] + 0.038, 0.475), (0.130, 0.041), "SQL filters", BLUE,
         fontsize=5.2)
    chip(ax, (xs[2] + 0.038, 0.416), (0.130, 0.041), "<text_0> query",
         GREEN, fontsize=5.2)
    chip(ax, (xs[2] + 0.038, 0.357), (0.130, 0.041), "<image_0> query",
         ORANGE, fontsize=5.2)
    chip(ax, (xs[2] + 0.038, 0.298), (0.130, 0.041), "ground-truth IDs",
         RED, fontsize=5.2)

    # Stage 4: train/dev/test and buckets.
    text(ax, xs[3] + 0.020, 0.750, "Evaluate by split and\nquery difficulty",
         fontsize=5.4, color=GRAY, fontweight="bold", linespacing=1.05)
    split_y = 0.580
    split_w = 0.050
    for i, (label, face) in enumerate([
        ("train", "#DCE7F4"),
        ("dev", "#F2F2F2"),
        ("test", "#F2F2F2"),
    ]):
        xx = xs[3] + 0.026 + i * 0.060
        round_box(ax, (xx, split_y), (split_w, 0.070),
                  edge="#999999", face=face, lw=0.55, radius=0.010)
        text(ax, xx + split_w / 2, split_y + 0.035, label,
             ha="center", va="center", fontsize=5.7,
             fontweight="bold" if label == "train" else "normal")
    text(ax, xs[3] + 0.026, 0.535, "difficulty buckets", fontsize=5.4,
         color=GRAY, fontweight="bold")
    buckets = ["single-condition", "compositional", "cross-modal", "hard negatives"]
    for i, bucket in enumerate(buckets):
        yy = 0.435 - i * 0.057
        round_box(ax, (xs[3] + 0.026, yy), (0.154, 0.038),
                  edge="#BBBBBB", face="#F7F7F7", lw=0.45, radius=0.008)
        text(ax, xs[3] + 0.103, yy + 0.019, bucket,
             ha="center", va="center", fontsize=5.1, color=INK)

    out_pdf = os.path.join(OUT_DIR, "fig_benchmark_construction.pdf")
    out_png = os.path.join(OUT_DIR, "fig_benchmark_construction.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    generate()
