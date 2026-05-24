#!/usr/bin/env python3
"""Generate a compact case-study figure for LocalSearch-R1."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch, Rectangle


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


def card(ax, xy, wh, title, edge="#333333", face="white"):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, zorder=1,
    ))
    ax.add_patch(Rectangle(
        (x, y + h - 0.070), w, 0.070,
        transform=ax.transAxes, facecolor=LIGHT_GRAY,
        edgecolor=edge, linewidth=0.7, zorder=2,
    ))
    ax.text(x + w / 2, y + h - 0.035, title,
            ha="center", va="center", fontsize=8.7, fontweight="bold",
            transform=ax.transAxes, zorder=3)


def text(ax, x, y, s, **kwargs):
    defaults = dict(ha="left", va="top", fontsize=8, color="#111111",
                    transform=ax.transAxes, zorder=5)
    defaults.update(kwargs)
    ax.text(x, y, s, **defaults)


def highlight(ax, x, y, w, h, color, alpha=0.18):
    ax.add_patch(Rectangle(
        (x, y), w, h, transform=ax.transAxes,
        facecolor=color, edgecolor="none", alpha=alpha, zorder=3,
    ))


def mini_box(ax, xy, wh, label, color, face_alpha=0.12, fontsize=4.9,
             weight="bold"):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=0.55, edgecolor=color, facecolor=color,
        alpha=face_alpha, transform=ax.transAxes, zorder=2,
    ))
    text(ax, x + w / 2, y + h / 2, label, ha="center", va="center",
         fontsize=fontsize, color=color, fontweight=weight, linespacing=0.95)


def arrow(ax, start, end, color="#777777"):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=7,
        linewidth=0.65, color=color, zorder=3,
        shrinkA=1.5, shrinkB=1.5,
    ))


def generate():
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left panel: query and program.
    card(ax, (0.025, 0.055), (0.58, 0.88), "Generated Retrieval Program")

    ax.add_patch(Rectangle((0.055, 0.826), 0.006, 0.027,
                           transform=ax.transAxes, facecolor=GRAY,
                           edgecolor="none", zorder=3))
    text(ax, 0.067, 0.849, "Natural language query", fontsize=6.7,
         fontweight="bold", color=GRAY)
    query = ("Find red Nike running shoes under $100 with good reviews\n"
             "that look like classic sneakers")
    ax.add_patch(FancyBboxPatch(
        (0.055, 0.692), 0.515, 0.085,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        linewidth=0.6, edgecolor="#BBBBBB", facecolor="#FFFFFF",
        transform=ax.transAxes, zorder=2,
    ))
    text(ax, 0.072, 0.755, query, fontsize=6.15, linespacing=1.05)

    ax.add_patch(Rectangle((0.055, 0.638), 0.006, 0.027,
                           transform=ax.transAxes, facecolor=GRAY,
                           edgecolor="none", zorder=3))
    text(ax, 0.067, 0.661, "Program", fontsize=6.7,
         fontweight="bold", color=GRAY)
    code_bg = FancyBboxPatch(
        (0.055, 0.160), 0.515, 0.440,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=0.55, edgecolor="#DDDDDD", facecolor="#FCFCFC",
        transform=ax.transAxes, zorder=1.8,
    )
    ax.add_patch(code_bg)
    code_x = 0.080
    code_y = 0.570
    line_h = 0.0405
    mono = dict(fontfamily="DejaVu Sans Mono", fontsize=5.45)

    # SQL block with semantic highlights.
    lines = [
        ("SQL:", "#111111"),
        ("  WHERE brand = 'Nike'", BLUE),
        ("    AND price < 100", BLUE),
        ("    AND average_rating >= 4.0", BLUE),
        ("    AND product_id IN <text_0>", GREEN),
        ("    AND product_id IN <image_0>", ORANGE),
        ("", "#111111"),
        ("Retrieval placeholders:", "#111111"),
        ('  <text_0>  = "red running shoes"', GREEN),
        ('  <image_0> = "classic red Nike sneaker"', ORANGE),
    ]
    for idx, color, width in [
        (1, BLUE, 0.455), (2, BLUE, 0.455), (3, BLUE, 0.455),
        (4, GREEN, 0.455), (5, ORANGE, 0.455),
        (8, GREEN, 0.455), (9, ORANGE, 0.455),
    ]:
        highlight(ax, code_x - 0.010, code_y - idx * line_h - 0.023,
                  width, 0.031, color, 0.14 if color == BLUE else 0.16)
    for i, (line, color) in enumerate(lines):
        weight = "bold" if line in ["SQL:", "Retrieval placeholders:"] else "normal"
        size = 5.25 if weight == "bold" else mono["fontsize"]
        text(ax, code_x, code_y - i * line_h, line, color=color,
             fontweight=weight, fontfamily=mono["fontfamily"], fontsize=size)

    # Right panel: evidence table.
    card(ax, (0.625, 0.055), (0.35, 0.88), "Execution Evidence")
    table_x = 0.642
    table_y = 0.790
    col_w = [0.098, 0.160, 0.064]
    row_h = 0.062
    headers = ["Constraint", "Evidence", "Outcome"]
    rows = [
        ("brand='Nike'", "P001.brand = Nike", "pass", BLUE),
        ("price<100", "P001.price = $89", "pass", BLUE),
        ("average_\nrating>=4.0", "P001.average_rating=4.5", "pass", BLUE),
        ("<text_0>", "P001 in text cand.", "pass", GREEN),
        ("<image_0>", "P001 in image cand.", "pass", ORANGE),
        ("Final\nSQL", "P001\nRed Nike shoe", "correct", RED),
    ]

    # Header.
    x0 = table_x
    y0 = table_y
    for j, header in enumerate(headers):
        w = col_w[j]
        ax.add_patch(Rectangle((x0, y0), w, row_h, facecolor="#F2F2F2",
                               edgecolor="#777777", linewidth=0.55,
                               transform=ax.transAxes, zorder=2))
        text(ax, x0 + w / 2, y0 + row_h * 0.50, header,
             ha="center", va="center", fontsize=5.55, fontweight="bold")
        x0 += w

    # Rows.
    for i, (constraint, evidence, outcome, color) in enumerate(rows):
        x0 = table_x
        y = table_y - (i + 1) * row_h
        bg = "#FFFFFF" if i % 2 == 0 else "#FAFAFA"
        if i == len(rows) - 1:
            bg = "#FFF1F1"
        cells = [constraint, evidence, outcome]
        for j, cell in enumerate(cells):
            w = col_w[j]
            ax.add_patch(Rectangle((x0, y), w, row_h, facecolor=bg,
                                   edgecolor="#777777", linewidth=0.55,
                                   transform=ax.transAxes, zorder=2))
            tcolor = color if j == 0 or i == len(rows) - 1 else "#111111"
            fweight = "bold" if i == len(rows) - 1 or j == 2 else "normal"
            text(ax, x0 + w / 2, y + row_h * 0.50, cell,
                 ha="center", va="center", fontsize=5.0, color=tcolor,
                 fontweight=fweight, linespacing=0.92)
            x0 += w

    # Compact fusion summary fills the lower-right space without repeating the
    # full system overview.
    text(ax, 0.645, 0.372, "Logical fusion", fontsize=5.8,
         fontweight="bold", color=GRAY, va="bottom")
    mini_box(ax, (0.650, 0.286), (0.108, 0.047), "metadata\nfilters", BLUE,
             fontsize=4.6)
    mini_box(ax, (0.650, 0.214), (0.108, 0.047), "text\ncandidates", GREEN,
             fontsize=4.6)
    mini_box(ax, (0.650, 0.142), (0.108, 0.047), "image\ncandidates", ORANGE,
             fontsize=4.6)

    fusion_x, fusion_y, fusion_w, fusion_h = 0.792, 0.217, 0.042, 0.064
    ax.add_patch(FancyBboxPatch(
        (fusion_x, fusion_y), fusion_w, fusion_h,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        linewidth=0.65, edgecolor="#777777", facecolor="#F5F5F5",
        transform=ax.transAxes, zorder=2,
    ))
    text(ax, fusion_x + fusion_w / 2, fusion_y + fusion_h / 2,
         "AND", ha="center", va="center", fontsize=5.1,
         color="#555555", fontweight="bold")

    final_x, final_y, final_w, final_h = 0.862, 0.198, 0.098, 0.092
    ax.add_patch(FancyBboxPatch(
        (final_x, final_y), final_w, final_h,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        linewidth=0.65, edgecolor=RED, facecolor="#FFF1F1",
        transform=ax.transAxes, zorder=2,
    ))
    text(ax, final_x + final_w / 2, final_y + final_h / 2,
         "P001\nRed Nike shoe", ha="center", va="center",
         fontsize=5.3, color=RED, fontweight="bold", linespacing=0.92)
    fusion_center = (fusion_x + fusion_w / 2, fusion_y + fusion_h / 2)
    fusion_inputs = [
        ((0.759, 0.310), (fusion_x - 0.002, fusion_y + fusion_h * 0.80)),
        ((0.759, 0.238), (fusion_x - 0.002, fusion_y + fusion_h * 0.50)),
        ((0.759, 0.166), (fusion_x - 0.002, fusion_y + fusion_h * 0.20)),
    ]
    for start, end in fusion_inputs:
        arrow(ax, start, end)
    arrow(ax, (fusion_x + fusion_w, fusion_center[1]),
          (final_x, final_y + final_h / 2), color=RED)
    text(ax, 0.910, 0.150, "all constraints satisfied",
         ha="center", fontsize=4.75, color=RED, fontweight="bold")

    out_pdf = os.path.join(OUT_DIR, "fig_case_study.pdf")
    out_png = os.path.join(OUT_DIR, "fig_case_study.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    generate()
