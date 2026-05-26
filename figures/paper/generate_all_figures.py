#!/usr/bin/env python3
"""Generate all paper-quality figures for LocalSearch-R1 (EMNLP 2026).

Output: PDF vector figures in figures/paper/
Style: ACL/EMNLP conference standard — compact, professional palette, readable at column width.

v4: Aligned all figures with 实验结果分析报告.md (2026-05-24).
  - Fixed: Search-R1 (0.435), DeepRetrieval (0.625), SQR email (0.630)
  - Updated: E3/E5 to match doc (add SQL-R1, 8B SFT; remove GPT-5.5 from ecom e3/e5)
  - Updated: E6 (add SFT 8B), E8 (4B vs 8B), E9 (fix GPT-5.5 data)
  - Added: E10 OOD evaluation figure
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))

# ── Category color scheme (same category = same hue) ────────────────────────
OURS_COLOR   = '#D62728'   # red — always for "Ours"
OURS_COLOR2  = '#E45756'   # lighter red for DAPO
SFT_COLOR    = '#FF7F0E'   # orange
SFT8B_COLOR  = '#FFB366'   # lighter orange for 8B SFT
LLM_COLOR    = '#4E79A7'   # steel blue (all commercial LLMs)
B2_COLOR     = '#59A14F'   # green (LLM-enhanced retrieval)
B3_COLOR     = '#B07AA1'   # purple (RL-based retrieval: DeepRetrieval, SQL-R1, Search-R1)
B1_COLOR     = '#76B7B2'   # teal (traditional retrieval)
B4_COLOR     = '#BAB0AC'   # gray (GraphRAG)
FEWSHOT_COLOR = '#E7CB94'  # tan


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    png_path = path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300)
    print(f'  -> {name} + {os.path.basename(png_path)}')
    plt.close(fig)


def _load_json(rel_path):
    with open(os.path.join(ROOT_DIR, rel_path), 'r') as f:
        return json.load(f)


def _maybe_load_json(rel_path):
    path = os.path.join(ROOT_DIR, rel_path)
    if not os.path.exists(path):
        return None
    return _load_json(rel_path)


def _bar_label(ax, bars, vals, dy=0.008, fontsize=5.5, bold_first=0):
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, val + dy, f'{val:.3f}',
                ha='center', va='bottom', fontsize=fontsize,
                fontweight='bold' if i < bold_first else 'normal')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: Main Results — E-commerce (Hit@1, from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_main_ecommerce():
    groups = [
        ('Ours', [
            ('GRPO (4B)',     0.809, OURS_COLOR),
            ('DAPO (4B)',     0.808, OURS_COLOR2),
        ]),
        ('SFT', [
            ('SFT (4B)',      0.680, SFT_COLOR),
        ]),
        ('Commercial LLMs', [
            ('GPT-5.5',      0.667, LLM_COLOR),
            ('Opus 4.7',     0.635, '#6A9FCA'),
            ('Gemini 3.1',   0.587, '#93BAD9'),
            ('GPT-5.4',      0.581, '#BBCFE8'),
        ]),
        ('LLM-Enhanced', [
            ('RankGPT',      0.768, B2_COLOR),
            ('SQR',          0.606, '#82C47B'),
            ('IRCoT',        0.565, '#ABD9A5'),
            ('HyDE',         0.549, '#C8E6C2'),
        ]),
        ('Structured Query', [
            ('DeepRetrieval', 0.625, B3_COLOR),
            ('SQL-R1',        0.620, '#C4A8CC'),
            ('Search-R1',     0.435, '#E0D5E3'),
        ]),
        ('Traditional / Graph', [
            ('Reranker',     0.618, B1_COLOR),
            ('LightRAG',    0.588, '#8BB7B4'),
            ('BM25',        0.571, '#A0D0CC'),
            ('GraphRAG',    0.500, B4_COLOR),
        ]),
    ]

    labels, vals, colors, spans = [], [], [], []
    pos = 0
    for group, items in groups:
        start = pos
        for label, val, color in items:
            if val is None:
                continue
            labels.append(label)
            vals.append(val)
            colors.append(color)
            pos += 1
        spans.append((group, start, pos))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    bg_fills = ['#FFF2F2', '#FFF8F0', '#F0F4FA', '#F0FAF0', '#F5F0F7', '#F0FAFA']
    for i, (group, s, e) in enumerate(spans):
        if e <= s:
            continue
        ax.axvspan(s - 0.5, e - 0.5, color=bg_fills[i % len(bg_fills)], zorder=0)
        ax.text((s + e - 1) / 2, 0.885, group, ha='center', va='top', fontsize=5.8,
                color='#666666', fontstyle='italic', zorder=1)

    bars = ax.bar(x, vals, 0.68, color=colors, edgecolor='white', linewidth=0.35, zorder=2)
    _bar_label(ax, bars, vals, bold_first=2)
    ax.set_ylabel('Hit@1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha='right', fontsize=6.3)
    ax.set_ylim(0.37, 0.90)
    ax.set_title('E-commerce Retrieval Performance (Hit@1)', fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout()
    save(fig, 'fig_main_ecommerce.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Main Results — Email (Hit@1, from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_main_email():
    groups = [
        ('Ours', [
            ('DAPO (4B)',    0.909, OURS_COLOR),
            ('GRPO (4B)',    0.906, OURS_COLOR2),
        ]),
        ('SFT', [
            ('SFT (4B)',     0.891, SFT_COLOR),
        ]),
        ('Commercial LLMs', [
            ('GPT-5.5',     0.855, LLM_COLOR),
            ('GPT-5.4',     0.854, '#6A9FCA'),
            ('Opus 4.7',    0.830, '#93BAD9'),
        ]),
        ('LLM-Enhanced', [
            ('RankGPT',     0.888, B2_COLOR),
            ('SQR',         0.630, '#82C47B'),
            ('IRCoT',       0.511, '#ABD9A5'),
            ('HyDE',        0.483, '#C8E6C2'),
        ]),
        ('Traditional / KG', [
            ('Reranker',    0.836, B1_COLOR),
            ('BM25',        0.731, '#A0D0CC'),
            ('LightRAG',   0.389, '#8BB7B4'),
            ('Search-R1',  0.377, '#E0D5E3'),
        ]),
    ]

    labels, vals, colors, spans = [], [], [], []
    pos = 0
    for group, items in groups:
        start = pos
        for label, val, color in items:
            if val is None:
                continue
            labels.append(label)
            vals.append(val)
            colors.append(color)
            pos += 1
        spans.append((group, start, pos))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    bg_fills = ['#FFF2F2', '#FFF8F0', '#F0F4FA', '#F0FAF0', '#F0FAFA']
    for i, (group, s, e) in enumerate(spans):
        if e <= s:
            continue
        ax.axvspan(s - 0.5, e - 0.5, color=bg_fills[i % len(bg_fills)], zorder=0)
        ax.text((s + e - 1) / 2, 0.955, group, ha='center', va='top', fontsize=5.8,
                color='#666666', fontstyle='italic', zorder=1)

    bars = ax.bar(x, vals, 0.68, color=colors, edgecolor='white', linewidth=0.35, zorder=2)
    _bar_label(ax, bars, vals, dy=0.004, bold_first=2)
    ax.set_ylabel('Hit@1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha='right', fontsize=6.3)
    ax.set_ylim(0.34, 0.97)
    ax.set_title('Email Retrieval Performance (Hit@1)', fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout()
    save(fig, 'fig_main_email.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: E3 + E5 Combined — E-commerce (from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e3_e5_combined_ecommerce():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # E3: by complexity level (doc table)
    methods_e3 = ['8B SFT', 'GRPO', 'DAPO', 'SFT', 'RankGPT', 'Reranker', 'SQL-R1', 'BM25']
    data_e3 = {
        '8B SFT':     [0.873, 0.816, 0.717],
        'GRPO':       [0.864, 0.797, 0.720],
        'DAPO':       [0.860, 0.800, 0.717],
        'SFT':        [0.752, 0.646, 0.598],
        'RankGPT':    [0.717, 0.803, 0.802],
        'Reranker':   [0.473, 0.706, 0.741],
        'SQL-R1':     [0.619, 0.637, 0.590],
        'BM25':       [0.453, 0.632, 0.690],
    }
    colors_e3 = [SFT8B_COLOR, OURS_COLOR, OURS_COLOR2, SFT_COLOR, B2_COLOR, B1_COLOR, B3_COLOR, '#A0D0CC']

    levels = ['L1 (Simple)', 'L2 (Medium)', 'L3 (Complex)']
    n = len(data_e3)
    x = np.arange(3)
    w = 0.8 / n

    for i, name in enumerate(methods_e3):
        offset = (i - n/2 + 0.5) * w
        ax1.bar(x + offset, data_e3[name], w * 0.9, label=name, color=colors_e3[i],
                edgecolor='white', linewidth=0.3)

    ax1.set_ylabel('Hit@1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels, fontsize=7)
    ax1.set_ylim(0.30, 0.92)
    ax1.set_title('(a) By Query Complexity', fontsize=9, fontweight='bold')

    # E5: by modality (doc table)
    methods_e5 = ['8B SFT', 'GRPO', 'DAPO', 'SFT', 'SQL-R1', 'Reranker', 'BM25']
    data_e5 = {
        '8B SFT':     [0.851, 0.684, 0.756],
        'GRPO':       [0.843, 0.690, 0.716],
        'DAPO':       [0.843, 0.690, 0.711],
        'SFT':        [0.743, 0.632, 0.323],
        'SQL-R1':     [0.866, 0.303, 0.048],
        'Reranker':   [0.310, 0.759, 0.819],
        'BM25':       [0.256, 0.759, 0.692],
    }
    colors_e5 = [SFT8B_COLOR, OURS_COLOR, OURS_COLOR2, SFT_COLOR, B3_COLOR, B1_COLOR, '#A0D0CC']

    modalities = ['Structured', 'Text', 'Image']
    n2 = len(data_e5)
    w2 = 0.8 / n2

    for i, name in enumerate(methods_e5):
        offset = (i - n2/2 + 0.5) * w2
        ax2.bar(x + offset, data_e5[name], w2 * 0.9, label=name, color=colors_e5[i],
                edgecolor='white', linewidth=0.3)

    ax2.set_ylabel('Hit@1')
    ax2.set_xticks(x)
    ax2.set_xticklabels(modalities, fontsize=7)
    ax2.set_ylim(0.0, 0.95)
    ax2.set_title('(b) By Retrieval Modality', fontsize=9, fontweight='bold')

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    all_labels = dict(zip(labels + l2, handles + h2))
    fig.legend(all_labels.values(), all_labels.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=8, frameon=False, fontsize=6)

    fig.suptitle('E-commerce: Performance Breakdown', fontsize=11, fontweight='bold', y=1.14)
    fig.tight_layout()
    save(fig, 'fig_e3_e5_combined_ecommerce.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: E3 + E5 Combined — Email (from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e3_e5_combined_email():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # E3: by level (doc table — only 4 methods + Reranker)
    methods_e3 = ['DAPO', 'GRPO', 'SFT', 'Reranker', 'BM25']
    data_e3 = {
        'DAPO':       [0.987, 0.887, 0.798],
        'GRPO':       [0.987, 0.883, 0.793],
        'SFT':        [0.987, 0.865, 0.755],
        'Reranker':   [0.804, 0.869, 0.831],
        'BM25':       [0.647, 0.794, 0.771],
    }
    colors_e3 = [OURS_COLOR, OURS_COLOR2, SFT_COLOR, B1_COLOR, '#A0D0CC']

    levels = ['L1 (Simple)', 'L2 (Medium)', 'L3 (Complex)']
    n = len(data_e3)
    x = np.arange(3)
    w = 0.8 / n

    for i, name in enumerate(methods_e3):
        offset = (i - n/2 + 0.5) * w
        ax1.bar(x + offset, data_e3[name], w * 0.9, label=name, color=colors_e3[i],
                edgecolor='white', linewidth=0.3)

    ax1.set_ylabel('Hit@1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels, fontsize=7)
    ax1.set_ylim(0.55, 1.02)
    ax1.set_title('(a) By Query Complexity', fontsize=9, fontweight='bold')

    # E5: by modality — email has only Structured + Text
    methods_e5 = ['DAPO', 'GRPO', 'SFT', 'Reranker', 'BM25']
    data_e5 = {
        'DAPO':       [0.885, 0.891],
        'GRPO':       [0.882, 0.889],
        'SFT':        [0.863, 0.871],
        'Reranker':   [0.768, 0.848],
        'BM25':       [0.638, 0.687],
    }
    colors_e5 = [OURS_COLOR, OURS_COLOR2, SFT_COLOR, B1_COLOR, '#A0D0CC']

    modalities = ['Structured', 'Text']
    n2 = len(data_e5)
    x2 = np.arange(2)
    w2 = 0.7 / n2

    for i, name in enumerate(methods_e5):
        offset = (i - n2/2 + 0.5) * w2
        ax2.bar(x2 + offset, data_e5[name], w2 * 0.9, label=name, color=colors_e5[i],
                edgecolor='white', linewidth=0.3)

    ax2.set_ylabel('Hit@1')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(modalities, fontsize=7)
    ax2.set_ylim(0.55, 0.92)
    ax2.set_title('(b) By Retrieval Modality', fontsize=9, fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    all_labels = dict(zip(labels + l2, handles + h2))
    fig.legend(all_labels.values(), all_labels.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=5, frameon=False, fontsize=6.5)

    fig.suptitle('Email: Performance Breakdown', fontsize=11, fontweight='bold', y=1.14)
    fig.tight_layout()
    save(fig, 'fig_e3_e5_combined_email.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: Paradigm Comparison — Dual Scenario (from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_paradigm_comparison():
    methods = [
        'Retrieval Program\nSynthesis (Ours)',
        'RankGPT\n(LLM Rerank)',
        'DeepRetrieval\n(Boolean RL)',
        'SQR\n(LLM Filter)',
        'SQL-R1\n(SQL-only RL)',
        'BGE-Reranker\n(Neural Rerank)',
        'BM25\n(Lexical)',
        'GraphRAG\n(Knowledge Graph)',
        'Search-R1\n(BM25 RL)',
    ]
    ecom  = [0.809, 0.768, 0.625, 0.606, 0.620, 0.618, 0.571, 0.500, 0.435]
    email = [0.909, 0.888, None,  0.630, 0.546, 0.836, 0.731, None,  0.377]

    colors_bar = [OURS_COLOR, B2_COLOR, B3_COLOR, B2_COLOR, B3_COLOR, B1_COLOR, B1_COLOR, B4_COLOR, B3_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8),
                                     gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.08})

    y = np.arange(len(methods))
    h = 0.6

    bars1 = ax1.barh(y, ecom, height=h, color=colors_bar, edgecolor='white', linewidth=0.3)
    for i, (bar, val) in enumerate(zip(bars1, ecom)):
        fw = 'bold' if i == 0 else 'normal'
        ax1.text(val + 0.006, i, f'{val:.3f}', va='center', fontsize=6.5, fontweight=fw)
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=6.5)
    ax1.set_xlabel('Hit@1')
    ax1.set_xlim(0.35, 0.88)
    ax1.invert_yaxis()
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')
    ax1.get_yticklabels()[0].set_fontweight('bold')
    ax1.get_yticklabels()[0].set_color(OURS_COLOR)

    email_vals, email_cols, email_y = [], [], []
    for i, val in enumerate(email):
        if val is not None:
            email_vals.append(val)
            email_cols.append(colors_bar[i])
            email_y.append(y[i])

    bars2 = ax2.barh(email_y, email_vals, height=h, color=email_cols, edgecolor='white', linewidth=0.3)
    for bar, val in zip(bars2, email_vals):
        ypos = bar.get_y() + bar.get_height() / 2
        fw = 'bold' if val == max(email_vals) else 'normal'
        ax2.text(val + 0.006, ypos, f'{val:.3f}', va='center', fontsize=6.5, fontweight=fw)

    for i, val in enumerate(email):
        if val is None:
            ax2.text(0.50, y[i], 'N/A', va='center', ha='center', fontsize=6.5, color='#999999', fontstyle='italic')

    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Hit@1')
    ax2.set_xlim(0.30, 0.97)
    ax2.set_ylim(len(methods) - 0.5, -0.5)
    ax2.set_title('(b) Email', fontsize=9, fontweight='bold')

    fig.suptitle('Retrieval Paradigm Comparison', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_paradigm_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6: RL Progression — zoom-in on RL stage
# ═══════════════════════════════════════════════════════════════════════════════
def fig_rl_progression():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    stages = ['SFT', 'GRPO\nstep200', 'GRPO\nstep300', 'DAPO\nstep420']
    ecom_h1 = [0.680, 0.806, 0.809, 0.808]
    ecom_mr = [0.705, 0.833, 0.835, 0.833]

    x = np.arange(len(stages))
    ax1.plot(x, ecom_h1, 'o-', color=OURS_COLOR, linewidth=2, markersize=7, label='Hit@1', zorder=3)
    ax1.plot(x, ecom_mr, 's--', color=OURS_COLOR2, linewidth=1.5, markersize=6, label='MRR', zorder=3)
    ax1.fill_between(x, 0.65, ecom_h1, alpha=0.08, color=OURS_COLOR)
    for i, (h, m) in enumerate(zip(ecom_h1, ecom_mr)):
        ax1.annotate(f'{h:.3f}', (i, h), textcoords='offset points', xytext=(0, 9),
                     fontsize=7, ha='center', color=OURS_COLOR, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0.65, 0.87)
    ax1.legend(loc='lower right', frameon=False, fontsize=7)
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')

    stages_e = ['SFT', 'GRPO\nstep843', 'DAPO\nstep843']
    email_h1 = [0.891, 0.906, 0.909]
    email_mr = [0.892, 0.907, 0.910]

    x2 = np.arange(len(stages_e))
    ax2.plot(x2, email_h1, 'o-', color=OURS_COLOR, linewidth=2, markersize=7, label='Hit@1', zorder=3)
    ax2.plot(x2, email_mr, 's--', color=OURS_COLOR2, linewidth=1.5, markersize=6, label='MRR', zorder=3)
    ax2.fill_between(x2, 0.87, email_h1, alpha=0.08, color=OURS_COLOR)
    for i, (h, m) in enumerate(zip(email_h1, email_mr)):
        ax2.annotate(f'{h:.3f}', (i, h), textcoords='offset points', xytext=(0, 9),
                     fontsize=7, ha='center', color=OURS_COLOR, fontweight='bold')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(stages_e, fontsize=7)
    ax2.set_ylim(0.87, 0.925)
    ax2.legend(loc='lower right', frameon=False, fontsize=7)
    ax2.set_title('(b) Email', fontsize=9, fontweight='bold')

    fig.suptitle('RL Training Progression (Zoom-in)', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_rl_progression.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7: Few-shot DSL failure (from 实验结果分析报告 B5)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fewshot_failure():
    methods = ['GPT-5.5\nfew-shot', 'Opus 4.7\nfew-shot', 'GPT-5.5\nzero-shot', 'SFT\n(4B)', 'GRPO\n(4B)']
    hit1      = [0.534,  0.420,  0.667, 0.680, 0.809]
    exec_rate = [0.914,  0.733,  1.000, 0.997, 0.997]

    colors = [FEWSHOT_COLOR, FEWSHOT_COLOR, '#F5B7B1', '#E74C3C', '#B71C1C']
    fig, ax1 = plt.subplots(figsize=(4.7, 2.6))
    x = np.arange(len(methods))
    bars = ax1.bar(x, hit1, 0.58, color=colors, edgecolor='white', linewidth=0.3, label='Hit@1')
    _bar_label(ax1, bars, hit1, dy=0.01, fontsize=6, bold_first=0)

    ax2 = ax1.twinx()
    ax2.plot(x, exec_rate, 'D-', color='#333333', markersize=5, linewidth=1.2, label='Exec Rate')
    ax2.spines['right'].set_visible(True)

    ax1.set_ylabel('Hit@1')
    ax2.set_ylabel('Execution Rate', color='#333333')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=7)
    ax1.set_ylim(0.35, 0.86)
    ax2.set_ylim(0.65, 1.03)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False, fontsize=7)
    ax1.set_title('Few-shot DSL vs SFT vs RL (Full Test Set)', fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout()
    save(fig, 'fig_fewshot_failure.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8: Cross-scenario comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cross_scenario():
    methods = ['DAPO\n(Ours)', 'GRPO\n(Ours)', 'SFT', 'GPT-5.5', 'Opus 4.7', 'BGE-\nReranker', 'BM25']
    ecom    = [0.808, 0.809, 0.680, 0.667, 0.635, 0.618, 0.571]
    email   = [0.909, 0.906, 0.891, 0.855, 0.830, 0.836, 0.731]

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.bar(x - w/2, ecom,  w, label='E-commerce', color=OURS_COLOR, alpha=0.8,
           edgecolor='white', linewidth=0.3)
    ax.bar(x + w/2, email, w, label='Email',      color=LLM_COLOR, alpha=0.8,
           edgecolor='white', linewidth=0.3)

    for i in range(len(methods)):
        gap = (email[i] - ecom[i]) * 100
        ax.text(i, max(ecom[i], email[i]) + 0.012, f'+{gap:.1f}pp',
                ha='center', fontsize=5.5, color='#444444')

    ax.set_ylabel('Hit@1')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylim(0.45, 0.98)
    ax.legend(loc='upper right', frameon=False)
    ax.set_title('Cross-Scenario Comparison: E-commerce vs Email', fontsize=10, fontweight='bold', pad=6)

    fig.tight_layout()
    save(fig, 'fig_cross_scenario.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9: E6 — Orchestration Analysis (from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e6_orchestration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    colors_act = ['#4E79A7', '#59A14F', '#E15759']

    # E-commerce (doc E6 table: SFT 4B, GRPO 4B, DAPO 4B, SFT 8B, GPT-5.5, Opus 4.7)
    methods_ec = ['SFT (4B)', 'GRPO (4B)', 'DAPO (4B)', 'SFT (8B)', 'GPT-5.5', 'Opus 4.7']
    sql_ec     = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
    text_ec    = [0.434, 0.623, 0.623, 0.471, 0.553, 0.513]
    image_ec   = [0.310, 0.000, 0.000, 0.263, 0.191, 0.326]

    x = np.arange(len(methods_ec))
    w = 0.25

    ax1.bar(x - w, sql_ec,   w, label='SQL Filter', color=colors_act[0], edgecolor='white', linewidth=0.3)
    ax1.bar(x,     text_ec,  w, label='Text Search', color=colors_act[1], edgecolor='white', linewidth=0.3)
    ax1.bar(x + w, image_ec, w, label='Image Search', color=colors_act[2], edgecolor='white', linewidth=0.3)

    ax1.set_ylabel('Action Frequency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_ec, rotation=30, ha='right', fontsize=6.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')

    # Email (no image channel)
    methods_em = ['SFT (4B)', 'GRPO (4B)', 'DAPO (4B)', 'GPT-5.5', 'Opus 4.7']
    sql_em     = [1.000, 1.000, 1.000, 1.000, 1.000]
    text_em    = [0.736, 0.730, 0.745, 0.723, 0.717]

    x2 = np.arange(len(methods_em))

    ax2.bar(x2 - w/2, sql_em,  w, label='SQL Filter', color=colors_act[0], edgecolor='white', linewidth=0.3)
    ax2.bar(x2 + w/2, text_em, w, label='Text Search', color=colors_act[1], edgecolor='white', linewidth=0.3)

    ax2.set_ylabel('Action Frequency')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods_em, rotation=30, ha='right', fontsize=6.5)
    ax2.set_ylim(0, 1.15)
    ax2.set_title('(b) Email', fontsize=9, fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=7.5)

    fig.suptitle('Retrieval Action Orchestration by Method', fontsize=11, fontweight='bold', y=1.12)
    fig.tight_layout()
    save(fig, 'fig_e6_orchestration.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10: E7 — Modality Selection Accuracy (bar chart from doc + confusion matrix)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e7_modality_selection():
    # Bar chart of accuracy scores (doc E7 table)
    methods = ['SFT 4B', 'SFT 8B', 'Opus 4.7', 'GPT-5.5', 'GRPO 4B', 'DAPO 4B']
    accuracies = [0.808, 0.780, 0.721, 0.709, 0.597, 0.597]
    colors = [SFT_COLOR, SFT8B_COLOR, '#93BAD9', LLM_COLOR, OURS_COLOR, OURS_COLOR2]

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, 0.6, color=colors, edgecolor='white', linewidth=0.3)

    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.008, f'{val:.3f}',
                ha='center', va='bottom', fontsize=6.5)

    ax.set_ylabel('Modality Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylim(0.50, 0.88)
    ax.set_title('Modality Selection Accuracy', fontsize=10, fontweight='bold', pad=6)

    fig.tight_layout()
    save(fig, 'fig_e7_modality_selection.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11: E8 — Model Scaling (4B vs 8B, from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e8_model_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # E-commerce: 4B vs 8B across training stages
    stages = ['Base', 'SFT', 'GRPO', 'DAPO']
    h1_4b = [0.117, 0.680, 0.809, 0.808]
    h1_8b = [0.132, 0.820, 0.820, 0.810]

    x = np.arange(len(stages))
    w = 0.3

    c_4b = OURS_COLOR
    c_8b = '#4E79A7'

    for i, v in enumerate(h1_4b):
        ax1.bar(i - w/2, v, w, color=c_4b, edgecolor='white', linewidth=0.3,
                label='4B' if i == 0 else None)
        ax1.text(i - w/2, v + 0.012, f'{v:.3f}', ha='center', fontsize=5.5,
                fontweight='bold' if i >= 2 else 'normal', color=c_4b)

    for i, v in enumerate(h1_8b):
        ax1.bar(i + w/2, v, w, color=c_8b, edgecolor='white', linewidth=0.3,
                label='8B' if i == 0 else None)
        ax1.text(i + w/2, v + 0.012, f'{v:.3f}', ha='center', fontsize=5.5,
                fontweight='bold' if i >= 1 else 'normal', color=c_8b)

    # Show ΔRL annotations
    ax1.annotate('ΔRL=+12.9pp', xy=(2.15, 0.809), xytext=(3.2, 0.60),
                fontsize=6, color=c_4b, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=c_4b, lw=0.8))
    ax1.annotate('ΔRL=+0.0pp', xy=(2.15, 0.820), xytext=(3.2, 0.70),
                fontsize=6, color=c_8b, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=c_8b, lw=0.8))

    ax1.set_ylabel('Hit@1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7)
    ax1.set_ylim(0, 0.92)
    ax1.legend(loc='upper left', frameon=False, fontsize=7)
    ax1.set_title('(a) E-commerce: 4B vs 8B', fontsize=9, fontweight='bold')

    # Email (4B only, no 8B data)
    stages_em = ['Base', 'SFT', 'GRPO', 'DAPO']
    h1_em_4b  = [0.243, 0.891, 0.906, 0.909]

    ax2.bar(np.arange(len(stages_em)), h1_em_4b, 0.5,
            color=['#93BAD9', SFT_COLOR, OURS_COLOR, OURS_COLOR2],
            edgecolor='white', linewidth=0.3)
    for i, v in enumerate(h1_em_4b):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=6, fontweight='bold' if i >= 2 else 'normal')

    ax2.set_ylabel('Hit@1')
    ax2.set_xticks(np.arange(len(stages_em)))
    ax2.set_xticklabels(stages_em, fontsize=7)
    ax2.set_ylim(0, 0.99)
    ax2.set_title('(b) Email (4B)', fontsize=9, fontweight='bold')

    fig.suptitle('Model Size & Training Stage Comparison', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_e8_model_scaling.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 12: E9 — Error Analysis (from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e9_error_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.0))

    c_success = '#59A14F'
    c_exec_fail = '#E15759'
    c_format_fail = '#BAB0AC'

    # E-commerce (doc E9 table)
    methods_ec = ['Base\n(4B)', 'Few-shot\nGPT-5.5', 'Few-shot\nOpus 4.7', 'SFT\n(4B)', 'GRPO\n(4B)', 'DAPO\n(4B)', 'GPT-5.4', 'GPT-5.5', 'Opus 4.7']
    total_ec      = [3000, 3000, 3000, 3000, 3000, 3000, 2998, 1000, 2998]
    format_ok_ec  = [949,  2885, 2352, 3000, 2999, 2998, 2998, 1000, 2998]
    exec_ok_ec    = [506,  2741, 2200, 2992, 2992, 2989, 2998, 1000, 2996]

    success_ec     = [e/t for e, t in zip(exec_ok_ec, total_ec)]
    exec_fail_ec   = [(f-e)/t for f, e, t in zip(format_ok_ec, exec_ok_ec, total_ec)]
    format_fail_ec = [(t-f)/t for t, f in zip(total_ec, format_ok_ec)]

    x = np.arange(len(methods_ec))
    ax1.bar(x, success_ec, 0.58, label='Success', color=c_success, edgecolor='white', linewidth=0.3)
    ax1.bar(x, exec_fail_ec, 0.58, bottom=success_ec, label='Exec Fail', color=c_exec_fail, edgecolor='white', linewidth=0.3)
    bottom2 = [s+e for s, e in zip(success_ec, exec_fail_ec)]
    ax1.bar(x, format_fail_ec, 0.58, bottom=bottom2, label='Format Fail', color=c_format_fail, edgecolor='white', linewidth=0.3)

    for i, s in enumerate(success_ec):
        ax1.text(i, s/2, f'{s:.1%}', ha='center', va='center', fontsize=5, fontweight='bold', color='white')

    ax1.set_ylabel('Proportion')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_ec, fontsize=5.5, rotation=30, ha='right')
    ax1.set_ylim(0, 1.05)
    ax1.set_title('(a) E-commerce', fontsize=10, fontweight='bold')

    # Email (doc E9 table)
    methods_em = ['Base\n(4B)', 'SFT\n(4B)', 'GRPO\n(4B)', 'DAPO\n(4B)', 'GPT-5.4', 'GPT-5.5', 'Opus 4.7']
    total_em      = [3000, 3000, 3000, 3000, 2999, 3000, 3000]
    format_ok_em  = [744,  2899, 2999, 2999, 2997, 2998, 2901]
    exec_ok_em    = [744,  2878, 2979, 2980, 2997, 2998, 2897]

    success_em     = [e/t for e, t in zip(exec_ok_em, total_em)]
    exec_fail_em   = [(f-e)/t for f, e, t in zip(format_ok_em, exec_ok_em, total_em)]
    format_fail_em = [(t-f)/t for t, f in zip(total_em, format_ok_em)]

    x2 = np.arange(len(methods_em))
    ax2.bar(x2, success_em, 0.58, label='Success', color=c_success, edgecolor='white', linewidth=0.3)
    ax2.bar(x2, exec_fail_em, 0.58, bottom=success_em, label='Exec Fail', color=c_exec_fail, edgecolor='white', linewidth=0.3)
    bottom2e = [s+e for s, e in zip(success_em, exec_fail_em)]
    ax2.bar(x2, format_fail_em, 0.58, bottom=bottom2e, label='Format Fail', color=c_format_fail, edgecolor='white', linewidth=0.3)

    for i, s in enumerate(success_em):
        ax2.text(i, s/2, f'{s:.1%}', ha='center', va='center', fontsize=5.5, fontweight='bold', color='white')

    ax2.set_ylabel('Proportion')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods_em, fontsize=6, rotation=25, ha='right')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('(b) Email', fontsize=10, fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=7.5)

    fig.suptitle('DSL Generation Error Analysis', fontsize=12, fontweight='bold', y=1.12)
    fig.tight_layout()
    save(fig, 'fig_e9_error_analysis.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 13: E10 — OOD Evaluation (from 实验结果分析报告)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e10_ood():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # E-commerce OOD
    methods_ec = ['4B Base', '4B SFT', '4B GRPO\nstep300', '4B DAPO\nstep420']
    iid_ec  = [0.117, 0.680, 0.809, 0.808]
    ood_ec  = [0.113, 0.683, 0.783, 0.793]

    x = np.arange(len(methods_ec))
    w = 0.3

    ax1.bar(x - w/2, iid_ec, w, label='IID (N=3000)', color=OURS_COLOR, alpha=0.85,
            edgecolor='white', linewidth=0.3)
    ax1.bar(x + w/2, ood_ec, w, label='OOD (N=300)', color=LLM_COLOR, alpha=0.85,
            edgecolor='white', linewidth=0.3)

    for i in range(len(methods_ec)):
        delta = (ood_ec[i] - iid_ec[i]) * 100
        sign = '+' if delta >= 0 else ''
        y_top = max(iid_ec[i], ood_ec[i]) + 0.045
        ax1.text(i, y_top, f'{sign}{delta:.1f}pp', ha='center', fontsize=5.5, color='#444444', fontweight='bold')

    ax1.set_ylabel('Hit@1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_ec, fontsize=6.5)
    ax1.set_ylim(0, 0.93)
    ax1.legend(loc='upper left', frameon=False, fontsize=7)
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')

    # Email OOD
    methods_em = ['4B Base', '4B SFT', '4B GRPO\nstep843', '4B DAPO\nstep843']
    iid_em  = [0.243, 0.891, 0.906, 0.909]
    ood_em  = [0.217, 0.873, 0.883, 0.883]

    x2 = np.arange(len(methods_em))

    ax2.bar(x2 - w/2, iid_em, w, label='IID (N=3000)', color=OURS_COLOR, alpha=0.85,
            edgecolor='white', linewidth=0.3)
    ax2.bar(x2 + w/2, ood_em, w, label='OOD (N=300)', color=LLM_COLOR, alpha=0.85,
            edgecolor='white', linewidth=0.3)

    for i in range(len(methods_em)):
        delta = (ood_em[i] - iid_em[i]) * 100
        sign = '+' if delta >= 0 else ''
        y_top = max(iid_em[i], ood_em[i]) + 0.030
        ax2.text(i, y_top, f'{sign}{delta:.1f}pp', ha='center', fontsize=5.5, color='#444444', fontweight='bold')

    ax2.set_ylabel('Hit@1')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods_em, fontsize=6.5)
    ax2.set_ylim(0.15, 1.0)
    ax2.legend(loc='upper left', frameon=False, fontsize=7)
    ax2.set_title('(b) Email', fontsize=9, fontweight='bold')

    fig.suptitle('OOD Generalization: IID vs Human-Annotated Queries', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_e10_ood.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 14: E2 — Action Space Ablation
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e2_action_space():
    variants = [
        ('Text only', 'results/ecommerce/e2/eval_sft_text_only.json', '#93BAD9'),
        ('SQL only', 'results/ecommerce/e2/eval_sft_sql_only.json', '#B07AA1'),
        ('SQL + image', 'results/ecommerce/e2/eval_sft_sql_image.json', '#E7CB94'),
        ('SQL + text', 'results/ecommerce/e2/eval_sft_sql_text.json', '#59A14F'),
        ('Full DSL SFT', 'results/ecommerce/eval_sft_4b_gt_v2_nocase.json', SFT_COLOR),
        ('Full DSL GRPO', 'results/ecommerce/eval_rl_step300_gt_v2.json', OURS_COLOR),
    ]

    labels, h1, mrr, colors = [], [], [], []
    for label, rel, color in variants:
        d = _maybe_load_json(rel)
        if not d:
            continue
        overall = d.get('overall', d)
        labels.append(label)
        h1.append(overall.get('hit_at_1', overall.get('hit1')))
        mrr.append(overall.get('mrr'))
        colors.append(color)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    y = np.arange(len(labels))
    ax1.barh(y, h1, color=colors, edgecolor='white', linewidth=0.3)
    ax2.barh(y, mrr, color=colors, edgecolor='white', linewidth=0.3)
    for ax, vals, title in [(ax1, h1, 'Hit@1'), (ax2, mrr, 'MRR')]:
        for i, v in enumerate(vals):
            ax.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=6.5,
                    fontweight='bold' if labels[i] == 'Full DSL GRPO' else 'normal')
        ax.set_xlabel(title)
        ax.set_xlim(0.35, 0.86)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, fontweight='bold')
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    fig.suptitle('Action Space Ablation (E-commerce)', fontsize=11, fontweight='bold', y=1.03)
    fig.tight_layout()
    save(fig, 'fig_e2_action_space.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 15: Dataset Query Statistics
# ═══════════════════════════════════════════════════════════════════════════════
def fig_dataset_query_stats():
    e3 = _load_json('results/analysis/e3_complexity_breakdown.json')
    e5 = _load_json('results/analysis/e5_modality_ablation.json')

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7))
    scenarios = [
        ('E-commerce', 'ecommerce', 'GRPO_4B_step300'),
        ('Email', 'email', 'DAPO_4B_step843'),
    ]
    level_names = ['L1', 'L2', 'L3']
    for row, (title, scenario, key) in enumerate(scenarios):
        level_counts = [e3[scenario][key][str(i)]['n'] for i in [1, 2, 3]]
        ax = axes[row, 0]
        bars = ax.bar(level_names, level_counts, color=[LLM_COLOR, B2_COLOR, OURS_COLOR],
                      edgecolor='white', linewidth=0.3)
        total = sum(level_counts)
        for bar, c in zip(bars, level_counts):
            ax.text(bar.get_x() + bar.get_width()/2, c + total * 0.015,
                    f'{c}\n({c/total:.0%})', ha='center', fontsize=6.5)
        ax.set_ylabel('# Queries')
        ax.set_title(f'{title}: Complexity', fontsize=9, fontweight='bold')
        ax.set_ylim(0, max(level_counts) * 1.22)

        modality = e5[scenario][key]
        items = sorted(((k, v['n']) for k, v in modality.items()), key=lambda x: x[1], reverse=True)
        labels = [k.replace('+', '+\n') for k, _ in items[:7]]
        counts = [v for _, v in items[:7]]
        ax = axes[row, 1]
        y = np.arange(len(labels))
        ax.barh(y, counts, color='#76B7B2', edgecolor='white', linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.2)
        ax.invert_yaxis()
        ax.set_xlabel('# Queries')
        ax.set_title(f'{title}: Modality Coverage', fontsize=9, fontweight='bold')
        for i, c in enumerate(counts):
            ax.text(c + max(counts) * 0.015, i, str(c), va='center', fontsize=6.2)
        ax.set_xlim(0, max(counts) * 1.2)

    fig.suptitle('Benchmark Query Distribution', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_dataset_query_stats.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 16: Training Dynamics (GRPO checkpoints + email validation)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_training_dynamics_paper():
    e1 = _load_json('results/analysis/e1_main_results.json')
    email_summary = _load_json('figures/training_curves/grpo_email_summary.json')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    steps = [200, 300, 550, 1200]
    vals = [0.806, 0.809, 0.798, 0.003]
    ax1.plot(steps, vals, 'o-', color=OURS_COLOR, linewidth=1.8, markersize=5, label='GRPO')
    ax1.scatter([420], [0.808], marker='D', s=45, color=OURS_COLOR2, label='DAPO step420', zorder=4)
    ax1.annotate('best', (300, 0.809), textcoords='offset points', xytext=(0, 10),
                 ha='center', fontsize=7, fontweight='bold', color=OURS_COLOR)
    ax1.annotate('collapse', (1200, 0.003), textcoords='offset points', xytext=(-8, 10),
                 ha='right', fontsize=7, color='#333333')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Test Hit@1')
    ax1.set_ylim(0, 0.88)
    ax1.set_title('(a) E-commerce RL Checkpoints', fontsize=9, fontweight='bold')
    ax1.legend(frameon=False, fontsize=7, loc='center right')

    val_traj = email_summary['val_acc_trajectory']
    x = [p['global_step'] for p in val_traj]
    y = [p['val_acc'] for p in val_traj]
    ax2.plot(x, y, 'o-', color=LLM_COLOR, linewidth=1.6, markersize=4, label='GRPO val reward')
    ax2.axhline(email_summary['gold_dsl_reward_ceiling_approx'], color='#777777',
                linestyle='--', linewidth=1.0, label='Gold DSL ceiling')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Validation Reward')
    ax2.set_ylim(0.66, 0.705)
    ax2.set_title('(b) Email GRPO Validation', fontsize=9, fontweight='bold')
    ax2.legend(frameon=False, fontsize=7, loc='lower right')

    fig.suptitle('RL Training Dynamics', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_training_dynamics_paper.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating paper figures (v4 — aligned with 实验结果分析报告)...\n')
    fig_main_ecommerce()             # 1
    fig_main_email()                  # 2
    fig_e2_action_space()             # 3  E2 action-space ablation
    fig_e3_e5_combined_ecommerce()    # 4  E3+E5 ecommerce
    fig_e3_e5_combined_email()        # 5  E3+E5 email
    fig_paradigm_comparison()         # 6  paradigm comparison
    fig_rl_progression()              # 7  RL progression
    fig_fewshot_failure()             # 8  few-shot failure
    fig_cross_scenario()              # 9  cross-scenario
    fig_e6_orchestration()            # 10 E6 orchestration
    fig_e7_modality_selection()       # 11 E7 modality selection
    fig_e8_model_scaling()            # 12 E8 model scaling (4B vs 8B)
    fig_e9_error_analysis()           # 13 E9 error analysis
    fig_e10_ood()                     # 14 E10 OOD evaluation (NEW)
    fig_dataset_query_stats()         # 15 dataset statistics
    fig_training_dynamics_paper()     # 16 training dynamics
    print(f'\n  All 16 figures saved to {OUT_DIR}/')
