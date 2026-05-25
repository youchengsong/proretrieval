#!/usr/bin/env python3
"""Generate all paper-quality figures for LocalSearch-R1 (EMNLP 2026).

Output: PDF vector figures in figures/paper/
Style: ACL/EMNLP conference standard — compact, professional palette, readable at column width.

v2: Revised per review feedback.
  - Deleted: radar, training_stability (synthetic data), grpo_overfitting (exposes collapse),
             standalone e3/e5 ecommerce (redundant with combined)
  - Modified: main results (Hit@1 only, category coloring), paradigm (dual-scenario),
              fewshot (red gradient), cross_scenario (pp not %), e3_email (removed fake RankGPT),
              rl_progression (zoom-in on RL stage)
  - Added: e3_e5_combined_email
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

# ── Category color scheme (same category = same hue) ────────────────────────
OURS_COLOR   = '#D62728'   # red — always for "Ours"
OURS_COLOR2  = '#E45756'   # lighter red for DAPO
SFT_COLOR    = '#FF7F0E'   # orange
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


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1: Main Results — E-commerce (Hit@1 only, category coloring)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_main_ecommerce():
    groups = [
        ('Ours', [
            ('GRPO (4B)',  0.809, OURS_COLOR),
            ('DAPO (4B)',  0.808, OURS_COLOR2),
        ]),
        ('SFT', [
            ('SFT (4B)',   0.680, SFT_COLOR),
        ]),
        ('Commercial LLMs', [
            ('GPT-5.5',   0.667, LLM_COLOR),
            ('Opus 4.7',  0.635, '#6A9FCA'),
            ('GPT-5.4',   0.581, '#93BAD9'),
            ('Gemini 3.1', 0.587, '#BBCFE8'),
        ]),
        ('LLM-Enhanced', [
            ('RankGPT',   0.768, B2_COLOR),
            ('IRCoT',     0.565, '#82C47B'),
            ('SQR',       0.606, '#ABD9A5'),
            ('HyDE',      0.549, '#C8E6C2'),
        ]),
        ('RL-Retrieval', [
            ('SQL-R1',    0.620, B3_COLOR),
            ('DeepRetrieval', 0.625, '#CCBDD1'),
            ('Search-R1', 0.435, '#E0D5E3'),
        ]),
        ('Traditional', [
            ('Reranker',  0.618, B1_COLOR),
            ('LightRAG',  0.588, '#C8C0B8'),
            ('BM25',      0.571, '#A0D0CC'),
            ('GraphRAG',  0.500, B4_COLOR),
        ]),
    ]

    all_labels, all_vals, all_colors = [], [], []
    separators = []
    bg_regions = []
    pos = 0
    for gname, items in groups:
        start = pos
        for label, val, col in items:
            all_labels.append(label)
            all_vals.append(val)
            all_colors.append(col)
            pos += 1
        bg_regions.append((gname, start, pos))
        separators.append(pos - 0.5)

    x = np.arange(len(all_labels))

    fig, ax = plt.subplots(figsize=(7.0, 2.6))

    # Background shading for categories
    bg_fills = ['#FFF2F2', '#FFF8F0', '#F0F4FA', '#F0FAF0', '#F5F0F7', '#F0FAFA']
    for i, (gname, s, e) in enumerate(bg_regions):
        ax.axvspan(s - 0.5, e - 0.5, color=bg_fills[i], zorder=0)
        ax.text((s + e - 1) / 2, 0.38, gname, ha='center', fontsize=6, color='#666666',
                fontstyle='italic', zorder=1)

    bars = ax.bar(x, all_vals, 0.7, color=all_colors, edgecolor='white', linewidth=0.4, zorder=2)

    # Value labels on top
    for i, (bar, val) in enumerate(zip(bars, all_vals)):
        fontw = 'bold' if i < 2 else 'normal'
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.008, f'{val:.3f}',
                ha='center', va='bottom', fontsize=5.5, fontweight=fontw, zorder=3)

    ax.set_ylabel('Hit@1')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=40, ha='right', fontsize=6.5)
    ax.set_ylim(0.37, 0.88)
    ax.set_title('E-commerce Retrieval Performance (Hit@1)', fontsize=10, fontweight='bold', pad=6)

    fig.tight_layout()
    save(fig, 'fig_main_ecommerce.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2: Main Results — Email (Hit@1 only, category coloring)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_main_email():
    groups = [
        ('Ours', [
            ('DAPO (4B)',  0.909, OURS_COLOR),
            ('GRPO (4B)',  0.906, OURS_COLOR2),
        ]),
        ('SFT', [
            ('SFT (4B)',   0.891, SFT_COLOR),
        ]),
        ('Commercial LLMs', [
            ('GPT-5.5',   0.855, LLM_COLOR),
            ('GPT-5.4',   0.854, '#6A9FCA'),
            ('Opus 4.7',  0.830, '#93BAD9'),
        ]),
        ('LLM-Enhanced', [
            ('RankGPT',   0.888, B2_COLOR),
        ]),
        ('Traditional + Structured Query', [
            ('Reranker',  0.836, B1_COLOR),
            ('SQR',       0.630, B2_COLOR),
            ('Search-R1', 0.377, '#CCBDD1'),
            ('BM25',      0.731, '#A0D0CC'),
        ]),
    ]

    all_labels, all_vals, all_colors = [], [], []
    bg_regions = []
    pos = 0
    for gname, items in groups:
        start = pos
        for label, val, col in items:
            all_labels.append(label)
            all_vals.append(val)
            all_colors.append(col)
            pos += 1
        bg_regions.append((gname, start, pos))

    x = np.arange(len(all_labels))

    fig, ax = plt.subplots(figsize=(5.5, 2.6))

    bg_fills = ['#FFF2F2', '#FFF8F0', '#F0F4FA', '#F0FAF0', '#F0FAFA']
    for i, (gname, s, e) in enumerate(bg_regions):
        ax.axvspan(s - 0.5, e - 0.5, color=bg_fills[i], zorder=0)
        ax.text((s + e - 1) / 2, 0.705, gname, ha='center', fontsize=6, color='#666666',
                fontstyle='italic', zorder=1)

    bars = ax.bar(x, all_vals, 0.7, color=all_colors, edgecolor='white', linewidth=0.4, zorder=2)

    for i, (bar, val) in enumerate(zip(bars, all_vals)):
        fontw = 'bold' if i < 2 else 'normal'
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.004, f'{val:.3f}',
                ha='center', va='bottom', fontsize=5.5, fontweight=fontw, zorder=3)

    ax.set_ylabel('Hit@1')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=40, ha='right', fontsize=6.5)
    ax.set_ylim(0.70, 0.94)
    ax.set_title('Email Retrieval Performance (Hit@1)', fontsize=10, fontweight='bold', pad=6)

    fig.tight_layout()
    save(fig, 'fig_main_email.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3: E3 + E5 Combined — E-commerce (two panels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e3_e5_combined_ecommerce():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    methods_e3 = ['GRPO', 'DAPO', 'SFT', 'GPT-5.5', 'RankGPT', 'Reranker']
    data_e3 = {
        'GRPO':       [0.864, 0.797, 0.720],
        'DAPO':       [0.860, 0.800, 0.717],
        'SFT':        [0.752, 0.646, 0.598],
        'GPT-5.5':    [0.666, 0.680, 0.642],
        'RankGPT':    [0.717, 0.803, 0.802],
        'Reranker':   [0.473, 0.706, 0.741],
    }
    colors_e3 = [OURS_COLOR, OURS_COLOR2, SFT_COLOR, LLM_COLOR, B2_COLOR, B1_COLOR]

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

    methods_e5 = ['GRPO', 'DAPO', 'SFT', 'SQL-R1', 'Reranker']
    data_e5 = {
        'GRPO':       [0.843, 0.690, 0.716],
        'DAPO':       [0.843, 0.690, 0.711],
        'SFT':        [0.743, 0.632, 0.323],
        'SQL-R1':     [0.866, 0.303, 0.048],
        'Reranker':   [0.310, 0.759, 0.819],
    }
    colors_e5 = [OURS_COLOR, OURS_COLOR2, SFT_COLOR, B3_COLOR, B1_COLOR]

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
    ax2.set_ylim(0.20, 0.90)
    ax2.set_title('(b) By Retrieval Modality', fontsize=9, fontweight='bold')

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    all_labels = dict(zip(labels + l2, handles + h2))
    fig.legend(all_labels.values(), all_labels.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=6, frameon=False, fontsize=6.5)

    fig.suptitle('E-commerce: Performance Breakdown', fontsize=11, fontweight='bold', y=1.14)
    fig.tight_layout()
    save(fig, 'fig_e3_e5_combined_ecommerce.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4: E3 + E5 Combined — Email (two panels, no Image modality)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e3_e5_combined_email():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # E3: by level — no RankGPT (only has N=300 average, no per-level breakdown)
    methods_e3 = ['DAPO', 'GRPO', 'SFT', 'GPT-5.5', 'Opus 4.7', 'Reranker', 'BM25']
    data_e3 = {
        'DAPO':       [0.987, 0.887, 0.798],
        'GRPO':       [0.987, 0.883, 0.793],
        'SFT':        [0.987, 0.865, 0.755],
        'GPT-5.5':    [0.923, 0.838, 0.753],
        'Opus 4.7':   [0.925, 0.815, 0.669],
        'Reranker':   [0.804, 0.869, 0.831],
        'BM25':       [0.647, 0.794, 0.771],
    }
    colors_e3 = [OURS_COLOR, OURS_COLOR2, SFT_COLOR, LLM_COLOR, '#6A9FCA', B1_COLOR, '#A0D0CC']

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
    methods_e5 = ['DAPO', 'GRPO', 'SFT', 'GPT-5.5', 'Reranker', 'BM25']
    data_e5 = {
        'DAPO':       [0.885, 0.891],
        'GRPO':       [0.882, 0.889],
        'SFT':        [0.863, 0.871],
        'GPT-5.5':    [0.822, 0.856],
        'Reranker':   [0.832, 0.860],
        'BM25':       [0.744, 0.764],
    }
    colors_e5 = [OURS_COLOR, OURS_COLOR2, SFT_COLOR, LLM_COLOR, B1_COLOR, '#A0D0CC']

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
    ax2.set_ylim(0.70, 0.92)
    ax2.set_title('(b) By Retrieval Modality', fontsize=9, fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    all_labels = dict(zip(labels + l2, handles + h2))
    fig.legend(all_labels.values(), all_labels.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=7, frameon=False, fontsize=6.5)

    fig.suptitle('Email: Performance Breakdown', fontsize=11, fontweight='bold', y=1.14)
    fig.tight_layout()
    save(fig, 'fig_e3_e5_combined_email.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5: Paradigm Comparison — Dual Scenario (ecommerce + email side by side)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_paradigm_comparison():
    e1 = _load_json('results/analysis/e1_main_results.json')
    # Methods with results on BOTH scenarios where possible
    methods = [
        'Retrieval Program\nSynthesis (Ours)',
        'RankGPT\n(LLM Rerank)',
        'DeepRetrieval\n(Boolean RL)',
        'SQL-R1\n(SQL-only)',
        'BGE-Reranker\n(Neural Rerank)',
        'SQR\n(LLM Filter)',
        'BM25\n(Lexical)',
        'GraphRAG\n(Knowledge Graph)',
        'Search-R1\n(BM25 RL)',
    ]
    ecom  = [
        _m(e1, 'ecommerce', 'SFT_8B'),
        _m(e1, 'ecommerce', 'B2_RankGPT'),
        _m(e1, 'ecommerce', 'B3_DeepRetrieval'),
        _m(e1, 'ecommerce', 'B3_SQL-R1'),
        _m(e1, 'ecommerce', 'B1_Reranker'),
        _m(e1, 'ecommerce', 'B3_SQR'),
        _m(e1, 'ecommerce', 'B1_BM25'),
        _m(e1, 'ecommerce', 'B4_GraphRAG'),
        _m(e1, 'ecommerce', 'B3_Search-R1_step400'),
    ]
    email = [
        _m(e1, 'email', 'DAPO_4B_step843'),
        _m(e1, 'email', 'B2_RankGPT'),
        None,
        _m(e1, 'email', 'B3_SQL-R1'),
        _m(e1, 'email', 'B1_Reranker'),
        _m(e1, 'email', 'B3_SQR'),
        _m(e1, 'email', 'B1_BM25'),
        None,
        _m(e1, 'email', 'B3_Search-R1'),
    ]

    colors_ecom  = [OURS_COLOR, B2_COLOR, B3_COLOR, B3_COLOR, B1_COLOR, B2_COLOR, B1_COLOR, B4_COLOR, B3_COLOR]
    colors_email = [OURS_COLOR, B2_COLOR, B3_COLOR, B3_COLOR, B1_COLOR, B2_COLOR, B1_COLOR, B4_COLOR, B3_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8),
                                     gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.08})

    y = np.arange(len(methods))
    h = 0.6

    # E-commerce panel
    bars1 = ax1.barh(y, ecom, height=h, color=colors_ecom, edgecolor='white', linewidth=0.3)
    for i, (bar, val) in enumerate(zip(bars1, ecom)):
        fw = 'bold' if i == 0 else 'normal'
        ax1.text(val + 0.006, i, f'{val:.3f}', va='center', fontsize=6.5, fontweight=fw)
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=6.5)
    ax1.set_xlabel('Hit@1')
    ax1.set_xlim(0.38, 0.88)
    ax1.invert_yaxis()
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')
    ax1.get_yticklabels()[0].set_fontweight('bold')
    ax1.get_yticklabels()[0].set_color(OURS_COLOR)

    # Email panel
    email_vals = []
    email_cols = []
    email_y = []
    for i, val in enumerate(email):
        if val is not None:
            email_vals.append(val)
            email_cols.append(colors_email[i])
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
    ax2.set_xlim(0.38, 0.97)
    ax2.set_ylim(len(methods) - 0.5, -0.5)
    ax2.set_title('(b) Email', fontsize=9, fontweight='bold')

    fig.suptitle('Retrieval Paradigm Comparison', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_paradigm_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6: RL Progression — zoom-in on RL stage (SFT→GRPO steps→DAPO)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_rl_progression():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # E-commerce: SFT → GRPO step200 → GRPO step300 → DAPO step420
    stages = ['SFT', 'GRPO\nstep200', 'GRPO\nstep300', 'DAPO\nstep420']
    ecom_h1 = [0.680, 0.806, 0.809, 0.808]
    ecom_mr = [0.705, 0.833, 0.835, 0.833]

    x = np.arange(len(stages))
    ax1.plot(x, ecom_h1, 'o-', color=OURS_COLOR, linewidth=2, markersize=7, label='Hit@1', zorder=3)
    ax1.plot(x, ecom_mr, 's--', color=OURS_COLOR2, linewidth=1.5, markersize=6, label='MRR', zorder=3)
    ax1.fill_between(x, 0.65, ecom_h1, alpha=0.08, color=OURS_COLOR)
    for i, (h, m) in enumerate(zip(ecom_h1, ecom_mr)):
        ax1.annotate(f'{h:.3f}', (i, h), textcoords='offset points', xytext=(0, -15),
                     fontsize=7, ha='center', color=OURS_COLOR, fontweight='bold')
        ax1.annotate(f'{m:.3f}', (i, m), textcoords='offset points', xytext=(0, 9),
                     fontsize=6.2, ha='center', color=OURS_COLOR2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0.65, 0.87)
    ax1.legend(loc='lower right', frameon=False, fontsize=7)
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')

    # Email: SFT → GRPO step843 → DAPO step843
    stages_e = ['SFT', 'GRPO\nstep843', 'DAPO\nstep843']
    email_h1 = [0.891, 0.906, 0.909]
    email_mr = [0.892, 0.907, 0.910]

    x2 = np.arange(len(stages_e))
    ax2.plot(x2, email_h1, 'o-', color=OURS_COLOR, linewidth=2, markersize=7, label='Hit@1', zorder=3)
    ax2.plot(x2, email_mr, 's--', color=OURS_COLOR2, linewidth=1.5, markersize=6, label='MRR', zorder=3)
    ax2.fill_between(x2, 0.87, email_h1, alpha=0.08, color=OURS_COLOR)
    for i, (h, m) in enumerate(zip(email_h1, email_mr)):
        ax2.annotate(f'{h:.3f}', (i, h), textcoords='offset points', xytext=(0, -15),
                     fontsize=7, ha='center', color=OURS_COLOR, fontweight='bold')
        ax2.annotate(f'{m:.3f}', (i, m), textcoords='offset points', xytext=(0, 9),
                     fontsize=6.2, ha='center', color=OURS_COLOR2)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(stages_e, fontsize=7)
    ax2.set_ylim(0.87, 0.925)
    ax2.legend(loc='lower right', frameon=False, fontsize=7)
    ax2.set_title('(b) Email', fontsize=9, fontweight='bold')

    fig.suptitle('RL Training Progression (Zoom-in)', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_rl_progression.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8: Cross-scenario comparison (pp not %)
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
        gap = (email[i] - ecom[i]) * 100  # percentage points
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
# Fig 9: E6 — Orchestration Behavior Analysis (retrieval action frequencies)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e6_orchestration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # E-commerce
    methods_ec = ['Base (4B)', 'SFT (4B)', 'GRPO (4B)', 'DAPO (4B)', 'GPT-5.5', 'Opus 4.7']
    sql_ec     = [1.000, 1.000, 1.000, 1.000, 1.000, 0.9997]
    text_ec    = [0.168, 0.434, 0.623, 0.623, 0.553, 0.513]
    image_ec   = [0.113, 0.310, 0.000, 0.000, 0.191, 0.326]
    colors_act = ['#4E79A7', '#59A14F', '#E15759']

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

    # Email
    methods_em = ['Base (4B)', 'SFT (4B)', 'GRPO (4B)', 'DAPO (4B)', 'GPT-5.5', 'Opus 4.7']
    sql_em     = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
    text_em    = [0.605, 0.736, 0.730, 0.745, 0.723, 0.717]

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
# Fig 10: E9 — Error Analysis (format/exec success stacked bar)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e9_error_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    c_success = '#59A14F'
    c_exec_fail = '#E15759'
    c_format_fail = '#BAB0AC'

    # E-commerce
    methods_ec = ['Base\n(4B)', 'SFT\n(4B)', 'GRPO\n(4B)', 'DAPO\n(4B)', 'GPT-5.5', 'Opus 4.7']
    total_ec      = [3000, 3000, 3000, 3000, 2998, 2998]
    format_ok_ec  = [949,  3000, 2999, 2998, 2998, 2998]
    exec_ok_ec    = [506,  2992, 2992, 2989, 2998, 2996]

    success_ec     = [e/t for e, t in zip(exec_ok_ec, total_ec)]
    exec_fail_ec   = [(f-e)/t for f, e, t in zip(format_ok_ec, exec_ok_ec, total_ec)]
    format_fail_ec = [(t-f)/t for t, f in zip(total_ec, format_ok_ec)]

    x = np.arange(len(methods_ec))
    ax1.bar(x, success_ec, 0.65, label='Success', color=c_success, edgecolor='white', linewidth=0.3)
    ax1.bar(x, exec_fail_ec, 0.65, bottom=success_ec, label='Exec Fail', color=c_exec_fail, edgecolor='white', linewidth=0.3)
    bottom2 = [s+e for s, e in zip(success_ec, exec_fail_ec)]
    ax1.bar(x, format_fail_ec, 0.65, bottom=bottom2, label='Format Fail', color=c_format_fail, edgecolor='white', linewidth=0.3)

    for i, s in enumerate(success_ec):
        ax1.text(i, s/2, f'{s:.1%}', ha='center', va='center', fontsize=5.5, fontweight='bold', color='white')

    ax1.set_ylabel('Proportion')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_ec, fontsize=6)
    ax1.set_ylim(0, 1.05)
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')

    # Email
    methods_em = ['Base\n(4B)', 'SFT\n(4B)', 'GRPO\n(4B)', 'DAPO\n(4B)', 'GPT-5.5', 'Opus 4.7']
    total_em      = [3000, 3000, 3000, 3000, 3000, 3000]
    format_ok_em  = [744,  2899, 2999, 2999, 2998, 2901]
    exec_ok_em    = [744,  2878, 2979, 2980, 2998, 2897]

    success_em     = [e/t for e, t in zip(exec_ok_em, total_em)]
    exec_fail_em   = [(f-e)/t for f, e, t in zip(format_ok_em, exec_ok_em, total_em)]
    format_fail_em = [(t-f)/t for t, f in zip(total_em, format_ok_em)]

    x2 = np.arange(len(methods_em))
    ax2.bar(x2, success_em, 0.65, label='Success', color=c_success, edgecolor='white', linewidth=0.3)
    ax2.bar(x2, exec_fail_em, 0.65, bottom=success_em, label='Exec Fail', color=c_exec_fail, edgecolor='white', linewidth=0.3)
    bottom2e = [s+e for s, e in zip(success_em, exec_fail_em)]
    ax2.bar(x2, format_fail_em, 0.65, bottom=bottom2e, label='Format Fail', color=c_format_fail, edgecolor='white', linewidth=0.3)

    for i, s in enumerate(success_em):
        ax2.text(i, s/2, f'{s:.1%}', ha='center', va='center', fontsize=5.5, fontweight='bold', color='white')

    ax2.set_ylabel('Proportion')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods_em, fontsize=6)
    ax2.set_ylim(0, 1.05)
    ax2.set_title('(b) Email', fontsize=9, fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=7.5)

    fig.suptitle('DSL Generation Error Analysis', fontsize=11, fontweight='bold', y=1.12)
    fig.tight_layout()
    save(fig, 'fig_e9_error_analysis.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11: E8 — Model Scaling (report-aligned 4B vs 8B comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e8_model_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))

    stages = ['Base', 'SFT', 'GRPO\nbest', 'DAPO\nbest']
    h1_4b = [0.117, 0.680, 0.809, 0.808]
    h1_8b = [0.132, 0.820]

    x = np.arange(len(stages))
    w = 0.34
    c_4b = OURS_COLOR
    c_8b = '#4E79A7'

    bars4 = ax1.bar(x - w/2, h1_4b, w, color=c_4b, edgecolor='white',
                    linewidth=0.3, label='4B')
    bars8 = ax1.bar(x[:2] + w/2, h1_8b, w, color=c_8b, edgecolor='white',
                    linewidth=0.3, label='8B')
    for bars, vals, color in [(bars4, h1_4b, c_4b), (bars8, h1_8b, c_8b)]:
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.010, f'{v:.3f}',
                     ha='center', fontsize=5.8, color=color,
                     fontweight='bold' if v >= 0.80 else 'normal')
    ax1.set_ylabel('Hit@1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7)
    ax1.set_ylim(0, 0.90)
    ax1.legend(loc='upper left', frameon=False, fontsize=7)
    ax1.set_title('(a) E-commerce: 4B vs 8B', fontsize=9, fontweight='bold')
    ax1.annotate('8B SFT > 4B GRPO',
                 xy=(1 + w/2, h1_8b[1]), xytext=(1.55, 0.61),
                 arrowprops=dict(arrowstyle='->', color='#555555', lw=0.7),
                 fontsize=6.5, color='#333333')

    email_stages = ['Base', 'SFT', 'GRPO', 'DAPO']
    email_h1 = [0.243, 0.891, 0.906, 0.909]
    x2 = np.arange(len(email_stages))
    colors = [B1_COLOR, SFT_COLOR, OURS_COLOR, OURS_COLOR2]
    bars = ax2.bar(x2, email_h1, 0.58, color=colors, edgecolor='white', linewidth=0.3)
    for bar, v in zip(bars, email_h1):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.010, f'{v:.3f}',
                 ha='center', fontsize=5.8,
                 fontweight='bold' if v == max(email_h1) else 'normal')
    ax2.set_ylabel('Hit@1')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(email_stages, fontsize=7)
    ax2.set_ylim(0, 0.96)
    ax2.set_title('(b) Email: 4B Training Stages', fontsize=9, fontweight='bold')

    fig.suptitle('Model Scale & Training Stage Comparison', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_e8_model_scaling.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Latest-data overrides and new script-generated analysis figures
# ═══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))


def _load_json(rel_path):
    with open(os.path.join(ROOT_DIR, rel_path), 'r') as f:
        return json.load(f)


def _maybe_load_json(rel_path):
    path = os.path.join(ROOT_DIR, rel_path)
    if not os.path.exists(path):
        return None
    return _load_json(rel_path)


def _m(d, scenario, key, metric='hit1', default=None):
    try:
        return d[scenario][key][metric]
    except KeyError:
        return default


def _overall_metric(rel_path, metric='hit_at_1', default=None):
    d = _maybe_load_json(rel_path)
    if not d:
        return default
    if 'overall' in d:
        return d['overall'].get(metric, default)
    return d.get(metric, default)


def _ecommerce_gpt55(metric='hit_at_1'):
    metric_map = {'hit1': 'hit_at_1', 'mrr': 'mrr', 'hit3': 'hit_at_3', 'exec_rate': 'exec_success_rate'}
    return _overall_metric('results/ecommerce/eval_predictions_gpt55_full.json',
                           metric_map.get(metric, metric), None)


def _bar_label(ax, bars, vals, dy=0.008, fontsize=5.5, bold_first=0):
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, val + dy, f'{val:.3f}',
                ha='center', va='bottom', fontsize=fontsize,
                fontweight='bold' if i < bold_first else 'normal')


# Override Fig 1 with latest results/analysis/e1_main_results.json.
def fig_main_ecommerce():
    e1 = _load_json('results/analysis/e1_main_results.json')

    groups = [
        ('Ours', [
            ('SFT (8B)', _m(e1, 'ecommerce', 'SFT_8B'), '#B71C1C'),
            ('GRPO (4B)', _m(e1, 'ecommerce', 'GRPO_4B_step300'), OURS_COLOR),
            ('DAPO (4B)', _m(e1, 'ecommerce', 'DAPO_4B_step420'), OURS_COLOR2),
        ]),
        ('SFT', [
            ('SFT (4B)', _m(e1, 'ecommerce', 'SFT_4B'), SFT_COLOR),
        ]),
        ('Commercial LLMs', [
            ('GPT-5.5', _ecommerce_gpt55('hit1'), LLM_COLOR),
            ('Opus 4.7', _m(e1, 'ecommerce', 'Opus_4.7'), '#6A9FCA'),
            ('Gemini 3.1', _m(e1, 'ecommerce', 'Gemini_3.1Pro'), '#93BAD9'),
            ('GPT-5.4', _m(e1, 'ecommerce', 'GPT-5.4'), '#BBCFE8'),
        ]),
        ('LLM-Enhanced', [
            ('RankGPT', _m(e1, 'ecommerce', 'B2_RankGPT'), B2_COLOR),
            ('SQR', _m(e1, 'ecommerce', 'B3_SQR'), '#82C47B'),
            ('IRCoT', _m(e1, 'ecommerce', 'B2_IRCoT'), '#ABD9A5'),
            ('HyDE', _m(e1, 'ecommerce', 'B2_HyDE'), '#C8E6C2'),
        ]),
        ('RL / Program Retrieval', [
            ('SQL-R1', _m(e1, 'ecommerce', 'B3_SQL-R1'), B3_COLOR),
            ('DeepRetrieval', _m(e1, 'ecommerce', 'B3_DeepRetrieval'), '#CCBDD1'),
            ('Search-R1', _m(e1, 'ecommerce', 'B3_Search-R1_step400'), '#E0D5E3'),
        ]),
        ('Traditional / Graph', [
            ('Reranker', _m(e1, 'ecommerce', 'B1_Reranker'), B1_COLOR),
            ('LightRAG', _m(e1, 'ecommerce', 'B4_LightRAG'), '#8BB7B4'),
            ('BM25', _m(e1, 'ecommerce', 'B1_BM25'), '#A0D0CC'),
            ('GraphRAG', _m(e1, 'ecommerce', 'B4_GraphRAG'), B4_COLOR),
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


# Override Fig 2 with latest results/analysis/e1_main_results.json.
def fig_main_email():
    e1 = _load_json('results/analysis/e1_main_results.json')
    groups = [
        ('Ours', [
            ('DAPO (4B)', _m(e1, 'email', 'DAPO_4B_step843'), OURS_COLOR),
            ('GRPO (4B)', _m(e1, 'email', 'GRPO_4B_step843'), OURS_COLOR2),
        ]),
        ('SFT', [
            ('SFT (4B)', _m(e1, 'email', 'SFT_4B'), SFT_COLOR),
        ]),
        ('Commercial LLMs', [
            ('GPT-5.5', _m(e1, 'email', 'GPT-5.5'), LLM_COLOR),
            ('GPT-5.4', _m(e1, 'email', 'GPT-5.4'), '#6A9FCA'),
            ('Opus 4.7', _m(e1, 'email', 'Opus_4.7'), '#93BAD9'),
        ]),
        ('LLM-Enhanced', [
            ('RankGPT', _m(e1, 'email', 'B2_RankGPT'), B2_COLOR),
            ('IRCoT', _m(e1, 'email', 'B2_IRCoT'), '#ABD9A5'),
        ]),
        ('Traditional', [
            ('Reranker', _m(e1, 'email', 'B1_Reranker'), B1_COLOR),
            ('BM25', _m(e1, 'email', 'B1_BM25'), '#A0D0CC'),
            ('RRF', _m(e1, 'email', 'B1_RRF'), '#B7DCD9'),
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
    fig, ax = plt.subplots(figsize=(6.2, 2.65))
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
    ax.set_xticklabels(labels, rotation=38, ha='right', fontsize=6.5)
    ax.set_ylim(0.48, 0.97)
    ax.set_title('Email Retrieval Performance (Hit@1)', fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout()
    save(fig, 'fig_main_email.pdf')


# Override Figure 2 with the latest training paradigm comparison values.
def fig_fewshot_failure():
    methods = ['GPT-5.5', 'SFT', 'GRPO', 'DAPO']
    hit1 = [0.667, 0.680, 0.809, 0.808]
    exec_rate = [1.000, 0.997, 0.997, 0.996]

    colors = [LLM_COLOR, SFT_COLOR, OURS_COLOR, OURS_COLOR2]
    fig, ax1 = plt.subplots(figsize=(4.2, 2.55))
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
    ax2.set_ylim(0.65, 1.08)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False, fontsize=7)
    ax1.set_title('Training Paradigm Comparison (E-commerce)', fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout()
    save(fig, 'fig_fewshot_failure.pdf')


# Override Fig 9 with current orchestration analysis JSON.
def fig_e6_orchestration():
    e6 = _load_json('results/analysis/e6_orchestration_analysis.json')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    colors_act = ['#4E79A7', '#59A14F', '#E15759']

    methods_ec = [
        ('Base (4B)', 'base4b'),
        ('SFT (4B)', 'sft_4b'),
        ('SFT (8B)', 'sft8b'),
        ('GRPO (4B)', 'grpo4b_step300'),
        ('DAPO (4B)', 'dapo4b_step420'),
        ('GPT-5.5', 'gpt55_full'),
        ('Opus 4.7', 'opus47'),
    ]
    x = np.arange(len(methods_ec))
    w = 0.25
    for offset, action, label, color in [
        (-w, 'sql_filter', 'SQL Filter', colors_act[0]),
        (0, 'text_search', 'Text Search', colors_act[1]),
        (w, 'image_search', 'Image Search', colors_act[2]),
    ]:
        vals = [e6['ecommerce'][key]['action_freq'].get(action, 0.0) for _, key in methods_ec]
        ax1.bar(x + offset, vals, w, label=label, color=color, edgecolor='white', linewidth=0.3)
    ax1.set_ylabel('Action Frequency')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m for m, _ in methods_ec], rotation=30, ha='right', fontsize=6.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')

    methods_em = [
        ('Base (4B)', 'base_email'),
        ('SFT (4B)', 'sft_email'),
        ('GRPO (4B)', 'grpo_step843'),
        ('DAPO (4B)', 'dapo_step843'),
        ('GPT-5.5', 'gpt55'),
        ('Opus 4.7', 'opus47'),
    ]
    x2 = np.arange(len(methods_em))
    for offset, action, label, color in [
        (-w/2, 'sql_filter', 'SQL Filter', colors_act[0]),
        (w/2, 'text_search', 'Text Search', colors_act[1]),
    ]:
        vals = [e6['email'][key]['action_freq'].get(action, 0.0) for _, key in methods_em]
        ax2.bar(x2 + offset, vals, w, label=label, color=color, edgecolor='white', linewidth=0.3)
    ax2.set_ylabel('Action Frequency')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([m for m, _ in methods_em], rotation=30, ha='right', fontsize=6.5)
    ax2.set_ylim(0, 1.15)
    ax2.set_title('(b) Email', fontsize=9, fontweight='bold')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle('Retrieval Action Orchestration by Method', fontsize=11, fontweight='bold', y=1.12)
    fig.tight_layout()
    save(fig, 'fig_e6_orchestration.pdf')


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


def fig_training_dynamics_paper():
    e1 = _load_json('results/analysis/e1_main_results.json')
    email_summary = _load_json('figures/training_curves/grpo_email_summary.json')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    steps = [200, 300, 550, 1200]
    vals = [
        _overall_metric('results/ecommerce/eval_rl_step200_gt_v2.json', 'hit_at_1', 0.806),
        _m(e1, 'ecommerce', 'GRPO_4B_step300'),
        _overall_metric('results/ecommerce/eval_rl_step550_gt_v2.json', 'hit_at_1', 0.798),
        _overall_metric('results/ecommerce/eval_rl_step1200_gt_v2.json', 'hit_at_1', 0.003),
    ]
    ax1.plot(steps, vals, 'o-', color=OURS_COLOR, linewidth=1.8, markersize=5, label='GRPO')
    dapo = _m(e1, 'ecommerce', 'DAPO_4B_step420')
    ax1.scatter([420], [dapo], marker='D', s=45, color=OURS_COLOR2, label='DAPO step420', zorder=4)
    ax1.annotate('best', (300, vals[1]), textcoords='offset points', xytext=(0, 10),
                 ha='center', fontsize=7, fontweight='bold', color=OURS_COLOR)
    ax1.annotate('collapse', (1200, vals[-1]), textcoords='offset points', xytext=(-8, 10),
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


def fig_e7_modality_selection():
    methods = ['SFT\n4B', 'Opus\n4.7', 'GPT-5.5', 'GRPO\n4B', 'DAPO\n4B']
    acc = [0.808, 0.721, 0.709, 0.597, 0.597]
    hit1 = [0.680, 0.635, 0.667, 0.809, 0.808]
    colors = [SFT_COLOR, '#6A9FCA', LLM_COLOR, OURS_COLOR, OURS_COLOR2]

    fig, ax1 = plt.subplots(figsize=(5.8, 2.8))
    x = np.arange(len(methods))
    bars = ax1.bar(x, acc, 0.58, color=colors, edgecolor='white', linewidth=0.3,
                   label='Modality accuracy')
    for bar, v in zip(bars, acc):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.010, f'{v:.3f}',
                 ha='center', fontsize=6,
                 fontweight='bold' if v == max(acc) else 'normal')
    ax1.set_ylabel('Modality Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=7)
    ax1.set_ylim(0.50, 0.86)

    ax2 = ax1.twinx()
    ax2.plot(x, hit1, 'D--', color='#333333', markersize=4.5, linewidth=1.1,
             label='Hit@1')
    ax2.spines['right'].set_visible(True)
    ax2.set_ylabel('Hit@1')
    ax2.set_ylim(0.55, 0.86)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper right', fontsize=7)
    ax1.set_title('E7: Modality Selection Accuracy vs Retrieval Performance',
                  fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout()
    save(fig, 'fig_e7_modality_selection.pdf')


def fig_e10_ood_generalization():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    panels = [
        ('E-commerce', ['Base', 'SFT', 'GRPO', 'DAPO'],
         [0.117, 0.680, 0.809, 0.808],
         [0.113, 0.683, 0.783, 0.793],
         [B1_COLOR, SFT_COLOR, OURS_COLOR, OURS_COLOR2]),
        ('Email', ['Base', 'SFT', 'GRPO', 'DAPO'],
         [0.243, 0.891, 0.906, 0.909],
         [0.217, 0.873, 0.883, 0.883],
         [B1_COLOR, SFT_COLOR, OURS_COLOR, OURS_COLOR2]),
    ]

    for ax, (title, labels, iid, ood, colors) in zip(axes, panels):
        x = np.arange(len(labels))
        w = 0.34
        bars1 = ax.bar(x - w/2, iid, w, color=colors, alpha=0.45,
                       edgecolor='white', linewidth=0.3, label='IID')
        bars2 = ax.bar(x + w/2, ood, w, color=colors, alpha=0.95,
                       edgecolor='white', linewidth=0.3, label='OOD')
        for i, (a, b) in enumerate(zip(iid, ood)):
            delta = (b - a) * 100
            ax.text(i, max(a, b) + 0.018, f'{delta:+.1f}pp',
                    ha='center', fontsize=6, color='#444444',
                    fontweight='bold' if labels[i] in ['GRPO', 'DAPO'] else 'normal')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 0.98)
        ax.set_ylabel('Hit@1')
        ax.legend(frameon=False, fontsize=7, loc='upper left')

    fig.suptitle('E10: OOD Generalization (N=300)', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_e10_ood_generalization.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating paper figures (v3 — latest JSON + new scripted analyses)...\n')
    fig_main_ecommerce()            # 1
    fig_main_email()                 # 2
    fig_e2_action_space()            # 3  (NEW: P0 action-space ablation)
    fig_e3_e5_combined_ecommerce()   # 3
    fig_e3_e5_combined_email()       # 4
    fig_paradigm_comparison()        # 5
    fig_rl_progression()             # 6
    fig_fewshot_failure()            # 7
    fig_cross_scenario()             # 8
    fig_e6_orchestration()           # 9  (NEW: orchestration behavior)
    fig_e7_modality_selection()      # 10 (NEW: diagnostic modality selection)
    fig_e9_error_analysis()          # 10 (NEW: error analysis)
    fig_e8_model_scaling()           # 11 (NEW: model scaling)
    fig_e10_ood_generalization()     # 12 (NEW: OOD generalization)
    fig_dataset_query_stats()        # 12 (NEW: benchmark query statistics)
    fig_training_dynamics_paper()    # 13 (NEW: paper-quality training dynamics)
    print(f'\n  All 15 figures saved to {OUT_DIR}/')
