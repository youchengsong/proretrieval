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
            ('Opus 4.7',  0.642, '#6A9FCA'),
            ('GPT-5.4',   0.581, '#93BAD9'),
            ('Gemini 3.1', 0.576, '#BBCFE8'),
        ]),
        ('LLM-Enhanced', [
            ('RankGPT',   0.762, B2_COLOR),
            ('IRCoT',     0.547, '#82C47B'),
            ('SQR',       0.607, '#ABD9A5'),
            ('HyDE',      0.534, '#C8E6C2'),
        ]),
        ('RL-Retrieval', [
            ('SQL-R1',    0.780, B3_COLOR),
            ('DeepRetrieval', 0.638, '#CCBDD1'),
            ('Search-R1', 0.459, '#E0D5E3'),
        ]),
        ('Traditional', [
            ('Reranker',  0.593, B1_COLOR),
            ('BM25',      0.548, '#A0D0CC'),
            ('GraphRAG',  0.493, B4_COLOR),
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
        ax.text((s + e - 1) / 2, 0.39, gname, ha='center', fontsize=6, color='#666666',
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
            ('RankGPT',   0.890, B2_COLOR),
        ]),
        ('Traditional + RL', [
            ('Reranker',  0.836, B1_COLOR),
            ('DeepRetrieval', 0.745, '#CCBDD1'),
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

    methods_e3 = ['GRPO', 'DAPO', 'SFT', 'GPT-5.5', 'RankGPT', 'Reranker', 'BM25']
    data_e3 = {
        'GRPO':       [0.864, 0.797, 0.720],
        'DAPO':       [0.859, 0.800, 0.717],
        'SFT':        [0.752, 0.644, 0.602],
        'GPT-5.5':    [0.666, 0.680, 0.642],
        'RankGPT':    [0.710, 0.803, 0.794],
        'Reranker':   [0.455, 0.691, 0.703],
        'BM25':       [0.435, 0.623, 0.653],
    }
    colors_e3 = [OURS_COLOR, OURS_COLOR2, SFT_COLOR, LLM_COLOR, B2_COLOR, B1_COLOR, '#A0D0CC']

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

    methods_e5 = ['GRPO', 'DAPO', 'SFT', 'GPT-5.5', 'Reranker', 'BM25']
    data_e5 = {
        'GRPO':       [0.843, 0.690, 0.716],
        'DAPO':       [0.843, 0.690, 0.711],
        'SFT':        [0.743, 0.634, 0.320],
        'GPT-5.5':    [0.724, 0.607, 0.481],
        'Reranker':   [0.558, 0.779, 0.789],
        'BM25':       [0.516, 0.761, 0.724],
    }
    colors_e5 = [OURS_COLOR, OURS_COLOR2, SFT_COLOR, LLM_COLOR, B1_COLOR, '#A0D0CC']

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
               loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=7, frameon=False, fontsize=6.5)

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
    # Methods with results on BOTH scenarios where possible
    methods = [
        'Retrieval Program\nSynthesis (Ours)',
        'SQL-R1\n(SQL-only RL)',
        'RankGPT\n(LLM Rerank)',
        'DeepRetrieval\n(Boolean RL)',
        'SQR\n(LLM Filter)',
        'BGE-Reranker\n(Neural Rerank)',
        'BM25\n(Lexical)',
        'GraphRAG\n(Knowledge Graph)',
        'Search-R1\n(BM25 RL)',
    ]
    ecom  = [0.809, 0.780, 0.762, 0.638, 0.607, 0.593, 0.548, 0.493, 0.459]
    email = [0.909, None,  0.890, 0.745, 0.617, 0.836, 0.731, None,  None]

    colors_ecom  = [OURS_COLOR, B3_COLOR, B2_COLOR, B3_COLOR, B2_COLOR, B1_COLOR, B1_COLOR, B4_COLOR, B3_COLOR]
    colors_email = [OURS_COLOR, B3_COLOR, B2_COLOR, B3_COLOR, B2_COLOR, B1_COLOR, B1_COLOR, B4_COLOR, B3_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)

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

    # Mark N/A entries
    for i, val in enumerate(email):
        if val is None:
            ax2.text(0.50, y[i], 'N/A', va='center', ha='center', fontsize=6.5, color='#999999', fontstyle='italic')

    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Hit@1')
    ax2.set_xlim(0.38, 0.97)
    ax2.invert_yaxis()
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
        ax1.annotate(f'{h:.3f}', (i, h), textcoords='offset points', xytext=(0, 9),
                     fontsize=7, ha='center', color=OURS_COLOR, fontweight='bold')

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
# Fig 7: Few-shot DSL failure (red gradient for trained methods, neutral title)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fewshot_failure():
    methods = ['GPT-5.5\n(few-shot)', 'Opus 4.7\n(few-shot)', 'GPT-5.5\n(zero-shot)', 'SFT\n(4B)', 'GRPO\n(4B)']
    hit1      = [0.253, 0.250, 0.667, 0.680, 0.809]
    exec_rate = [0.480, 0.503, 1.000, 0.997, 0.997]

    # Few-shot: tan; trained methods: red gradient (light → dark)
    colors = [FEWSHOT_COLOR, FEWSHOT_COLOR, '#F5B7B1', '#E74C3C', '#B71C1C']

    fig, ax1 = plt.subplots(figsize=(4.5, 2.5))
    x = np.arange(len(methods))

    bars = ax1.bar(x, hit1, 0.55, color=colors, edgecolor='white', linewidth=0.3, label='Hit@1', zorder=2)

    ax2 = ax1.twinx()
    ax2.plot(x, exec_rate, 'D-', color='#333333', markersize=5, linewidth=1.2, label='Exec Rate', zorder=3)
    ax2.spines['right'].set_visible(True)

    ax1.set_ylabel('Hit@1')
    ax2.set_ylabel('Execution Rate', color='#333333')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=7)
    ax1.set_ylim(0, 0.90)
    ax2.set_ylim(0.3, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False, fontsize=7)

    ax1.set_title('Few-shot DSL vs SFT vs RL', fontsize=10, fontweight='bold', pad=6)

    fig.tight_layout()
    save(fig, 'fig_fewshot_failure.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8: Cross-scenario comparison (pp not %)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cross_scenario():
    methods = ['DAPO\n(Ours)', 'GRPO\n(Ours)', 'SFT', 'GPT-5.5', 'Opus 4.7', 'BGE-\nReranker', 'BM25']
    ecom    = [0.808, 0.809, 0.680, 0.667, 0.642, 0.593, 0.548]
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
    methods_ec = ['Base\n(4B)', 'Few-shot\nGPT-5.5', 'Few-shot\nOpus 4.7', 'SFT\n(4B)', 'GRPO\n(4B)', 'DAPO\n(4B)', 'GPT-5.5', 'Opus 4.7']
    total_ec      = [3000, 3000, 3000, 3000, 3000, 3000, 1000, 2998]
    format_ok_ec  = [949,  2885, 2352, 3000, 2999, 2998, 1000, 2998]
    exec_ok_ec    = [506,  2741, 2200, 2992, 2992, 2989, 1000, 2996]

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
# Fig 11: E8 — Model Scaling (1.7B vs 4B across training stages)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_e8_model_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # E-commerce
    stages = ['Base', 'SFT', 'GRPO', 'DAPO']
    h1_1_7b = [None,  0.680, None,  None]
    h1_4b   = [0.117, 0.680, 0.809, 0.808]

    x = np.arange(len(stages))
    w = 0.3

    c_small = '#93BAD9'
    c_large = OURS_COLOR

    # 1.7B bars (only where data exists)
    for i, v in enumerate(h1_1_7b):
        if v is not None:
            ax1.bar(i - w/2, v, w, color=c_small, edgecolor='white', linewidth=0.3,
                    label='1.7B' if i == 1 else None)
            ax1.text(i - w/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=6, color=c_small)

    for i, v in enumerate(h1_4b):
        ax1.bar(i + w/2, v, w, color=c_large, edgecolor='white', linewidth=0.3,
                label='4B' if i == 0 else None)
        ax1.text(i + w/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=6, fontweight='bold' if i >= 2 else 'normal')

    ax1.set_ylabel('Hit@1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7)
    ax1.set_ylim(0, 0.90)
    ax1.legend(loc='upper left', frameon=False, fontsize=7)
    ax1.set_title('(a) E-commerce', fontsize=9, fontweight='bold')

    # Email (no 1.7B data — show training progression)
    stages_em = ['Base', 'SFT', 'GRPO', 'DAPO']
    h1_em_4b  = [0.243, 0.891, 0.906, 0.909]

    ax2.bar(np.arange(len(stages_em)), h1_em_4b, 0.5, color=[c_small, SFT_COLOR, OURS_COLOR, OURS_COLOR2],
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
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating paper figures (v2 — revised)...\n')
    fig_main_ecommerce()            # 1
    fig_main_email()                 # 2
    fig_e3_e5_combined_ecommerce()   # 3
    fig_e3_e5_combined_email()       # 4
    fig_paradigm_comparison()        # 5
    fig_rl_progression()             # 6
    fig_fewshot_failure()            # 7
    fig_cross_scenario()             # 8
    fig_e6_orchestration()           # 9  (NEW: orchestration behavior)
    fig_e9_error_analysis()          # 10 (NEW: error analysis)
    fig_e8_model_scaling()           # 11 (NEW: model scaling)
    print(f'\n  All 11 figures saved to {OUT_DIR}/')
