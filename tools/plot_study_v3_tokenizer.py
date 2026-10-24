"""Paper figure: how each tokenizer represents a word after one orthographic edit (CPU audit only)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

MODELS = ['MARBERT', 'MARBERTv2', 'AraBERTv02-twitter', 'CAMeLBERT-DA', 'XLM-R-base']
LABELS = {'Qwen2.5-7B-Instruct': 'Qwen2.5-7B', 'AraBERTv02-twitter': 'AraBERTv02-tw'}
# Fixed categorical order (reference palette slots 1-5) with a distinct marker as secondary encoding.
SERIES = [('tatweel/single/non_cue', 'Tatweel', '#2a78d6', 'o'),
          ('elongationx2/single/non_cue', 'Elongation ×2', '#eb6834', 's'),
          ('elongationx4/single/non_cue', 'Elongation ×4', '#1baf7a', 'D'),
          ('fold/single/non_cue', 'Hamza/ta/ya folding', '#eda100', '^'),
          ('typo/single/non_cue', 'Keyboard typo', '#e87ba4', 'v')]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def panel(ax, summary, title):
    data = summary.set_index(['model', 'condition'])
    y = {m: len(MODELS) - 1 - i for i, m in enumerate(MODELS)}
    offsets = [-.24, -.12, 0, .12, .24]
    for (condition, label, color, marker), dy in zip(SERIES, offsets):
        for model in MODELS:
            row = data.loc[(model, condition)]
            unk = row.edited_word_unk_rate > .5
            ax.scatter(row.mean_fragmentation, y[model] + dy, s=34, marker=marker, zorder=3,
                       facecolor='white' if unk else color, edgecolor=color, linewidth=1.6)
            if unk and condition.startswith('tatweel'):
                ax.annotate('→ [UNK]', (row.mean_fragmentation, y[model] + dy), xytext=(6, -1),
                            textcoords='offset points', fontsize=7, color=MUTED, va='center')
    ax.axhline(.5, color=MUTED, linewidth=.6, linestyle=(0, (2, 2)))
    ax.axvline(1, color=MUTED, linewidth=.8)
    ax.set_yticks(list(y.values()), [LABELS.get(m, m) for m in y], fontsize=8, color=INK)
    ax.set_xlim(.5, 4.5)
    ax.set_xlabel('Tokens for the edited word ÷ tokens for the original', fontsize=8, color=INK)
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.grid(axis='x', color=GRID, linewidth=.6)
    ax.tick_params(axis='x', labelsize=7, colors=MUTED)
    ax.tick_params(axis='y', length=0)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.text(4.45, .62, 'Arabic WordPiece', fontsize=7, color=MUTED, ha='right', va='bottom')
    ax.text(4.45, .38, 'Multilingual', fontsize=7, color=MUTED, ha='right', va='top')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arsarcasm', default='reports/study-v3/tokenizer_summary.csv')
    parser.add_argument('--arsas', default='reports/study-v3/arsas_tokenizer_summary.csv')
    parser.add_argument('--output', default='paper/figures/tokenizer_edits.pdf')
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    panel(axes[0], pd.read_csv(args.arsarcasm), 'ArSarcasm-v2 validation (n ≈ 2,000 per edit)')
    panel(axes[1], pd.read_csv(args.arsas), 'ArSAS validation (n ≈ 1,600 per edit)')
    handles = [plt.Line2D([], [], linestyle='', marker=m, markersize=6, color=c, label=l) for _, l, c, m in SERIES]
    handles.append(plt.Line2D([], [], linestyle='', marker='o', markersize=6, markerfacecolor='white',
                              markeredgecolor=MUTED, markeredgewidth=1.4, label='hollow = word becomes [UNK]'))
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=7.5, handletextpad=.3,
               columnspacing=1.0, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .08, 1, 1))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    fig.savefig(Path(args.output).with_suffix('.png'), dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
