"""Turn the Study v3 public analysis bundle into paper tables, a figure and preregistered verdicts.

Input is the text-free folder (or its ZIP) written by the notebook's analysis cell, i.e.
`<dataset>-<all|cue-tweets>-{robustness_cells,contrasts,natural_noise_cells}.csv` plus
`<dataset>-<model>-s<seed>-validation_metrics.json`. Everything is derived mechanically
from those files; no number is typed by hand.

    python tools/report_study_v3.py --public path/to/<stamp>-public.zip --output paper/generated
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

DATASETS = {'arsarcasm': 'ArSarcasm-v2', 'arsas': 'ArSAS'}
WORDPIECE = ['MARBERT', 'MARBERTv2', 'AraBERTv02-twitter', 'CAMeLBERT-DA']
ENCODERS = WORDPIECE + ['XLM-R-base']
MODEL_ORDER = ENCODERS + ['Qwen2.5-7B-Instruct']
SHORT = {'AraBERTv02-twitter': 'AraBERT-tw', 'CAMeLBERT-DA': 'CAMeL-DA', 'XLM-R-base': 'XLM-R',
         'Qwen2.5-7B-Instruct': 'Qwen2.5'}
REFERENCE = 'XLM-R-base'
PRESERVING = ('tatweel', 'elongation', 'fold')
CONDITION_LABELS = {
    'tatweel/single/non_cue': 'Tatweel, 1 word', 'tatweel/dense/non_cue': 'Tatweel, 30\\% of words',
    'tatweel/single/cue': 'Tatweel on negation cue',
    'elongationx2/single/non_cue': 'Elongation $\\times$2, 1 word', 'elongationx4/single/non_cue': 'Elongation $\\times$4, 1 word',
    'elongationx2/dense/non_cue': 'Elongation $\\times$2, 30\\% of words',
    'elongationx2/single/cue': 'Elongation $\\times$2 on negation cue',
    'fold/single/non_cue': 'Folding, 1 word', 'fold/dense/non_cue': 'Folding, 30\\% of words',
    'typo/single/non_cue': 'Typo (control), 1 word', 'typo/dense/non_cue': 'Typo (control), 30\\% of words'}
TOKENIZER = {'arsarcasm': 'reports/study-v3/tokenizer_summary.csv', 'arsas': 'reports/study-v3/arsas_tokenizer_summary.csv'}


def load_bundle(path):
    """Return {file name: DataFrame or dict} for a folder or ZIP bundle."""
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            items = {Path(n).name: archive.read(n) for n in archive.namelist() if not n.endswith('/')}
    else:
        items = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    import io
    bundle = {}
    for name, data in items.items():
        if name.endswith('.csv'):
            bundle[name] = pd.read_csv(io.BytesIO(data))
        elif name.endswith('.json'):
            bundle[name] = json.loads(data)
    return bundle


def ordered(models):
    return [m for m in MODEL_ORDER if m in set(models)] + sorted(set(models) - set(MODEL_ORDER))


def clean_performance(bundle):
    rows = []
    for name, value in bundle.items():
        match = re.fullmatch(r'(arsarcasm|arsas)-(.+)-s(\d+)-validation_metrics\.json', name)
        if match and isinstance(value, dict) and value.get('status') == 'completed':
            rows.append({'dataset': match[1], 'model': match[2], 'seed': int(match[3]),
                         'macro_f1': value['macro_f1'], 'accuracy': value['accuracy']})
    return pd.DataFrame(rows, columns=['dataset', 'model', 'seed', 'macro_f1', 'accuracy'])


def latex_clean(frame):
    summary = frame.groupby(['dataset', 'model']).macro_f1.agg(['mean', 'std', 'size']).reset_index()
    lines = ['\\begin{tabular}{lrr}', '\\toprule', 'Model & ' + ' & '.join(DATASETS.values()) + ' \\\\', '\\midrule']
    for model in ordered(summary.model):
        cells = []
        for dataset in DATASETS:
            row = summary[(summary.dataset == dataset) & (summary.model == model)]
            cells.append('--' if row.empty else f"{100 * row['mean'].iloc[0]:.1f} $\\pm$ {100 * row['std'].iloc[0]:.1f}")
        lines.append(f"{SHORT.get(model, model)} & {' & '.join(cells)} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def latex_flips(cells, dataset):
    raw = cells[(cells.defense == 'RAW')].set_index(['condition', 'model'])
    models = ordered(cells.model)
    lines = ['\\begin{tabular}{l' + 'r' * len(models) + '}', '\\toprule',
             'Condition & ' + ' & '.join(SHORT.get(m, m) for m in models) + ' \\\\', '\\midrule']
    for condition, label in CONDITION_LABELS.items():
        values = []
        for model in models:
            if (condition, model) not in raw.index or pd.isna(raw.loc[(condition, model), 'flip_rate']):
                values.append('--')
                continue
            row = raw.loc[(condition, model)]
            values.append(f"{100 * row.flip_rate:.1f}")
        lines.append(f"{label} & {' & '.join(values)} \\\\")
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def verdict(row, direction):
    """Supported when the 95% interval excludes zero in the stated direction."""
    if pd.isna(row.get('low')):
        return 'not computed (n<30)'
    if direction > 0:
        return 'supported' if row['low'] > 0 else ('opposite' if row['high'] < 0 else 'not supported')
    return 'supported' if row['high'] < 0 else ('opposite' if row['low'] > 0 else 'not supported')


def hypotheses(contrasts, dataset):
    rows = []

    def add(hypothesis, model, label, row, direction):
        rows.append({'dataset': dataset, 'hypothesis': hypothesis, 'model': model, 'test': label,
                     'n': row.get('n'), 'difference': row.get('difference'), 'low': row.get('low'),
                     'high': row.get('high'), 'verdict': verdict(row, direction)})

    raw = contrasts[contrasts.get('defense', pd.Series(index=contrasts.index, dtype=object)).eq('RAW')]
    for _, row in raw[raw.kind == 'density_contrasts'].iterrows():
        if row.a.startswith(PRESERVING) and row.model in ENCODERS + ['Qwen2.5-7B-Instruct']:
            add('H1', row.model, f'{row.a} minus {row.b}', row, +1)
    for _, row in contrasts[(contrasts.kind == 'model_contrasts') & (contrasts.condition == 'tatweel/dense/non_cue')].iterrows():
        if row.a in WORDPIECE and row.b == REFERENCE:
            add('H2', row.a, f'{row.a} minus {row.b}, tatweel/dense', row, +1)
    for _, row in raw[(raw.kind == 'cue_contrasts') & raw.a.str.startswith('tatweel')].iterrows():
        if row.model in WORDPIECE:
            add('H3', row.model, f'{row.a} minus {row.b}', row, +1)
    defense = contrasts[contrasts.kind == 'defense_contrasts']
    for _, row in defense[(defense.a == 'STD') & (defense.b == 'RAW')].iterrows():
        if row.model in ENCODERS and str(row.condition).startswith(('tatweel', 'elongation')):
            add('H4', row.model, f'STD minus RAW, {row.condition}', row, -1)
    for _, row in defense[(defense.a == 'NASN2') & (defense.b == 'NO_CONTEXT')].iterrows():
        if row.model in ENCODERS:
            add('H5', row.model, f'NASN2 minus NO_CONTEXT, {row.condition}', row, -1)
    return pd.DataFrame(rows)


def summarize(verdicts):
    """One line per hypothesis and dataset, following the preregistered decision rules."""
    lines = []
    for (dataset, hypothesis), part in verdicts.groupby(['dataset', 'hypothesis'], sort=True):
        supported = part.verdict.eq('supported')
        if hypothesis == 'H5':
            # Claim only if NASN2 beats NO_CONTEXT, per encoder, in >= 3 of 5 encoders (any condition counts once).
            wins = part[supported].model.nunique()
            status = f'NASN2 advantage in {wins}/{part.model.nunique()} encoders; claim allowed: {wins >= 3}'
        else:
            per_model = part.groupby('model').verdict.apply(lambda v: v.eq('supported').all())
            status = (f'{int(per_model.sum())}/{len(per_model)} models supported in every test; '
                      f'{int(supported.sum())}/{len(part)} individual tests supported')
        lines.append(f'- {DATASETS.get(dataset, dataset)} {hypothesis}: {status}')
    return '\n'.join(lines) + '\n'


def rq2(cells, dataset, tokenizer_path):
    if not tokenizer_path or not Path(tokenizer_path).is_file():
        return None
    path = Path(tokenizer_path)
    tokens = pd.read_csv(path)
    merged = cells[cells.defense == 'RAW'].merge(tokens, on=['model', 'condition'], how='inner')
    merged = merged.dropna(subset=['flip_rate'])
    if len(merged) < 5:
        return None
    return {'dataset': dataset, 'cells': len(merged),
            'spearman_flip_vs_unk_rate': merged.flip_rate.corr(merged.edited_word_unk_rate, method='spearman'),
            'spearman_flip_vs_fragmentation': merged.flip_rate.corr(merged.mean_fragmentation, method='spearman')}


def figure(cells_by_dataset, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    conditions = list(CONDITION_LABELS)
    fig, axes = plt.subplots(1, len(cells_by_dataset), figsize=(7.2, 4.2), sharey=True, squeeze=False)
    for ax, (dataset, cells) in zip(axes[0], cells_by_dataset.items()):
        raw = cells[cells.defense == 'RAW'].set_index(['condition', 'model'])
        models = ordered(cells.model)
        step = .7 / max(len(models), 1)
        for k, model in enumerate(models):
            for i, condition in enumerate(conditions):
                if (condition, model) not in raw.index or pd.isna(raw.loc[(condition, model), 'flip_rate']):
                    continue
                row = raw.loc[(condition, model)]
                y = len(conditions) - 1 - i + (k - (len(models) - 1) / 2) * step
                ax.plot([100 * row.flip_rate_low, 100 * row.flip_rate_high], [y, y], color=colors[k % 6], linewidth=1)
                ax.scatter(100 * row.flip_rate, y, color=colors[k % 6], marker=markers[k % 6], s=16, zorder=3,
                           label=SHORT.get(model, model) if i == 0 else None)
        ax.set_yticks(range(len(conditions)), [CONDITION_LABELS[c].replace('\\%', '%').replace('$\\times$', '×')
                                                for c in reversed(conditions)], fontsize=7)
        ax.set_xlabel('Predictions changed by the edit (%)', fontsize=8)
        ax.set_title(DATASETS.get(dataset, dataset), fontsize=9, loc='left')
        ax.grid(axis='x', color='#e4e3df', linewidth=.6)
        ax.tick_params(axis='x', labelsize=7)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output, bbox_inches='tight')
    fig.savefig(Path(output).with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def write(path, text):
    Path(path).write_text(text, encoding='utf-8', newline='\n')


def report(public, output, tokenizer=TOKENIZER):
    bundle = load_bundle(public)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clean = clean_performance(bundle)
    if not clean.empty:
        write(output / 'clean_performance.tex', latex_clean(clean))
    cells_by_dataset, verdicts, correlations = {}, [], []
    for dataset in DATASETS:
        cells = bundle.get(f'{dataset}-all-robustness_cells.csv')
        contrasts = bundle.get(f'{dataset}-all-contrasts.csv')
        if cells is None:
            continue
        cells_by_dataset[dataset] = cells
        write(output / f'flip_rates_{dataset}.tex', latex_flips(cells, dataset))
        if contrasts is not None:
            verdicts.append(hypotheses(contrasts, dataset))
        correlation = rq2(cells, dataset, tokenizer.get(dataset, ''))
        if correlation:
            correlations.append(correlation)
    verdicts = pd.concat(verdicts, ignore_index=True) if verdicts else pd.DataFrame()
    verdicts.to_csv(output / 'hypotheses.csv', index=False, lineterminator='\n')
    pd.DataFrame(correlations).to_csv(output / 'rq2_tokenizer_correlations.csv', index=False, lineterminator='\n')
    write(output / 'hypotheses_summary.md', summarize(verdicts) if not verdicts.empty else 'No contrasts found.\n')
    if cells_by_dataset:
        figure(cells_by_dataset, output / 'flip_rates.pdf')
    return verdicts, correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--public', required=True, help='Public analysis folder or ZIP from the notebook')
    parser.add_argument('--output', default='paper/generated')
    args = parser.parse_args()
    verdicts, correlations = report(args.public, args.output)
    print((Path(args.output) / 'hypotheses_summary.md').read_text(encoding='utf-8'))
    print(pd.DataFrame(correlations).to_string(index=False))


if __name__ == '__main__':
    main()
