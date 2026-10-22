import json
import zipfile

import numpy as np
import pandas as pd

from nasn import study_v3
from nasn.study_v3 import LABELS, analyze, build_variants, load_config, unique_inputs
from tests.test_study_v3 import frame
from tools.report_study_v3 import report


def bundle(tmp_path):
    """Public bundle produced by the real analysis code from two fake models."""
    config = {**load_config(), 'bootstrap_repetitions': 50, 'minimum_cell_size': 5}
    variants = build_variants(frame(60), config)
    path = tmp_path / 'variants.csv.gz'
    variants.to_csv(path, index=False)
    inputs = unique_inputs(study_v3.read_variants(path), config['defenses'])
    files = []
    for model, seed in [('MARBERT', 1), ('MARBERT', 2), ('XLM-R-base', 1)]:
        # MARBERT changes class whenever tatweel reaches it; XLM-R never does.
        probs = np.full((len(inputs), 3), .1)
        flipped = inputs.input_text.str.contains('ـ').to_numpy() & (model == 'MARBERT')
        probs[np.arange(len(inputs)), np.where(flipped, 0, 2)] = .8
        out = tmp_path / f'{model}{seed}'
        out.mkdir()
        table = inputs.drop(columns='input_text').assign(model=model, seed=seed)
        for i, label in enumerate(LABELS):
            table[f'p_{label}'] = probs[:, i]
        table.to_csv(out / 'predictions.csv.gz', index=False)
        files.append(out / 'predictions.csv.gz')
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    args = type('A', (), {'output': tmp_path / 'analysis', 'variants': path, 'predictions': files,
                          'config': config_path, 'cue_only': False})
    analyze(args, config)
    archive = tmp_path / 'public.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name in ['robustness_cells.csv', 'contrasts.csv', 'natural_noise_cells.csv']:
            z.write(tmp_path / 'analysis' / name, f'arsarcasm-all-{name}')
        for model, seed, f1 in [('MARBERT', 42, .75), ('MARBERT', 43, .77), ('XLM-R-base', 42, .70)]:
            z.writestr(f'arsarcasm-{model}-s{seed}-validation_metrics.json',
                       json.dumps({'status': 'completed', 'macro_f1': f1, 'accuracy': .8}))
    return archive


def test_report_verdicts_tables_and_figure(tmp_path):
    verdicts, _ = report(bundle(tmp_path), tmp_path / 'out', tokenizer={})
    v = verdicts.set_index(['hypothesis', 'model', 'test']).verdict
    # MARBERT flips on every tatweel edit while XLM-R never does.
    assert v[('H2', 'MARBERT', 'MARBERT minus XLM-R-base, tatweel/dense')] == 'supported'
    # STD strips tatweel, so MARBERT stops flipping: lower flip rate under STD.
    assert v[('H4', 'MARBERT', 'STD minus RAW, tatweel/dense/non_cue')] == 'supported'
    # Cue and non-cue tatweel both flip every time: no difference.
    assert v[('H3', 'MARBERT', 'tatweel/single/cue minus tatweel/single/non_cue')] == 'not supported'
    out = tmp_path / 'out'
    clean = (out / 'clean_performance.tex').read_text(encoding='utf-8')
    assert 'MARBERT & 76.0 $\\pm$ 1.4' in clean and 'XLM-R' in clean
    flips = (out / 'flip_rates_arsarcasm.tex').read_text(encoding='utf-8')
    assert 'Tatweel, 1 word & 100.0 & 0.0' in flips
    assert (out / 'flip_rates.pdf').stat().st_size > 1000
    summary = (out / 'hypotheses_summary.md').read_text(encoding='utf-8')
    assert 'ArSarcasm-v2 H2: 1/1 models supported' in summary
    assert pd.read_csv(out / 'hypotheses.csv').verdict.notna().all()
