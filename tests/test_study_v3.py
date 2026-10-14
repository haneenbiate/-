import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from nasn import study_v3
from nasn.study_v3 import LABELS, analyze, build_variants, defend, load_config, macro_f1, text_hash, unique_inputs


def frame(n=40):
    texts = ['الفيلم كان جميل جدا ولا أحب النهاية', 'مو حلو أبدا هالمسلسل', 'كثير رائعة الحلقة الأخيرة', 'الطقس اليوم عادي']
    return pd.DataFrame({'tweet': [texts[i % 4] + f' {i}' for i in range(n)], 'source_row_id': [f'r{i}' for i in range(n)],
                         'group_id': [f'g{i // 2}' for i in range(n)], 'partition': 'validation',
                         'sentiment': [LABELS[i % 3] for i in range(n)], 'dialect': 'msa', 'sarcasm': 'FALSE'})


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    gold, pred = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    assert macro_f1(gold, pred) == pytest.approx(f1_score(gold, pred, average='macro', labels=[0, 1, 2]))
    assert macro_f1(gold, np.stack([pred, gold]))[1] == 1


def test_defenses():
    assert defend('حلـــووووو', 'STD') == 'حلو'
    assert defend('حلـو', 'RAW') == 'حلـو'
    assert 'ـ' not in defend('حلـو جدا', 'NO_CONTEXT')


def test_variants_and_unique_inputs():
    config = load_config()
    variants = build_variants(frame(8), config)
    assert len(variants) == 8 * len(config['conditions'])
    assert variants.groupby('source_row_id').variant_id.nunique().eq(len(config['conditions'])).all()
    cue = variants[variants.condition.eq('elongationx2/single/cue') & variants.applicable]
    assert set(cue.source_row_id) == {'r0', 'r1', 'r4', 'r5'}
    inputs = unique_inputs(variants, ['RAW', 'STD'])
    assert not inputs.duplicated(['defense', 'raw_sha256']).any()


def test_analyze_end_to_end(tmp_path):
    config = load_config()
    variants = build_variants(frame(), config)
    path = tmp_path / 'variants.csv.gz'
    variants.to_csv(path, index=False)
    inputs = unique_inputs(study_v3.read_variants(path), ['RAW', 'STD'])
    files = []
    for model, seed in [('fake', 1), ('fake', 2), ('XLM-R-base', 1)]:
        # 'fake' changes class whenever its input contains tatweel; the reference never does.
        probs = np.full((len(inputs), 3), .1)
        flipped = inputs.input_text.str.contains('ـ').to_numpy() & (model == 'fake')
        probs[np.arange(len(inputs)), np.where(flipped, 0, 2)] = .8
        out = tmp_path / f'{model}{seed}'
        out.mkdir()
        frame_ = inputs.drop(columns='input_text').assign(model=model, seed=seed)
        for i, label in enumerate(LABELS):
            frame_[f'p_{label}'] = probs[:, i]
        frame_.to_csv(out / 'predictions.csv.gz', index=False)
        files.append(out / 'predictions.csv.gz')
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps({**config, 'bootstrap_repetitions': 50, 'minimum_cell_size': 5}), encoding='utf-8')
    args = type('A', (), {'output': tmp_path / 'out', 'variants': path, 'predictions': files, 'config': config_path, 'cue_only': False})
    analyze(args, {**config, 'bootstrap_repetitions': 50, 'minimum_cell_size': 5})
    cells = pd.read_csv(tmp_path / 'out/robustness_cells.csv').set_index(['model', 'defense', 'condition'])
    assert cells.loc[('fake', 'RAW', 'tatweel/single/non_cue'), 'flip_rate'] == 1
    assert cells.loc[('fake', 'RAW', 'tatweel/single/non_cue'), 'flip_rate_low'] == 1
    assert cells.loc[('fake', 'STD', 'tatweel/single/non_cue'), 'flip_rate'] == 0
    assert cells.loc[('fake', 'RAW', 'fold/dense/non_cue'), 'flip_rate'] == 0
    assert cells.loc[('fake', 'RAW', 'tatweel/single/non_cue'), 'seeds'] == 2
    contrasts = pd.read_csv(tmp_path / 'out/contrasts.csv')
    cue = contrasts[(contrasts.kind == 'cue_contrasts') & (contrasts.a == 'tatweel/single/cue') & (contrasts.model == 'fake')
                    & (contrasts.defense == 'RAW')].iloc[0]
    assert cue.difference == 0 and cue.n == 20
    std = contrasts[(contrasts.kind == 'defense_contrasts') & (contrasts.model == 'fake') & (contrasts.a == 'STD')
                    & (contrasts.condition == 'tatweel/dense/non_cue')].iloc[0]
    assert std.difference == -1
    between = contrasts[(contrasts.kind == 'model_contrasts') & (contrasts.condition == 'tatweel/dense/non_cue')].iloc[0]
    assert (between.a, between.b, between.difference, between.low) == ('fake', 'XLM-R-base', 1, 1)


def test_test_partition_locked():
    with pytest.raises(Exception, match='locked'):
        study_v3.load_split('test', load_config())
