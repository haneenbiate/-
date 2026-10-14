import hashlib

import pandas as pd
import pytest

from nasn.annotation_v3 import agreement, intake


def test_agreement_formulas():
    perfect = agreement(['yes', 'no'] * 5, ['yes', 'no'] * 5, ['yes', 'no'])
    assert perfect['cohen_kappa'] == 1 and perfect['gwet_ac1'] == 1
    # Prevalence paradox: high agreement, kappa near zero, AC1 high.
    a = ['yes'] * 19 + ['no']
    b = ['yes'] * 18 + ['no', 'yes']
    result = agreement(a, b, ['yes', 'no'])
    assert result['agreement'] == pytest.approx(.9)
    assert result['cohen_kappa'] < .1 < .85 < result['gwet_ac1']


def package(tmp_path, n=40):
    original = [f'نص {i}' for i in range(n)]
    modified = [f'نـص {i}' for i in range(n)]
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    conditions = ['typo/single/non_cue' if i % 4 == 0 else 'tatweel/single/non_cue' for i in range(n)]
    pd.DataFrame({'item': [f'item{i:04d}' for i in range(n)], 'variant_id': [f'v{i}' for i in range(n)],
                  'source_row_id': [f'r{i}' for i in range(n)], 'condition': conditions,
                  'clean_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in original],
                  'perturbed_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in modified]}
                 ).to_csv(pkg / 'annotation_key_private.csv', index=False, encoding='utf-8-sig')
    sheet = pd.DataFrame({'item': [f'item{i:04d}' for i in range(n)], 'original': original, 'modified': modified})
    for c in ['meaning_preserved', 'sentiment_preserved', 'modified_sentiment', 'negation_present', 'negation_affected', 'notes']:
        sheet[c] = ''
    sheet.to_csv(pkg / 'annotation_sheet_A.csv', index=False, encoding='utf-8-sig')
    return pkg, sheet, conditions


def filled(sheet, conditions, flip=()):
    out = sheet.copy()
    out['meaning_preserved'] = ['no' if c.startswith('typo') else 'yes' for c in conditions]
    out['sentiment_preserved'] = 'yes'
    out['modified_sentiment'] = ['POS' if i not in flip else 'NEG' for i in range(len(out))]
    out['negation_present'] = 'no'
    out['negation_affected'] = 'na'
    return out


def test_intake(tmp_path):
    pkg, sheet, conditions = package(tmp_path)
    filled(sheet, conditions).to_csv(tmp_path / 'A.csv', index=False, encoding='utf-8-sig')
    filled(sheet, conditions, flip={25, 30}).to_csv(tmp_path / 'B.csv', index=False, encoding='utf-8-sig')
    summary = intake(pkg, tmp_path / 'A.csv', tmp_path / 'B.csv', tmp_path / 'out')
    assert summary['needs_adjudication'] == 2 and summary['agreed_all_fields'] == 38
    assert summary['attention_check_typo_marked_meaning_changed'] == {'A': 1.0, 'B': 1.0}
    assert summary['agreed_preserved_rate_by_condition']['typo/single/non_cue'] == 0
    queue = pd.read_csv(tmp_path / 'out/adjudication_queue.csv', encoding='utf-8-sig')
    assert queue.item.tolist() == ['item0025', 'item0030']


def test_intake_rejects_edited_text_and_bad_values(tmp_path):
    pkg, sheet, conditions = package(tmp_path)
    good = filled(sheet, conditions)
    edited = good.copy()
    edited.loc[3, 'modified'] = 'شيء آخر'
    edited.to_csv(tmp_path / 'A.csv', index=False, encoding='utf-8-sig')
    good.to_csv(tmp_path / 'B.csv', index=False, encoding='utf-8-sig')
    with pytest.raises(Exception, match='edited'):
        intake(pkg, tmp_path / 'A.csv', tmp_path / 'B.csv', tmp_path / 'o1')
    bad = good.copy()
    bad.loc[0, 'meaning_preserved'] = 'maybe'
    bad.to_csv(tmp_path / 'A.csv', index=False, encoding='utf-8-sig')
    with pytest.raises(Exception, match='invalid'):
        intake(pkg, tmp_path / 'A.csv', tmp_path / 'B.csv', tmp_path / 'o2')
