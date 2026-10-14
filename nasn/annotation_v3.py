"""Study v3 human annotation intake: integrity checks, agreement, attention check, adjudication queue.

Usage after both annotators return their sheets:
    python -m nasn.annotation_v3 --package runs/study-v3/annotation-v1 \
        --a returned/annotation_sheet_A.csv --b returned/annotation_sheet_B.csv --output runs/study-v3/annotation-v1-intake
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preflight import require

FIELDS = {'meaning_preserved': ['yes', 'no', 'unsure'], 'sentiment_preserved': ['yes', 'no', 'unsure'],
          'modified_sentiment': ['NEG', 'NEU', 'POS'], 'negation_present': ['yes', 'no'],
          'negation_affected': ['yes', 'no', 'na']}
PILOT_ITEMS = 20


def read_sheet(path, package):
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    blank = pd.read_csv(Path(package) / 'annotation_sheet_A.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    require(sheet.item.tolist() == blank.item.tolist(), f'{path}: items missing, added or reordered')
    require(sheet.original.tolist() == blank.original.tolist() and sheet.modified.tolist() == blank.modified.tolist(),
            f'{path}: tweet text was edited')
    for field, values in FIELDS.items():
        sheet[field] = sheet[field].str.strip()
        bad = sheet[~sheet[field].isin(values)]
        require(bad.empty, f'{path}: {field} has missing/invalid values in {bad.item.head(5).tolist()}')
    bad = sheet[(sheet.negation_present == 'no') & (sheet.negation_affected != 'na')]
    require(bad.empty, f'{path}: negation_affected must be na without negation: {bad.item.head(5).tolist()}')
    return sheet


def agreement(a, b, categories):
    """Observed agreement, Cohen's kappa and Gwet's AC1 for two raters."""
    a, b = np.asarray(a), np.asarray(b)
    observed = float((a == b).mean())
    pa = np.array([(a == c).mean() for c in categories])
    pb = np.array([(b == c).mean() for c in categories])
    chance_kappa = float((pa * pb).sum())
    pi = (pa + pb) / 2
    chance_ac1 = float((pi * (1 - pi)).sum() / (len(categories) - 1))
    kappa = (observed - chance_kappa) / (1 - chance_kappa) if chance_kappa < 1 else float('nan')
    ac1 = (observed - chance_ac1) / (1 - chance_ac1) if chance_ac1 < 1 else float('nan')
    return {'n': len(a), 'agreement': observed, 'cohen_kappa': kappa, 'gwet_ac1': ac1}


def intake(package, path_a, path_b, output):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), f'Output exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(Path(package) / 'annotation_key_private.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    a, b = read_sheet(path_a, package), read_sheet(path_b, package)
    digest = lambda t: hashlib.sha256(t.encode()).hexdigest()
    require(a.original.map(digest).tolist() == key.clean_sha256.tolist() and
            a.modified.map(digest).tolist() == key.perturbed_sha256.tolist(), 'Package key does not match sheets')
    main = key.index >= PILOT_ITEMS  # pilot items are discussed by annotators, so excluded from agreement
    rows = [{'field': f, 'scope': 'all_main', **agreement(a[f][main], b[f][main], v)} for f, v in FIELDS.items()]
    for condition, part in key[main].groupby('condition'):
        rows += [{'field': 'meaning_preserved', 'scope': condition, **agreement(a.meaning_preserved[part.index],
                  b.meaning_preserved[part.index], FIELDS['meaning_preserved'])}]
    pd.DataFrame(rows).to_csv(output / 'agreement.csv', index=False)
    typo = key.condition.str.startswith('typo/')
    attention = {name: float(sheet.meaning_preserved[typo].eq('no').mean()) for name, sheet in [('A', a), ('B', b)]}
    fields = list(FIELDS)
    differs = (a[fields] != b[fields]).any(axis=1)
    queue = key[['item', 'condition']].join(a[['original', 'modified']])
    for name, sheet in [('A', a), ('B', b)]:
        queue = queue.join(sheet[fields + ['notes']].add_suffix(f'_{name}'))
    queue[differs].drop(columns='condition').to_csv(output / 'adjudication_queue.csv', index=False, encoding='utf-8-sig')
    agreed = key[~differs][['item', 'variant_id', 'condition']].assign(**{f: a[f][~differs] for f in fields})
    agreed.to_csv(output / 'agreed_decisions.csv', index=False)
    preserved = agreed.assign(preserved=agreed.meaning_preserved.eq('yes') & agreed.sentiment_preserved.eq('yes'))
    summary = {'items': len(key), 'pilot_items_excluded_from_agreement': PILOT_ITEMS,
               'agreed_all_fields': int((~differs).sum()), 'needs_adjudication': int(differs.sum()),
               'attention_check_typo_marked_meaning_changed': attention,
               'agreed_preserved_rate_by_condition': preserved.groupby('condition').preserved.mean().round(4).to_dict(),
               'sheet_sha256': {'A': hashlib.sha256(Path(path_a).read_bytes()).hexdigest(),
                                'B': hashlib.sha256(Path(path_b).read_bytes()).hexdigest()}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--package', required=True)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    intake(args.package, args.a, args.b, args.output)


if __name__ == '__main__':
    main()
