"""Paper tables for the Study v3 human validation (agreement and per-condition preservation).

    python -m tools.report_annotation_v3 --package runs/study-v3/annotation-v1 --a A.xlsx --b B.xlsx \
        [--final runs/study-v3/annotation-v1-final/final_decisions.csv] --output paper/generated
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from nasn.annotation_v3 import FIELDS, PILOT_ITEMS, agreement, read_sheet
from tools.report_study_v3 import CONDITION_LABELS

FIELD_LABELS = {'meaning_preserved': 'Meaning preserved', 'sentiment_preserved': 'Sentiment preserved',
                'modified_sentiment': 'Sentiment of edited text', 'negation_present': 'Negation present',
                'negation_affected': 'Negation affected by edit'}


def report(package, path_a, path_b, final, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    a, b = read_sheet(path_a, package), read_sheet(path_b, package)
    key = pd.read_csv(Path(package) / 'annotation_key_private.csv', encoding='utf-8-sig')
    main = key.index >= PILOT_ITEMS
    rows = [{'field': f, **agreement(a[f][main], b[f][main], v)} for f, v in FIELDS.items()]
    table = ['\\begin{tabular}{lrrr}', '\\toprule', 'Field & Agreement & $\\kappa$ & AC1 \\\\', '\\midrule']
    for row in rows:
        table.append(f"{FIELD_LABELS[row['field']]} & {100 * row['agreement']:.1f}\\% & {row['cohen_kappa']:.2f} & {row['gwet_ac1']:.2f} \\\\")
    table += ['\\bottomrule', '\\end{tabular}']
    (output / 'human_agreement.tex').write_text('\n'.join(table) + '\n', encoding='utf-8', newline='\n')

    frame = key[['item', 'condition']].copy()
    for name, sheet in [('a', a), ('b', b)]:
        frame[f'meaning_{name}'] = sheet.meaning_preserved.eq('yes').to_numpy()
        frame[f'sentiment_{name}'] = sheet.sentiment_preserved.eq('yes').to_numpy()
    frame['both_meaning'] = frame.meaning_a & frame.meaning_b
    frame['both_preserved'] = frame.both_meaning & frame.sentiment_a & frame.sentiment_b
    has_final = final is not None and Path(final).is_file()
    if has_final:
        decided = pd.read_csv(final).set_index('item')
        frame['final_preserved'] = frame.item.map(decided.preserved).astype(bool)
    head = 'Condition & Meaning (A / B) & Both: meaning & Both: meaning + sentiment' + (' & Final' if has_final else '')
    table = ['\\begin{tabular}{l' + 'r' * (4 if has_final else 3) + '}', '\\toprule', head + ' \\\\', '\\midrule']
    summary = {}
    for condition, label in CONDITION_LABELS.items():
        part = frame[frame.condition == condition]
        if part.empty:
            continue
        cells = [f"{part.meaning_a.sum()} / {part.meaning_b.sum()}", f"{part.both_meaning.sum()}", f"{part.both_preserved.sum()}"]
        if has_final:
            cells.append(f"{part.final_preserved.sum()}")
        table.append(f"{label} & {' & '.join(cells)} \\\\")
        summary[condition] = {'n': len(part), 'meaning_a': int(part.meaning_a.sum()), 'meaning_b': int(part.meaning_b.sum()),
                              'both_meaning': int(part.both_meaning.sum()), 'both_preserved': int(part.both_preserved.sum()),
                              **({'final_preserved': int(part.final_preserved.sum())} if has_final else {})}
    table += ['\\bottomrule', '\\end{tabular}']
    (output / 'human_preservation.tex').write_text('\n'.join(table) + '\n', encoding='utf-8', newline='\n')
    (output / 'human_validation.json').write_text(json.dumps({'agreement': rows, 'by_condition': summary,
                                                             'adjudicated': has_final}, indent=2) + '\n',
                                                 encoding='utf-8', newline='\n')
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--package', required=True)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--final')
    parser.add_argument('--output', default='paper/generated')
    args = parser.parse_args()
    rows, summary = report(args.package, args.a, args.b, args.final, args.output)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(summary).T.to_string())


if __name__ == '__main__':
    main()
