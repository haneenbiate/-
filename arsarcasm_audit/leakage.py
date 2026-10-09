"""Leakage-safe manifests retaining immutable official data and annotations."""
import argparse
import csv
import json
import math
import platform
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit

from .audit import (COMMIT, HASHES, LABELS, REPOSITORY, DataValidationError,
                    baseline_partitions, digest, fetch, load_csv, near_pairs, sha256)

SEED = 42
CANDIDATES = 512

def canonical_text(text):
    """NFC + removal of U+0640 tatweel + collapse Unicode whitespace. Nothing else."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text.replace('\u0640', ''))).strip()

def group_id(text):
    return 'ct1_' + digest(canonical_text(text).encode('utf-8'))

def attach_ids(frame, source):
    result = frame.copy()
    result['source_split'] = source
    result['row_id'] = np.arange(len(result))
    result['source_row_id'] = [f'{source}:{i:05d}' for i in range(len(result))]
    result['text_sha256'] = result.tweet.map(lambda x: digest(x.encode('utf-8')))
    result['canonical'] = result.tweet.map(canonical_text)
    if result.canonical.eq('').any():
        raise DataValidationError(f'{source}: canonical text is empty after tatweel removal')
    result['group_id'] = result.tweet.map(group_id)
    return result

def group_split(eligible, fraction=.2, seed=SEED):
    """Optimize candidate group splits using eligible training sentiment only."""
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise DataValidationError('Validation fraction must be in (0, 1)')
    if not 0 <= seed <= 2**32 - 1:
        raise DataValidationError('Seed outside sklearn random-state range')
    if eligible.group_id.nunique() < 2:
        raise DataValidationError('At least two eligible canonical groups are needed')
    labels = list(LABELS['sentiment'])
    total = eligible.sentiment.value_counts().reindex(labels, fill_value=0).to_numpy()
    best = None
    splitter = GroupShuffleSplit(n_splits=CANDIDATES, test_size=fraction, random_state=seed)
    for order, (train, val) in enumerate(splitter.split(eligible, groups=eligible.group_id)):
        val_counts = eligible.iloc[val].sentiment.value_counts().reindex(labels, fill_value=0).to_numpy()
        train_counts = total - val_counts
        # Presence first, then per-class target count error + total size error.
        score = (int(((train_counts == 0) & (total > 0)).sum()),
                 int(((val_counts == 0) & (total > 0)).sum()),
                 float(np.mean(np.abs(val_counts - fraction * total) / np.maximum(total, 1))
                       + abs(len(val) - fraction * len(eligible)) / len(eligible)), order)
        if best is None or score < best[0]:
            best = (score, train, val)
    _, train, val = best
    assignments = pd.Series('train', index=eligible.index)
    assignments.loc[eligible.iloc[val].index] = 'validation'
    return assignments, {'candidates': CANDIDATES, 'chosen_candidate': best[0][3],
                         'missing_train_classes': best[0][0], 'missing_validation_classes': best[0][1],
                         'balance_objective': best[0][2]}

def write_csv(path, records, columns):
    pd.DataFrame(records, columns=columns).to_csv(path, index=False, encoding='utf-8')

MANIFEST = ['source_split', 'row_id', 'source_row_id', 'group_id', 'text_sha256',
            'partition', 'sentiment', 'sarcasm', 'dialect']

def overlap_audit(frame, stage):
    summaries, members = [], []
    for kind, key in (('exact_text', 'tweet'), ('canonical_text', 'canonical')):
        for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            a, b = frame[frame.partition == left], frame[frame.partition == right]
            common = sorted(set(a[key]) & set(b[key]))
            left_rows = right_rows = pairs = conflicts = conflict_groups = 0
            for text in common:
                x, y = a[a[key] == text], b[b[key] == text]
                n_pairs = len(x) * len(y)
                equal_pairs = sum(int(x.sentiment.eq(label).sum()) * int(y.sentiment.eq(label).sum())
                                  for label in LABELS['sentiment'])
                left_rows += len(x)
                right_rows += len(y)
                pairs += n_pairs
                conflicts += n_pairs - equal_pairs
                conflict_groups += int(n_pairs != equal_pairs)
                for _, row in pd.concat([x, y]).iterrows():
                    members.append({'stage': stage, 'kind': kind, 'scope': f'{left}_{right}',
                                    'source_row_id': row.source_row_id, 'group_id': row.group_id,
                                    'partition': row.partition, 'sentiment': row.sentiment,
                                    'text_sha256': row.text_sha256})
            summaries.append({'stage': stage, 'kind': kind, 'scope': f'{left}_{right}',
                              'groups': len(common), 'left_rows': left_rows, 'right_rows': right_rows,
                              'row_pairs': pairs, 'conflicting_sentiment_groups': conflict_groups,
                              'conflicting_sentiment_pairs': conflicts})
    return summaries, members

def group_audit(frame):
    records = []
    for kind, key in (('exact_text', 'tweet'), ('canonical_text', 'canonical')):
        for text, group in frame.groupby(key, sort=True):
            if len(group) < 2:
                continue
            records.append({'kind': kind, 'comparison_sha256': digest(text.encode('utf-8')),
                            'group_id': group.iloc[0].group_id, 'rows': len(group),
                            'official_train_rows': int(group.source_split.eq('official_train').sum()),
                            'official_test_rows': int(group.source_split.eq('official_test').sum()),
                            'sentiment_labels': '|'.join(sorted(group.sentiment.unique())),
                            'conflicting_sentiment': group.sentiment.nunique() > 1,
                            'source_row_ids': '|'.join(group.source_row_id)})
    return records

def distributions(frame):
    records = []
    slices = {name: frame[frame.partition == name] for name in ('train', 'validation', 'test', 'quarantine')}
    slices.update({name: frame[frame.source_split == name] for name in ('official_train', 'official_test')})
    for name, subset in slices.items():
        for column, labels in LABELS.items():
            for label in labels:
                count = int(subset[column].eq(label).sum())
                records.append({'split': name, 'field': column, 'label': label, 'count': count,
                                'proportion': count / len(subset) if len(subset) else 0})
    return records

def prepare(training, testing, output, fraction=.2, seed=SEED, require_official=True):
    training, testing, output = Path(training), Path(testing), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise DataValidationError(f'Output must be an empty/new directory: {output}')
    source_hashes = {'official_train': sha256(training), 'official_test': sha256(testing)}
    if require_official:
        for source, name in (('official_train', 'training_data.csv'), ('official_test', 'testing_data.csv')):
            if source_hashes[source] != HASHES[name]:
                raise DataValidationError(f'{source}: does not match pinned official SHA-256')
    train_raw, test_raw = load_csv(training), load_csv(testing)
    if require_official and (len(train_raw), len(test_raw)) != (12548, 3000):
        raise DataValidationError('Pinned official data must have 12,548 training and 3,000 test rows')
    train, test = attach_ids(train_raw, 'official_train'), attach_ids(test_raw, 'official_test')
    train['baseline_partition'] = baseline_partitions(train, .2, SEED)
    test['baseline_partition'] = 'test'
    # Test LABELS are deliberately absent from the quarantine decision and split optimizer.
    test_groups = set(test.group_id)
    overlap = train.group_id.isin(test_groups)
    train['partition'] = 'quarantine'
    eligible = train[~overlap]
    assignments, choice = group_split(eligible, fraction, seed)
    train.loc[eligible.index, 'partition'] = assignments
    test['partition'] = 'test'
    train['quarantine_reason'] = ''
    raw_test_texts = set(test.tweet)
    for i in train.index[overlap]:
        train.loc[i, 'quarantine_reason'] = ('exact_test_text_overlap' if train.loc[i, 'tweet'] in raw_test_texts
                                            else 'canonical_test_text_overlap')
    all_rows = pd.concat([train, test], ignore_index=True)
    baseline = all_rows.copy()
    baseline['partition'] = baseline.baseline_partition
    summaries, overlap_members = overlap_audit(baseline, 'reconstructed_task01')
    new_summary, new_members = overlap_audit(all_rows, 'task02')
    summaries += new_summary
    overlap_members += new_members
    # Runtime safety invariants, independently checked by tests too.
    active = all_rows[all_rows.partition.isin(['train', 'validation'])]
    if set(active.group_id) & test_groups:
        raise DataValidationError('Internal invariant: test overlap in active training data')
    if set(active[active.partition == 'train'].group_id) & set(active[active.partition == 'validation'].group_id):
        raise DataValidationError('Internal invariant: train-validation canonical overlap')
    if len(train) != len(active) + int(overlap.sum()) or test.row_id.tolist() != list(range(len(test_raw))):
        raise DataValidationError('Internal invariant: lost source rows or changed test order')
    output.mkdir(parents=True, exist_ok=True)
    all_rows[MANIFEST].to_csv(output / 'split_manifest.csv', index=False)
    baseline[MANIFEST].to_csv(output / 'baseline_manifest.csv', index=False)
    for partition in ('train', 'validation', 'test'):
        all_rows.loc[all_rows.partition == partition, MANIFEST].to_csv(output / f'{partition}_manifest.csv', index=False)
    train.loc[overlap, MANIFEST + ['quarantine_reason']].to_csv(output / 'quarantine_manifest.csv', index=False)
    transitions = train[MANIFEST + ['baseline_partition', 'quarantine_reason']].copy()
    transitions['assignment_changed'] = transitions.partition != transitions.baseline_partition
    transitions.to_csv(output / 'training_deviations.csv', index=False)
    official_train_groups = set(train.group_id)
    exact_train_texts = set(train.tweet)
    evaluation = test[MANIFEST].copy()
    evaluation['overlaps_official_train_canonical'] = test.group_id.isin(official_train_groups)
    evaluation['overlaps_official_train_exact'] = test.tweet.isin(exact_train_texts)
    evaluation['test_group_size'] = test.groupby('group_id').group_id.transform('size')
    evaluation['canonical_representative'] = ~test.group_id.duplicated()
    evaluation['unique_nonoverlap'] = evaluation.canonical_representative & ~evaluation.overlaps_official_train_canonical
    evaluation['evaluation_category'] = np.where(evaluation.overlaps_official_train_canonical,
        'overlapping_official_train', np.where(evaluation.canonical_representative,
        'unique_nonoverlap', 'repeated_test_only'))
    evaluation.to_csv(output / 'test_evaluation_manifest.csv', index=False)
    all_rows[MANIFEST].to_csv(output / 'group_members.csv', index=False)
    duplicate_groups = group_audit(all_rows)
    write_csv(output / 'duplicate_groups.csv', duplicate_groups,
              ['kind', 'comparison_sha256', 'group_id', 'rows', 'official_train_rows', 'official_test_rows',
               'sentiment_labels', 'conflicting_sentiment', 'source_row_ids'])
    write_csv(output / 'overlap_statistics.csv', summaries,
              ['stage', 'kind', 'scope', 'groups', 'left_rows', 'right_rows', 'row_pairs',
               'conflicting_sentiment_groups', 'conflicting_sentiment_pairs'])
    write_csv(output / 'overlap_members.csv', overlap_members,
              ['stage', 'kind', 'scope', 'source_row_id', 'group_id', 'partition', 'sentiment', 'text_sha256'])
    class_stats = distributions(all_rows)
    write_csv(output / 'distributions.csv', class_stats, ['split', 'field', 'label', 'count', 'proportion'])
    near_total = 0
    near_cols = ['left_source_row_id', 'right_source_row_id', 'left_group_id', 'right_group_id',
                 'left_partition', 'right_partition', 'cosine_similarity', 'review_only']
    with (output / 'near_duplicate_review.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=near_cols)
        writer.writeheader()
        for left, right, similarity in near_pairs(all_rows.canonical.tolist()):
            a, b = all_rows.iloc[left], all_rows.iloc[right]
            writer.writerow(dict(zip(near_cols, [a.source_row_id, b.source_row_id, a.group_id, b.group_id,
                a.partition, b.partition, similarity, True])))
            near_total += 1
    if source_hashes != {'official_train': sha256(training), 'official_test': sha256(testing)}:
        raise DataValidationError('Source files changed during preparation; discard partial outputs')
    counts = {p: int(all_rows.partition.eq(p).sum()) for p in ('train', 'validation', 'test', 'quarantine')}
    counts.update({'official_train': len(train), 'official_test': len(test), 'source_rows_deleted': 0,
                   'quarantine_exact_match': int(train.quarantine_reason.eq('exact_test_text_overlap').sum()),
                   'quarantine_canonical_only': int(train.quarantine_reason.eq('canonical_test_text_overlap').sum()),
                   'unique_nonoverlap_test_rows': int(evaluation.unique_nonoverlap.sum()),
                   'overlapping_test_rows': int(evaluation.overlaps_official_train_canonical.sum()),
                   'near_duplicate_review_pairs': near_total})
    metadata = {'source_repository': REPOSITORY, 'source_commit': COMMIT, 'source_sha256': source_hashes,
        'official_checksum_enforced': require_official, 'seed': seed, 'validation_fraction_target': fraction,
        'validation_fraction_actual': counts['validation'] / len(eligible), 'counts': counts,
        'canonicalization': 'remove U+0640; NFC; collapse Unicode whitespace; strip; no casefold or letter folding',
        'group_id': 'ct1_ + SHA-256 of canonical UTF-8 text',
        'source_row_id': 'official_train/test + zero-based data-record position; anchored by source_sha256',
        'split_method': '512 GroupShuffleSplit candidates; lexicographic training-only class/size objective',
        'split_selection': choice, 'baseline': {'status': 'reconstructed, not recovered original artifact',
            'method': 'train_test_split; stratify=sentiment; test_size=0.2; random_state=42'},
        'test_labels_used_for_split': False, 'near_duplicates_merged': False,
        'near_method': 'audit-only canonical char TF-IDF 3-5 grams, cosine >= 0.9, sparse blocks of 128',
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
                     'scikit-learn': sklearn.__version__, 'unicode': unicodedata.unidata_version}}
    artifact_hashes = {p.name: sha256(p) for p in sorted(output.glob('*.csv'))}
    metadata['artifact_sha256'] = artifact_hashes
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report(output, metadata, summaries, class_stats, duplicate_groups)
    return metadata

def report(output, metadata, summaries, stats, groups):
    lines = ['# Task 02: leakage-safe splitting methodology', '',
        'No model was trained. No evaluation scores or model-selection results are claimed.', '',
        '## Source and baseline recovery', '',
        f"Official source: {REPOSITORY}, commit `{COMMIT}`. Checksums are in metadata.json.",
        'The restarted workspace contained only its initial README. Earlier Task 01 code, tests, reports and '
        'manifests were absent. Baseline assignments were reconstructed from the previously documented '
        'seed-42, 20% sentiment-stratified train_test_split, not read from recovered artifacts. '
        'Baseline overlap values below are newly measured on that reconstruction.', '',
        '## Canonical grouping', '',
        'Remove U+0640 tatweel, apply Unicode NFC, collapse Unicode whitespace and strip leading/trailing '
        'whitespace. Do not casefold, apply NFKC, fold hamza/alef/ya/ta marbuta, strip diacritics, '
        'remove punctuation/URLs/negation, or otherwise clean text. Canonical keys are for grouping only; '
        'raw CSVs and annotations remain unchanged. NFC combines canonically equivalent decompositions, '
        'not distinct Arabic letters. IDs are full SHA-256 canonical text hashes prefixed ct1_. '
        'Row IDs are zero-based CSV data-record positions anchored by file checksums.', '',
        '## Quarantine and group-disjoint assignment', '',
        'Quarantine every official training row whose canonical group occurs in official test text. '
        'Only test text is consulted for this decision. Quarantine rows are excluded from both fitting '
        'and validation, but retained in the master manifest with explicit reasons. All other training '
        'rows are retained, including duplicate rows and conflicting annotations; no majority label, '
        'deduplication or relabeling is applied. Each eligible canonical group stays wholly in one partition.',
        f"Seed: {metadata['seed']}. Target validation row fraction: {metadata['validation_fraction_target']}. "
        f"Actual fraction after quarantine: {metadata['validation_fraction_actual']:.6f}.",
        'Generate 512 GroupShuffleSplit candidates from eligible official training data. Group counts '
        'target the requested fraction; row/class balance is approximate because group sizes differ. '
        'Rank candidates lexicographically by number of missing training classes, number of missing '
        'validation classes, then mean per-class validation count error (divided by each class total) '
        'plus row-fraction error. Break ties by candidate order. Only eligible training sentiment '
        'labels enter this optimization; test labels, test scores and near-duplicate findings do not. '
        'The finite search does not guarantee global optimal balance. Missing-class counts are in metadata.', '',
        '## Deviations from official training', '',
        'No source records are deleted or rewritten. The sole eligibility exclusion is test-text '
        'quarantine. The remaining official training rows are assigned to new group-disjoint train and '
        'validation partitions; these differ from the old row-wise split. training_deviations.csv lists '
        'every original training row, its old/new assignment, and any quarantine reason. '
        'quarantine_manifest.csv enumerates every excluded row. No conflict or near-duplicate exclusions '
        'are applied. Deviations are in manifests only.', '',
        '| Count | Value |', '| --- | ---: |']
    lines += [f'| {key} | {value} |' for key, value in metadata['counts'].items()]
    lines += ['', '## Exact and canonical overlap', '',
        'A group is one shared comparison text. Row pairs count every cross-partition combination. '
        'A conflicting pair has unequal sentiment labels; a conflicting group has at least one such '
        'cross-partition pair. Quarantine is excluded from Task 02 train/validation overlap checks.', '',
        '| Stage | Comparison | Scope | Groups | Left rows | Right rows | Pairs | Conflict groups | Conflict pairs |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {s['stage']} | {s['kind']} | {s['scope']} | {s['groups']} | {s['left_rows']} | "
              f"{s['right_rows']} | {s['row_pairs']} | {s['conflicting_sentiment_groups']} | "
              f"{s['conflicting_sentiment_pairs']} |" for s in summaries]
    lines += ['', '## Distributions', '', '| Split | Field | Label | Count | Proportion |',
              '| --- | --- | --- | ---: | ---: |']
    lines += [f"| {s['split']} | {s['field']} | {s['label']} | {s['count']} | {s['proportion']:.6f} |" for s in stats]
    lines += ['', '## Duplicates and label conflicts', '']
    for kind in ('exact_text', 'canonical_text'):
        subset = [g for g in groups if g['kind'] == kind]
        lines.append(f"{kind}: {len(subset)} duplicate groups, "
                     f"{sum(g['conflicting_sentiment'] for g in subset)} with conflicting sentiment labels. "
                     'See duplicate_groups.csv for all member IDs and labels, group_members.csv for assignments.')
    lines += ['', '## Official and secondary evaluation manifests', '',
        'test_manifest.csv retains all official test rows, order and annotations. Input checksums are '
        'verified before and after preparation. test_evaluation_manifest.csv also retains every test row; '
        'it marks exact/canonical training overlap, canonical group size, and the first source row as '
        'a deterministic canonical representative. unique_nonoverlap identifies representatives with '
        'no original-training overlap. The exclusive categories are overlapping_official_train, '
        'unique_nonoverlap, and repeated_test_only. These flags are text-only. “Unique” here means '
        'canonical identity, not semantic uniqueness.',
        'Use validation alone for model/hyperparameter selection. After freezing all choices, report '
        'evaluation on the full official test and optionally the predeclared secondary unique_nonoverlap '
        'slice. Neither test labels, overlap-label conflicts nor metrics may drive selection. No test '
        'examples are moved into training or validation.', '', '## Near-duplicate inspection only', '',
        'near_duplicate_review.csv uses canonical char TF-IDF 3–5 grams and exhaustive blockwise '
        'cosine >= 0.9 on combined source text, excluding equal canonical keys. It does not change '
        'groups, eligibility or assignments. Short texts below three characters have no near features. '
        'It is lexical, not semantic equivalence, with possible false positives/negatives. Features '
        'built from test text are audit-only. No manual review or automatic merging is claimed. '
        'Canonical disjointness does not establish absence of semantic leakage.', '',
        '## Reproduction and evidence', '',
        'Run commands in README.md. Outputs require a new/empty directory. metadata.json records '
        'seed, algorithm, source hashes, versions and CSV artifact hashes. Automated tests cover '
        'meaning-sensitive character preservation, quarantine, group disjointness, label conflicts, '
        'source immutability, deterministic manifests and test-label independence. No future '
        'publication or model behavior is implied.', '']
    (output / 'methodology.md').write_text('\n'.join(lines), encoding='utf-8')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    download = sub.add_parser('fetch')
    download.add_argument('--destination', type=Path, default=Path('data/raw'))
    split = sub.add_parser('split')
    split.add_argument('--training', type=Path, default=Path('data/raw/training_data.csv'))
    split.add_argument('--testing', type=Path, default=Path('data/raw/testing_data.csv'))
    split.add_argument('--output', type=Path, default=Path('reports/task02'))
    split.add_argument('--seed', type=int, choices=[SEED], default=SEED)
    split.add_argument('--validation-fraction', type=float, default=.2)
    args = parser.parse_args(argv)
    try:
        if args.command == 'fetch':
            fetch(args.destination)
            print(f'Verified pinned official files: {args.destination}')
        else:
            metadata = prepare(args.training, args.testing, args.output, args.validation_fraction, args.seed)
            print(json.dumps(metadata['counts'], indent=2))
            print(f'Methodology: {args.output}/methodology.md')
    except (OSError, ValueError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')
