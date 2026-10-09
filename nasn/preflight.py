"""Validate existing Task 02 artifacts. Never regenerate or repair splits."""
import json
import math
import platform
import unicodedata
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from arsarcasm_audit.audit import COMMIT, HASHES, LABELS, DataValidationError, digest, load_csv, sha256
from arsarcasm_audit.leakage import attach_ids, canonical_text

EXPECTED = {'train': 10037, 'validation': 2509, 'quarantine': 2, 'test': 3000}
LABEL2ID = {'NEG': 0, 'NEU': 1, 'POS': 2}


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def load_partition(raw, manifest, partition):
    """Join saved positions to raw text; no normalization of model inputs."""
    needed = ['row_id', 'source_row_id', 'group_id', 'text_sha256', 'sentiment', 'dialect', 'sarcasm', 'partition', 'source_split']
    require(set(needed) <= set(manifest), f'{partition}: missing manifest columns')
    ids = pd.to_numeric(manifest.row_id, errors='raise')
    require((ids == ids.astype(int)).all(), f'{partition}: noninteger row IDs')
    require(not ids.duplicated().any(), f'{partition}: duplicated row IDs')
    require(ids.between(0, len(raw) - 1).all(), f'{partition}: row ID outside source')
    require(manifest.partition.eq(partition).all(), f'{partition}: unexpected partition')
    require(manifest.source_split.eq('official_train' if partition != 'test' else 'official_test').all(),
            f'{partition}: incorrect source split')
    joined = raw.iloc[ids.astype(int).to_numpy()].reset_index(drop=True).copy()
    expected = attach_ids(raw, 'official_train' if partition != 'test' else 'official_test').iloc[ids.astype(int).to_numpy()]
    for column in ['source_row_id', 'group_id', 'text_sha256', 'sentiment', 'dialect', 'sarcasm']:
        require(manifest[column].tolist() == expected[column].astype(str).tolist(), f'{partition}: {column} mismatch')
    for column in ['source_row_id', 'group_id', 'text_sha256', 'row_id', 'partition']:
        joined[column] = manifest[column].to_numpy()
    return joined


def library_versions():
    values = {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
              'scikit-learn': sklearn.__version__}
    for name in ['torch', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors', 'pytest']:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def audit_near(pairs, master, threshold=.95):
    lookup = master.set_index('source_row_id')
    require(not lookup.index.duplicated().any(), 'Master manifest has duplicate source IDs')
    require(not pairs[['left_source_row_id', 'right_source_row_id']].duplicated().any(), 'Duplicate near pairs')
    records = []
    for _, pair in pairs.iterrows():
        require(pair.left_source_row_id in lookup.index and pair.right_source_row_id in lookup.index,
                'Near pair refers to unknown row')
        a, b = lookup.loc[pair.left_source_row_id], lookup.loc[pair.right_source_row_id]
        score = float(pair.cosine_similarity)
        require(math.isfinite(score) and .9 <= score <= 1.000001, 'Invalid near-pair similarity')
        for side, row in [('left', a), ('right', b)]:
            require(pair[f'{side}_partition'] == row.partition and pair[f'{side}_group_id'] == row.group_id,
                    'Near-pair split/group membership mismatch')
        require(pair.left_source_row_id != pair.right_source_row_id and a.group_id != b.group_id,
                'Near pair contains self or canonical identity')
        cross = a.partition != b.partition
        conflict = a.sentiment != b.sentiment
        active_cross = {a.partition, b.partition} <= {'train', 'validation', 'test'} and cross
        high_risk = active_cross and (score >= threshold or conflict)
        records.append({**pair.to_dict(), 'left_sentiment': a.sentiment, 'right_sentiment': b.sentiment,
                        'sentiment_conflict': conflict, 'cross_split': cross,
                        'active_cross_split': active_cross, 'high_risk_manual_review': high_risk,
                        'review_reason': ('cross_split_label_conflict' if high_risk and conflict else
                                          'cross_split_similarity_ge_0.95' if high_risk else '')})
    return pd.DataFrame(records)


def preflight(config, output):
    directory, output = Path(config['split_directory']), Path(output)
    require(Path(directory / 'metadata.json').is_file(), 'Task 02 metadata missing; no reconstruction permitted')
    require(sha256(directory / 'metadata.json') == config['task02_metadata_sha256'], 'Task 02 metadata hash changed')
    require(sha256(directory / 'methodology.md') == config['task02_methodology_sha256'], 'Task 02 methodology hash changed')
    metadata = json.loads((directory / 'metadata.json').read_text())
    require(metadata['source_commit'] == COMMIT, 'Incorrect source revision')
    require(metadata['seed'] == 42, 'Task 02 seed mismatch')
    require(config['label2id'] == LABEL2ID and set(LABEL2ID) == set(LABELS['sentiment']), 'Invalid label mapping')
    hashes = {}
    for name, expected_hash in metadata['artifact_sha256'].items():
        require((directory / name).is_file(), f'Required Task 02 artifact missing: {name}')
        hashes[name] = sha256(directory / name)
        require(hashes[name] == expected_hash, f'Task 02 checksum mismatch: {name}')
    required = ['train_manifest.csv', 'validation_manifest.csv', 'quarantine_manifest.csv', 'test_manifest.csv',
                'split_manifest.csv', 'near_duplicate_review.csv']
    require(set(required) <= hashes.keys(), 'Task 02 checksum registry is incomplete')
    raw = {}
    for source, filename, cfg in [('official_train', 'training_data.csv', 'raw_training'),
                                  ('official_test', 'testing_data.csv', 'raw_testing')]:
        require(Path(config[cfg]).is_file(), f'Missing raw source {config[cfg]}; fetch pinned data, never regenerate splits')
        require(sha256(config[cfg]) == HASHES[filename] == metadata['source_sha256'][source], 'Source checksum mismatch')
        raw[source] = load_csv(config[cfg])
    require(len(raw['official_train']) == 12548 and len(raw['official_test']) == 3000, 'Official source count mismatch')
    current = library_versions()
    for library in ['python', 'pandas', 'numpy', 'scikit-learn']:
        require(current[library] == metadata['versions'][library], f'Task 02 runtime version mismatch: {library}')
    require(unicodedata.unidata_version == metadata['versions']['unicode'], 'Unicode database version mismatch')
    parts, frames = {}, {}
    for part, count in EXPECTED.items():
        file = 'quarantine_manifest.csv' if part == 'quarantine' else f'{part}_manifest.csv'
        parts[part] = pd.read_csv(directory / file, dtype=str, keep_default_na=False)
        require(len(parts[part]) == count == metadata['counts'][part], f'{part}: expected {count} rows')
        frames[part] = load_partition(raw['official_test' if part == 'test' else 'official_train'], parts[part], part)
    master = pd.read_csv(directory / 'split_manifest.csv', dtype=str, keep_default_na=False)
    require(len(master) == 15548 and not master.source_row_id.duplicated().any(), 'Master row coverage invalid')
    joined = pd.concat(parts.values(), ignore_index=True)
    for col in master.columns:
        require(master.set_index('source_row_id')[col].sort_index().equals(
            joined.set_index('source_row_id')[col].sort_index()) if col != 'source_row_id' else
            set(master.source_row_id) == set(joined.source_row_id), f'Master/partition mismatch: {col}')
    require(parts['test'].row_id.astype(int).tolist() == list(range(3000)), 'Official test order changed')
    train_ids = pd.concat([parts[p] for p in ['train', 'validation', 'quarantine']]).row_id.astype(int).tolist()
    require(sorted(train_ids) == list(range(12548)), 'Official training row coverage invalid')
    overlap = []
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        exact = len(set(frames[a].tweet) & set(frames[b].tweet))
        canonical = len(set(frames[a].tweet.map(canonical_text)) & set(frames[b].tweet.map(canonical_text)))
        require(exact == canonical == 0, f'{a}/{b}: exact or canonical overlap')
        overlap.append({'scope': f'{a}_{b}', 'exact_groups': exact, 'canonical_groups': canonical})
    test_keys = set(frames['test'].tweet.map(canonical_text))
    all_train = attach_ids(raw['official_train'], 'official_train')
    actual_quarantine = set(all_train.loc[all_train.canonical.isin(test_keys), 'source_row_id'])
    require(actual_quarantine == set(parts['quarantine'].source_row_id), 'Quarantine is not the complete test-overlap set')
    pairs = pd.read_csv(directory / 'near_duplicate_review.csv', dtype=str)
    require(len(pairs) == 753, 'Expected 753 near pairs')
    reviewed = audit_near(pairs, master, config['high_risk_similarity'])
    grouped = reviewed.groupby(['left_partition', 'right_partition'], sort=True).agg(
        pairs=('cosine_similarity', 'size'), label_conflicts=('sentiment_conflict', 'sum'),
        high_risk=('high_risk_manual_review', 'sum')).reset_index()
    reviewed['similarity_bin'] = pd.cut(reviewed.cosine_similarity.astype(float), [.9, .95, .99, 1.000001],
                                        right=False, labels=['0.90-0.95', '0.95-0.99', '0.99-1.00'])
    summary = {'status': 'passed', 'source_commit': COMMIT, 'counts': EXPECTED, 'overlap': overlap,
               'manifest_sha256': hashes, 'source_sha256': metadata['source_sha256'],
               'near_pairs': len(reviewed), 'near_cross_split': int(reviewed.active_cross_split.sum()),
               'near_label_conflicts': int(reviewed.sentiment_conflict.sum()),
               'high_risk_manual_review_pairs': int(reviewed.high_risk_manual_review.sum()),
               'high_risk_rule': 'active cross-split AND (score >= 0.95 OR unequal sentiment labels)',
               'similarity_bins': reviewed.similarity_bin.value_counts(sort=False).to_dict(),
               'versions': current, 'test_evaluated': False, 'splits_reconstructed': False,
               'label2id': LABEL2ID, 'label_mapping_origin': 'Dataset has string labels only; v1.0 explicitly assigns NEG=0, NEU=1, POS=2'}
    output.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(output / 'near_duplicate_audit.csv', index=False)
    grouped.to_csv(output / 'near_duplicate_summary.csv', index=False)
    reviewed[reviewed.high_risk_manual_review].to_csv(output / 'manual_review_queue.csv', index=False)
    (output / 'preflight.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Official test and quarantine frames never reach the training/evaluation interface.
    return frames['train'], frames['validation'], summary
