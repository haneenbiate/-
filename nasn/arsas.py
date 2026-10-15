"""ArSAS (Elmadany, Mubarak and Magdy, 2018) as the Study v3 second dataset.

Pinned bytes, exclusion of tweets shared with ArSarcasm-v2, and a deterministic
group-level stratified train/validation/test split. Manifests hold tweet IDs,
hashes and labels only; tweet text stays in the ignored data/ directory.
"""
import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from arsarcasm_audit.audit import load_csv, sha256
from arsarcasm_audit.leakage import canonical_text, group_id
from .preflight import require

URL = 'https://homepages.inf.ed.ac.uk/wmagdy/Resources/ArSAS.zip'
ZIP_SHA256 = '1d517bfddf2fb766ab6ca2bc98a816f1339e68d7bb04d281959358d8e015d8f1'
MEMBER = 'ArSAS..txt'
MEMBER_SHA256 = 'e1f6081f2a4dafab422834efc2ed97c17840154d700aa5b8309a623e900a0567'
LABELS = {'Negative': 'NEG', 'Neutral': 'NEU', 'Positive': 'POS'}
TEST_FRACTION, VALIDATION_FRACTION, SEED = .2, .1, 42
MANIFEST = Path(__file__).resolve().parents[1] / 'reports/arsas/manifest.csv'


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read()
    require(hashlib.sha256(content).hexdigest() == ZIP_SHA256, 'ArSAS archive checksum changed')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        (destination / MEMBER).write_bytes(archive.read(MEMBER))
    require(sha256(destination / MEMBER) == MEMBER_SHA256, 'ArSAS file checksum changed')
    return destination / MEMBER


def read(path):
    require(sha256(path) == MEMBER_SHA256, f'ArSAS file checksum mismatch: {path}')
    frame = pd.read_csv(path, sep='\t', encoding='utf-8-sig', dtype=str, keep_default_na=False, quoting=3)
    frame.columns = [c.lstrip('#').strip() for c in frame.columns]
    frame = frame.rename(columns={'Tweet_text': 'tweet'})
    frame['Speech_act_label_confidence'] = frame.Speech_act_label_confidence.str.strip()
    require(len(frame) == 21064, 'Unexpected ArSAS rows')
    return frame


def linked_groups(frame):
    """Union rows sharing canonical text or tweet ID (ArSAS repeats 360 IDs, some re-labelled)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for text_group, tweet_id in zip(frame.text_group, frame.Tweet_ID):
        parent[find('t:' + text_group)] = find('i:' + tweet_id)
    roots = [find('t:' + g) for g in frame.text_group]
    return ['as1_' + hashlib.sha256(r.encode()).hexdigest() for r in roots]


def overlap_key(text):
    """Stricter than Task 02 canonical text: also ignores URLs, mentions and hashtags."""
    return re.sub(r'\s+', ' ', re.sub(r'https?://\S+|[@#]\S+', '', canonical_text(text))).strip()


def build_manifest(arsas_path, arsarcasm_directory):
    frame = read(arsas_path)
    frame['source_row_id'] = [f'arsas:{i:05d}' for i in range(len(frame))]
    frame['text_sha256'] = frame.tweet.map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    frame['text_group'] = frame.tweet.map(group_id)
    frame['group_id'] = linked_groups(frame)
    frame['sentiment'] = frame.Sentiment_label.map(LABELS).fillna('')
    known = set()
    for name in ['training_data.csv', 'testing_data.csv']:
        known |= set(load_csv(Path(arsarcasm_directory) / name).tweet.map(overlap_key))
    keys = frame.tweet.map(overlap_key)
    conflicts = frame[frame.sentiment != ''].groupby('group_id').sentiment.nunique()
    frame['exclusion'] = ''
    frame.loc[frame.group_id.isin(conflicts[conflicts > 1].index), 'exclusion'] = 'label_conflict_within_group'
    frame.loc[keys.str.len().ge(10) & keys.isin(known), 'exclusion'] = 'shared_with_arsarcasm_v2'
    frame.loc[frame.Sentiment_label.eq('Mixed'), 'exclusion'] = 'mixed_sentiment'
    # Exclusion is group-wide: one excluded member removes every copy of that text.
    excluded = frame[frame.exclusion != ''].groupby('group_id').exclusion.first()
    frame.loc[frame.group_id.isin(excluded.index) & frame.exclusion.eq(''), 'exclusion'] = 'duplicate_of_excluded'
    eligible = frame[frame.exclusion == ''].drop_duplicates('group_id')[['group_id', 'sentiment']].copy()
    eligible['_rank'] = eligible.group_id.map(lambda g: hashlib.sha256(f'{SEED}|arsas|{g}'.encode()).hexdigest())
    assignment = {}
    for _, part in eligible.sort_values('_rank').groupby('sentiment'):
        n_test, n_validation = round(TEST_FRACTION * len(part)), round(VALIDATION_FRACTION * len(part))
        for i, g in enumerate(part.group_id):
            assignment[g] = 'test' if i < n_test else 'validation' if i < n_test + n_validation else 'train'
    frame['partition'] = frame.group_id.map(assignment).where(frame.exclusion.eq(''), 'excluded')
    frame['row_id'] = range(len(frame))
    return frame[['row_id', 'source_row_id', 'Tweet_ID', 'group_id', 'text_sha256', 'sentiment', 'Sentiment_label',
                  'Sentiment_label_confidence', 'Topic', 'partition', 'exclusion']]


def load_partition(arsas_path, partition, manifest_path=MANIFEST):
    """Rows of one registered partition joined to the pinned raw text."""
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    frame = read(arsas_path)
    require(manifest.Tweet_ID.tolist() == frame.Tweet_ID.tolist() and len(manifest) == len(frame),
            'ArSAS manifest rows differ from source')
    require(manifest.text_sha256.tolist() == frame.tweet.map(lambda t: hashlib.sha256(t.encode()).hexdigest()).tolist(),
            'ArSAS text hashes differ from manifest')
    rows = manifest.partition.eq(partition).to_numpy()
    require(rows.any(), f'Unknown ArSAS partition: {partition}')
    result = manifest[rows].reset_index(drop=True)
    result['tweet'] = frame.tweet[rows].to_numpy()
    return result.assign(sarcasm='NA', dialect='unknown')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fetch'); p.add_argument('--destination', default='data/arsas')
    p = sub.add_parser('split'); p.add_argument('--arsas', default=f'data/arsas/{MEMBER}')
    p.add_argument('--arsarcasm-directory', default='data/raw'); p.add_argument('--output', default=str(MANIFEST.parent))
    args = parser.parse_args(argv)
    if args.command == 'fetch':
        print(fetch(args.destination))
        return
    output = Path(args.output)
    require(not (output / 'manifest.csv').exists(), f'Manifest exists; choose a new directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.arsas, args.arsarcasm_directory)
    manifest.to_csv(output / 'manifest.csv', index=False, lineterminator='\n')
    counts = manifest.groupby(['partition', 'sentiment']).size().unstack(fill_value=0)
    metadata = {'source': URL, 'zip_sha256': ZIP_SHA256, 'file_sha256': MEMBER_SHA256, 'seed': SEED,
                'fractions': {'test': TEST_FRACTION, 'validation': VALIDATION_FRACTION},
                'label_map': LABELS, 'exclusions': manifest.exclusion.value_counts().to_dict(),
                'counts': {p: {k: int(v) for k, v in row.items()} for p, row in counts.iterrows()},
                'manifest_sha256': sha256(output / 'manifest.csv'),
                'method': 'Group = rows linked by Task 02 canonical text or a repeated tweet ID. Mixed, within-group label conflicts and any text whose '
                          'URL/mention/hashtag-free canonical form (>=10 chars) occurs in ArSarcasm-v2 train or test '
                          'are excluded group-wide. Remaining groups are ranked by SHA-256(seed|arsas|group) within '
                          'each label: first 20% test, next 10% validation, rest train.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
