"""Pinned source, strict CSV loading, baseline splitting and audit-only similarity."""
import csv
import hashlib
import io
import re
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

COMMIT = 'eb85513e133cfc67e30aca4968fadbe2594fcb42'
REPOSITORY = 'https://github.com/iabufarha/ArSarcasm-v2'
HASHES = {
    'training_data.csv': '1da727eb763459f436c4e61f52c6a0bc63d5300af8fcc99b5675ae4616dae04a',
    'testing_data.csv': '0f0a77ba8a4a0a9370846a42a93a63244092f1fb30a909595b70e17e299bcf6c',
}
LABELS = {'sentiment': ('NEG', 'NEU', 'POS'), 'sarcasm': ('FALSE', 'TRUE'),
          'dialect': ('msa', 'egypt', 'gulf', 'levant', 'magreb')}
REQUIRED = ('tweet', 'sarcasm', 'sentiment', 'dialect')

class DataValidationError(ValueError):
    """Dataset or configuration violates preparation requirements."""

def digest(content):
    return hashlib.sha256(content).hexdigest()

def sha256(path):
    return digest(Path(path).read_bytes())

def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = destination / name
        if path.exists():
            if sha256(path) != expected:
                raise DataValidationError(f'Refusing to overwrite {path}: checksum mismatch')
            continue
        url = f'https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/{COMMIT}/ArSarcasm-v2/{name}'
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if digest(content) != expected:
            raise DataValidationError(f'{name}: download checksum mismatch')
        path.write_bytes(content)

def load_csv(path):
    try:
        content = Path(path).read_text(encoding='utf-8-sig')
        records = csv.reader(io.StringIO(content), strict=True)
        header = next(records)
        if len(header) != len(set(header)):
            raise DataValidationError(f'{path}: duplicate column names')
        if set(REQUIRED) - set(header):
            raise DataValidationError(f'{path}: missing columns {sorted(set(REQUIRED) - set(header))}')
        for row_id, row in enumerate(records):
            if len(row) != len(header):
                raise DataValidationError(f'{path}: row {row_id}: expected {len(header)} fields, got {len(row)}')
        frame = pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False)
        if frame.empty:
            raise DataValidationError(f'{path}: empty dataset')
        for column in frame:
            invalid = frame[column].str.strip().eq('')
            if invalid.any():
                raise DataValidationError(f'{path}: missing {column} at row IDs {frame.index[invalid].tolist()[:10]}')
        for column, allowed in LABELS.items():
            invalid = ~frame[column].isin(allowed)
            if invalid.any():
                raise DataValidationError(f'{path}: invalid {column} labels at row IDs {frame.index[invalid].tolist()[:10]}')
        return frame
    except (OSError, UnicodeError, csv.Error, StopIteration, pd.errors.ParserError) as exc:
        raise DataValidationError(f'Cannot load {path}: {exc}') from exc

def baseline_partitions(training, fraction=.2, seed=42):
    """Reconstruct Task 01's documented row-wise sentiment-stratified split."""
    try:
        _, val = train_test_split(training.index.to_numpy(), test_size=fraction,
                                  random_state=seed, stratify=training.sentiment)
    except ValueError as exc:
        raise DataValidationError(f'Cannot reconstruct baseline split: {exc}') from exc
    val_ids = set(val)
    return pd.Series(['validation' if i in val_ids else 'train' for i in training.index], index=training.index)

def near_pairs(texts, threshold=.9, block_size=128):
    """Audit-only char TF-IDF similarities; never used to assign groups/splits."""
    try:
        matrix = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=False).fit_transform(texts)
    except ValueError as exc:
        if 'empty vocabulary' in str(exc):
            return
        raise
    for start in range(0, len(texts), block_size):
        scores = (matrix[start:start + block_size] @ matrix.T).tocsr()
        scores.data[scores.data < threshold] = 0
        scores.eliminate_zeros()
        scores = scores.tocoo()
        for local, right, score in zip(scores.row, scores.col, scores.data):
            left, right = start + int(local), int(right)
            if left < right and texts[left] != texts[right]:
                yield left, right, float(score)
