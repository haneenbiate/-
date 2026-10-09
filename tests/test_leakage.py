import io
import json

import pandas as pd
import pytest

from arsarcasm_audit.audit import DataValidationError, fetch, load_csv, sha256
from arsarcasm_audit.leakage import (attach_ids, canonical_text, group_id, group_split,
                                   overlap_audit, prepare)

@pytest.fixture
def inputs(tmp_path):
    records = [dict(tweet=f'نص عربي {label} برقم {i}', sentiment=label,
                    sarcasm='FALSE', dialect='msa')
               for label in ('NEG', 'NEU', 'POS') for i in range(20)]
    # Duplicate, conflict within training, and a canonical-only duplicate.
    records += [dict(records[0]), dict(records[0], sentiment='POS'),
                dict(records[5], tweet='  ' + records[5]['tweet'].replace('عربي', 'عـربي') + '  ')]
    training = pd.DataFrame(records)
    testing = pd.DataFrame([dict(records[2]), dict(records[3], tweet=records[3]['tweet'].replace('عربي', 'عـربي')),
                           dict(tweet='نص اختبار فريد', sentiment='NEU', sarcasm='TRUE', dialect='gulf'),
                           dict(tweet='نص اختبار فريد', sentiment='NEG', sarcasm='FALSE', dialect='msa')])
    train, test = tmp_path / 'train.csv', tmp_path / 'test.csv'
    training.to_csv(train, index=False)
    testing.to_csv(test, index=False)
    return train, test

@pytest.mark.parametrize('left,right', [('نـص عربي', 'نص عربي'), (' نص\n عربي\t', 'نص عربي'),
    ('ا\u0654حب', 'أحب'), ('نص\u00a0عربي', 'نص عربي'), ('اـ\u0654حب', 'أحب')])
def test_canonical_equivalence(left, right):
    assert canonical_text(left) == canonical_text(right)
    assert group_id(left) == group_id(right)
    assert canonical_text(canonical_text(left)) == canonical_text(left)

@pytest.mark.parametrize('left,right', [('احب', 'أحب'), ('إحب', 'أحب'), ('لا أحب', 'أحب'),
    ('ليس جيد', 'جيد'), ('سيئ', 'جيد'), ('فتى', 'فتي'), ('مدرسة', 'مدرسه'),
    ('علم', 'عَلَم'), ('نص!', 'نص'), ('ABC', 'abc'), ('ﻻ', 'لا')])
def test_meaning_sensitive_distinctions(left, right):
    assert canonical_text(left) != canonical_text(right)
    assert group_id(left) != group_id(right)

def test_source_and_test_immutable_and_coverage(inputs, tmp_path):
    train, test = inputs
    before = (train.read_bytes(), test.read_bytes())
    out = tmp_path / 'out'
    metadata = prepare(train, test, out, require_official=False)
    assert before == (train.read_bytes(), test.read_bytes())
    manifest = pd.read_csv(out / 'split_manifest.csv', dtype={'sarcasm': str})
    source = manifest[manifest.source_split == 'official_train']
    official_test = pd.read_csv(out / 'test_manifest.csv')
    assert sorted(source.row_id) == list(range(len(load_csv(train))))
    assert official_test.row_id.tolist() == list(range(len(load_csv(test))))
    assert set(official_test.partition) == {'test'}
    assert not manifest.source_row_id.duplicated().any()
    active = manifest[manifest.partition.isin(['train', 'validation'])]
    assert not set(active.group_id) & set(official_test.group_id)
    fitting = active[active.partition == 'train']
    validation = active[active.partition == 'validation']
    assert not set(fitting.group_id) & set(validation.group_id)
    assert set(fitting.sentiment) == set(validation.sentiment) == {'NEG', 'NEU', 'POS'}
    for part in (fitting, validation):
        assert not part.group_id.isin(official_test.group_id).any()
    assert metadata['source_sha256'] == {'official_train': sha256(train), 'official_test': sha256(test)}
    assert metadata['seed'] == 42
    raw_sources = {'official_train': load_csv(train), 'official_test': load_csv(test)}
    for _, row in manifest.iterrows():
        original = raw_sources[row.source_split].iloc[row.row_id]
        assert (row.sentiment, row.sarcasm, row.dialect) == (
            original.sentiment, original.sarcasm, original.dialect)
    quarantine = pd.read_csv(out / 'quarantine_manifest.csv')
    assert set(quarantine.row_id) == {2, 3}
    assert set(quarantine.quarantine_reason) == {'exact_test_text_overlap', 'canonical_test_text_overlap'}
    raw_train = attach_ids(load_csv(train), 'official_train')
    raw_test = attach_ids(load_csv(test), 'official_test')
    assert set(quarantine.row_id) == set(raw_train.loc[raw_train.group_id.isin(raw_test.group_id), 'row_id'])

def test_test_evaluation_categories(inputs, tmp_path):
    prepare(*inputs, tmp_path / 'out', require_official=False)
    evaluation = pd.read_csv(tmp_path / 'out/test_evaluation_manifest.csv')
    assert evaluation.evaluation_category.tolist() == [
        'overlapping_official_train', 'overlapping_official_train', 'unique_nonoverlap', 'repeated_test_only']
    assert evaluation.unique_nonoverlap.tolist() == [False, False, True, False]
    assert evaluation.test_group_size.tolist() == [1, 1, 2, 2]
    assert evaluation.overlaps_official_train_exact.tolist() == [True, False, False, False]
    assert evaluation.canonical_representative.tolist() == [True, True, True, False]

def test_duplicate_conflicts_retained(inputs, tmp_path):
    prepare(*inputs, tmp_path / 'out', require_official=False)
    groups = pd.read_csv(tmp_path / 'out/duplicate_groups.csv')
    assert groups.conflicting_sentiment.any()
    manifest = pd.read_csv(tmp_path / 'out/split_manifest.csv')
    duplicates = manifest[(manifest.source_split == 'official_train') & manifest.row_id.isin([0, 60, 61])]
    assert len(duplicates) == 3
    assert duplicates.partition.nunique() == 1
    assert set(duplicates.sentiment) == {'NEG', 'POS'}

def test_overlap_pair_conflicts_are_cross_partition():
    frame = pd.DataFrame({'tweet': ['نص'] * 4, 'sentiment': ['NEG', 'POS', 'NEG', 'POS'],
                          'partition': ['train', 'train', 'validation', 'test']})
    frame = attach_ids(frame, 'official_train')
    summary, members = overlap_audit(frame, 'baseline')
    found = {s['scope']: s for s in summary if s['kind'] == 'exact_text'}
    assert found['train_validation']['row_pairs'] == 2
    assert found['train_validation']['conflicting_sentiment_pairs'] == 1
    assert found['train_test']['conflicting_sentiment_pairs'] == 1
    assert found['validation_test']['conflicting_sentiment_pairs'] == 1
    assert members

def test_test_labels_cannot_change_active_assignments(inputs, tmp_path):
    train, test = inputs
    prepare(train, test, tmp_path / 'first', require_official=False)
    labels = load_csv(test)
    labels['sentiment'] = ['POS', 'NEG', 'NEG', 'POS']
    labels.to_csv(test, index=False)
    prepare(train, test, tmp_path / 'second', require_official=False)
    for name in ('train_manifest.csv', 'validation_manifest.csv', 'quarantine_manifest.csv', 'training_deviations.csv'):
        assert (tmp_path / 'first' / name).read_bytes() == (tmp_path / 'second' / name).read_bytes()
    one, two = [json.loads((tmp_path / name / 'metadata.json').read_text()) for name in ('first', 'second')]
    assert one['split_selection'] == two['split_selection']

def test_reproducible_artifacts_and_no_overwrite(inputs, tmp_path):
    for name in ('a', 'b'):
        prepare(*inputs, tmp_path / name, require_official=False)
    for file in (tmp_path / 'a').iterdir():
        assert file.read_bytes() == (tmp_path / 'b' / file.name).read_bytes()
    with pytest.raises(DataValidationError, match='empty/new'):
        prepare(*inputs, tmp_path / 'a', require_official=False)

def test_near_duplicates_not_merged_or_quarantined(tmp_path):
    from arsarcasm_audit.audit import near_pairs
    base = 'هذا نص عربي طويل جدا للتجربة واختبار التشابه في التغريدات'
    texts = [base, base + '!']
    assert list(near_pairs(texts, .8))
    assert group_id(texts[0]) != group_id(texts[1])
    rows = [dict(tweet=text, sentiment='NEG', sarcasm='FALSE', dialect='msa') for text in texts]
    frame = attach_ids(pd.DataFrame(rows), 'official_train')
    assignment, _ = group_split(frame)
    assert set(assignment) == {'train', 'validation'}

def test_test_near_match_remains_eligible(inputs, tmp_path):
    train, test = inputs
    base = 'هذا نص عربي طويل جدا للتجربة واختبار التشابه في التغريدات'
    training, testing = load_csv(train), load_csv(test)
    training.loc[0, 'tweet'] = base
    testing.loc[2, 'tweet'] = base + '!'
    training.to_csv(train, index=False)
    testing.to_csv(test, index=False)
    out = tmp_path / 'out'
    prepare(train, test, out, require_official=False)
    master = pd.read_csv(out / 'split_manifest.csv')
    target = master[master.source_row_id == 'official_train:00000'].iloc[0]
    assert target.partition in {'train', 'validation'}
    pairs = pd.read_csv(out / 'near_duplicate_review.csv')
    assert ((pairs.left_source_row_id == 'official_train:00000') &
            (pairs.right_source_row_id == 'official_test:00002')).any()

def test_raw_canonical_disjointness_recomputed_from_sources(inputs, tmp_path):
    train, test = inputs
    out = tmp_path / 'out'
    prepare(train, test, out, require_official=False)
    source = load_csv(train)
    train_rows = pd.read_csv(out / 'train_manifest.csv').row_id
    val_rows = pd.read_csv(out / 'validation_manifest.csv').row_id
    fit_texts = {canonical_text(source.loc[i, 'tweet']) for i in train_rows}
    val_texts = {canonical_text(source.loc[i, 'tweet']) for i in val_rows}
    test_texts = set(load_csv(test).tweet.map(canonical_text))
    assert not fit_texts & val_texts
    assert not fit_texts & test_texts
    assert not val_texts & test_texts

@pytest.mark.parametrize('column,value', [('sentiment', 'negative'), ('sarcasm', 'false'),
    ('dialect', 'unknown'), ('tweet', ''), ('tweet', ' ')])
def test_invalid_data(inputs, column, value):
    train, _ = inputs
    frame = pd.read_csv(train, keep_default_na=False, dtype=str)
    frame.loc[0, column] = value
    frame.to_csv(train, index=False)
    with pytest.raises(DataValidationError): load_csv(train)

@pytest.mark.parametrize('content', ['tweet,sentiment\nنص,NEG\n',
    'tweet,tweet,sarcasm,sentiment,dialect\n', 'tweet,sarcasm,sentiment,dialect\n',
    'tweet,sarcasm,sentiment,dialect\nنص,FALSE,NEG\n',
    'tweet,sarcasm,sentiment,dialect\n"unfinished'])
def test_bad_schema(tmp_path, content):
    path = tmp_path / 'bad.csv'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(DataValidationError): load_csv(path)

def test_checksum_enforced_and_no_outputs(inputs, tmp_path):
    with pytest.raises(DataValidationError, match='SHA-256'):
        prepare(*inputs, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()

def test_impossible_split_and_empty_canonical():
    frame = pd.DataFrame([dict(tweet='واحد', sentiment='NEG', sarcasm='FALSE', dialect='msa')])
    with pytest.raises(DataValidationError, match='two eligible'):
        group_split(attach_ids(frame, 'official_train'))
    frame.loc[0, 'tweet'] = 'ـــ'
    with pytest.raises(DataValidationError, match='canonical text is empty'):
        attach_ids(frame, 'official_train')

@pytest.mark.parametrize('fraction', [0, 1, float('nan')])
def test_invalid_fraction(inputs, fraction):
    frame = attach_ids(load_csv(inputs[0]), 'official_train')
    with pytest.raises(DataValidationError): group_split(frame, fraction)

def test_download_integrity(tmp_path, monkeypatch):
    import arsarcasm_audit.audit as module
    payload = b'fixture'
    monkeypatch.setattr(module, 'HASHES', {'training_data.csv': module.digest(payload)})
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: io.BytesIO(payload))
    fetch(tmp_path)
    fetch(tmp_path)
    (tmp_path / 'training_data.csv').write_bytes(b'wrong')
    with pytest.raises(DataValidationError, match='overwrite'): fetch(tmp_path)
    (tmp_path / 'training_data.csv').unlink()
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: io.BytesIO(b'corrupt'))
    with pytest.raises(DataValidationError, match='checksum'): fetch(tmp_path)
    assert not (tmp_path / 'training_data.csv').exists()
