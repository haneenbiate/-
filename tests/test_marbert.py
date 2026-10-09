"""Offline software tests. Tiny random BERT is not a MARBERT experiment."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from arsarcasm_audit.audit import DataValidationError, sha256
from nasn.preflight import LABEL2ID, audit_near, load_partition, preflight
from nasn.metrics import calculate, prediction_frame, subgroup_metrics
from nasn.marbert import (check_config, encoded_dataset, fit, train_epoch)
from arsarcasm_audit.leakage import attach_ids

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def config():
    conf = json.loads((ROOT / 'configs/marbert_v1.json').read_text())
    for field in ['raw_training', 'raw_testing', 'split_directory']:
        conf[field] = str(ROOT / conf[field])
    return conf

@pytest.fixture
def tiny(tmp_path):
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'نص', 'جيد', 'سيئ', 'لا']
    file = tmp_path / 'vocab.txt'
    file.write_text('\n'.join(vocab) + '\n')
    tokenizer = BertTokenizerFast(vocab_file=str(file), do_lower_case=False)
    model = BertForSequenceClassification(BertConfig(vocab_size=len(vocab), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=3,
        label2id=LABEL2ID, id2label={i: label for label, i in LABEL2ID.items()},
        hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    return tokenizer, model


def example_frame(partition='train'):
    frame = pd.DataFrame({'tweet': ['نص جيد', 'نص سيئ', 'لا جيد', 'نص', 'جيد'],
        'sentiment': ['POS', 'NEG', 'NEU', 'NEU', 'POS'], 'sarcasm': ['FALSE'] * 5, 'dialect': ['msa'] * 5})
    frame = attach_ids(frame, 'official_train')
    frame['partition'] = partition
    return frame


def test_actual_preflight(config, tmp_path):
    train, val, summary = preflight(config, tmp_path)
    assert (len(train), len(val)) == (10037, 2509)
    assert summary['status'] == 'passed'
    assert summary['near_pairs'] == 753
    assert summary['splits_reconstructed'] is False
    assert summary['test_evaluated'] is False
    assert train.tweet.iloc[0]  # raw text loaded, no replacement with canonical key


def test_preflight_missing_and_tampered(config, tmp_path):
    config['split_directory'] = str(tmp_path / 'absent')
    with pytest.raises(DataValidationError, match='missing'):
        preflight(config, tmp_path / 'output')
    (tmp_path / 'absent').mkdir()
    (tmp_path / 'absent/metadata.json').write_text('{}')
    with pytest.raises(DataValidationError, match='hash changed'):
        preflight(config, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_manifest_loader_checks_original_labels():
    raw = example_frame()[['tweet','sentiment','sarcasm','dialect']]
    manifest = attach_ids(raw, 'official_train').astype(str)
    manifest['partition'] = 'train'
    loaded = load_partition(raw, manifest, 'train')
    assert loaded.tweet.tolist() == raw.tweet.tolist()
    manifest.loc[0,'sentiment'] = 'NEG'
    with pytest.raises(DataValidationError, match='sentiment mismatch'):
        load_partition(raw, manifest, 'train')


def test_mapping_and_known_metrics():
    assert LABEL2ID == {'NEG': 0, 'NEU': 1, 'POS': 2}
    metrics, classes, matrix = calculate([0,1,2], [0,0,2])
    assert metrics['accuracy'] == pytest.approx(2/3)
    assert metrics['macro_f1'] == pytest.approx(5/9)
    assert metrics['weighted_f1'] == pytest.approx(5/9)
    assert matrix.tolist() == [[1,0,0],[1,0,0],[0,0,1]]
    assert [item['label'] for item in classes] == ['NEG','NEU','POS']
    assert [item['support'] for item in classes] == [1,1,1]
    with pytest.raises(DataValidationError): calculate([3], [0])
    with pytest.raises(DataValidationError): calculate([], [])


def test_prediction_confidences_and_subgroup_limits():
    frame = example_frame('validation')
    probs = np.tile([.1,.2,.7], (len(frame),1))
    result = prediction_frame(frame, probs)
    assert result.confidence.tolist() == [.7] * 5
    assert result.predicted_label.tolist() == ['POS'] * 5
    assert set(result.gold_id) == {0,1,2}
    assert all(g['status'] == 'insufficient_sample_size' for g in subgroup_metrics(result, minimum=30))
    assert all(g['status'] == 'computed' for g in subgroup_metrics(result, minimum=3))
    with pytest.raises(DataValidationError): prediction_frame(frame, probs * 2)


def test_near_audit_flags_not_filters():
    master = pd.DataFrame({'source_row_id':['a','b'], 'group_id':['g1','g2'],
                          'partition':['train','validation'], 'sentiment':['NEG','POS']})
    pairs = pd.DataFrame([{'left_source_row_id':'a','right_source_row_id':'b', 'left_group_id':'g1',
        'right_group_id':'g2','left_partition':'train','right_partition':'validation','cosine_similarity':.91}])
    result = audit_near(pairs, master)
    assert len(result) == 1 and bool(result.iloc[0].high_risk_manual_review)
    assert bool(result.iloc[0].sentiment_conflict)
    assert result.iloc[0].review_reason == 'cross_split_label_conflict'
    pairs.loc[0,'left_partition'] = 'test'
    with pytest.raises(DataValidationError, match='membership mismatch'):
        audit_near(pairs, master)


def test_raw_text_and_test_exclusion(tiny):
    tokenizer, _ = tiny
    frame = example_frame('test')
    with pytest.raises(DataValidationError, match='cannot enter'):
        encoded_dataset(frame, tokenizer, 128)
    frame.partition = 'train'
    class RecordingTokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == frame.tweet.tolist()
            return tokenizer(texts, **kwargs)
    encoded = encoded_dataset(frame, RecordingTokenizer(), 128)
    assert len(encoded) == len(frame)
    assert [int(encoded[i]['labels']) for i in range(len(encoded))] == [2,0,1,1,2]


def test_model_shape_and_local_checkpoint_reload(tiny, tmp_path):
    tokenizer, model = tiny
    model.eval()
    tokens = tokenizer(['نص جيد','لا جيد'], padding=True, return_tensors='pt')
    with torch.inference_mode():
        expected = model(**tokens).logits
    assert expected.shape == (2,3)
    model.save_pretrained(tmp_path / 'checkpoint', safe_serialization=True)
    tokenizer.save_pretrained(tmp_path / 'checkpoint')
    loaded = BertForSequenceClassification.from_pretrained(tmp_path / 'checkpoint', local_files_only=True)
    tok = BertTokenizerFast.from_pretrained(tmp_path / 'checkpoint', local_files_only=True)
    loaded.eval()
    with torch.inference_mode():
        actual = loaded(**tok(['نص جيد','لا جيد'], padding=True, return_tensors='pt')).logits
    assert torch.allclose(expected, actual, atol=1e-6)
    assert loaded.config.label2id == LABEL2ID


def test_gradient_accumulation_matches_full_batch_including_remainder(tiny):
    import copy
    tokenizer, first = tiny
    second = copy.deepcopy(first)
    dataset = encoded_dataset(example_frame(), tokenizer, 16)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    opt1 = torch.optim.SGD(first.parameters(), lr=.001)
    opt2 = torch.optim.SGD(second.parameters(), lr=.001)
    loss1 = train_epoch(first, loader, opt1, 'cpu', 4, 1.0)
    loss2 = train_epoch(second, loader, opt2, 'cpu', 2, 1.0)
    assert loss1 == pytest.approx(loss2, abs=1e-5)
    for a,b in zip(first.parameters(),second.parameters()):
        assert torch.allclose(a,b,atol=1e-6)


def test_offline_tiny_end_to_end_pipeline(tiny, config, tmp_path, monkeypatch):
    import transformers
    import nasn.marbert as runner
    tokenizer, model = tiny
    fixture = tmp_path / 'fixture'
    model.save_pretrained(fixture)
    tokenizer.save_pretrained(fixture)
    original_model = transformers.AutoModelForSequenceClassification.from_pretrained
    original_tok = transformers.AutoTokenizer.from_pretrained
    def model_loader(path, **kwargs):
        if path == 'UBC-NLP/MARBERT':
            return original_model(fixture, local_files_only=True,
                                  output_loading_info=kwargs.get('output_loading_info', False))
        return original_model(path, **kwargs)
    def token_loader(path, **kwargs):
        if path == 'UBC-NLP/MARBERT':
            return original_tok(fixture, local_files_only=True)
        return original_tok(path, **kwargs)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, 'from_pretrained', model_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', token_loader)
    monkeypatch.setattr(runner, 'resolve_revision', lambda cfg: 'a' * 40)
    output = tmp_path / 'toy-run'
    output.mkdir()
    status = fit(example_frame('train'), example_frame('validation'), config, output)
    assert status['checkpoint_reload_verified']
    assert not status['test_evaluated']
    assert (output / 'best_checkpoint/model.safetensors').is_file()
    predictions = pd.read_csv(output / 'validation_predictions.csv')
    assert len(predictions) == 5
    history = pd.read_csv(output / 'epoch_history.csv')
    best_epoch = int(history.loc[history.macro_f1.idxmax(), 'epoch'])
    assert status['best_epoch'] == best_epoch
    assert pd.read_csv(output / 'confusion_matrix.csv', index_col=0).to_numpy().sum() == 5
    for name, expected in json.loads((output / 'checkpoint_hashes.json').read_text()).items():
        assert sha256(output / 'best_checkpoint' / name) == expected


def test_config_changes_fail(config):
    check_config(config)
    config['learning_rate'] = 1e-3
    with pytest.raises(DataValidationError, match='parameter changed'):
        check_config(config)

def test_diagnostic_queries_are_redacted():
    from nasn.marbert import safe_error
    error = Exception("host='example.com' failed GET /weights?Signature=secret&Policy=value "
                      "https://example.com/file?token=secret (403 Forbidden)")
    clean = safe_error(error)
    assert '403 Forbidden' in clean and 'example.com' in clean
    assert 'secret' not in clean and 'Signature=' not in clean and 'Policy=' not in clean
