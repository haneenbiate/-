"""Validation-only metrics with explicit fixed label support and subgroup limits."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from .preflight import LABEL2ID, require


def calculate(gold, predicted):
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    require(len(gold) > 0 and gold.shape == predicted.shape and gold.ndim == 1, 'Metric input shape mismatch')
    require(np.isin(gold, [0, 1, 2]).all() and np.isin(predicted, [0, 1, 2]).all(), 'Metrics contain invalid labels')
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=[0, 1, 2], zero_division=0)
    overall = {'accuracy': float(accuracy_score(gold, predicted)),
               'macro_f1': float(f1_score(gold, predicted, labels=[0, 1, 2], average='macro', zero_division=0)),
               'weighted_f1': float(f1_score(gold, predicted, labels=[0, 1, 2], average='weighted', zero_division=0)),
               'n': len(gold)}
    classes = [{'label': label, 'label_id': i, 'precision': float(precision[i]), 'recall': float(recall[i]),
                'f1': float(f1[i]), 'support': int(support[i])} for label, i in LABEL2ID.items()]
    return overall, classes, confusion_matrix(gold, predicted, labels=[0, 1, 2])


def prediction_frame(validation, probabilities):
    probabilities = np.asarray(probabilities)
    require(probabilities.shape == (len(validation), 3), 'Expected N x 3 probabilities')
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all(),
            'Invalid model probabilities')
    require(np.allclose(probabilities.sum(axis=1), 1, atol=1e-5), 'Model probabilities must sum to one')
    output = validation[['source_row_id', 'row_id', 'group_id', 'sentiment', 'sarcasm', 'dialect']].copy()
    output = output.rename(columns={'sentiment': 'gold_label'})
    output['gold_id'] = output.gold_label.map(LABEL2ID)
    require(output.gold_id.notna().all(), 'Prediction frame gold label mapping failure')
    output['predicted_id'] = probabilities.argmax(axis=1)
    output['predicted_label'] = output.predicted_id.map({v: k for k, v in LABEL2ID.items()})
    output['confidence'] = probabilities.max(axis=1)
    for label, i in LABEL2ID.items():
        output[f'probability_{label}'] = probabilities[:, i]
    return output


def subgroup_metrics(predictions, minimum=30):
    results = []
    for field in ['sarcasm', 'dialect']:
        for value, part in predictions.groupby(field, sort=True):
            row = {'field': field, 'value': str(value), 'n': len(part), 'minimum_sample_size': minimum}
            if len(part) < minimum:
                row.update(status='insufficient_sample_size', accuracy=None, macro_f1=None, weighted_f1=None)
            else:
                overall, per_class, _ = calculate(part.gold_id, part.predicted_id)
                row.update(status='computed', **{k: overall[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']},
                           present_gold_classes=int(part.gold_id.nunique()), per_class=per_class)
            results.append(row)
    return results
