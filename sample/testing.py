import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve


def load_kpi(path: str, **kwargs):
    df = pd.read_csv(path, **kwargs)
    return df.get('timestamp'), df.get('value'), df.get('label', np.zeros_like(df['value'], dtype=np.int32))


def adjust_scores(labels, scores, delay=None, inplace=False) -> np.ndarray:
    if np.shape(scores) != np.shape(labels):
        raise ValueError('`labels` and `scores` must have same shape')
    if delay is None:
        delay = len(scores)
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    is_anomaly = labels[0] == 1
    adjusted_scores = np.copy(scores) if not inplace else scores
    pos = 0
    for part in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, part)
            adjusted_scores[pos: ptr] = np.max(adjusted_scores[pos: ptr])
            adjusted_scores[ptr: part] = np.maximum(adjusted_scores[ptr: part], adjusted_scores[pos])
        is_anomaly = not is_anomaly
        pos = part
    part = len(labels)
    if is_anomaly:
        ptr = min(pos + delay + 1, part)
        adjusted_scores[pos: part] = np.max(adjusted_scores[pos: ptr])
    return adjusted_scores


def ignore_missing(series_list, missing):
    ret = []
    for series in series_list:
        series = np.copy(series)
        ret.append(series[missing != 1])
    return tuple(ret)


def best_f1score(labels, scores):
    precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=scores)
    f1score = 2 * precision * recall / np.clip(precision + recall, a_min=1e-8, a_max=None)

    best_threshold = thresholds[np.argmax(f1score)]
    best_precision = precision[np.argmax(f1score)]
    best_recall = recall[np.argmax(f1score)]

    return best_threshold, best_precision, best_recall, np.max(f1score)
