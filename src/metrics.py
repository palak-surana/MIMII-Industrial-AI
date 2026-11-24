import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve


def compute_metrics(y, p, thr=0.5, step_sec=None):
    y = np.asarray(y).astype(int); yh = (np.asarray(p) >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    m = dict(accuracy=div(tp + tn, len(y)), precision=prec, recall=rec, sensitivity=rec,
             specificity=div(tn, tn + fp), f1=div(2 * tp, 2 * tp + fp + fn),
             auroc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
             tp=tp, fp=fp, tn=tn, fn=fn)
    if step_sec:   # false alarms per operating hour of NORMAL machine time
        hours = (y == 0).sum() * step_sec / 3600.0
        m["false_alarms_per_hour"] = fp / hours if hours else float("nan")
    return m


def best_threshold(y, p):
    """Threshold maximising F1 on (y, p); returns (thr, f1). Used on the validation machine only."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5, 0.0
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(th[i]), float(f1[i])


def choose_threshold(y, p, mode="f1", target_spec=0.90):
    """Decision threshold from the VALIDATION machine. 'f1': maximise F1 (can be extreme/unstable when the
    validation machine is hard); 'spec': the score below which target_spec of validation-normal windows fall."""
    y = np.asarray(y).astype(int); p = np.asarray(p)
    if mode == "spec" and (y == 0).any():
        return float(np.quantile(p[y == 0], target_spec))
    return best_threshold(y, p)[0]
