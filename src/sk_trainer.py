"""Fast stand-in classifier (logistic regression on per-Mel mean/std). ONLY for smoke-testing the
pipeline on machines without PyTorch. Real experiments must use the CNN-LSTM (--trainer torch)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .metrics import compute_metrics, choose_threshold


def feats(X, mean, std):
    Z = (np.asarray(X, dtype=np.float32) - mean) / std
    return np.concatenate([Z.mean(2), Z.std(2)], axis=1)


class SklearnTrainer:
    def __init__(self, cfg, mean, std, seed=0):
        self.cfg, self.mean, self.std, self.seed = cfg, mean, std, seed

    def fit(self, X, y, Xva, yva):
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5, random_state=self.seed)
        self.clf.fit(feats(X, self.mean, self.std), y)
        pv = self.predict_proba(Xva)
        self.thr = choose_threshold(yva, pv, getattr(self.cfg, "thr_mode", "f1"), getattr(self.cfg, "target_spec", 0.9))
        return dict(val_f1=compute_metrics(yva, pv, self.thr)["f1"], best_epoch=0, thr=self.thr)

    def predict_proba(self, X):
        return self.clf.predict_proba(feats(X, self.mean, self.std))[:, 1] if len(X) else np.zeros(0)
