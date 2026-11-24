"""Optional comparison baseline: Random Forest on hand-crafted statistical features."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def handcrafted(X):
    out = []
    for i in range(0, len(X), 2048):
        db = np.asarray(X[i:i + 2048], dtype=np.float32)                  # (N, mels, T) in dB
        pw = 10 ** (db / 10)
        e = pw.sum(1)                                                     # frame energy (N, T)
        rms = np.sqrt((e ** 2).mean(1)) + 1e-12
        crest = e.max(1) / rms
        z = (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-12)
        kurt = (z ** 4).mean(1)
        cent = ((np.arange(db.shape[1])[None, :, None] * pw).sum(1) / (e + 1e-12)).mean(1)   # spectral centroid (mel index)
        out.append(np.concatenate([db.mean(2), db.std(2), np.stack([np.log(rms), crest, kurt, cent], 1)], 1))
    return np.concatenate(out)


def rf_baseline(Xtr, ytr, seed=0):
    rf = RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf.fit(handcrafted(Xtr), ytr)
    return lambda X: rf.predict_proba(handcrafted(X))[:, 1]
