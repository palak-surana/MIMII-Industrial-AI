
"""Phases 5-7: Label pool, random baseline and entropy-based uncertainty sampling."""

import numpy as np


def entropy(p, eps=1e-8):
    """Calculate normalized binary entropy in the range [0, 1]."""

    p = np.asarray(p, dtype=np.float64)

    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite.")

    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    p = np.clip(p, eps, 1 - eps)

    return (
        -(p * np.log(p) + (1 - p) * np.log(1 - p))
    ) / np.log(2)


class LabelPool:
    """Manage labelled and unlabelled training samples."""

    def __init__(self, y_hidden, init_frac=0.10, seed=0):

        self._y = np.asarray(y_hidden).astype(int)

        if self._y.ndim != 1 or len(self._y) == 0:
            raise ValueError("y_hidden must be a non-empty 1D array.")

        if not 0 < init_frac <= 1:
            raise ValueError("init_frac must be in (0, 1].")

        if not np.isin(self._y, [0, 1]).all():
            raise ValueError("Labels must be binary: 0 or 1.")

        n = len(self._y)
        rng = np.random.default_rng(seed)

        n_init = min(max(int(round(init_frac * n)), 4), n)

        pos = np.where(self._y == 1)[0]
        neg = np.where(self._y == 0)[0]

        if len(pos) == 0 or len(neg) == 0:
            init = rng.choice(n, n_init, replace=False)

        else:
            n_pos = int(np.clip(
                round(n_init * len(pos) / n),
                min(2, len(pos)),
                min(len(pos), n_init)
            ))

            n_neg = n_init - n_pos

            if n_neg > len(neg):
                n_neg = len(neg)
                n_pos = n_init - n_neg

            init = np.concatenate([
                rng.choice(pos, n_pos, replace=False),
                rng.choice(neg, n_neg, replace=False)
            ])

        self.labelled = np.zeros(n, dtype=bool)
        self.labelled[init] = True

        self.round_labelled = np.full(n, -1, dtype=int)
        self.round_labelled[init] = 0

        self.query_log = []

    n = property(lambda self: len(self._y))
    n_labelled = property(lambda self: int(self.labelled.sum()))
    frac_labelled = property(lambda self: self.n_labelled / self.n)
    labelled_idx = property(lambda self: np.where(self.labelled)[0])
    unlabelled_idx = property(lambda self: np.where(~self.labelled)[0])

    def labels_of_labelled(self, idx):

        idx = np.asarray(idx, dtype=int)

        if idx.ndim != 1:
            raise ValueError("Indices must be a 1D array.")

        if np.any(idx < 0) or np.any(idx >= self.n):
            raise IndexError("Label index out of range.")

        if not self.labelled[idx].all():
            raise ValueError("Attempted to read a hidden label.")

        return self._y[idx]

    def annotate(self, idx, round_no):
        """Reveal true labels for selected, previously unlabelled samples."""

        idx = np.asarray(idx, dtype=int)

        if idx.ndim != 1:
            raise ValueError("Indices must be a 1D array.")

        if len(idx) == 0:
            return np.array([], dtype=int)

        if len(np.unique(idx)) != len(idx):
            raise ValueError("Duplicate indices in annotation request.")

        if np.any(idx < 0) or np.any(idx >= self.n):
            raise IndexError("Annotation index out of range.")

        if self.labelled[idx].any():
            raise ValueError(
                "Attempted to annotate an already labelled sample."
            )

        self.labelled[idx] = True
        self.round_labelled[idx] = round_no

        self.query_log.append({
            "round": int(round_no),
            "n": int(len(idx))
        })

        return self._y[idx]


def validate_indices(indices, name="indices"):
    """Validate a one-dimensional collection of unique indices."""

    indices = np.asarray(indices, dtype=int)

    if indices.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicates.")

    if np.any(indices < 0):
        raise IndexError(f"{name} cannot contain negative indices.")

    return indices


def select_random(unl_idx, n, rng):
    """Select random unlabelled samples without replacement."""

    unl_idx = validate_indices(unl_idx, "unlabelled indices")

    if n < 0:
        raise ValueError("n cannot be negative.")

    if n == 0 or len(unl_idx) == 0:
        return np.array([], dtype=int), None

    return rng.choice(
        unl_idx,
        size=min(n, len(unl_idx)),
        replace=False
    ), None


def select_uncertain(unl_idx, probs, n):
    """Select samples with the highest predictive entropy."""

    unl_idx = validate_indices(unl_idx, "unlabelled indices")
    probs = np.asarray(probs, dtype=np.float64)

    if n < 0:
        raise ValueError("n cannot be negative.")

    if probs.ndim != 1:
        raise ValueError("Probabilities must be a 1D array.")

    if len(unl_idx) != len(probs):
        raise ValueError(
            "Probability count must match unlabelled samples."
        )

    if not np.isfinite(probs).all():
        raise ValueError("Probabilities must be finite.")

    if np.any((probs < 0) | (probs > 1)):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    n = min(n, len(unl_idx))

    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    uncertainty = entropy(probs)

    top = np.argpartition(-uncertainty, n - 1)[:n]
    top = top[np.argsort(-uncertainty[top])]

    return unl_idx[top], uncertainty[top]
