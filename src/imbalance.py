
"""Training-only class imbalance handling for industrial machine fault detection."""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def resample_training_data(X, y, method="original", seed=0, k_neighbors=5):
    """
    Resample training data only.

    Parameters
    ----------
    X : array
        Log-Mel features with shape (samples, mel_bands, time_frames).
    y : array
        Binary labels: normal=0, faulty=1.
    method : str
        original, undersample, or smote.
    seed : int
        Random seed for reproducibility.
    k_neighbors : int
        Number of neighbors used by SMOTE.

    Returns
    -------
    X_resampled, y_resampled
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(np.int64)

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must contain both classes: 0 and 1.")

    if method == "original":
        return X, y

    original_shape = X.shape[1:]
    X_flat = np.asarray(X, dtype=np.float32).reshape(len(X), -1)

    if method == "undersample":
        sampler = RandomUnderSampler(random_state=seed)

    elif method == "smote":
        minority_count = int(np.bincount(y).min())

        if minority_count <= k_neighbors:
            raise ValueError(
                f"SMOTE needs at least {k_neighbors + 1} minority samples. "
                f"Found {minority_count}."
            )

        sampler = SMOTE(
            random_state=seed,
            k_neighbors=k_neighbors
        )

    else:
        raise ValueError(
            "Unknown method. Use original, undersample, or smote."
        )

    X_resampled, y_resampled = sampler.fit_resample(X_flat, y)

    X_resampled = X_resampled.reshape(
        len(X_resampled), *original_shape
    ).astype(np.float16)

    y_resampled = np.asarray(y_resampled, dtype=np.int64)

    print(f"\nResampling method: {method}")
    print(f"Before: {len(y):,} samples | "
          f"Normal: {(y == 0).sum():,} | Faulty: {(y == 1).sum():,}")
    print(f"After:  {len(y_resampled):,} samples | "
          f"Normal: {(y_resampled == 0).sum():,} | "
          f"Faulty: {(y_resampled == 1).sum():,}")

    return X_resampled, y_resampled
