
"""Phase 3: machine-wise split (train / val / test are DIFFERENT machine IDs)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Data:
    Xtr: np.ndarray
    ytr: np.ndarray
    mtr: pd.DataFrame

    Xva: np.ndarray
    yva: np.ndarray
    mva: pd.DataFrame

    Xte: np.ndarray
    yte: np.ndarray
    mte: pd.DataFrame

    mean: np.ndarray
    std: np.ndarray
    step_sec: float


def calculate_normalization(X, chunk_size=2048):
    """Calculate exact global per-Mel-band mean and std from training data only."""

    n_samples, n_mels, n_frames = X.shape

    total_sum = np.zeros(n_mels, dtype=np.float64)
    total_sq_sum = np.zeros(n_mels, dtype=np.float64)
    total_count = 0

    for start in range(0, n_samples, chunk_size):
        chunk = np.asarray(
            X[start:start + chunk_size],
            dtype=np.float64
        )

        total_sum += chunk.sum(axis=(0, 2))
        total_sq_sum += np.square(chunk).sum(axis=(0, 2))

        total_count += chunk.shape[0] * chunk.shape[2]

    mean = total_sum / total_count

    variance = total_sq_sum / total_count - np.square(mean)
    variance = np.maximum(variance, 0.0)

    std = np.sqrt(variance)

    return (
        mean.astype(np.float32)[:, None],
        (std + 1e-6).astype(np.float32)[:, None]
    )


def load_data(cfg, window=None):

    w = cfg.window_secs if window is None else window

    X = np.load(
        Path(cfg.processed_dir) / f"X_w{w:g}s.npy",
        mmap_mode="r"
    )

    meta = pd.read_csv(
        Path(cfg.processed_dir) / f"meta_w{w:g}s.csv"
    )

    y = meta["label"].to_numpy().astype(int)

    tr = set(cfg.train_ids)
    va = set(cfg.val_ids)
    te = set(cfg.test_ids)

    assert not (tr & va or tr & te or va & te), (
        "Machine IDs overlap between splits!"
    )

    sel = {
        "tr": np.where(meta["id"].isin(tr))[0],
        "va": np.where(meta["id"].isin(va))[0],
        "te": np.where(meta["id"].isin(te))[0]
    }

    for name in ("tr", "va", "te"):
        assert len(sel[name]) > 0, (
            f"No samples found for {name} split. "
            "Check machine IDs and metadata."
        )

    if cfg.pool_cap and len(sel["tr"]) > cfg.pool_cap:
        sel["tr"] = np.sort(
            np.random.default_rng(12345).choice(
                sel["tr"],
                cfg.pool_cap,
                replace=False
            )
        )

    def get_split(name):
        idx = sel[name]

        return (
            np.asarray(X[idx]),
            y[idx],
            meta.iloc[idx].reset_index(drop=True)
        )

    (Xtr, ytr, mtr) = get_split("tr")
    (Xva, yva, mva) = get_split("va")
    (Xte, yte, mte) = get_split("te")

    if cfg.per_id_norm:

        for X_, m_ in (
            (Xtr, mtr),
            (Xva, mva),
            (Xte, mte)
        ):
            norm_by_machine(X_, m_)

        mean = np.zeros((Xtr.shape[1], 1), dtype=np.float32)
        std = np.ones((Xtr.shape[1], 1), dtype=np.float32)

    else:
        mean, std = calculate_normalization(Xtr)

    step_sec = float(meta["step_sec"].iloc[0])

    return Data(
        Xtr, ytr, mtr,
        Xva, yva, mva,
        Xte, yte, mte,
        mean, std,
        step_sec
    )


def norm_by_machine(X, meta, chunk_size=2048):
    """Standardise each machine/id group using its own audio, without labels."""

    key = meta["machine"].astype(str) + "/" + meta["id"].astype(str)

    for group in key.unique():

        idx = np.where(key.to_numpy() == group)[0]

        group_mean, group_std = calculate_normalization(
            X[idx],
            chunk_size
        )

        for start in range(0, len(idx), chunk_size):

            selected = idx[start:start + chunk_size]

            X[selected] = (
                (
                    np.asarray(X[selected], dtype=np.float32)
                    - group_mean
                ) / group_std
            ).astype(X.dtype)


def describe_split(d):

    for name, y in (
        ("train pool", d.ytr),
        ("validation", d.yva),
        ("test (unseen machine)", d.yte)
    ):

        print(
            f"{name:24s} {len(y):7d} segments, "
            f"faulty = {y.mean() * 100:.1f}%"
        )
