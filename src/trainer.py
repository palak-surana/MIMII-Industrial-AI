
"""CNN-LSTM training and prediction wrapper with optional imbalance handling."""

import copy
import numpy as np
import torch
import torch.nn as nn

from .model import CNNLSTM
from .metrics import compute_metrics, best_threshold, choose_threshold
from .imbalance import resample_training_data


def get_device(name="auto"):
    return torch.device(
        "cuda" if name == "auto" and torch.cuda.is_available()
        else ("cpu" if name == "auto" else name)
    )


class TorchTrainer:
    def __init__(self, cfg, mean, std, seed=0):
        self.cfg = cfg
        self.seed = seed
        self.device = get_device(cfg.device)

        self.mean = torch.from_numpy(
            np.asarray(mean, dtype=np.float32)
        ).to(self.device)

        self.std = torch.from_numpy(
            np.asarray(std, dtype=np.float32)
        ).to(self.device)

        self.model = None
        self.thr = 0.5

    def _prep(self, xb):
        x = torch.from_numpy(
            np.asarray(xb, dtype=np.float32)
        ).to(self.device)

        return (x - self.mean) / self.std.clamp_min(1e-6)

    def predict_proba(self, X, bs=512):
        if self.model is None:
            raise RuntimeError("Train the model before prediction.")

        self.model.eval()
        out = []

        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = self._prep(X[i:i + bs])
                probs = torch.sigmoid(self.model(xb))
                out.append(probs.cpu().numpy())

        return np.concatenate(out) if out else np.zeros(0)

    def fit(self, X, y, Xva, yva):
        c = self.cfg

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        rng = np.random.default_rng(self.seed)

        y = np.asarray(y).astype(np.int64)
        yva = np.asarray(yva).astype(np.int64)

        # Apply imbalance handling ONLY to labelled training data.
        method = getattr(c, "imbalance_method", "original")

        X_train, y_train = resample_training_data(
            X,
            y,
            method=method,
            seed=self.seed
        )

        print(
            f"\nTraining CNN-LSTM | Method: {method} | "
            f"Device: {self.device} | Samples: {len(y_train):,}",
            flush=True
        )

        self.model = CNNLSTM(
            X_train.shape[1],
            hidden=c.hidden,
            dropout=c.dropout
        ).to(self.device)

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=c.lr
        )

        # Original data: use positive-class weighting.
        # Resampled data: do not apply additional weighting.
        if method == "original":
            n_pos = max(int(y_train.sum()), 1)
            pw = float(np.clip(
                (len(y_train) - n_pos) / n_pos,
                1,
                20
            ))
        else:
            pw = 1.0

        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pw,
                dtype=torch.float32,
                device=self.device
            )
        )

        best = -1.0
        best_state = None
        bad = 0
        best_ep = 0

        for ep in range(c.epochs):
            self.model.train()
            perm = rng.permutation(len(X_train))

            for i in range(0, len(X_train), c.batch_size):
                idx = perm[i:i + c.batch_size]

                # BatchNorm requires at least two samples.
                if len(idx) < 2:
                    continue

                opt.zero_grad()

                xb = self._prep(X_train[idx])

                yb = torch.from_numpy(
                    y_train[idx].astype(np.float32)
                ).to(self.device)

                loss = loss_fn(self.model(xb), yb)

                loss.backward()
                opt.step()

            # Validation data is never resampled.
            pv = self.predict_proba(Xva)

            if c.select_metric == "auroc":
                from sklearn.metrics import roc_auc_score

                score = (
                    roc_auc_score(yva, pv)
                    if 0 < yva.sum() < len(yva)
                    else 0.0
                )
            else:
                score = best_threshold(yva, pv)[1]

            print(
                f"Epoch {ep + 1}/{c.epochs} | "
                f"Validation score: {score:.4f}",
                flush=True
            )

            if score > best:
                best = score
                best_state = copy.deepcopy(
                    self.model.state_dict()
                )
                bad = 0
                best_ep = ep + 1
            else:
                bad += 1

                if bad >= c.patience:
                    print("Early stopping triggered.", flush=True)
                    break

        if best_state is None:
            raise RuntimeError("Training failed to produce a valid model.")

        self.model.load_state_dict(best_state)

        # Select decision threshold using validation data only.
        pv = self.predict_proba(Xva)

        self.thr = choose_threshold(
            yva,
            pv,
            c.thr_mode,
            c.target_spec
        )

        val_metrics = compute_metrics(
            yva,
            pv,
            self.thr
        )

        print(
            f"Training complete | Best epoch: {best_ep} | "
            f"Validation F1: {val_metrics['f1']:.4f} | "
            f"Threshold: {self.thr:.4f}",
            flush=True
        )

        return dict(
            val_f1=val_metrics["f1"],
            best_epoch=best_ep,
            thr=self.thr,
            imbalance_method=method,
            train_samples=len(y_train),
            positive_weight=pw
        )

    def save(self, path, extra=None):
        torch.save(
            dict(
                state=self.model.state_dict(),
                mean=self.mean.cpu().numpy(),
                std=self.std.cpu().numpy(),
                n_mels=self.mean.shape[0],
                hidden=self.cfg.hidden,
                dropout=self.cfg.dropout,
                imbalance_method=getattr(
                    self.cfg, "imbalance_method", "original"
                ),
                extra=extra or {}
            ),
            path
        )


def load_model(path, device="cpu"):
    ck = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model = CNNLSTM(
        ck["n_mels"],
        hidden=ck["hidden"],
        dropout=ck["dropout"]
    )

    model.load_state_dict(ck["state"])

    return model.to(device).eval(), ck
