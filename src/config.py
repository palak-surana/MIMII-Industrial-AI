
"""Central configuration. Every field can be overridden from the command line,
e.g. python -m src.preprocess --machines pump fan --snr 6_dB
"""

import argparse
from dataclasses import dataclass, field, fields, MISSING


@dataclass
class Config:
    # ---- paths ----
    data_root: str = "data/mimii"
    processed_dir: str = "data/processed"
    results_dir: str = "results"

    # ---- dataset ----
    snr: str = "-6_dB"
    machines: list = field(default_factory=lambda: ["pump"])
    train_ids: list = field(default_factory=lambda: ["id_00", "id_02"])
    val_ids: list = field(default_factory=lambda: ["id_04"])
    test_ids: list = field(default_factory=lambda: ["id_06"])

    # ---- preprocessing ----
    sr: int = 16000
    channel: int = 0
    band_lo: float = 50.0
    band_hi: float = 7500.0
    n_fft: int = 1024
    hop: int = 512
    n_mels: int = 64
    windows: list = field(default_factory=lambda: [1.0, 2.0])
    window_secs: float = 1.0
    overlap: float = 0.5

    # ---- model / training ----
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 1
    hidden: int = 64
    dropout: float = 0.3
    device: str = "auto"

    # ---- active learning ----
    init_frac: float = 0.10
    batch_frac: float = 0.10
    max_frac: float = 0.50
    target_f1: float = 0.80

    # One seed for the initial pilot experiment.
    seeds: list = field(default_factory=lambda: [0])

    per_id_norm: int = 0
    select_metric: str = "f1"
    thr_mode: str = "f1"
    target_spec: float = 0.90
    pool_cap: int = 0

    # ---- trainer ----
    trainer: str = "torch"

    # Keep original imbalance handling for this pilot.
    imbalance_method: str = "original"

    # ---- experiment stage ----
    stage: str = "curves"


def parse_args(description="", argv=None):
    p = argparse.ArgumentParser(description=description)

    for f in fields(Config):
        d = (
            f.default
            if f.default is not MISSING
            else f.default_factory()
        )

        if isinstance(d, list):
            p.add_argument(
                f"--{f.name}",
                nargs="+",
                type=type(d[0]),
                default=d
            )
        else:
            p.add_argument(
                f"--{f.name}",
                type=type(d),
                default=d
            )

    return Config(**vars(p.parse_args(argv)))
