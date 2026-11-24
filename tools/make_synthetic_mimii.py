"""Generates a tiny FAKE dataset in the exact MIMII folder layout, for smoke-testing the pipeline
when the real data is not available. NOT for reporting results.
   python tools/make_synthetic_mimii.py --out data/synthetic"""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import wavfile

SR, SECS = 16000, 10


def clip(rng, f0, faulty, ch):
    t = np.arange(SR * SECS) / SR
    x = 0.05 * rng.standard_normal((len(t), ch))
    for h in (1, 2, 3):
        x += 0.04 / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6))[:, None]
    if faulty:   # extra high-frequency bursts + broadband noise
        env = (np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t) > 0.3)[:, None]
        x += env * 0.05 * np.sin(2 * np.pi * rng.uniform(2800, 3200) * t)[:, None]
        x += 0.03 * rng.standard_normal(x.shape)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic"); ap.add_argument("--machine", default="pump")
    ap.add_argument("--snr", default="6_dB"); ap.add_argument("--normal", type=int, default=24)
    ap.add_argument("--abnormal", type=int, default=8); ap.add_argument("--channels", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for k, mid in enumerate(["id_00", "id_02", "id_04", "id_06"]):
        f0 = 250 + 30 * k                      # every machine "sounds" a bit different (domain shift)
        for name, n in (("normal", a.normal), ("abnormal", a.abnormal)):
            d = Path(a.out) / f"{a.snr}_{a.machine}" / a.machine / mid / name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                wavfile.write(d / f"{i:08d}.wav", SR, clip(rng, f0, name == "abnormal", a.channels))
    print("synthetic dataset written to", a.out)
