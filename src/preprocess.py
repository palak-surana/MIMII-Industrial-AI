"""Phase 2: raw MIMII WAV -> band-passed log-Mel windows with clip-level labels.
Output (per window size w):  data/processed/X_w{w}s.npy  (N, n_mels, T) float16
                              data/processed/meta_w{w}s.csv
plus clips.csv (one row per recording) and mel.npz (filterbank / band edges)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
from .config import parse_args
from .features import mel_filterbank, bandpass, log_mel


def locate(cfg, machine):
    root = Path(cfg.data_root)
    for c in (root / f"{cfg.snr}_{machine}" / machine, root / machine, root / f"{cfg.snr}_{machine}"):
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Cannot find {machine} under {root} (expected {cfg.snr}_{machine}/{machine}/id_xx/normal/*.wav)")


def read_wav(path):
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32)
    return sr, (x[:, None] if x.ndim == 1 else x)


def main(argv=None):
    cfg = parse_args("Preprocess MIMII", argv)
    out = Path(cfg.processed_dir); out.mkdir(parents=True, exist_ok=True)
    fb, hz = mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.band_lo, cfg.band_hi)
    np.savez(out / "mel.npz", fb=fb, hz=hz)
    ids = sorted(set(cfg.train_ids + cfg.val_ids + cfg.test_ids))
    clips, X = [], {w: [] for w in cfg.windows}
    meta = {w: [] for w in cfg.windows}
    for m in cfg.machines:
        base = locate(cfg, m)
        for mid in ids:
            for lab, name in ((0, "normal"), (1, "abnormal")):
                files = sorted((base / mid / name).glob("*.wav"))
                print(f"{m} {mid} {name}: {len(files)} files", flush=True)
                for k, f in enumerate(files):
                    sr, x = read_wav(f)
                    if sr != cfg.sr:
                        raise ValueError(f"{f}: sample rate {sr} != {cfg.sr}")
                    sig = x.mean(1) if cfg.channel < 0 else x[:, cfg.channel]
                    clips.append(dict(machine=m, id=mid, label=lab, file=f.name, seconds=len(sig) / sr,
                                      n_channels=x.shape[1], rms=float(np.sqrt((sig ** 2).mean()))))
                    S = log_mel(bandpass(sig, sr, cfg.band_lo, cfg.band_hi), cfg.n_fft, cfg.hop, fb)
                    for w in cfg.windows:
                        wf = int(w * sr // cfg.hop); step = max(1, int(wf * (1 - cfg.overlap)))
                        for s in range(0, S.shape[1] - wf + 1, step):
                            X[w].append(S[:, s:s + wf].astype(np.float16))
                            meta[w].append(dict(machine=m, id=mid, label=lab, file=f.name, start_frame=s,
                                                step_sec=step * cfg.hop / sr))
                    if k and k % 200 == 0:
                        print(f"   ...{k}/{len(files)}", flush=True)
    pd.DataFrame(clips).to_csv(out / "clips.csv", index=False)
    for w in cfg.windows:
        np.save(out / f"X_w{w:g}s.npy", np.stack(X[w]))
        pd.DataFrame(meta[w]).to_csv(out / f"meta_w{w:g}s.csv", index=False)
        print(f"window {w:g}s: {len(X[w])} segments")


if __name__ == "__main__":
    main()
