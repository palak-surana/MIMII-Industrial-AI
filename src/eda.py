"""Phase 1: required EDA. Needs preprocessing to have been run. Writes figures + CSVs to results/eda/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from .config import parse_args
from .preprocess import locate, read_wav


def main(argv=None):
    cfg = parse_args("EDA", argv)
    out = Path(cfg.results_dir) / "eda"; out.mkdir(parents=True, exist_ok=True)
    P = Path(cfg.processed_dir)
    clips = pd.read_csv(P / "clips.csv"); clips["cond"] = clips.label.map({0: "normal", 1: "abnormal"})
    # 1-4: counts, class balance, duration
    cnt = clips.pivot_table(index=["machine", "id"], columns="cond", values="file", aggfunc="count", fill_value=0)
    cnt["total"] = cnt.sum(axis=1); cnt["abnormal_%"] = 100 * cnt.get("abnormal", 0) / cnt["total"]
    hrs = clips.groupby(["machine", "id", "cond"]).seconds.sum().unstack(fill_value=0) / 3600
    cnt.to_csv(out / "recordings_per_machine_id.csv"); hrs.to_csv(out / "operating_hours.csv")
    print(cnt, "\nOperating hours:\n", hrs.round(2), sep="\n")
    print("clip duration (s):", clips.seconds.describe()[["min", "mean", "max"]].round(2).to_dict(),
          "| channels:", clips.n_channels.unique().tolist(), "| sr:", cfg.sr, "| SNR folder:", cfg.snr)
    cnt[[c for c in ("normal", "abnormal") if c in cnt]].plot.bar(figsize=(7, 3.8), title="Normal vs faulty recordings per machine ID")
    plt.tight_layout(); plt.savefig(out / "class_distribution.png", dpi=150); plt.close()
    clips.seconds.hist(bins=30, figsize=(5, 3)); plt.title("Clip duration (s)"); plt.savefig(out / "duration_hist.png", dpi=150); plt.close()
    clips.boxplot(column="rms", by="cond", figsize=(4.5, 3.5)); plt.suptitle(""); plt.title("Clip RMS: normal vs abnormal")
    plt.savefig(out / "rms_by_condition.png", dpi=150); plt.close()
    # 5-7: waveform examples (same machine ID) + channel comparison
    m, mid = cfg.machines[0], cfg.train_ids[0]; base = locate(cfg, m)
    fn = sorted((base / mid / "normal").glob("*.wav"))[0]; fa = sorted((base / mid / "abnormal").glob("*.wav"))[0]
    (sr, xn), (_, xa) = read_wav(fn), read_wav(fa)
    t = np.arange(len(xn)) / sr
    fig, ax = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    ax[0].plot(t, xn[:, 0], lw=.4); ax[0].set_title(f"{m} {mid} NORMAL ({fn.name})")
    ax[1].plot(np.arange(len(xa)) / sr, xa[:, 0], lw=.4, c="C3"); ax[1].set_title(f"{m} {mid} FAULTY ({fa.name})"); ax[1].set_xlabel("s")
    plt.tight_layout(); plt.savefig(out / "waveform_normal_vs_faulty.png", dpi=150); plt.close()
    ch = xn.shape[1]; fig, ax = plt.subplots(ch, 1, figsize=(9, 1.1 * ch + 1), sharex=True)
    for i in range(ch):
        ax[i if ch > 1 else 0].plot(t, xn[:, i], lw=.3); ax[i if ch > 1 else 0].set_ylabel(f"ch{i}", rotation=0, labelpad=18)
    plt.suptitle("Microphone-channel comparison (same recording)"); plt.savefig(out / "channel_comparison.png", dpi=150); plt.close()
    # 8: frequency spectrum (Welch PSD) + mean log-Mel spectrum, normal vs faulty
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for x, lab, c in ((xn[:, 0], "normal", "C0"), (xa[:, 0], "faulty", "C3")):
        f, p = welch(x, sr, nperseg=2048); ax[0].semilogy(f, p, c=c, label=lab, lw=.8)
    ax[0].set_title("Welch PSD (one clip each)"); ax[0].set_xlabel("Hz"); ax[0].legend()
    mel = np.load(P / "mel.npz"); w = cfg.window_secs
    X = np.load(P / f"X_w{w:g}s.npy", mmap_mode="r"); meta = pd.read_csv(P / f"meta_w{w:g}s.csv")
    centre = mel["hz"][1:-1]
    for lab, c, name in ((0, "C0", "normal"), (1, "C3", "faulty")):
        idx = np.where((meta.label == lab) & (meta.id == mid))[0][:500]
        ax[1].plot(centre, np.asarray(X[idx], dtype=np.float32).mean((0, 2)), c=c, label=name)
    ax[1].set_title(f"Mean log-Mel spectrum ({mid})"); ax[1].set_xlabel("Hz (Mel centre)"); ax[1].set_ylabel("dB"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out / "spectrum_normal_vs_faulty.png", dpi=150); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a, lab, name in zip(ax, (0, 1), ("normal", "faulty")):
        i = np.where((meta.label == lab) & (meta.id == mid))[0][0]
        a.imshow(np.asarray(X[i], dtype=np.float32), origin="lower", aspect="auto"); a.set_title(f"log-Mel window: {name}")
    plt.tight_layout(); plt.savefig(out / "logmel_examples.png", dpi=150); plt.close()
    print("EDA written to", out)


if __name__ == "__main__":
    main()
