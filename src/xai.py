"""Phase 9 (XAI): Integrated Gradients on the log-Mel input -> influential frequency bands + time interval.
  python -m src.xai --model results/models/agent_final.pt --index 123      (index into the TEST windows)"""
import numpy as np
import torch
from .features import mel_filterbank


def integrated_gradients(model, x, steps=50):
    """x: normalised tensor (n_mels, T) on the model's device. Attribution w.r.t. the faulty LOGIT."""
    model.eval(); x = x.unsqueeze(0); base = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps + 1, device=x.device)[1:].view(-1, 1, 1)
    xs = (base + alphas * (x - base)).detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):          # cuDNN LSTM backward needs train mode otherwise
        grads = torch.autograd.grad(model(xs).sum(), xs)[0]
    return ((x - base) * grads.mean(0, keepdim=True)).squeeze(0).detach().cpu().numpy()      # (n_mels, T)


def summarise(attr, hz_edges, sr, hop, start_frame=0, top_k=3, interval_s=0.5):
    """Top-k frequency bands and the most influential time interval (seconds into the clip)."""
    pos = np.maximum(attr, 0)
    band = pos.sum(1); chosen = []
    for i in np.argsort(-band):                              # pick separated peaks
        if all(abs(i - j) > 2 for j in chosen):
            chosen.append(int(i))
        if len(chosen) == top_k:
            break
    bands = [(float(hz_edges[i]), float(hz_edges[i + 2])) for i in chosen]
    frame = pos.sum(0); w = max(1, int(interval_s * sr / hop)); w = min(w, len(frame))
    s = int(np.argmax(np.convolve(frame, np.ones(w), "valid")))
    t0 = (start_frame + s) * hop / sr
    return dict(bands_hz=bands, interval_s=(t0, t0 + w * hop / sr))


def explain(model, x_raw, mean, std, cfg, start_frame=0, device="cpu"):
    """x_raw: (n_mels, T) log-Mel in dB. Returns probability, attribution map, summary."""
    xn = torch.from_numpy(((np.asarray(x_raw, np.float32) - mean) / std)).to(device)
    with torch.no_grad():
        p = float(torch.sigmoid(model(xn.unsqueeze(0)))[0])
    attr = integrated_gradients(model, xn)
    _, hz = mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.band_lo, cfg.band_hi)
    return p, attr, summarise(attr, hz, cfg.sr, cfg.hop, start_frame)


def main(argv=None):
    import argparse, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from .config import parse_args
    from .data import load_data
    from .trainer import load_model
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--index", type=int, default=-1)
    a, rest = ap.parse_known_args(argv); cfg = parse_args("xai", rest)
    d = load_data(cfg); model, ck = load_model(a.model)
    i = a.index
    if i < 0:                                              # default: most confident faulty test window
        from .metrics import compute_metrics
        with torch.no_grad():
            pr = np.concatenate([torch.sigmoid(model(torch.from_numpy((d.Xte[j:j + 256].astype(np.float32) - ck["mean"]) / ck["std"]))).numpy()
                                 for j in range(0, len(d.Xte), 256)])
        i = int(np.argmax(pr * (d.yte == 1)))
    p, attr, s = explain(model, d.Xte[i], ck["mean"], ck["std"], cfg, int(d.mte.start_frame[i]))
    m = d.mte.iloc[i]
    print(f"{m.machine} / {m.id}  true={'faulty' if d.yte[i] else 'healthy'}  P(faulty)={p:.2f}")
    print("influential bands (Hz):", [f"{lo:.0f}-{hi:.0f}" for lo, hi in s["bands_hz"]], "| interval (s): %.1f-%.1f" % s["interval_s"])
    out = f"{cfg.results_dir}/xai_test_{i}.png"
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].imshow(d.Xte[i].astype(np.float32), origin="lower", aspect="auto"); ax[0].set_title("log-Mel input")
    ax[1].imshow(np.abs(attr), origin="lower", aspect="auto", cmap="magma"); ax[1].set_title("Integrated-Gradients |attribution|")
    for a_ in ax: a_.set_xlabel("frame"); a_.set_ylabel("Mel band")
    plt.tight_layout(); plt.savefig(out, dpi=150); print("saved", out)


if __name__ == "__main__":
    main()
