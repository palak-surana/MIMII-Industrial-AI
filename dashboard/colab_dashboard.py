"""Notebook dashboard (Colab / Jupyter) for the agentic active-learning project.

    import sys; sys.path.insert(0, "/content/mimii_al")
    from dashboard.colab_dashboard import launch
    launch(results_dir=f"{R}/v2", processed_dir=P, data_root="/content/mimii_al/data/mimii", window_secs=2.0, v1_dir=R)

Tabs: 1 Results (random vs active, tables, tests) | 2 Agent decisions (round-by-round) | 3 Inspect a segment
(spectrogram, waveform, prediction, uncertainty, annotation status, Integrated Gradients).
All rendering functions below are plain functions (no widgets) so they can be tested / reused."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.active_learning import entropy


def _json(p):
    return json.load(open(p)) if Path(p).exists() else None


# ------------------------------------------------------------------ results tab
def curves_figure(df, target=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (col, lab) in zip(ax, (("test_f1", "F1 on unseen machine"), ("test_auroc", "AUROC on unseen machine"))):
        for s, name, c in (("random", "Random", "tab:blue"), ("uncertainty", "Agentic active learning", "tab:red")):
            g = df[df.strategy == s].groupby("round").agg(x=("frac_labelled", "mean"), m=(col, "mean"), sd=(col, "std")).fillna(0)
            a.plot(g.x * 100, g.m, "-o", c=c, label=name)
            a.fill_between(g.x * 100, g.m - g.sd, g.m + g.sd, color=c, alpha=.15)
        if target is not None and col == "test_f1":
            a.axhline(target, ls="--", c="gray", lw=1, label=f"pre-set target {target}")
        a.set_xlabel("% of training pool labelled"); a.set_ylabel(lab); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def auc_table(df):
    """Exploratory: mean over the whole labelling budget (area under the learning curve), paired by seed."""
    rows = []
    for col in ("test_f1", "test_auroc"):
        a = df.groupby(["strategy", "seed"])[col].mean().unstack(0)
        d = a["uncertainty"] - a["random"]
        rows.append(dict(metric=col, random=a["random"].mean(), active=a["uncertainty"].mean(), diff=d.mean(),
                         diff_std=d.std(), active_better_in=f"{int((d > 0).sum())}/{len(d)} seeds",
                         paired_t_p=stats.ttest_rel(a["uncertainty"], a["random"]).pvalue if len(d) >= 3 else np.nan))
    return pd.DataFrame(rows).round(3)


def exp1_table(j):
    rows = []
    for w in [k for k in j if k.endswith("s") and isinstance(j[k], dict)]:
        for m in ("cnn_lstm", "random_forest"):
            t = j[w][m]["test"]; sd = j[w][m].get("test_std")
            f = lambda k: f"{t[k]:.3f}" + (f" ± {sd[k]:.3f}" if sd else "")
            rows.append({"window": w, "model": m, "test F1": f("f1"), "test AUROC": f("auroc"),
                         "seeds": j[w][m].get("n_seeds", 1)})
    return pd.DataFrame(rows)


def load_results(results_dir):
    R = Path(results_dir); rd = lambda p: pd.read_csv(R / p) if (R / p).exists() else None
    return dict(rounds=rd("all_rounds.csv"), summary=_json(R / "summary.json"), table=rd("table_f1.csv"),
                final=rd("exp6_final_cross_machine.csv"), per_seed=rd("final_f1_per_seed.csv"),
                exp1=_json(R / "exp1_windows.json") or _json(R.parent / "exp1_windows.json"),
                log=_json(R / "agent_log.json"))


# ------------------------------------------------------------------ agent tab
def _card(label, value):
    return (f"<div style='display:inline-block;margin:4px 10px 4px 0;padding:6px 12px;border:1px solid #ccc;"
            f"border-radius:8px;min-width:110px'><div style='font-size:11px;color:#666'>{label}</div>"
            f"<div style='font-size:18px;font-weight:600'>{value}</div></div>")


def agent_round_html(log, r):
    x = log[r]
    cards = (_card("Round", x["round"]) + _card("Currently labelled", f"{x['n_labelled']:,}") +
             _card("Unlabelled", f"{x['n_unlabelled']:,}") + _card("Validation F1", f"{x['val_f1']:.2f}") +
             _card("Target F1", x.get("target_f1")) + _card("Threshold", f"{x.get('thr', float('nan')):.3f}"))
    body = f"<div>{cards}</div><p><b>Next action:</b> {x['decision']}<br><b>Selection method:</b> {x['selection_method']}</p>"
    if x.get("candidates"):
        def _u(c):
            return "" if c["uncertainty"] is None else "%.3f" % c["uncertainty"]
        rows = "".join("<tr><td>%s</td><td>%s</td></tr>" % (c["segment"], _u(c)) for c in x["candidates"])
        body += f"<p><b>Top candidates sent to the (simulated) engineer</b></p><table border=1 cellpadding=4><tr><th>segment</th><th>uncertainty</th></tr>{rows}</table>"
    return body


def agent_figure(log, r):
    n = [x["n_labelled"] for x in log]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(n, [x["val_f1"] for x in log], "-o", label="validation F1 (agent's stop rule)")
    ax.plot(n, [x["test"]["f1"] for x in log], "-s", label="test F1 (unseen machine; reporting only)")
    if log[0].get("target_f1"):
        ax.axhline(log[0]["target_f1"], ls="--", c="gray", lw=1, label="target F1")
    ax.axvline(n[r], c="k", alpha=.3); ax.set_xlabel("labelled segments"); ax.set_ylabel("F1"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ segment tab
def render_segment(X, y, meta, probs, thr, i, status, mel_hz, cfg, explain_fn=None, wav_loader=None):
    """Returns (html, figure) for window i. probs = P(faulty) for the whole split."""
    p = float(probs[i]); m = meta.iloc[i]
    cards = (_card("Machine", f"{m.machine} / {m.id}") + _card("True label", "faulty" if y[i] else "healthy") +
             _card("Predicted", "faulty" if p >= thr else "healthy") + _card("P(faulty)", f"{p:.3f}") +
             _card("Uncertainty", f"{float(entropy(p)):.2f}") + _card("Annotation", status))
    html = f"<div>{cards}</div><p style='font-size:12px;color:#555'>clip {m.file}, starts {m.start_frame * cfg.hop / cfg.sr:.1f}s into the recording; decision threshold {thr:.3f}; uncertainty = binary entropy around P=0.5 (what the agent ranks by)</p>"
    wav = None
    if wav_loader is not None:
        try:
            wav = wav_loader(m, X.shape[2])
        except Exception:
            wav = None
    attr = None
    if explain_fn is not None:
        _, attr, s = explain_fn(X[i], int(m.start_frame))
        html += ("<p><b>Integrated Gradients:</b> influential bands "
                 + ", ".join(f"{lo:.0f}-{hi:.0f} Hz" for lo, hi in s["bands_hz"])
                 + " | most influential interval %.1f-%.1f s (exploratory; shows what the model uses, not the physical cause)</p>" % s["interval_s"])
    panels = 1 + (wav is not None) + (attr is not None)
    fig, ax = plt.subplots(1, panels, figsize=(5.2 * panels, 3.4)); ax = np.atleast_1d(ax); k = 0
    T = X.shape[2] * cfg.hop / cfg.sr; centres = mel_hz[1:-1]; ticks = np.linspace(0, len(centres) - 1, 6).astype(int)
    img = lambda a_, M, t: (a_.imshow(M, origin="lower", aspect="auto", extent=[0, T, 0, len(centres)]),
                            a_.set_yticks(ticks + .5), a_.set_yticklabels([f"{centres[j]:.0f}" for j in ticks]),
                            a_.set_xlabel("s"), a_.set_ylabel("Hz"), a_.set_title(t))
    img(ax[k], X[i].astype(np.float32), "log-Mel (per-machine normalised)"); k += 1
    if wav is not None:
        sig, sr = wav; ax[k].plot(np.arange(len(sig)) / sr, sig, lw=.4); ax[k].set_title("waveform of this window"); ax[k].set_xlabel("s"); k += 1
    if attr is not None:
        img(ax[k], np.abs(attr), "|Integrated-Gradients attribution|")
    fig.tight_layout()
    return html, fig


# ------------------------------------------------------------------ widget layer
def launch(results_dir, processed_dir, data_root="data/mimii", window_secs=2.0, v1_dir=None, device=None):
    import ipywidgets as W
    from IPython.display import display, clear_output, HTML
    from src.config import Config
    from src.data import load_data

    R = Path(results_dir); res = load_results(R); v1 = load_results(v1_dir) if v1_dir else None
    cfg = Config(processed_dir=str(processed_dir), results_dir=str(R), data_root=str(data_root), window_secs=window_secs)

    # ---- tab 1: results
    out1 = W.Output(); proto = W.ToggleButtons(options=[("v2 (final protocol)", "v2")] + ([("v1 (original)", "v1")] if v1 else []))

    def draw_results(_=None):
        r = res if proto.value == "v2" else v1
        with out1:
            clear_output(wait=True)
            if r["rounds"] is None:
                display(HTML("<b>No all_rounds.csv here - run the analysis stage first.</b>")); return
            tgt = (r["summary"] or {}).get("target_f1")
            fig = curves_figure(r["rounds"], tgt); display(fig); plt.close(fig)
            display(HTML("<h4>Test F1 at each budget (mean ± std over seeds)</h4>")); display(r["table"])
            display(HTML("<h4>Averaged over the whole budget, paired by seed (exploratory)</h4>")); display(auc_table(r["rounds"]))
            if r["final"] is not None:
                display(HTML("<h4>Final budget: cross-machine metrics</h4>")); display(r["final"].set_index(r["final"].columns[0]).T.round(3))
            if r["per_seed"] is not None:
                display(HTML("<h4>Final F1 per seed</h4>")); display(r["per_seed"].round(3))
            if r["exp1"]:
                display(HTML("<h4>Supervised reference and window size (Experiments 1-2)</h4>")); display(exp1_table(r["exp1"]))
    proto.observe(draw_results, names="value"); draw_results()
    tab1 = W.VBox([proto, out1])

    # ---- tab 2: agent log
    out2 = W.Output(); log = res["log"]
    if log:
        sl = W.IntSlider(value=len(log) - 1, min=0, max=len(log) - 1, description="Round", continuous_update=False)

        def draw_agent(_=None):
            with out2:
                clear_output(wait=True); display(HTML(agent_round_html(log, sl.value)))
                fig = agent_figure(log, sl.value); display(fig); plt.close(fig)
        sl.observe(draw_agent, names="value"); draw_agent(); tab2 = W.VBox([sl, out2])
    else:
        tab2 = W.HTML("<b>No agent_log.json in this folder. Run src.run_agent_demo first.</b>")

    # ---- tab 3: segment inspector
    mp = R / "models" / "agent_final.pt"
    if not mp.exists():
        tab3 = W.HTML("<b>No models/agent_final.pt here. Run src.run_agent_demo (torch trainer) first.</b>")
    else:
        import torch
        from src.trainer import load_model
        from src.xai import explain
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, ck = load_model(mp, dev); d = load_data(cfg)
        thr = log[-1].get("thr", 0.5) if log else 0.5
        mel_hz = np.load(Path(processed_dir) / "mel.npz")["hz"]
        pool_state = R / "agent_pool.npz"; rl = np.load(pool_state)["round_labelled"] if pool_state.exists() else None

        def predict(X, bs=512):
            with torch.no_grad():
                return np.concatenate([torch.sigmoid(model(torch.from_numpy((X[j:j + bs].astype(np.float32) - ck["mean"]) / ck["std"]).to(dev))).cpu().numpy()
                                       for j in range(0, len(X), bs)])
        pre = {"pool": predict(d.Xtr), "test": predict(d.Xte)}
        try:
            from src.preprocess import locate, read_wav
            def wav_loader(m, wf):
                sub = "abnormal" if m.label else "normal"
                sr, x = read_wav(locate(cfg, m.machine) / m.id / sub / m.file)
                sig = x.mean(1) if cfg.channel < 0 else x[:, cfg.channel]; s0 = int(m.start_frame) * cfg.hop
                return sig[s0:s0 + wf * cfg.hop], sr
        except Exception:
            wav_loader = None
        split = W.ToggleButtons(options=[("Training pool", "pool"), ("Unseen test machine", "test")])
        idx = W.BoundedIntText(value=0, min=0, max=len(d.Xtr) - 1, description="Index")
        ig = W.Checkbox(value=False, description="Explain (Integrated Gradients)")
        b_f, b_n, b_u = W.Button(description="Random faulty"), W.Button(description="Random normal"), W.Button(description="Most uncertain unlabelled" )
        out3 = W.Output(); rng = np.random.default_rng(0)
        data = lambda: (d.Xtr, d.ytr, d.mtr) if split.value == "pool" else (d.Xte, d.yte, d.mte)

        def draw_seg(_=None):
            X, y, meta = data(); idx.max = len(X) - 1; i = min(idx.value, len(X) - 1)
            status = "n/a (held-out machine)"
            if split.value == "pool" and rl is not None:
                status = "unlabelled (hidden)" if rl[i] < 0 else ("initial seed set" if rl[i] == 0 else f"labelled in round {rl[i]}")
            ex = (lambda x, s: explain(model, x, ck["mean"], ck["std"], cfg, s, dev)) if ig.value else None
            html, fig = render_segment(X, y, meta, pre[split.value], thr, i, status, mel_hz, cfg, ex, wav_loader)
            with out3:
                clear_output(wait=True); display(HTML(html)); display(fig); plt.close(fig)

        def pick(kind):
            def f(_):
                X, y, meta = data(); p = pre[split.value]
                if kind == "unc":
                    ok = (rl < 0) if (split.value == "pool" and rl is not None) else np.ones(len(X), bool)
                    idx.value = int(np.where(ok)[0][np.argmax(entropy(p[ok]))])
                else:
                    idx.value = int(rng.choice(np.where(y == (1 if kind == "f" else 0))[0]))
                draw_seg()
            return f
        b_f.on_click(pick("f")); b_n.on_click(pick("n")); b_u.on_click(pick("unc"))
        for w_ in (split, idx, ig):
            w_.observe(draw_seg, names="value")
        draw_seg()
        tab3 = W.VBox([split, W.HBox([idx, b_f, b_n, b_u]), ig, out3])

    ui = W.Tab(children=[tab1, tab2, tab3])
    for k, t in enumerate(("Random vs active learning", "Agent decisions", "Inspect a segment")):
        ui.set_title(k, t)
    display(ui)
    return ui
