"""Tests the dashboard's plain rendering functions (no widgets/torch). Run: python -m tests.test_dashboard <results_dir> <processed_dir>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
from src.config import Config
from src.data import load_data
from dashboard.colab_dashboard import *

R, P = sys.argv[1], sys.argv[2]
res = load_results(R); cfg = Config(processed_dir=P, window_secs=1.0)
fig = curves_figure(res["rounds"], 0.9); print("curves ok", len(fig.axes))
print(auc_table(res["rounds"]).to_string(index=False))
print(exp1_table(res["exp1"]).to_string(index=False))
log = res["log"]; h = agent_round_html(log, 0); assert "Next action" in h; agent_figure(log, len(log) - 1); print("agent ok", len(log), "rounds")
d = load_data(cfg); mel = np.load(f"{P}/mel.npz")["hz"]; rng = np.random.default_rng(0)
probs = rng.random(len(d.Xte))
fake_explain = lambda x, s: (0.9, rng.normal(size=x.shape), dict(bands_hz=[(100., 200.)], interval_s=(1., 1.5)))
html, fig = render_segment(d.Xte, d.yte, d.mte, probs, 0.5, 3, "n/a (held-out machine)", mel, cfg, fake_explain, None)
assert "Integrated" in html and len(fig.axes) == 2
html, fig = render_segment(d.Xtr, d.ytr, d.mtr, rng.random(len(d.Xtr)), 0.5, 0, "labelled in round 1", mel, cfg, None,
                           lambda m, wf: (np.sin(np.arange(wf * cfg.hop) / 50), cfg.sr))
assert len(fig.axes) == 2; print("segment ok"); print("ALL DASHBOARD TESTS PASSED")
