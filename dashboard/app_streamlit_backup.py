"""Streamlit dashboard.   streamlit run dashboard/app.py -- --window_secs 1.0
(arguments after `--` are the same as every other script)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.config import parse_args
from src.active_learning import entropy

cfg = parse_args("dashboard", sys.argv[1:]); R = Path(cfg.results_dir)
st.set_page_config(page_title="Agentic Active Learning - Machine Faults", layout="wide")
st.title("Annotation-Efficient Machine Fault Detection (CNN-LSTM + Agentic Active Learning)")
page = st.sidebar.radio("View", ["Random vs Active Learning", "Agent decisions", "Inspect a segment"])


@st.cache_resource
def get_data():
    from src.data import load_data
    return load_data(cfg)


if page == "Random vs Active Learning":
    summ = json.load(open(R / "summary.json")) if (R / "summary.json").exists() else None
    if not summ:
        st.warning("Run `python -m src.run_experiments --stage all` first.")
    else:
        t = summ["labels_to_target_test"]
        c = st.columns(4); c[0].metric("Target F1", summ["target_f1"])
        c[1].metric("Labels needed - Random", f"{t['random_pct']:.0f}%"); c[2].metric("Labels needed - Active", f"{t['active_pct']:.0f}%")
        c[3].metric("Labels saved", f"{t['annotation_saving_pct']:.1f}%")
        l, r = st.columns(2)
        l.image(str(R / "f1_curve.png")); r.dataframe(pd.read_csv(R / "table_f1.csv"), hide_index=True)
        if (R / "exp1_windows.json").exists():
            st.subheader("Experiment 1/2 - window size & supervised reference"); st.json(json.load(open(R / "exp1_windows.json")), expanded=False)
        if (R / "exp6_final_cross_machine.csv").exists():
            st.subheader("Experiment 6 - cross-machine metrics at max budget"); st.dataframe(pd.read_csv(R / "exp6_final_cross_machine.csv"))

elif page == "Agent decisions":
    if not (R / "agent_log.json").exists():
        st.warning("Run `python -m src.run_agent_demo` first.")
    else:
        log = json.load(open(R / "agent_log.json")); rnd = st.slider("Round", 0, len(log) - 1, len(log) - 1); r = log[rnd]
        c = st.columns(5); c[0].metric("Round", r["round"]); c[1].metric("Labelled", f"{r['n_labelled']:,}")
        c[2].metric("Unlabelled", f"{r['n_unlabelled']:,}"); c[3].metric("Validation F1", f"{r['val_f1']:.2f}"); c[4].metric("Target F1", r["target_f1"])
        st.info(f"**Agent decision:** {r['decision']}  \n**Selection method:** {r['selection_method']}")
        if r.get("candidates"):
            st.write("Top candidates sent to the simulated maintenance engineer"); st.dataframe(pd.DataFrame(r["candidates"]), hide_index=True)
        st.line_chart(pd.DataFrame({"val F1": [x["val_f1"] for x in log], "test F1 (unseen machine)": [x["test"]["f1"] for x in log]},
                                   index=[x["n_labelled"] for x in log]))

else:
    from src.trainer import load_model
    import torch
    d = get_data(); mp = R / "models" / "agent_final.pt"
    if not mp.exists():
        st.warning("Run `python -m src.run_agent_demo` (torch trainer) first.")
    else:
        model, ck = load_model(mp)
        split = st.radio("Split", ["Training pool (annotation status shown)", "Unseen test machine"], horizontal=True)
        pool_view = split.startswith("Training")
        X, y, meta = (d.Xtr, d.ytr, d.mtr) if pool_view else (d.Xte, d.yte, d.mte)
        i = st.number_input("Segment index", 0, len(X) - 1, 0)
        with torch.no_grad():
            p = float(torch.sigmoid(model(torch.from_numpy((X[i:i + 1].astype(np.float32) - ck["mean"]) / ck["std"])))[0])
        m = meta.iloc[i]
        status = "n/a (held-out machine)"
        if pool_view and (R / "agent_pool.npz").exists():
            rl = np.load(R / "agent_pool.npz")["round_labelled"][i]
            status = "unlabelled (hidden)" if rl < 0 else ("initial labelled seed" if rl == 0 else f"labelled in round {rl}")
        c = st.columns(6); c[0].metric("Machine", f"{m.machine} / {m.id}"); c[1].metric("True label", "faulty" if y[i] else "healthy")
        c[2].metric("Predicted", "faulty" if p >= .5 else "healthy"); c[3].metric("P(faulty)", f"{p:.2f}")
        c[4].metric("Uncertainty", f"{float(entropy(p)):.2f}"); c[5].metric("Annotation", status)
        fig, ax = plt.subplots(figsize=(7, 3)); ax.imshow(X[i].astype(np.float32), origin="lower", aspect="auto"); ax.set_xlabel("frame"); ax.set_ylabel("Mel band")
        st.pyplot(fig)
        if st.checkbox("Explain with Integrated Gradients"):
            from src.xai import explain
            _, attr, s = explain(model, X[i], ck["mean"], ck["std"], cfg, int(m.start_frame))
            st.write("Influential bands (Hz):", ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in s["bands_hz"]),
                     "| interval: %.1f-%.1f s" % s["interval_s"])
            fig, ax = plt.subplots(figsize=(7, 3)); ax.imshow(np.abs(attr), origin="lower", aspect="auto", cmap="magma"); st.pyplot(fig)
