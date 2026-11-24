"""Logic tests (no PyTorch needed): pool / selection / agents / budget / analysis on hard synthetic data.
Run:  python -m tests.test_logic"""
import numpy as np, pandas as pd
from src.config import Config
from src.data import Data
from src.sk_trainer import SklearnTrainer
from src.active_learning import LabelPool, entropy, select_uncertain
from src.agents import Coordinator, EvaluationBudgetAgent
from src.analysis import analyse


def make(n, rng, shift):
    y = (rng.random(n) < 0.2).astype(int)
    X = rng.normal(0, 1, (n, 8, 10)) + shift
    X[:, 2, :] += y[:, None] * rng.normal(1.0, 0.8, n)[:, None]       # weak, noisy fault signature
    meta = pd.DataFrame(dict(machine="pump", id="id_00", label=y, file="a.wav", start_frame=0, step_sec=0.5))
    return X.astype(np.float32), y, meta


def build():
    rng = np.random.default_rng(0)
    (Xtr, ytr, mtr), (Xva, yva, mva), (Xte, yte, mte) = make(1500, rng, 0), make(500, rng, .3), make(500, rng, -.3)
    return Data(Xtr, ytr, mtr, Xva, yva, mva, Xte, yte, mte, Xtr.mean((0, 2))[:, None], Xtr.std((0, 2))[:, None] + 1e-6, 0.5)


def test_entropy_and_selection():
    p = np.array([0.98, 0.52, 0.04])                      # the example from the assignment
    idx, u = select_uncertain(np.array([10, 11, 12]), p, 1)
    assert idx[0] == 11 and entropy(0.5) > entropy(0.98) > entropy(0.999999)


def test_pool_no_leakage():
    y = (np.random.default_rng(1).random(1000) < .2).astype(int)
    pool = LabelPool(y, 0.1, seed=3)
    assert pool.n_labelled == 100 and 0 < y[pool.labelled_idx].mean() < 0.5
    try:
        pool.labels_of_labelled(pool.unlabelled_idx[:5]); raise SystemExit("leak!")
    except AssertionError:
        pass
    pool.annotate(pool.unlabelled_idx[:50], 1)
    assert pool.n_labelled == 150


def test_agents_and_budget():
    d = build(); fac = lambda s: SklearnTrainer(None, d.mean, d.std, s); logs = []
    for seed in range(3):
        for strat in ("random", "uncertainty"):
            pool = LabelPool(d.ytr, 0.1, seed); first = set(pool.labelled_idx)
            c = Coordinator(d, pool, fac, strat, EvaluationBudgetAgent(0.5, 0.05, 0.8, stop_on_target=False), seed, verbose=False)
            log = c.run(); logs += log
            assert log[0]["n_labelled"] == 150 and log[-1]["n_labelled"] == 750          # stops exactly at 50 %
            assert [r["n_labelled"] for r in log] == list(range(150, 751, 75))
            assert first <= set(pool.labelled_idx)
    df, rows, res = analyse(logs, 0.7, "results_test")
    print(*[(r["Labelled Data"], r["Random Selection F1"], r["Agentic Active Learning F1"]) for r in rows], sep="\n")
    print(res)
    # stop-on-target mode
    pool = LabelPool(d.ytr, 0.1, 0)
    log = Coordinator(d, pool, fac, "uncertainty", EvaluationBudgetAgent(0.5, 0.05, 0.01, True), 0, verbose=True).run()
    assert len(log) == 1 and "target" in log[0]["decision"]


if __name__ == "__main__":
    test_entropy_and_selection(); test_pool_no_leakage(); test_agents_and_budget(); print("\nALL TESTS PASSED")
