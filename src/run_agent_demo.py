"""Runs ONE full agentic cycle with early stopping at the target validation F1 and saves everything the dashboard needs:
   results/agent_log.json, results/agent_pool.npz, results/models/agent_final.pt
   python -m src.run_agent_demo --target_f1 0.80 --max_frac 0.5"""
from pathlib import Path
import numpy as np
from .config import parse_args
from .data import load_data, describe_split
from .active_learning import LabelPool
from .agents import Coordinator, EvaluationBudgetAgent, save_log
from .run_experiments import make_factory


def main(argv=None):
    cfg = parse_args("Agentic active-learning demo", argv)
    res = Path(cfg.results_dir); (res / "models").mkdir(parents=True, exist_ok=True)
    d = load_data(cfg); describe_split(d)
    pool = LabelPool(d.ytr, cfg.init_frac, cfg.seeds[0])
    coord = Coordinator(d, pool, make_factory(cfg, d), "uncertainty",
                        EvaluationBudgetAgent(cfg.max_frac, cfg.batch_frac, cfg.target_f1, stop_on_target=True), seed=cfg.seeds[0])
    log = coord.run(); save_log(log, res / "agent_log.json")
    np.savez(res / "agent_pool.npz", round_labelled=pool.round_labelled)
    if cfg.trainer == "torch":
        coord.trainer.save(res / "models" / "agent_final.pt")
    print(f"\nFinished: {log[-1]['decision']} | labels used {pool.n_labelled}/{pool.n} ({pool.frac_labelled*100:.0f}%)")


if __name__ == "__main__":
    main()
