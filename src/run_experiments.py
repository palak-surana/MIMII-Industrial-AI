
"""
Experiments 1-6: Industrial Machine Fault Detection.

exp1:
    1s vs 2s windows, fully supervised CNN-LSTM and Random Forest.

curves:
    Random vs uncertainty-based active learning.
    Validation F1 is used for comparison and budget decisions.

analysis:
    Validation F1 tables, annotation budget, saving estimates,
    statistical analysis and plots.

final_test:
    Evaluate selected final models on the unseen test machine.
    Test data is not used for training, threshold selection,
    active-learning selection or budget decisions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import parse_args
from .data import load_data, describe_split
from .metrics import compute_metrics, choose_threshold
from .active_learning import LabelPool
from .agents import Coordinator, EvaluationBudgetAgent
from . import analysis


def make_factory(cfg, data):
    """Create the selected trainer."""

    if cfg.trainer == "sklearn":
        from .sk_trainer import SklearnTrainer

        return lambda seed: SklearnTrainer(
            cfg, data.mean, data.std, seed
        )

    from .trainer import TorchTrainer

    return lambda seed: TorchTrainer(
        cfg, data.mean, data.std, seed
    )


def mean_metrics(metric_list):
    """Calculate mean of numeric metrics."""

    if not metric_list:
        return {}

    result = {}

    for key in metric_list[0]:

        values = [
            item[key]
            for item in metric_list
            if isinstance(item.get(key), (int, float, np.number))
        ]

        if values:
            result[key] = float(np.nanmean(values))

    return result


def std_metrics(metric_list, keys=("f1", "auroc", "precision", "recall")):
    """Calculate standard deviation across seeds."""

    result = {}

    for key in keys:

        values = [
            item[key]
            for item in metric_list
            if key in item
        ]

        result[key] = (
            float(np.nanstd(values))
            if values else float("nan")
        )

    return result


def save_json(data, path):
    """Save results as strict, standard-compliant JSON."""

    def clean_json(obj):
        if isinstance(obj, dict):
            return {
                key: clean_json(value)
                for key, value in obj.items()
            }

        if isinstance(obj, (list, tuple)):
            return [clean_json(value) for value in obj]

        if isinstance(obj, np.generic):
            return clean_json(obj.item())

        if isinstance(obj, float) and not np.isfinite(obj):
            return None

        return obj

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            clean_json(data),
            file,
            indent=2,
            allow_nan=False,
        )


# ---------------------------------------------------------
# EXPERIMENT 1: WINDOW SIZE COMPARISON
# ---------------------------------------------------------

def exp1(cfg, results_dir):
    """
    Compare 1-second and 2-second windows.

    Window selection uses validation F1.
    Test evaluation is performed only for the selected window.
    """

    from .baseline import rf_baseline

    window_results = {}
    best_window = None
    best_val_f1 = -float("inf")

    for window in cfg.windows:

        data = load_data(cfg, window)
        print(f"\n=== Window {window:g}s ===")
        describe_split(data)

        trainer_factory = make_factory(cfg, data)
        validation_runs = []

        for seed in cfg.seeds:

            trainer = trainer_factory(seed)

            trainer.fit(
                data.Xtr,
                data.ytr,
                data.Xva,
                data.yva,
            )

            validation_metrics = compute_metrics(
                data.yva,
                trainer.predict_proba(data.Xva),
                trainer.thr,
                data.step_sec,
            )

            validation_runs.append({
                "seed": seed,
                "threshold": float(trainer.thr),
                "validation": validation_metrics,
            })

            print(
                f"Seed {seed}: "
                f"Validation F1 = {validation_metrics['f1']:.3f}"
            )

        validation_values = [
            run["validation"] for run in validation_runs
        ]

        mean_val = mean_metrics(validation_values)

        window_results[f"{window:g}s"] = {
            "cnn_lstm": {
                "validation": mean_val,
                "validation_std": std_metrics(validation_values),
                "n_seeds": len(cfg.seeds),
                "per_seed": [
                    {
                        "seed": run["seed"],
                        "threshold": run["threshold"],
                        "validation_f1": run["validation"]["f1"],
                        "validation_auroc": run["validation"]["auroc"],
                    }
                    for run in validation_runs
                ],
            }
        }

        # Random Forest validation baseline.
        rf_predict = rf_baseline(data.Xtr, data.ytr)

        rf_val_probabilities = rf_predict(data.Xva)

        rf_threshold = choose_threshold(
            data.yva,
            rf_val_probabilities,
            cfg.thr_mode,
            cfg.target_spec,
        )

        rf_validation = compute_metrics(
            data.yva,
            rf_val_probabilities,
            rf_threshold,
            data.step_sec,
        )

        window_results[f"{window:g}s"]["random_forest"] = {
            "threshold": float(rf_threshold),
            "validation": rf_validation,
        }

        print(
            f"Random Forest validation F1: "
            f"{rf_validation['f1']:.3f}"
        )

        if mean_val["f1"] > best_val_f1:
            best_val_f1 = mean_val["f1"]
            best_window = window

    if best_window is None:
        raise RuntimeError("No valid window was selected.")

    # -----------------------------------------------------
    # Final test evaluation for the selected window only.
    # -----------------------------------------------------

    print(
        f"\nSelected window using validation F1: "
        f"{best_window:g}s"
    )

    final_data = load_data(cfg, best_window)
    final_factory = make_factory(cfg, final_data)

    cnn_test_runs = []

    for seed in cfg.seeds:

        trainer = final_factory(seed)

        trainer.fit(
            final_data.Xtr,
            final_data.ytr,
            final_data.Xva,
            final_data.yva,
        )

        test_probabilities = trainer.predict_proba(
            final_data.Xte
        )

        test_metrics = compute_metrics(
            final_data.yte,
            test_probabilities,
            trainer.thr,
            final_data.step_sec,
        )

        cnn_test_runs.append({
            "seed": seed,
            "test": test_metrics,
        })

        print(
            f"Final CNN-LSTM test | Seed {seed} | "
            f"F1: {test_metrics['f1']:.3f} | "
            f"AUROC: {test_metrics['auroc']:.3f}"
        )

    # Final Random Forest evaluation for selected window.
    rf_predict = rf_baseline(
        final_data.Xtr,
        final_data.ytr,
    )

    rf_val_probabilities = rf_predict(final_data.Xva)

    rf_threshold = choose_threshold(
        final_data.yva,
        rf_val_probabilities,
        cfg.thr_mode,
        cfg.target_spec,
    )

    rf_test_probabilities = rf_predict(final_data.Xte)

    rf_test_metrics = compute_metrics(
        final_data.yte,
        rf_test_probabilities,
        rf_threshold,
        final_data.step_sec,
    )

    output = {
        "window_comparison_validation": window_results,
        "selected_window": float(best_window),
        "selection_metric": "mean validation F1",
        "final_test_evaluation": {
            "cnn_lstm": cnn_test_runs,
            "random_forest": {
                "threshold": float(rf_threshold),
                "test": rf_test_metrics,
            },
        },
    }

    save_json(
        output,
        results_dir / "exp1_windows.json",
    )

    print("\nExperiment 1 completed.")
    print(f"Selected window: {best_window:g}s")
    print(f"Mean validation F1: {best_val_f1:.3f}")


# ---------------------------------------------------------
# EXPERIMENTS 3 AND 4: ACTIVE LEARNING CURVES
# ---------------------------------------------------------

def curves(cfg, results_dir):
    """
    Compare random and uncertainty sampling.

    No test data is accessed in this function.
    """

    data = load_data(cfg)
    describe_split(data)

    trainer_factory = make_factory(cfg, data)
    logs = []

    for seed in cfg.seeds:

        for strategy in ("random", "uncertainty"):

            print(
                f"\nStarting {strategy} active learning "
                f"with seed {seed}"
            )

            pool = LabelPool(
                data.ytr,
                cfg.init_frac,
                seed,
            )

            budget = EvaluationBudgetAgent(
                cfg.max_frac,
                cfg.batch_frac,
                cfg.target_f1,
                stop_on_target=False,
            )

            coordinator = Coordinator(
                data=data,
                pool=pool,
                trainer_factory=trainer_factory,
                strategy=strategy,
                budget_agent=budget,
                seed=seed,
                verbose=True,
            )

            strategy_logs = coordinator.run()

            logs.extend(strategy_logs)

            final_round = strategy_logs[-1]

            print(
                f"\nCompleted {strategy}, seed {seed}"
                f"\nFinal labelled fraction: "
                f"{final_round['frac_labelled'] * 100:.1f}%"
                f"\nFinal validation F1: "
                f"{final_round['val_f1']:.4f}"
            )

    output_path = (
        results_dir /
        f"curves_w{cfg.window_secs:g}s.json"
    )

    save_json(logs, output_path)

    print(f"\nActive-learning logs saved: {output_path}")


# ---------------------------------------------------------
# EXPERIMENT 5: VALIDATION ANALYSIS
# ---------------------------------------------------------

def run_analysis(cfg, results_dir):
    """
    Analyze active-learning curves using validation data only.
    """

    logs_path = (
        results_dir /
        f"curves_w{cfg.window_secs:g}s.json"
    )

    if not logs_path.exists():
        raise FileNotFoundError(
            f"Experiment logs not found: {logs_path}. "
            "Run the curves stage first."
        )

    with open(logs_path, "r", encoding="utf-8") as file:
        logs = json.load(file)

    df, rows, summary = analysis.analyse(
        logs,
        cfg.target_f1,
        results_dir,
    )

    print("\n==========================================")
    print("VALIDATION F1 VS ANNOTATION BUDGET")
    print("==========================================")

    print(
        f"{'Labelled':12s}"
        f"{'Random':22s}"
        f"{'Agentic AL':22s}"
    )

    for row in rows:

        print(
            f"{row['Labelled Data']:12s}"
            f"{row['Random Selection F1']:22s}"
            f"{row['Agentic Active Learning F1']:22s}"
        )

    target_result = summary["labels_to_target_validation"]

    print("\nTarget F1 analysis:")
    print(f"Target F1: {cfg.target_f1:.2f}")
    print(f"Random labels required: {target_result['random_pct']}")
    print(f"Active labels required: {target_result['active_pct']}")
    print(
        f"Annotation saving: "
        f"{target_result['annotation_saving_pct']}"
    )

    print("\nValidation-based analysis completed.")
    print(f"Results saved in: {results_dir}")

    return df, rows, summary


# ---------------------------------------------------------
# EXPERIMENT 6: FINAL UNSEEN MACHINE EVALUATION
# ---------------------------------------------------------

def final_test_evaluation(cfg, results_dir):
    """
    Evaluate each strategy at maximum annotation budget.

    The test machine is not used for selection, training,
    threshold selection, or active-learning decisions.
    """

    print("\n==========================================")
    print("FINAL UNSEEN-MACHINE TEST EVALUATION")
    print("==========================================")

    data = load_data(cfg)
    describe_split(data)

    trainer_factory = make_factory(cfg, data)

    final_results = {
        "evaluation": "final held-out machine evaluation",
        "test_machine_ids": cfg.test_ids,
        "maximum_annotation_fraction": cfg.max_frac,
        "strategies": {},
    }

    for seed in cfg.seeds:

        for strategy in ("random", "uncertainty"):

            print(
                f"\nFinal evaluation: {strategy}, seed {seed}"
            )

            pool = LabelPool(
                data.ytr,
                cfg.init_frac,
                seed,
            )

            budget = EvaluationBudgetAgent(
                cfg.max_frac,
                cfg.batch_frac,
                cfg.target_f1,
                stop_on_target=False,
            )

            coordinator = Coordinator(
                data=data,
                pool=pool,
                trainer_factory=trainer_factory,
                strategy=strategy,
                budget_agent=budget,
                seed=seed,
                verbose=False,
            )

            # Active learning uses training and validation only.
            coordinator.run()

            trainer = coordinator.trainer

            if trainer is None:
                raise RuntimeError(
                    "Final trained model was not available."
                )

            # Test data is accessed only after training and selection.
            test_probabilities = trainer.predict_proba(
                data.Xte
            )

            test_metrics = compute_metrics(
                data.yte,
                test_probabilities,
                trainer.thr,
                data.step_sec,
            )

            final_results["strategies"].setdefault(
                strategy, []
            )

            final_results["strategies"][strategy].append({
                "seed": seed,
                "labelled_fraction": pool.frac_labelled,
                "threshold": float(trainer.thr),
                "test_metrics": test_metrics,
            })

            print(
                f"Test F1: {test_metrics['f1']:.4f} | "
                f"Precision: {test_metrics['precision']:.4f} | "
                f"Recall: {test_metrics['recall']:.4f} | "
                f"Specificity: {test_metrics['specificity']:.4f} | "
                f"AUROC: {test_metrics['auroc']:.4f}"
            )

    save_json(
        final_results,
        results_dir / "final_test_evaluation.json",
    )

    print(
        "\nFinal test results saved to "
        "final_test_evaluation.json"
    )

    return final_results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main(argv=None):

    cfg = parse_args(
        "Run industrial machine fault detection experiments",
        argv,
    )

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.stage in ("exp1", "all"):
        exp1(cfg, results_dir)

    if cfg.stage in ("curves", "all"):
        curves(cfg, results_dir)

    if cfg.stage in ("analysis", "all"):

        run_analysis(cfg, results_dir)

        final_test_evaluation(
            cfg,
            results_dir,
        )


if __name__ == "__main__":
    main()
