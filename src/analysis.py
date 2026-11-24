
"""Experiments 3-6 analysis.

Active-learning comparisons use VALIDATION metrics only.
The unseen test machine is reserved for final evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


FRACS = (0.10, 0.20, 0.30, 0.40, 0.50)


def to_df(logs):
    """Convert coordinator logs into a validation-only DataFrame."""

    required = (
        "strategy",
        "seed",
        "round",
        "n_labelled",
        "frac_labelled",
        "val_f1",
    )

    rows = []

    for record in logs:
        missing = [key for key in required if key not in record]

        if missing:
            raise ValueError(
                f"Experiment log is missing fields: {missing}"
            )

        rows.append({key: record[key] for key in required})

    if not rows:
        raise ValueError("No experiment logs were provided.")

    return pd.DataFrame(rows)


def budget_table(df, col="val_f1", fracs=FRACS, tol=0.026):
    """Compare random and uncertainty strategies at annotation budgets."""

    rows = []

    for fraction in fracs:

        row = {
            "Labelled Data": f"{int(fraction * 100)}%"
        }

        for strategy, name in (
            ("random", "Random Selection"),
            ("uncertainty", "Agentic Active Learning"),
        ):

            values = []

            for _, group in df[df.strategy == strategy].groupby("seed"):

                index = (group.frac_labelled - fraction).abs().idxmin()

                actual_fraction = group.loc[index, "frac_labelled"]

                if abs(actual_fraction - fraction) <= tol:
                    values.append(float(group.loc[index, col]))

            row[f"{name} F1"] = (
                f"{np.mean(values):.3f} ± {np.std(values):.3f}"
                if values
                else "n/a"
            )

            row[f"_{strategy}"] = values

        rows.append(row)

    return rows


def labels_to_target(df, strategy, target, col="val_f1"):
    """First labelled fraction where the seed-averaged validation curve reaches target."""

    group = (
        df[df.strategy == strategy]
        .groupby("round")
        .agg(
            fraction=("frac_labelled", "mean"),
            score=(col, "mean"),
        )
    )

    hit = group[group.score >= target]

    if hit.empty:
        return float("nan")

    return float(hit.fraction.iloc[0])


def sustained(curve, fractions, target):
    """First fraction from which the average curve remains above target."""

    valid = np.asarray(curve) >= target

    for index in range(len(valid)):

        if valid[index:].all():
            return float(fractions[index])

    return float("nan")


def saving(labels_random, labels_agent):
    """Calculate percentage annotation saving."""

    if (
        not np.isfinite(labels_random)
        or not np.isfinite(labels_agent)
        or labels_random <= 0
    ):
        return float("nan")

    return (
        (labels_random - labels_agent)
        / labels_random
        * 100
    )


def sustained_saving(df, col="val_f1", target=0.80, B=2000, seed=0):
    """Bootstrap annotation-saving estimates over paired seeds."""

    matrices = {}

    for strategy in ("random", "uncertainty"):

        subset = df[df.strategy == strategy]

        matrices[strategy] = (
            subset.pivot(
                index="seed",
                columns="round",
                values=col,
            )
            .sort_index()
        )

    if not matrices["random"].index.equals(
        matrices["uncertainty"].index
    ):
        raise ValueError(
            "Random and uncertainty runs must use matching seeds."
        )

    if not matrices["random"].columns.equals(
        matrices["uncertainty"].columns
    ):
        raise ValueError(
            "Random and uncertainty runs must have matching rounds."
        )

    random_matrix = matrices["random"].to_numpy()
    active_matrix = matrices["uncertainty"].to_numpy()

    if (
        not np.isfinite(random_matrix).all()
        or not np.isfinite(active_matrix).all()
    ):
        raise ValueError(
            "Validation curves contain missing values. "
            "Check experiment logs before bootstrapping."
        )

    fractions = (
        df.groupby("round")
        .frac_labelled.mean()
        .sort_index()
        .to_numpy()
    )

    random_required = sustained(
        random_matrix.mean(axis=0),
        fractions,
        target,
    )

    active_required = sustained(
        active_matrix.mean(axis=0),
        fractions,
        target,
    )

    rng = np.random.default_rng(seed)
    bootstrap_savings = []

    for _ in range(B):

        indices = rng.integers(
            0,
            len(random_matrix),
            len(random_matrix),
        )

        random_fraction = sustained(
            random_matrix[indices].mean(axis=0),
            fractions,
            target,
        )

        active_fraction = sustained(
            active_matrix[indices].mean(axis=0),
            fractions,
            target,
        )

        value = saving(random_fraction, active_fraction)

        if np.isfinite(value):
            bootstrap_savings.append(value)

    result = {
        "metric": col,
        "target": float(target),
        "random_pct": (
            random_required * 100
            if np.isfinite(random_required)
            else None
        ),
        "active_pct": (
            active_required * 100
            if np.isfinite(active_required)
            else None
        ),
        "valid_bootstraps": len(bootstrap_savings),
        "B": B,
    }

    if bootstrap_savings:

        result.update({
            "saving_median_pct": float(
                np.median(bootstrap_savings)
            ),
            "saving_ci95_pct": [
                float(np.percentile(bootstrap_savings, 2.5)),
                float(np.percentile(bootstrap_savings, 97.5)),
            ],
        })

    return result


def plot_curves(
    df,
    target,
    path,
    col="val_f1",
    title="Validation F1 vs annotation budget",
):
    """Plot validation performance without accessing test metrics."""

    plt.figure(figsize=(7, 4.5))

    for strategy, name, color in (
        ("random", "Random Selection", "tab:blue"),
        ("uncertainty", "Agentic Active Learning", "tab:red"),
    ):

        group = (
            df[df.strategy == strategy]
            .groupby("round")
            .agg(
                x=("frac_labelled", "mean"),
                mean=(col, "mean"),
                std=(col, "std"),
            )
            .fillna(0)
        )

        plt.plot(
            group.x * 100,
            group["mean"],
            "-o",
            color=color,
            label=name,
        )

        plt.fill_between(
            group.x * 100,
            group["mean"] - group["std"],
            group["mean"] + group["std"],
            color=color,
            alpha=0.15,
        )

    plt.axhline(
        target,
        linestyle="--",
        color="gray",
        label=f"Target F1 = {target}",
    )

    plt.xlabel("Labelled training data (%)")
    plt.ylabel("Validation F1-score")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(path, dpi=150)
    plt.close()


def analyse(logs, target, out_dir, auroc_targets=None):
    """Analyse active-learning experiments using validation results only."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = to_df(logs)

    df.to_csv(
        out / "all_rounds.csv",
        index=False,
    )

    rows = budget_table(df, col="val_f1")

    pd.DataFrame([
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in rows
    ]).to_csv(
        out / "table_f1.csv",
        index=False,
    )

    # Paired exploratory significance tests across seeds.
    significance = {}

    for row in rows:

        random_values = row["_random"]
        active_values = row["_uncertainty"]

        if (
            len(random_values) == len(active_values)
            and len(random_values) >= 3
        ):

            difference = (
                np.asarray(active_values)
                - np.asarray(random_values)
            )

            # A zero-variance paired difference has no valid
            # t-statistic. Store None instead of NaN.
            if difference.std() > 0:
                test_result = stats.ttest_rel(
                    active_values,
                    random_values,
                )
                p_value = (
                    float(test_result.pvalue)
                    if np.isfinite(test_result.pvalue)
                    else None
                )
            else:
                p_value = None

            significance[row["Labelled Data"]] = {
                "mean_validation_f1_gain": float(
                    difference.mean()
                ),
                "paired_ttest_p_value": p_value,
                "n_seeds": len(random_values),
            }

    result = {
        "primary_metric": "validation_f1",
        "target_f1": float(target),
        "significance_paired_ttest": significance,
    }

    random_labels = labels_to_target(
        df,
        "random",
        target,
    )

    active_labels = labels_to_target(
        df,
        "uncertainty",
        target,
    )

    annotation_saving = saving(
        random_labels,
        active_labels,
    )

    result["labels_to_target_validation"] = {
        "random_pct": (
            random_labels * 100
            if np.isfinite(random_labels)
            else None
        ),
        "active_pct": (
            active_labels * 100
            if np.isfinite(active_labels)
            else None
        ),
        "annotation_saving_pct": (
            float(annotation_saving)
            if np.isfinite(annotation_saving)
            else None
        ),
    }

    result["sustained_validation_f1_target"] = (
        sustained_saving(
            df,
            col="val_f1",
            target=target,
        )
    )

    plot_curves(
        df,
        target,
        out / "val_f1_curve.png",
    )

    # Strict JSON: NaN and Infinity are not allowed.
    with open(
        out / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            allow_nan=False,
        )

    return df, rows, result
