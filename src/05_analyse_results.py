"""Analyse the cumulative model results for Évora.

This script reads the metrics, predictions, and processed target history from
the `results/` and `data/` folders beside this script. It compares
feature representations A and B, estimates uncertainty with a moving-block
bootstrap, and examines model performance over time.

Workflow:
1. Load the metrics, evaluation predictions, and processed target history.
2. Verify that both feature representations use the same evaluation rows.
3. Compare A and B, including bootstrap confidence intervals.
4. Summarise monthly Macro-F1 scores.
5. Save comparison tables and figures for reporting.

Outputs:
- `results/analysis_evora_A_vs_B.csv`: model comparison between representations
- `results/analysis_evora_bootstrap.csv`: bootstrap estimates and confidence intervals
- `results/analysis_evora_monthly_macro_f1.csv`: monthly Macro-F1 results
- `results/figures/evora_fig6_A_vs_B.png` through `evora_fig9_monthly_macro_f1.png`

The block bootstrap uses one-week blocks by default to retain short-term temporal
dependence. This script analyses existing results; it does not train models.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

CITY = "evora"
BLOCK_H = 168
N_BOOT = 500
SEED = 42
MODELS = [
    "Gaussian Naive Bayes",
    "Online Logistic Regression",
    "Hoeffding Tree",
    "Online Bagging",
    "Adaptive Random Forest",
]
SHORT = {
    "Gaussian Naive Bayes": "GNB",
    "Online Logistic Regression": "LogReg",
    "Hoeffding Tree": "HT",
    "Online Bagging": "Bagging",
    "Adaptive Random Forest": "ARF",
    "Baseline: persistence": "persistence",
    "Baseline: persistence (y_t-1)": "persistence",
}


def ensure_directories() -> None:
    """Create the results and figure output directories if needed."""
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """Load model metrics, predictions, and the aligned persistence baseline.

    Raises:
        ValueError: If representations A and B do not share the same evaluation
            timestamps and target labels.

    Returns:
        Metrics, prediction tables keyed by representation, true labels, and
        persistence-baseline predictions.
    """
    metrics = pd.read_csv(RESULTS_DIR / f"metrics_{CITY}.csv")
    predictions = {
        rep: pd.read_csv(RESULTS_DIR / f"predictions_{CITY}_rep{rep}.csv", parse_dates=["time"])
        for rep in ("A", "B")
    }

    if not predictions["A"]["time"].equals(predictions["B"]["time"]) or not np.array_equal(
        predictions["A"]["y_true"].to_numpy(),
        predictions["B"]["y_true"].to_numpy(),
    ):
        raise ValueError("Representations A and B must have the same evaluation timestamps and labels.")

    y_true = predictions["A"]["y_true"].to_numpy()
    if len(y_true) == 0:
        raise ValueError("Prediction files contain no evaluation rows.")

    processed = pd.read_csv(DATA_DIR / f"processed_{CITY}.csv", parse_dates=["time"])
    target_history = processed.set_index("time")["target_T6"]
    persistence = target_history.shift(1).reindex(predictions["A"]["time"]).to_numpy(dtype=int)
    return metrics, predictions, y_true, persistence


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return true-positive, true-negative, false-positive, and false-negative counts."""
    return (
        int(((y_true == 1) & (y_pred == 1)).sum()),
        int(((y_true == 0) & (y_pred == 0)).sum()),
        int(((y_true == 0) & (y_pred == 1)).sum()),
        int(((y_true == 1) & (y_pred == 0)).sum()),
    )


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Macro-F1 for the binary event and non-event classes."""
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)

    def divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision_event = divide(tp, tp + fp)
    recall_event = divide(tp, tp + fn)
    precision_nonevent = divide(tn, tn + fn)
    recall_nonevent = divide(tn, tn + fp)
    f1_event = divide(2 * precision_event * recall_event, precision_event + recall_event)
    f1_nonevent = divide(
        2 * precision_nonevent * recall_nonevent,
        precision_nonevent + recall_nonevent,
    )
    return (f1_event + f1_nonevent) / 2


def compare_representations(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build and save the per-model comparison of feature sets A and B."""
    rows = []
    for model in MODELS:
        result_a = metrics[(metrics["model"] == model) & (metrics["rep"] == "A")].iloc[0]
        result_b = metrics[(metrics["model"] == model) & (metrics["rep"] == "B")].iloc[0]
        rows.append(
            {
                "model": model,
                "macro_f1_A": result_a["macro_f1"],
                "macro_f1_B": result_b["macro_f1"],
                "delta_macro_f1": result_b["macro_f1"] - result_a["macro_f1"],
                "recall_A": result_a["recall_event"],
                "recall_B": result_b["recall_event"],
                "precision_A": result_a["precision_event"],
                "precision_B": result_b["precision_event"],
                "update_ms_A": result_a["update_ms_per_event"],
                "update_ms_B": result_b["update_ms_per_event"],
                "update_cost_ratio_B_over_A": (
                    result_b["update_ms_per_event"] / result_a["update_ms_per_event"]
                ),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / f"analysis_{CITY}_A_vs_B.csv", index=False)
    print(comparison.round(4).to_string(index=False))
    return comparison


def bootstrap_comparisons(
    predictions: dict[str, pd.DataFrame],
    y_true: np.ndarray,
    persistence: np.ndarray,
) -> pd.DataFrame:
    """Estimate Macro-F1 confidence intervals using a moving-block bootstrap."""
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    block_size = min(BLOCK_H, n)
    n_blocks = int(np.ceil(n / block_size))
    series = {
        (rep, model): predictions[rep][model].to_numpy()
        for rep in ("A", "B")
        for model in MODELS
    }

    def sample_indices() -> np.ndarray:
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        return (starts[:, None] + np.arange(block_size)).ravel()[:n]

    full = {key: macro_f1(y_true, values) for key, values in series.items()}
    full_persistence = macro_f1(y_true, persistence)
    boot = {key: np.empty(N_BOOT) for key in series}
    boot_persistence = np.empty(N_BOOT)

    for iteration in range(N_BOOT):
        indices = sample_indices()
        y_sample = y_true[indices]
        boot_persistence[iteration] = macro_f1(y_sample, persistence[indices])
        for key, values in series.items():
            boot[key][iteration] = macro_f1(y_sample, values[indices])

    def interval(values: np.ndarray) -> tuple[float, float]:
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    rows = []
    for model in MODELS:
        for rep in ("A", "B"):
            low, high = interval(boot[(rep, model)])
            rows.append(
                {
                    "comparison": "Macro-F1",
                    "model": model,
                    "rep": rep,
                    "estimate": full[(rep, model)],
                    "ci_low": low,
                    "ci_high": high,
                }
            )

        low, high = interval(boot[("B", model)] - boot[("A", model)])
        rows.append(
            {
                "comparison": "B minus A",
                "model": model,
                "rep": "-",
                "estimate": full[("B", model)] - full[("A", model)],
                "ci_low": low,
                "ci_high": high,
            }
        )

        for rep in ("A", "B"):
            low, high = interval(boot[(rep, model)] - boot_persistence)
            rows.append(
                {
                    "comparison": "model minus persistence",
                    "model": model,
                    "rep": rep,
                    "estimate": full[(rep, model)] - full_persistence,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    low, high = interval(boot_persistence)
    rows.append(
        {
            "comparison": "Macro-F1",
            "model": "Baseline: persistence",
            "rep": "-",
            "estimate": full_persistence,
            "ci_low": low,
            "ci_high": high,
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / f"analysis_{CITY}_bootstrap.csv", index=False)
    print("\n", result.round(4).to_string(index=False))
    return result


def save_monthly_scores(
    predictions: dict[str, pd.DataFrame],
    y_true: np.ndarray,
    persistence: np.ndarray,
) -> pd.DataFrame:
    """Calculate and save monthly Macro-F1 scores for both representations."""
    months = predictions["A"]["time"].dt.to_period("M").astype(str).to_numpy()
    rows = []
    for rep in ("A", "B"):
        for month in np.unique(months):
            selected = months == month
            for model in MODELS:
                rows.append(
                    {
                        "rep": rep,
                        "month": month,
                        "model": model,
                        "macro_f1": macro_f1(
                            y_true[selected],
                            predictions[rep][model].to_numpy()[selected],
                        ),
                    }
                )
            rows.append(
                {
                    "rep": rep,
                    "month": month,
                    "model": "Baseline: persistence",
                    "macro_f1": macro_f1(y_true[selected], persistence[selected]),
                }
            )

    monthly = pd.DataFrame(rows)
    monthly.to_csv(RESULTS_DIR / f"analysis_{CITY}_monthly_macro_f1.csv", index=False)
    return monthly


def save_figures(metrics: pd.DataFrame, predictions: dict[str, pd.DataFrame], monthly: pd.DataFrame) -> None:
    """Generate the representation, confusion-matrix, cost, and monthly plots."""
    positions = np.arange(len(MODELS))
    bar_width = 0.38
    labels = [SHORT[model] for model in MODELS]

    persistence_metrics = metrics[metrics["model"] == "Baseline: persistence (y_t-1)"].iloc[0]
    majority_metrics = metrics[metrics["model"] == "Baseline: majority class"].iloc[0]

    # Compare Macro-F1 and event recall for representations A and B.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, (column, title) in zip(
        axes,
        [("macro_f1", "Macro-F1"), ("recall_event", "Event-class recall")],
    ):
        for index, rep in enumerate(("A", "B")):
            values = [
                metrics[(metrics["model"] == model) & (metrics["rep"] == rep)][column].iloc[0]
                for model in MODELS
            ]
            axis.bar(
                positions + (index - 0.5) * bar_width,
                values,
                bar_width,
                label=f"Representation {rep}",
            )
        axis.axhline(persistence_metrics[column], color="r", ls="--", lw=1, label="Persistence (y_t-1)")
        axis.axhline(majority_metrics[column], color="gray", ls=":", lw=1, label="Majority class")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_title(title)
        axis.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{CITY.title()} - Representation A vs B")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{CITY}_fig6_A_vs_B.png", dpi=150)
    plt.close(fig)

    # Confusion matrices are normalised by the true class; annotations show counts.
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    for row, rep in enumerate(("A", "B")):
        for column, model in enumerate(MODELS):
            score = metrics[(metrics["model"] == model) & (metrics["rep"] == rep)].iloc[0]
            matrix = np.array(
                [[score["TN"], score["FP"]], [score["FN"], score["TP"]]],
                dtype=float,
            )
            row_totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
            axes[row, column].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for actual in range(2):
                for predicted in range(2):
                    value = normalized[actual, predicted]
                    axes[row, column].text(
                        predicted,
                        actual,
                        f"{int(matrix[actual, predicted])}\n({value:.0%})",
                        ha="center",
                        va="center",
                        color="white" if value > 0.5 else "black",
                        fontsize=9,
                    )
            axes[row, column].set_xticks([0, 1])
            axes[row, column].set_yticks([0, 1])
            axes[row, column].set_xticklabels(["prev. 0", "prev. 1"])
            axes[row, column].set_yticklabels(["real 0", "real 1"])
            axes[row, column].set_title(f"{SHORT[model]} ({rep})", fontsize=10)
    fig.suptitle(f"{CITY.title()} - Confusion matrices (percentage by true class)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{CITY}_fig7_confusion.png", dpi=150)
    plt.close(fig)

    # Compare per-event computational cost with Macro-F1.
    fig, axis = plt.subplots(figsize=(7, 5))
    for rep, marker in (("A", "o"), ("B", "s")):
        for model in MODELS:
            score = metrics[(metrics["model"] == model) & (metrics["rep"] == rep)].iloc[0]
            cost = score["predict_ms_per_event"] + score["update_ms_per_event"]
            axis.scatter(cost, score["macro_f1"], marker=marker, s=45)
            axis.annotate(
                f"{SHORT[model]} {rep}",
                (cost, score["macro_f1"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axis.axhline(persistence_metrics["macro_f1"], color="r", ls="--", lw=1, label="Persistence (y_t-1)")
    axis.set_xscale("log")
    axis.set_xlabel("Cost per event: prediction + update (ms, log scale)")
    axis.set_ylabel("Macro-F1")
    axis.set_title(f"{CITY.title()} - Cost vs. performance (circle = A, square = B)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{CITY}_fig8_cost_performance.png", dpi=150)
    plt.close(fig)

    # Show monthly Macro-F1 trends for both feature representations.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, rep in zip(axes, ("A", "B")):
        for model in MODELS + ["Baseline: persistence"]:
            subset = monthly[(monthly["rep"] == rep) & (monthly["model"] == model)]
            axis.plot(
                subset["month"],
                subset["macro_f1"],
                marker="o",
                ms=3,
                lw=1.5 if "Baseline" in model else 1,
                ls="--" if "Baseline" in model else "-",
                label=SHORT.get(model, "persistence"),
            )
        axis.set_ylabel(f"Macro-F1 (Repr. {rep})")
    axes[0].legend(fontsize=7, ncol=3)
    plt.setp(axes[1].get_xticklabels(), rotation=60, fontsize=7)
    fig.suptitle(f"{CITY.title()} - Monthly Macro-F1 during evaluation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{CITY}_fig9_monthly_macro_f1.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the result comparisons, bootstrap, monthly analysis, and plotting."""
    ensure_directories()
    metrics, predictions, y_true, persistence = load_inputs()
    compare_representations(metrics)
    bootstrap_comparisons(predictions, y_true, persistence)
    monthly = save_monthly_scores(predictions, y_true, persistence)
    save_figures(metrics, predictions, monthly)
    print("\nTables saved to results/analysis_*.csv | figures saved to results/figures/")


if __name__ == "__main__":
    main()
