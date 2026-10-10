"""Run prequential stream-model evaluation for Évora.

This script compares feature representations A and B using five online
classification models.

Workflow:
1. Load processed feature data, feature-set definitions, and the frozen cutoff
   from the `data/` and `results/` folders beside this script.
2. Create fresh model instances for each feature representation.
3. Evaluate each model sequentially, optionally learning from warm-up rows.
4. Compare model results with majority-class and persistence baselines.
5. Save metrics, predictions, and run configuration for later analysis.

Outputs:
- `results/metrics_evora.csv`: evaluation metrics and timing per representation
- `results/predictions_evora_repA.csv` and `..._repB.csv`: evaluation predictions
- `results/configuration_evora.json`: thresholds, package versions, and model setup
- `results/timing_evora_runs.csv` and `results/timing_evora_summary.csv`: standalone timing results

Set `SUBSET` to an integer to run only the first rows as a quick test.
"""

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from river import ensemble, forest, linear_model, naive_bayes, preprocessing, tree

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

CITY = "evora"
SEED = 42
N_MODELS = 5
LEARN_ON_WARMUP = True
SUBSET: int | None = None  # For example, set to 3000 for a quick test.
TARGET = "target_T6"


def ensure_output_directory() -> None:
    """Create the results directory if it does not already exist."""
    RESULTS_DIR.mkdir(exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any], pd.Timestamp]:
    """Load processed data, feature definitions, and frozen threshold metadata.

    Returns:
        The sorted feature dataframe, feature sets A/B, frozen threshold data,
        and the end of the warm-up period.
    """
    with (RESULTS_DIR / "frozen_thresholds.json").open("r", encoding="utf-8") as fh:
        frozen = json.load(fh)
    with (RESULTS_DIR / "feature_sets.json").open("r", encoding="utf-8") as fh:
        feature_sets = json.load(fh)

    data = pd.read_csv(DATA_DIR / f"processed_{CITY}.csv", parse_dates=["time"])
    data = data.sort_values("time").reset_index(drop=True)
    if SUBSET is not None:
        data = data.iloc[:SUBSET].copy()

    return data, feature_sets, frozen, pd.Timestamp(frozen["warmup_end"])


def make_models() -> dict[str, Any]:
    """Create fresh instances of the five required online classification models."""
    return {
        "Gaussian Naive Bayes": naive_bayes.GaussianNB(),
        "Online Logistic Regression": preprocessing.StandardScaler()
        | linear_model.LogisticRegression(),
        "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
        "Online Bagging": ensemble.BaggingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=N_MODELS,
            seed=SEED,
        ),
        "Adaptive Random Forest": forest.ARFClassifier(n_models=N_MODELS, seed=SEED),
    }


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Calculate binary classification metrics from the confusion matrix.

    Class 1 is the adverse-weather event. Undefined precision or recall values
    are reported as zero.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def divide(numerator: int, denominator: int) -> float:
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

    return {
        "accuracy": (tp + tn) / len(y_true),
        "macro_f1": (f1_nonevent + f1_event) / 2,
        "precision_event": precision_event,
        "recall_event": recall_event,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def evaluate_representation(
    data: pd.DataFrame,
    columns: list[str],
    representation: str,
    is_warm: np.ndarray,
    y_all: np.ndarray,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Run prequential evaluation and baselines for one feature representation.
    """
    feature_values = data[columns].to_numpy(dtype=float)
    models = make_models()
    predictions: dict[str, list[int]] = {name: [] for name in models}
    predict_time_ns = {name: 0 for name in models}
    update_time_ns = {name: 0 for name in models}
    print(f"\n=== Representation {representation} ({len(columns)} features) ===")

    for row in range(len(data)):
        features = dict(zip(columns, feature_values[row]))
        target = int(y_all[row])

        for name, model in models.items():
            if is_warm[row]:
                if LEARN_ON_WARMUP:
                    model.learn_one(features, target)
                continue

            start = time.perf_counter_ns()
            prediction = model.predict_one(features)
            predict_end = time.perf_counter_ns()
            prediction = 0 if prediction is None else int(prediction)
            predictions[name].append(prediction)

            update_start = time.perf_counter_ns()
            model.learn_one(features, target)
            update_end = time.perf_counter_ns()
            predict_time_ns[name] += predict_end - start
            update_time_ns[name] += update_end - update_start

        if row % 5000 == 0:
            print(f"  row {row}/{len(data)}")

    evaluation_indices = np.flatnonzero(~is_warm)
    y_evaluation = y_all[~is_warm]
    n_evaluation = len(y_evaluation)
    if n_evaluation == 0:
        raise ValueError("No evaluation rows available; check the warm-up cutoff or SUBSET.")

    rows = []
    for name in models:
        rows.append(
            {
                "city": CITY,
                "rep": representation,
                "model": name,
                "n_eval": n_evaluation,
                **summarize(y_evaluation, predictions[name]),
                "predict_ms_per_event": predict_time_ns[name] / n_evaluation / 1e6,
                "update_ms_per_event": update_time_ns[name] / n_evaluation / 1e6,
            }
        )

    majority_class = int(y_all[is_warm].mean() >= 0.5)
    persistence = y_all[evaluation_indices - 1]
    baselines = {
        "Baseline: majority class": np.full(n_evaluation, majority_class),
        "Baseline: persistence (y_t-1)": persistence,
    }
    for name, baseline_predictions in baselines.items():
        rows.append(
            {
                "city": CITY,
                "rep": representation,
                "model": name,
                "n_eval": n_evaluation,
                **summarize(y_evaluation, baseline_predictions),
                "predict_ms_per_event": 0.0,
                "update_ms_per_event": 0.0,
            }
        )

    prediction_table = pd.DataFrame(
        {
            "time": data.loc[~is_warm, "time"].values,
            "y_true": y_evaluation,
            **predictions,
        }
    )
    return rows, prediction_table


def save_configuration(
    frozen: dict[str, Any],
    feature_sets: dict[str, list[str]],
    row_count: int,
) -> None:
    """Write run settings and relevant package versions for reproducibility."""
    configuration = {
        "city": CITY,
        "seed": SEED,
        "n_models": N_MODELS,
        "learn_on_warmup": LEARN_ON_WARMUP,
        "warmup_end": frozen["warmup_end"],
        "tau_W_kmh": frozen["tau_W_kmh"],
        "rows": row_count,
        "python": platform.python_version(),
        "river": version("river"),
        "pandas": version("pandas"),
        "numpy": version("numpy"),
        "models": {name: repr(model) for name, model in make_models().items()},
        "feature_sets": feature_sets,
    }
    with (RESULTS_DIR / f"configuration_{CITY}.json").open("w", encoding="utf-8") as fh:
        json.dump(configuration, fh, indent=2)


def main() -> None:
    """Load inputs, evaluate both representations, and save all result files."""
    ensure_output_directory()
    data, feature_sets, frozen, warmup_end = load_inputs()
    is_warm = (data["time"] <= warmup_end).to_numpy()
    y_all = data[TARGET].astype(int).to_numpy()
    print(f"{CITY}: {len(data)} rows | warm-up={is_warm.sum()} | evaluation={(~is_warm).sum()}")

    metric_rows = []
    for representation, columns in feature_sets.items():
        rows, predictions = evaluate_representation(
            data,
            columns,
            representation,
            is_warm,
            y_all,
        )
        metric_rows.extend(rows)
        predictions.to_csv(
            RESULTS_DIR / f"predictions_{CITY}_rep{representation}.csv",
            index=False,
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / f"metrics_{CITY}.csv", index=False)
    print("\n", metrics.round(4).to_string(index=False))

    save_configuration(frozen, feature_sets, len(data))
    print("\nSaved metrics, predictions, and configuration in results/.")


if __name__ == "__main__":
    main()
