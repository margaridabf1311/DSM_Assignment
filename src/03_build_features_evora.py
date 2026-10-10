"""Build the T6 target and feature representations A and B.

This script reads Évora's raw weather data and the frozen thresholds created by
the warm-up script. Features at time t use only information available before t.

Workflow:
1. Load the frozen rain and wind thresholds.
2. Build the T6 target and feature representations for Évora.
3. Check that changing weather values at t does not change features at t.
4. Remove rows with incomplete lag or rolling-window features.
5. Save the processed data, evaluation stream, and feature-set definitions.

Outputs:
- `data/processed_evora.csv`: complete feature table with the target
- `data/eval_stream_evora.csv`: evaluation-period rows only
- `results/feature_sets.json`: column names for representations A and B
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

CITY = "evora"
RAW_VARIABLES = {
    "temperature": "T",
    "relative_humidity": "H",
    "precipitation": "P",
    "wind_speed": "W",
}


def ensure_directories() -> None:
    """Create the data and results folders if they do not already exist."""
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def load_frozen_thresholds() -> tuple[float, float, pd.Timestamp]:
    """Load the frozen thresholds and warm-up cutoff from the results folder.

    Returns:
        Wind threshold, rain threshold, and end of the warm-up period.
    """
    with (RESULTS_DIR / "frozen_thresholds.json").open("r", encoding="utf-8") as fh:
        frozen = json.load(fh)
    return (
        frozen["tau_W_kmh"],
        frozen["rain_threshold_mm"],
        pd.Timestamp(frozen["warmup_end"]),
    )


def build_features(
    df: pd.DataFrame,
    tau_w: float,
    rain_threshold: float,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create the current-time target and two strictly historical feature sets.

    Representation A contains the four weather variables at t-1. Representation
    B adds changes, rolling summaries, and counts computed from observations no
    later than t-1.

    Returns:
        The feature-and-target table, representation A column names, and
        representation B column names.
    """
    data = df.sort_values("time").reset_index(drop=True).copy()
    data["rain_t"] = (data["precipitation"] >= rain_threshold).astype(int)
    data["wind_t"] = (data["wind_speed"] >= tau_w).astype(int)
    data["target_T6"] = ((data["rain_t"] == 1) | (data["wind_t"] == 1)).astype(int)

    features = pd.DataFrame({"time": data["time"]})

    for variable, prefix in RAW_VARIABLES.items():
        features[f"{prefix}_lag1"] = data[variable].shift(1)
    representation_a = [column for column in features.columns if column != "time"]

    for variable, prefix in RAW_VARIABLES.items():
        previous = data[variable].shift(1)
        features[f"{prefix}_delta"] = previous - data[variable].shift(2)
        for window in (6, 24):
            features[f"{prefix}_mean_{window}h"] = previous.rolling(window).mean()
            features[f"{prefix}_std_{window}h"] = previous.rolling(window).std()

    features["rain_hours_24h"] = data["rain_t"].shift(1).rolling(24).sum()
    features["windy_hours_24h"] = data["wind_t"].shift(1).rolling(24).sum()
    features["adverse_hours_24h"] = data["target_T6"].shift(1).rolling(24).sum()
    features["precip_sum_6h"] = data["precipitation"].shift(1).rolling(6).sum()
    representation_b = [column for column in features.columns if column != "time"]

    output = pd.concat([features, data[["target_T6"]]], axis=1)
    return output, representation_a, representation_b


def check_no_leakage(
    df: pd.DataFrame,
    tau_w: float,
    rain_threshold: float,
    row: int | None = None,
) -> None:
    """Verify that changing input values at t leaves the features at t unchanged.
    """
    if df.empty:
        raise ValueError("Cannot check feature leakage on an empty dataset.")

    row = min(500, len(df) - 1) if row is None else row
    if not 0 <= row < len(df):
        raise IndexError(f"Leakage-check row {row} is outside the dataset.")

    ordered_data = df.sort_values("time").reset_index(drop=True)
    baseline, rep_a, rep_b = build_features(ordered_data, tau_w, rain_threshold)
    perturbed_data = ordered_data.copy()
    for variable in RAW_VARIABLES:
        perturbed_data.loc[row, variable] += 1000
    perturbed, _, _ = build_features(perturbed_data, tau_w, rain_threshold)

    columns = list(dict.fromkeys(rep_a + rep_b))
    unchanged = np.allclose(
        baseline.loc[row, columns].astype(float),
        perturbed.loc[row, columns].astype(float),
        equal_nan=True,
    )
    if not unchanged:
        raise AssertionError("Feature leakage detected: features at t depend on values at t.")


def process_city(
    tau_w: float,
    rain_threshold: float,
    warmup_end: pd.Timestamp,
) -> tuple[list[str], list[str]]:
    """Build, validate, and save the feature data for Évora.

    Returns the column names in feature representations A and B.
    """
    raw_path = DATA_DIR / f"raw_{CITY}.csv"
    raw = pd.read_csv(raw_path, parse_dates=["time"])
    check_no_leakage(raw, tau_w, rain_threshold)

    features, representation_a, representation_b = build_features(raw, tau_w, rain_threshold)
    clean = features.dropna().reset_index(drop=True)
    evaluation = clean[clean["time"] > warmup_end].reset_index(drop=True)
    warmup = clean[clean["time"] <= warmup_end]

    clean.to_csv(DATA_DIR / f"processed_{CITY}.csv", index=False)
    evaluation.to_csv(DATA_DIR / f"eval_stream_{CITY}.csv", index=False)

    warmup_rate = warmup["target_T6"].mean()
    evaluation_rate = evaluation["target_T6"].mean()
    print(
        f"{CITY}: warm-up={len(warmup)}h | evaluation={len(evaluation)}h | "
        f"T6 warm-up={warmup_rate:.1%} | T6 evaluation={evaluation_rate:.1%}"
    )
    return representation_a, representation_b


def main() -> None:
    """Build and save Évora's feature data and feature-set definitions."""
    ensure_directories()
    tau_w, rain_threshold, warmup_end = load_frozen_thresholds()

    representation_a, representation_b = process_city(tau_w, rain_threshold, warmup_end)
    feature_sets = {"A": representation_a, "B": representation_b}

    with (RESULTS_DIR / "feature_sets.json").open("w", encoding="utf-8") as fh:
        json.dump(feature_sets, fh, indent=2)

    print(f"Wind threshold used: {tau_w:.2f} km/h | warm-up ends: {warmup_end}")


if __name__ == "__main__":
    main()
