"""Exploratory analysis for the Coimbra and Évora stream datasets.

This script loads hourly weather data for each city, labels adverse hours using
the frozen pooled warm-up thresholds, and creates descriptive tables and figures.

Workflow:
1. Load the raw city data and frozen thresholds.
2. Add event labels and local-time fields for analysis.
3. Export descriptive, event, extreme-value, run-length, and evening summaries.
4. Generate plots of weather trends, event rates, and event persistence.

Outputs:
- `results/eda_<city>_descriptive.csv`: weather statistics by period
- `results/eda_<city>_target_composition.csv`: event rates and composition
- `results/eda_<city>_extremes.csv`: highest wind and precipitation records
- `results/eda_<city>_run_lengths.csv`: adverse and event-free run statistics
- `results/eda_<city>_evenings_by_year.csv` and `..._by_month.csv`: session summaries
- `results/eda_<city>_rolling.csv`: rolling event-rate summary
- `results/figures/<city>_fig*.png`: exploratory plots
"""

import json
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

CITIES = ("coimbra", "evora")
EVENING_HOURS = (21, 22, 23)  
SEASON_MONTHS = (5, 6, 7, 8, 9)  
ROLL_H = 24 * 30
VARS = ["temperature", "relative_humidity", "precipitation", "wind_speed"]


def ensure_directories() -> None:
    """Create the output folders used for CSV summaries and plots."""
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_frozen_thresholds() -> tuple[float, float, pd.Timestamp]:
    """Load the warm-up thresholds and evaluation cutoff stored by the earlier script.
    Returns a tuple with the wind threshold, rain threshold, and the warm-up end timestamp.
    """
    with (RESULTS_DIR / "frozen_thresholds.json").open("r", encoding="utf-8") as fh:
        frozen = json.load(fh)
    return frozen["tau_W_kmh"], frozen["rain_threshold_mm"], pd.Timestamp(frozen["warmup_end"])


def load_city_data(city: str, tau_w: float, rain_threshold: float, warm_end: pd.Timestamp) -> pd.DataFrame:
    """Load the raw city data and add columns describing adverse events.

    The added columns are used only for descriptive reporting and do not modify the
    underlying threshold definitions.
    """
    df = pd.read_csv(DATA_DIR / f"raw_{city}.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    df["period"] = np.where(df["time"] <= warm_end, "warmup", "evaluation")
    df["rain"] = (df["precipitation"] >= rain_threshold).astype(int)
    df["wind"] = (df["wind_speed"] >= tau_w).astype(int)
    df["T6"] = ((df["rain"] == 1) | (df["wind"] == 1)).astype(int)
    df["kind"] = np.select(
        [(df["rain"] == 1) & (df["wind"] == 1), df["rain"] == 1, df["wind"] == 1],
        ["rain+wind", "rain only", "wind only"],
        "none",
    )

    local = df["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Lisbon")
    df["hour_local"] = local.dt.hour
    df["date_local"] = local.dt.normalize().dt.tz_localize(None)
    df["ym"] = local.dt.strftime("%Y-%m")
    return df


def run_lengths(series: pd.Series, value: int) -> np.ndarray:
    """Return the lengths of consecutive runs for a binary signal.
    """
    values = series.astype(int).to_numpy()
    change = np.flatnonzero(np.diff(values)) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(values)]
    return (ends - starts)[values[starts] == value]


def save_summary_tables(df: pd.DataFrame, city: str) -> tuple[np.ndarray, np.ndarray]:
    """Export descriptive summary tables and run-length statistics.

    The outputs include common descriptive statistics by period, event composition,
    extreme observations, and evening-session summaries used in the report.
    """
    parts = []
    for label, subset in [
        ("warmup", df[df["period"] == "warmup"]),
        ("evaluation", df[df["period"] == "evaluation"]),
        ("full", df),
    ]:
        summary = subset[VARS].describe(percentiles=[0.05, 0.5, 0.95]).T
        summary.insert(0, "period", label)
        parts.append(summary)
    pd.concat(parts).to_csv(RESULTS_DIR / f"eda_{city}_descriptive.csv")

    freq = df.groupby("period").agg(
        hours=("T6", "size"),
        T6_rate=("T6", "mean"),
        rain_rate=("rain", "mean"),
        wind_rate=("wind", "mean"),
    )
    comp = df.groupby(["period", "kind"]).size().unstack(fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0).add_prefix("share_")
    event_summary = freq.join(comp)
    event_summary.to_csv(RESULTS_DIR / f"eda_{city}_target_composition.csv")
    print(event_summary.round(3).to_string())

    pd.concat(
        [
            df.nlargest(10, "wind_speed")[["time", "wind_speed", "precipitation"]].assign(kind="top_wind"),
            df.nlargest(10, "precipitation")[["time", "wind_speed", "precipitation"]].assign(kind="top_precip"),
        ]
    ).to_csv(RESULTS_DIR / f"eda_{city}_extremes.csv", index=False)

    adverse_runs = run_lengths(df["T6"], 1)
    free_runs = run_lengths(df["T6"], 0)
    pd.DataFrame(
        [
            {
                "run_type": "adverse (T6=1)",
                "n_runs": len(adverse_runs),
                "mean_h": float(adverse_runs.mean()),
                "median_h": float(np.median(adverse_runs)),
                "p90_h": float(np.percentile(adverse_runs, 90)),
                "max_h": float(adverse_runs.max()),
            },
            {
                "run_type": "event-free (T6=0)",
                "n_runs": len(free_runs),
                "mean_h": float(free_runs.mean()),
                "median_h": float(np.median(free_runs)),
                "p90_h": float(np.percentile(free_runs, 90)),
                "max_h": float(free_runs.max()),
            },
        ]
    ).to_csv(RESULTS_DIR / f"eda_{city}_run_lengths.csv", index=False)

    evening = df[df["hour_local"].isin(EVENING_HOURS)]
    by_day = evening.groupby("date_local").agg(
        n=("T6", "size"),
        adverse_hours=("T6", "sum"),
        disrupted=("T6", "max"),
    )
    by_day = by_day[by_day["n"] == len(EVENING_HOURS)].copy()
    by_day["year"], by_day["month"] = by_day.index.year, by_day.index.month
    season = by_day[by_day["month"].isin(SEASON_MONTHS)]

    season.groupby("year").agg(
        evenings=("disrupted", "size"),
        share_disrupted=("disrupted", "mean"),
        mean_adverse_hours=("adverse_hours", "mean"),
    ).to_csv(RESULTS_DIR / f"eda_{city}_evenings_by_year.csv")

    season.groupby(["year", "month"]).agg(
        evenings=("disrupted", "size"),
        share_disrupted=("disrupted", "mean"),
    ).to_csv(RESULTS_DIR / f"eda_{city}_evenings_by_month.csv")

    return adverse_runs, free_runs


def plot_temporal_series(df: pd.DataFrame, tau_w: float, warm_end: pd.Timestamp, city: str) -> None:
    """Plot the daily evolution of the main weather variables.

    The dashed line marks the end of the warm-up period, and the rainfall/wind
    threshold is shown for context.
    """
    daily = df.set_index("time").resample("D").agg(
        {"temperature": "mean", "relative_humidity": "mean", "precipitation": "sum", "wind_speed": "max"}
    )
    fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for axis, (column, label) in zip(
        ax,
        [
            ("temperature", "Mean daily temperature (°C)"),
            ("relative_humidity", "Mean relative humidity (%)"),
            ("precipitation", "Daily precipitation (mm)"),
            ("wind_speed", "Maximum wind speed (km/h)"),
        ],
    ):
        axis.plot(daily.index, daily[column], lw=0.8)
        axis.set_ylabel(label, fontsize=8)
        axis.axvline(warm_end, color="k", ls="--", lw=0.8)
    ax[3].axhline(tau_w, color="r", ls=":", label=f"tau_W = {tau_w:.1f} km/h")
    ax[3].legend(fontsize=8)
    fig.suptitle(f"{city.title()} - Daily weather trends (dashed line = end of warm-up)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{city}_fig1_temporal.png", dpi=150)
    plt.close(fig)


def plot_event_by_hour(df: pd.DataFrame, city: str) -> None:
    """Plot event rates by hour of day using local Lisbon time.

    This helps identify whether the adverse-event pattern is concentrated in a
    specific part of the evening or night.
    """
    hour_rates = df.groupby("hour_local")[["T6", "rain", "wind"]].mean()
    fig, axis = plt.subplots(figsize=(8, 4))
    hour_rates.plot(ax=axis, marker="o")
    axis.axvspan(min(EVENING_HOURS) - 0.5, max(EVENING_HOURS) + 0.5, color="orange", alpha=0.2, label="Session hours")
    axis.set_xlabel("Local hour (Europe/Lisbon)")
    axis.set_ylabel("Event rate")
    axis.legend()
    axis.set_title(f"{city.title()} - Event rate by hour of day")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{city}_fig2_hour_of_day.png", dpi=150)
    plt.close(fig)


def plot_monthly_event_rates(df: pd.DataFrame, city: str) -> None:
    """Plot the monthly composition of the T6 indicator.

    The stacked bars show how often events are driven by rain, wind, or both.
    """
    months = df["time"].dt.strftime("%Y-%m")
    counts = (
        df.assign(month=months)
        .groupby(["month", "kind"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["rain only", "wind only", "rain+wind", "none"], fill_value=0)
    )
    rate = counts.div(counts.sum(axis=1), axis=0)[["rain only", "wind only", "rain+wind"]]
    fig, axis = plt.subplots(figsize=(11, 4))
    rate.plot(kind="bar", stacked=True, ax=axis, width=0.85)
    first_eval = months[df["period"] == "evaluation"].iloc[0]
    axis.axvline(list(rate.index).index(first_eval) - 0.5, color="k", ls="--", lw=0.8)
    axis.set_xlabel("Month")
    axis.set_ylabel("Fraction of hours")
    axis.legend(title="Event type")
    axis.set_title(f"{city.title()} - Monthly T6 event composition (dashed line = start of evaluation)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{city}_fig3_monthly.png", dpi=150)
    plt.close(fig)


def plot_run_lengths(adverse_runs: np.ndarray, free_runs: np.ndarray, city: str) -> None:
    """Visualize the distribution of adverse and event-free run durations.

    This helps assess how persistent adverse periods are relative to normal periods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(np.clip(adverse_runs, 0, 48), bins=48)
    axes[0].set_title("Adverse-event runs (hours, capped at 48)")
    axes[1].hist(np.clip(free_runs, 0, 72), bins=72)
    axes[1].set_title("Event-free runs (hours, capped at 72)")
    for axis in axes:
        axis.set_xlabel("Duration (hours)")
        axis.set_ylabel("Number of runs")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{city}_fig4_run_lengths.png", dpi=150)
    plt.close(fig)


def plot_rolling_rate(df: pd.DataFrame, warm_end: pd.Timestamp, city: str) -> None:
    """Plot the 30-day rolling rate of the T6 adverse-event indicator.

    This highlights whether adverse conditions become more or less frequent over time.
    """
    rolling = df.set_index("time")["T6"].rolling(ROLL_H).mean()

    evaluation = rolling[rolling.index >= warm_end + pd.Timedelta(hours=ROLL_H)].dropna()
    pd.DataFrame(
        [
            {
                "window_h": ROLL_H,
                "min": evaluation.min(),
                "min_window_end": evaluation.idxmin(),
                "max": evaluation.max(),
                "max_window_end": evaluation.idxmax(),
                "mean": evaluation.mean(),
                "std": evaluation.std(),
            }
        ]
    ).to_csv(RESULTS_DIR / f"eda_{city}_rolling.csv", index=False)
    
    fig, axis = plt.subplots(figsize=(11, 3.5))
    axis.plot(rolling.index, rolling.values)
    axis.axvline(warm_end, color="k", ls="--", lw=0.8)
    axis.set_ylabel("T6 event rate (30-day rolling)")
    axis.set_title(f"{city.title()} - Rolling adverse-event rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{city}_fig5_rolling_rate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the full exploratory workflow for both cities.

    Each city's data produces its own descriptive outputs and exploratory figures.
    """
    ensure_directories()
    tau_w, rain_threshold, warm_end = load_frozen_thresholds()
    for city in CITIES:
        df = load_city_data(city, tau_w, rain_threshold, warm_end)
        adverse_runs, free_runs = save_summary_tables(df, city)

        plot_temporal_series(df, tau_w, warm_end, city)
        plot_event_by_hour(df, city)
        plot_monthly_event_rates(df, city)
        plot_run_lengths(adverse_runs, free_runs, city)
        plot_rolling_rate(df, warm_end, city)

    print("\nTabelas em results/eda_*.csv | figuras em results/figures/")


if __name__ == "__main__":
    main()
