"""# Pooled warm-up threshold estimation

## Workflow
1. Fetch archived Open-Meteo data for each city in `CITIES`.
2. Save raw CSV files and retrieval metadata under the `data/` directory.
3. Audit warm-up, evaluation, and full-period data quality.
4. Fail fast if warm-up data contains structural issues.
5. Keep only timestamps shared across all cities.
6. Compute the pooled warm-up wind quantile (`tau_W`) and store it in `results/`.

## Outputs
- `data/raw_<city>.csv`: hourly weather records per city
- `data/retrieval_metadata.json`: source metadata for each retrieval
- `results/data_quality.csv`: quality audit results
- `results/frozen_thresholds.json`: frozen warm-up threshold summary
"""

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

START_DATE = "2025-01-01"
END_DATE = "2026-09-30"
WARMUP_START = pd.Timestamp("2025-01-01 00:00:00")
WARMUP_END = pd.Timestamp("2025-03-31 23:00:00")
WIND_QUANTILE = 0.80
RAIN_THRESHOLD_MM = 0.1

CITIES = {
    "Coimbra": (40.2089, -8.4292),
    "Evora": (38.5714, -7.904),
}
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


def ensure_directories() -> None:
    """Create the directories used for raw data and result artifacts."""
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def fetch_city_data(city_name: str, lat: float, lon: float, retries: int = 3) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch and normalise Open-Meteo hourly weather data for one city and 
    returns a tuple containing the cleaned dataframe and retrieval metadata.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Python-Script"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            print(f"[{city_name}] attempt {attempt} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"Could not fetch data for {city_name}.")

    hourly = data["hourly"]
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "relative_humidity": hourly["relative_humidity_2m"],
            "precipitation": hourly["precipitation"],
            "wind_speed": hourly["wind_speed_10m"],
        }
    ).sort_values("time").reset_index(drop=True)
    df["city"] = city_name

    metadata = {
        "city": city_name,
        "url": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "grid_lat": data.get("latitude"),
        "grid_lon": data.get("longitude"),
        "units": data.get("hourly_units"),
    }
    return df, metadata


def audit_quality(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Return a quality audit table for warm-up, evaluation, and full periods.

    The audit highlights missing timestamps, timestamp duplication, null values,
    invalid humidity/wind/precipitation ranges, and the maximum wind speed.
    """
    full_index = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    rows: list[dict[str, Any]] = []
    periods = {
        "warmup": df[(df["time"] >= WARMUP_START) & (df["time"] <= WARMUP_END)],
        "evaluation": df[df["time"] > WARMUP_END],
        "full": df,
    }

    for label, subset in periods.items():
        if subset.empty:
            expected = 0
            start = end = pd.NaT
        else:
            start, end = subset["time"].min(), subset["time"].max()
            expected = len(full_index[(full_index >= start) & (full_index <= end)])

        rows.append(
            {
                "city": city,
                "period": label,
                "rows": len(subset),
                "expected": expected,
                "missing_timestamps": expected - subset["time"].nunique(),
                "duplicates": int(subset["time"].duplicated().sum()),
                "nulls": int(
                    subset[["temperature", "relative_humidity", "precipitation", "wind_speed"]]
                    .isna()
                    .sum()
                    .sum()
                ),
                "invalid_rh": int(((subset["relative_humidity"] < 0) | (subset["relative_humidity"] > 100)).sum()),
                "invalid_wind": int((subset["wind_speed"] < 0).sum()),
                "invalid_precip": int((subset["precipitation"] < 0).sum()),
                "wind_max": float(subset["wind_speed"].max()) if not subset.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the full data-fetch, quality-audit, and threshold-estimation pipeline."""
    ensure_directories()

    frames: dict[str, pd.DataFrame] = {}
    metadata: list[dict[str, Any]] = []

    for city, (lat, lon) in CITIES.items():
        print(f"Downloading {city}...")
        frame, meta = fetch_city_data(city, lat, lon)
        frames[city] = frame
        metadata.append(meta)
        frame.to_csv(DATA_DIR / f"raw_{city.lower()}.csv", index=False)

    with (DATA_DIR / "retrieval_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    quality = pd.concat([audit_quality(df, city) for city, df in frames.items()], ignore_index=True)
    quality_path = RESULTS_DIR / "data_quality.csv"
    quality.to_csv(quality_path, index=False)
    print(quality.to_string(index=False))

    bad = quality[
        (quality["period"] == "warmup")
        & (
            quality[
                [
                    "missing_timestamps",
                    "duplicates",
                    "nulls",
                    "invalid_rh",
                    "invalid_wind",
                    "invalid_precip",
                ]
            ].sum(axis=1)
            > 0
        )
    ]
    if not bad.empty:
        raise ValueError(f"Warm-up quality issues detected:\n{bad}")

    common_times = set.intersection(*(set(df["time"]) for df in frames.values()))
    if not common_times:
        raise ValueError("No common timestamps were found across cities.")

    frames = {
        city: df[df["time"].isin(common_times)].reset_index(drop=True)
        for city, df in frames.items()
    }

    warm_frames = {
        city: df[(df["time"] >= WARMUP_START) & (df["time"] <= WARMUP_END)]
        for city, df in frames.items()
    }
    pooled_wind = pd.concat([frame["wind_speed"] for frame in warm_frames.values()], ignore_index=True)
    tau_w = float(pooled_wind.quantile(WIND_QUANTILE))

    frozen = {
        "tau_W_kmh": tau_w,
        "wind_quantile": WIND_QUANTILE,
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "warmup_start": str(WARMUP_START),
        "warmup_end": str(WARMUP_END),
        "n_warmup_pooled": int(len(pooled_wind)),
        "quantile_method": "pandas linear interpolation",
        "cities": list(CITIES),
    }

    with (RESULTS_DIR / "frozen_thresholds.json").open("w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"tau_W (Q{WIND_QUANTILE:.2f}, pooled warm-up) = {tau_w:.2f} km/h  [saved to {RESULTS_DIR / 'frozen_thresholds.json'}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
