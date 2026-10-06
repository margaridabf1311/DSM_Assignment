"""
PA36 - Download, auditoria de qualidade e pooled warm-up (tau_W congelado)
Versão revista do código do grupo.
"""
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÃO (mesmos valores nos scripts dos dois elementos do par)
# -------------------------------------------------------------------------
START_DATE = "2024-01-01"            # SUGESTÃO: começar mais cedo para o warm-up cobrir 12 meses
END_DATE = "2026-09-30"
WARMUP_START = pd.Timestamp("2024-01-01 00:00:00")
WARMUP_END = pd.Timestamp("2024-12-31 23:00:00")   # warm-up = 1 ano completo (todas as estações)
WIND_Q = 0.80
RAIN_MM = 0.1

CITIES = {
    "Coimbra": (40.2089, -8.4292),
    "Evora": (38.5714, -7.904),
}
HOURLY_VARIABLES = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)


# -------------------------------------------------------------------------
# 2. DOWNLOAD (unidades explícitas, retry, metadados para auditoria)
# -------------------------------------------------------------------------
def fetch_city_data(city_name, lat, lon, retries=3):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Python-Script"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except Exception as e:
            print(f"[{city_name}] tentativa {attempt} falhou: {e}")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

    h = data["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "temperature": h["temperature_2m"],
        "relative_humidity": h["relative_humidity_2m"],
        "precipitation": h["precipitation"],
        "wind_speed": h["wind_speed_10m"],
    }).sort_values("time").reset_index(drop=True)
    df["city"] = city_name

    meta = dict(city=city_name, url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                grid_lat=data.get("latitude"), grid_lon=data.get("longitude"),
                units=data.get("hourly_units"))
    return df, meta


frames, metas = {}, []
for city, (lat, lon) in CITIES.items():
    print(f"A descarregar {city}...")
    frames[city], m = fetch_city_data(city, lat, lon)
    metas.append(m)
    frames[city].to_csv(f"data/raw_{city.lower()}.csv", index=False)
json.dump(metas, open("data/retrieval_metadata.json", "w"), indent=2)


# -------------------------------------------------------------------------
# 3. QUALIDADE DOS DADOS - stream completa, com resumo por período
# -------------------------------------------------------------------------
def audit(df, city):
    full_idx = pd.date_range(df.time.min(), df.time.max(), freq="h")
    rows = []
    for label, sub in {
        "warmup": df[(df.time >= WARMUP_START) & (df.time <= WARMUP_END)],
        "evaluation": df[df.time > WARMUP_END],
        "full": df,
    }.items():
        exp = len(full_idx[(full_idx >= sub.time.min()) & (full_idx <= sub.time.max())])
        rows.append(dict(
            city=city, period=label, rows=len(sub), expected=exp,
            missing_timestamps=exp - sub.time.nunique(),
            duplicates=int(sub.time.duplicated().sum()),
            nulls=int(sub[["temperature", "relative_humidity", "precipitation", "wind_speed"]].isna().sum().sum()),
            invalid_rh=int(((sub.relative_humidity < 0) | (sub.relative_humidity > 100)).sum()),
            invalid_wind=int((sub.wind_speed < 0).sum()),
            invalid_precip=int((sub.precipitation < 0).sum()),
            wind_max=sub.wind_speed.max(),        # extremos: inspecionar, NÃO remover
        ))
    return pd.DataFrame(rows)


quality = pd.concat([audit(df, c) for c, df in frames.items()], ignore_index=True)
quality.to_csv("results/data_quality.csv", index=False)
print(quality.to_string(index=False))

# Falha explícita se o warm-up tiver problemas (em vez de só imprimir)
bad = quality[(quality.period == "warmup") &
              (quality[["missing_timestamps", "duplicates", "nulls", "invalid_rh", "invalid_wind", "invalid_precip"]].sum(axis=1) > 0)]
assert bad.empty, f"Problemas de qualidade no warm-up:\n{bad}"

# Mesma cronologia nas duas cidades
common = set.intersection(*[set(df.time) for df in frames.values()])
frames = {c: df[df.time.isin(common)].reset_index(drop=True) for c, df in frames.items()}


# -------------------------------------------------------------------------
# 4. POOLED WARM-UP -> tau_W (calculado uma vez e GUARDADO)
# -------------------------------------------------------------------------
warm = {c: df[(df.time >= WARMUP_START) & (df.time <= WARMUP_END)] for c, df in frames.items()}
pooled_wind = pd.concat([w.wind_speed for w in warm.values()])
tau_W = float(pooled_wind.quantile(WIND_Q))

frozen = dict(tau_W_kmh=tau_W, wind_quantile=WIND_Q, rain_threshold_mm=RAIN_MM,
              warmup_start=str(WARMUP_START), warmup_end=str(WARMUP_END),
              n_warmup_pooled=int(len(pooled_wind)), quantile_method="pandas linear interpolation",
              cities=list(CITIES))
json.dump(frozen, open("results/frozen_thresholds.json", "w"), indent=2)

print("\n" + "=" * 60)
print(f"tau_W (Q{WIND_Q:.2f}, pooled warm-up) = {tau_W:.2f} km/h  [guardado em results/frozen_thresholds.json]")
print("=" * 60)