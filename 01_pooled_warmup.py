import json
import os
import urllib.parse
import urllib.request
import pandas as pd

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÕES
# -------------------------------------------------------------------------
START_DATE = "2025-01-01"
END_DATE = "2026-09-30"
WARMUP_START = "2025-01-01 00:00:00"
WARMUP_END = "2025-03-31 23:00:00"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

os.makedirs("data", exist_ok=True)


# -------------------------------------------------------------------------
# 2. FUNÇÃO DE DOWNLOAD
# -------------------------------------------------------------------------
def fetch_city_data(city_name, lat, lon):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    print(f"A descarregar dados de {city_name} via Open-Meteo...")

    req = urllib.request.Request(url, headers={"User-Agent": "Python-Script"})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))

    hourly = data["hourly"]
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "relative_humidity": hourly["relative_humidity_2m"],
            "precipitation": hourly["precipitation"],
            "wind_speed": hourly["wind_speed_10m"],
            "city": city_name,
        }
    )
    return df


# -------------------------------------------------------------------------
# 3. OBTER DADOS PARA COIMBRA E ÉVORA
# -------------------------------------------------------------------------
# Obtém Évora (para a tua análise) e Coimbra (necessária para o pooled warm-up)
df_evora = fetch_city_data("Evora", 38.5714, -7.9070)
df_coimbra = fetch_city_data("Coimbra", 40.2089, -8.4292)

# Guardar os brutos
df_evora.to_csv("data/raw_evora.csv", index=False)
df_coimbra.to_csv("data/raw_coimbra.csv", index=False)


# -------------------------------------------------------------------------
# 4. AUDITORIA DE QUALIDADE DOS DADOS (DATA QUALITY CHECK)
# -------------------------------------------------------------------------
def check_quality(df, city_name):
    print(f"\n--- AUDITORIA DE QUALIDADE DE DADOS: {city_name} ---")
    warmup_subset = df[
        (df["time"] >= WARMUP_START) & (df["time"] <= WARMUP_END)
    ].copy()

    expected = 2160  # 90 dias * 24 horas (Jan a Mar 2025)
    obtained = len(warmup_subset)
    duplicates = warmup_subset.duplicated(subset=["time"]).sum()
    nulls = warmup_subset.isnull().sum().sum()

    # 1. Verificação de Limites Físicos / Inconsistências de Sensor
    invalid_rh = (
        (warmup_subset["relative_humidity"] < 0)
        | (warmup_subset["relative_humidity"] > 100)
    ).sum()
    invalid_wind = (warmup_subset["wind_speed"] < 0).sum()
    invalid_precip = (warmup_subset["precipitation"] < 0).sum()
    total_invalids = invalid_rh + invalid_wind + invalid_precip

    # 2. Inspecionar Extremos Meteorológicos Legítimos (Outliers Físicos)
    max_wind = warmup_subset["wind_speed"].max()
    min_wind = warmup_subset["wind_speed"].min()
    q99_wind = warmup_subset["wind_speed"].quantile(0.99)

    print(f"• Registos de Warm-up obtidos : {obtained} / {expected}h")
    print(f"• Registos duplicados         : {duplicates}")
    print(f"• Valores em falta (NAs)      : {nulls}")
    print(f"• Erros Físicos/Sensor        : {total_invalids}")
    print(
        f"• Vento no Warm-up (Min/Max/Q99): {min_wind:.1f} / {max_wind:.1f} / {q99_wind:.1f} km/h"
    )

    if (
        obtained == expected
        and duplicates == 0
        and nulls == 0
        and total_invalids == 0
    ):
        print(
            f"-> STATUS {city_name}: APROVADO! Dados limpos e prontos para o Pooled Warm-Up."
        )
    else:
        print(
            f"-> STATUS {city_name}: ATENÇÃO - Inconsistências detetadas no Warm-Up!"
        )


# Execução
check_quality(df_coimbra, "Coimbra")
check_quality(df_evora, "Évora")


# -------------------------------------------------------------------------
# 5. POOLED WARM-UP & CÁLCULO DE tau_W CONGELADO
# -------------------------------------------------------------------------
w_coimbra = df_coimbra[
    (df_coimbra["time"] >= WARMUP_START) & (df_coimbra["time"] <= WARMUP_END)
]
w_evora = df_evora[
    (df_evora["time"] >= WARMUP_START) & (df_evora["time"] <= WARMUP_END)
]

pooled_wind = pd.concat([w_coimbra["wind_speed"], w_evora["wind_speed"]])
tau_W = pooled_wind.quantile(0.80)  # Quantil Q0.80 CONGELADO

print("\n" + "=" * 60)
print(f"POOLED WARM-UP CONCLUÍDO")
print(f"Limiar de Vento Forte Congelado (tau_W = Q0.80): {tau_W:.2f} km/h")
print("=" * 60 + "\n")
