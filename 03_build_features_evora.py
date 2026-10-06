"""
PA36 - Construção do alvo T6 e das representações A e B (apenas informação anterior a t).
Lê os CSV brutos e results/frozen_thresholds.json (gerados pelo script de warm-up).
"""
import json

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÃO LIDA DO JSON (nada de valores escritos à mão)
# -------------------------------------------------------------------------
frozen = json.load(open("results/frozen_thresholds.json"))
TAU_W = frozen["tau_W_kmh"]
RAIN_MM = frozen["rain_threshold_mm"]
WARMUP_END = pd.Timestamp(frozen["warmup_end"])
CITIES = ["evora"]   # a tua cidade; a parceira corre o MESMO script com ["coimbra"]

RAW_VARS = {"temperature": "T", "relative_humidity": "H", "precipitation": "P", "wind_speed": "W"}


# -------------------------------------------------------------------------
# 2. FUNÇÃO ÚNICA, aplicada às duas cidades (mesma lógica = comparação justa)
# -------------------------------------------------------------------------
def build(df):
    d = df.sort_values("time").reset_index(drop=True).copy()

    # Alvo T6 em t (NÃO entra nas features)
    d["rain_t"] = (d["precipitation"] >= RAIN_MM).astype(int)
    d["wind_t"] = (d["wind_speed"] >= TAU_W).astype(int)
    d["target_T6"] = ((d["rain_t"] == 1) | (d["wind_t"] == 1)).astype(int)

    f = pd.DataFrame({"time": d["time"]})

    # Representação A: valores em t-1
    for v, s in RAW_VARS.items():
        f[f"{s}_lag1"] = d[v].shift(1)
    rep_a = [c for c in f.columns if c != "time"]

    # Representação B: A + contexto temporal. Tudo termina em t-1.
    for v, s in RAW_VARS.items():
        past = d[v].shift(1)
        f[f"{s}_delta"] = past - d[v].shift(2)            # variação t-1 vs t-2
        for w in (6, 24):
            f[f"{s}_mean_{w}h"] = past.rolling(w).mean()
            f[f"{s}_std_{w}h"] = past.rolling(w).std()
    f["rain_hours_24h"] = d["rain_t"].shift(1).rolling(24).sum()
    f["windy_hours_24h"] = d["wind_t"].shift(1).rolling(24).sum()
    f["adverse_hours_24h"] = d["target_T6"].shift(1).rolling(24).sum()   # alvo PASSADO, não o atual
    f["precip_sum_6h"] = d["precipitation"].shift(1).rolling(6).sum()
    rep_b = [c for c in f.columns if c != "time"]

    out = pd.concat([f, d[["target_T6"]]], axis=1)
    return out, rep_a, rep_b


# -------------------------------------------------------------------------
# 3. TESTE DE LEAKAGE: alterar os valores em t não pode mudar as features em t
# -------------------------------------------------------------------------
def check_no_leakage(df, k=500):
    base, a, b = build(df)
    pert = df.copy()
    for v in RAW_VARS:
        pert.loc[k, v] = pert.loc[k, v] + 1000          # valor absurdo no instante k
    new, _, _ = build(pert)
    cols = a + b
    assert np.allclose(base.loc[k, cols].astype(float), new.loc[k, cols].astype(float), equal_nan=True), \
        "LEAKAGE: as features em t dependem de valores em t!"


# -------------------------------------------------------------------------
# 4. EXECUÇÃO
# -------------------------------------------------------------------------
feature_sets = {}
for city in CITIES:
    raw = pd.read_csv(f"data/raw_{city}.csv", parse_dates=["time"])
    check_no_leakage(raw)
    feats, rep_a, rep_b = build(raw)
    feature_sets = {"A": rep_a, "B": rep_b}

    # Remove só o arranque (NaN dos lags/janelas): mesmas linhas nas duas representações
    clean = feats.dropna().reset_index(drop=True)
    eval_stream = clean[clean["time"] > WARMUP_END].reset_index(drop=True)
    warm = clean[clean["time"] <= WARMUP_END]

    clean.to_csv(f"data/processed_{city}.csv", index=False)
    eval_stream.to_csv(f"data/eval_stream_{city}.csv", index=False)

    print(f"{city}: warm-up={len(warm)}h | avaliação={len(eval_stream)}h | "
          f"T6 warm-up={warm['target_T6'].mean():.1%} | T6 avaliação={eval_stream['target_T6'].mean():.1%}")

json.dump(feature_sets, open("results/feature_sets.json", "w"), indent=2)
print("tau_W usado:", TAU_W, "| warm-up termina em:", WARMUP_END)