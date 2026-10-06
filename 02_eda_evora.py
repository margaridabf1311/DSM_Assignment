"""
PA36 - Análise exploratória da stream de Évora (descritiva; não altera limiares nem o desenho).
Lê data/raw_<cidade>.csv e results/frozen_thresholds.json.
Para a Coimbra: a parceira muda CITY para "coimbra" e corre o MESMO script.

NOTA: não foi executado no ambiente onde foi escrito. Requer: pip install matplotlib
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÃO
# -------------------------------------------------------------------------
CITY = "evora"
EVENING_HOURS = (21, 22, 23)       # ASSUNÇÃO: horas locais de uma sessão noturna. Ajustem e justifiquem.
SEASON_MONTHS = (5, 6, 7, 8, 9)    # ASSUNÇÃO: época de cinema ao ar livre. Ajustem e justifiquem.
ROLL_H = 24 * 30                   # janela rolante de 30 dias
VARS = ["temperature", "relative_humidity", "precipitation", "wind_speed"]
OUT_T, OUT_F = "results", "results/figures"
os.makedirs(OUT_F, exist_ok=True)

frozen = json.load(open("results/frozen_thresholds.json"))
TAU_W, RAIN = frozen["tau_W_kmh"], frozen["rain_threshold_mm"]
WARM_END = pd.Timestamp(frozen["warmup_end"])

# -------------------------------------------------------------------------
# 2. STREAM + EVENTOS (apenas para descrição)
# -------------------------------------------------------------------------
df = pd.read_csv(f"data/raw_{CITY}.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
df["period"] = np.where(df.time <= WARM_END, "warmup", "evaluation")
df["rain"] = (df.precipitation >= RAIN).astype(int)
df["wind"] = (df.wind_speed >= TAU_W).astype(int)
df["T6"] = ((df.rain == 1) | (df.wind == 1)).astype(int)
df["kind"] = np.select([(df.rain == 1) & (df.wind == 1), df.rain == 1, df.wind == 1],
                       ["rain+wind", "rain only", "wind only"], "none")

local = df.time.dt.tz_localize("UTC").dt.tz_convert("Europe/Lisbon")   # hora local só para a análise
df["hour_local"] = local.dt.hour
df["date_local"] = local.dt.normalize().dt.tz_localize(None)
df["month_local"] = local.dt.month
df["year_local"] = local.dt.year
df["ym"] = local.dt.strftime("%Y-%m")

# -------------------------------------------------------------------------
# 3. TABELAS
# -------------------------------------------------------------------------
# 3.1 Estatísticas descritivas por período
parts = []
for label, sub in [("warmup", df[df.period == "warmup"]), ("evaluation", df[df.period == "evaluation"]), ("full", df)]:
    d = sub[VARS].describe(percentiles=[0.05, 0.5, 0.95]).T
    d.insert(0, "period", label)
    parts.append(d)
pd.concat(parts).to_csv(f"{OUT_T}/eda_{CITY}_descriptive.csv")

# 3.2 Frequência do alvo e composição do T6 (chuva / vento / ambos)
freq = df.groupby("period").agg(hours=("T6", "size"), T6_rate=("T6", "mean"),
                                rain_rate=("rain", "mean"), wind_rate=("wind", "mean"))
comp = df.groupby(["period", "kind"]).size().unstack(fill_value=0)
comp = comp.div(comp.sum(axis=1), axis=0).add_prefix("share_")
freq.join(comp).to_csv(f"{OUT_T}/eda_{CITY}_target_composition.csv")
print(freq.join(comp).round(3).to_string())

# 3.3 Extremos (inspecionar, NÃO remover)
pd.concat([df.nlargest(10, "wind_speed")[["time", "wind_speed", "precipitation"]].assign(kind="top_wind"),
           df.nlargest(10, "precipitation")[["time", "wind_speed", "precipitation"]].assign(kind="top_precip")]
          ).to_csv(f"{OUT_T}/eda_{CITY}_extremes.csv", index=False)


# 3.4 Persistência: duração de sequências consecutivas
def run_lengths(s, value):
    s = s.astype(int).values
    change = np.flatnonzero(np.diff(s)) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(s)]
    return (ends - starts)[s[starts] == value]


adverse_runs, free_runs = run_lengths(df.T6, 1), run_lengths(df.T6, 0)
stats = lambda r, n: dict(run_type=n, n_runs=len(r), mean_h=r.mean(), median_h=np.median(r),
                          p90_h=np.percentile(r, 90), max_h=r.max())
pd.DataFrame([stats(adverse_runs, "adverse (T6=1)"), stats(free_runs, "event-free (T6=0)")]
             ).to_csv(f"{OUT_T}/eda_{CITY}_run_lengths.csv", index=False)

# 3.5 Noites (hora local): proporção de noites com pelo menos 1 hora adversa na janela da sessão
ev = df[df.hour_local.isin(EVENING_HOURS)]
by_day = ev.groupby("date_local").agg(n=("T6", "size"), adverse_hours=("T6", "sum"), disrupted=("T6", "max"))
by_day = by_day[by_day.n == len(EVENING_HOURS)].copy()
by_day["year"], by_day["month"] = by_day.index.year, by_day.index.month
season = by_day[by_day.month.isin(SEASON_MONTHS)]
season.groupby("year").agg(evenings=("disrupted", "size"), share_disrupted=("disrupted", "mean"),
                           mean_adverse_hours=("adverse_hours", "mean")
                           ).to_csv(f"{OUT_T}/eda_{CITY}_evenings_by_year.csv")
season.groupby(["year", "month"]).agg(evenings=("disrupted", "size"), share_disrupted=("disrupted", "mean")
                                      ).to_csv(f"{OUT_T}/eda_{CITY}_evenings_by_month.csv")

# -------------------------------------------------------------------------
# 4. FIGURAS (cada uma deve responder a uma pergunta no relatório)
# -------------------------------------------------------------------------
# Fig 1: evolução temporal das 4 variáveis (diária)
daily = df.set_index("time").resample("D").agg({"temperature": "mean", "relative_humidity": "mean",
                                                 "precipitation": "sum", "wind_speed": "max"})
fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for a, (c, t) in zip(ax, [("temperature", "Temperatura média diária (°C)"),
                          ("relative_humidity", "Humidade relativa média (%)"),
                          ("precipitation", "Precipitação diária (mm)"),
                          ("wind_speed", "Vento máximo diário (km/h)")]):
    a.plot(daily.index, daily[c], lw=0.8)
    a.set_ylabel(t, fontsize=8)
    a.axvline(WARM_END, color="k", ls="--", lw=0.8)
ax[3].axhline(TAU_W, color="r", ls=":", label=f"tau_W = {TAU_W:.1f} km/h")
ax[3].legend(fontsize=8)
fig.suptitle(f"{CITY.title()} - evolução temporal (linha tracejada = fim do warm-up)")
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig1_temporal.png", dpi=150); plt.close(fig)

# Fig 2 (ORIGINAL): taxa de evento por hora do dia (hora local)
hr = df.groupby("hour_local")[["T6", "rain", "wind"]].mean()
fig, a = plt.subplots(figsize=(8, 4))
hr.plot(ax=a, marker="o")
a.axvspan(min(EVENING_HOURS) - 0.5, max(EVENING_HOURS) + 0.5, color="orange", alpha=0.2, label="janela da sessão")
a.set_xlabel("Hora local (Europe/Lisbon)"); a.set_ylabel("Taxa de evento"); a.legend()
a.set_title(f"{CITY.title()} - taxa de evento por hora do dia")
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig2_hour_of_day.png", dpi=150); plt.close(fig)

# Fig 3 (ORIGINAL): taxa mensal do T6, decomposta
cnt = df.groupby(["ym", "kind"]).size().unstack(fill_value=0).reindex(columns=["rain only", "wind only", "rain+wind", "none"], fill_value=0)
rate = cnt.div(cnt.sum(axis=1), axis=0)[["rain only", "wind only", "rain+wind"]]
fig, a = plt.subplots(figsize=(11, 4))
rate.plot(kind="bar", stacked=True, ax=a, width=0.85)
first_eval = df.loc[df.period == "evaluation", "ym"].iloc[0]
a.axvline(list(rate.index).index(first_eval) - 0.5, color="k", ls="--", lw=0.8)
a.set_ylabel("Fração de horas"); a.set_title(f"{CITY.title()} - taxa mensal de T6 (tracejado = início da avaliação)")
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig3_monthly.png", dpi=150); plt.close(fig)

# Fig 4 (ORIGINAL): persistência - duração das sequências
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(np.clip(adverse_runs, 0, 48), bins=48); ax[0].set_title("Sequências adversas (h, truncado a 48)")
ax[1].hist(np.clip(free_runs, 0, 72), bins=72); ax[1].set_title("Janelas sem evento (h, truncado a 72)")
for a in ax:
    a.set_xlabel("Duração (h)"); a.set_ylabel("Nº de sequências")
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig4_run_lengths.png", dpi=150); plt.close(fig)

# Fig 5: taxa rolante de T6 (30 dias)
roll = df.set_index("time")["T6"].rolling(ROLL_H).mean()
fig, a = plt.subplots(figsize=(11, 3.5))
a.plot(roll.index, roll.values)
a.axvline(WARM_END, color="k", ls="--", lw=0.8)
a.set_ylabel("Taxa T6 (30 dias)"); a.set_title(f"{CITY.title()} - taxa rolante do evento")
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig5_rolling_rate.png", dpi=150); plt.close(fig)

print("\nTabelas em results/eda_*.csv | figuras em results/figures/")