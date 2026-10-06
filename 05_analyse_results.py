"""
PA36 - Análise dos resultados dos modelos (Évora). Só lê ficheiros já gerados; não volta a treinar nada.
Lê: results/metrics_<cidade>.csv, results/predictions_<cidade>_rep{A,B}.csv, data/processed_<cidade>.csv
Para a Coimbra: a parceira muda CITY e corre o MESMO script.

NOTA: não foi executado no ambiente onde foi escrito. Requer: pip install matplotlib
"""
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
BLOCK_H = 168        # bootstrap por blocos de 1 semana (preserva a autocorrelação dentro do bloco)
N_BOOT = 500
SEED = 42
OUT_T, OUT_F = "results", "results/figures"
os.makedirs(OUT_F, exist_ok=True)
MODELS = ["Gaussian Naive Bayes", "Online Logistic Regression", "Hoeffding Tree",
          "Online Bagging", "Adaptive Random Forest"]
SHORT = {"Gaussian Naive Bayes": "GNB", "Online Logistic Regression": "LogReg", "Hoeffding Tree": "HT",
         "Online Bagging": "Bagging", "Adaptive Random Forest": "ARF"}

met = pd.read_csv(f"results/metrics_{CITY}.csv")
pred = {r: pd.read_csv(f"results/predictions_{CITY}_rep{r}.csv", parse_dates=["time"]) for r in ("A", "B")}
assert (pred["A"].time.values == pred["B"].time.values).all() and (pred["A"].y_true.values == pred["B"].y_true.values).all(), \
    "As representações A e B têm de ter a mesma história de avaliação"
y = pred["A"].y_true.values
n = len(y)

# Persistência y_{t-1} alinhada com a avaliação (usa o último rótulo do warm-up na 1.ª linha, como no script 04)
proc = pd.read_csv(f"data/processed_{CITY}.csv", parse_dates=["time"]).set_index("time")["target_T6"]
pers = proc.shift(1).reindex(pred["A"].time).values.astype(int)
pers_met = met[met.model.str.contains("persistência")].iloc[0]
maj_met = met[met.model.str.contains("maioritária")].iloc[0]


def counts(y, p):
    tp = ((y == 1) & (p == 1)).sum(); tn = ((y == 0) & (p == 0)).sum()
    fp = ((y == 0) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    return tp, tn, fp, fn


def macro_f1(y, p):
    tp, tn, fp, fn = counts(y, p)
    div = lambda a, b: a / b if b else 0.0
    p1, r1, p0, r0 = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fn), div(tn, tn + fp)
    return (div(2 * p1 * r1, p1 + r1) + div(2 * p0 * r0, p0 + r0)) / 2


# -------------------------------------------------------------------------
# 2. TABELA A vs B (tudo igual exceto a representação)
# -------------------------------------------------------------------------
rows = []
for m in MODELS:
    a = met[(met.model == m) & (met.rep == "A")].iloc[0]
    b = met[(met.model == m) & (met.rep == "B")].iloc[0]
    rows.append(dict(model=m,
                     macro_f1_A=a.macro_f1, macro_f1_B=b.macro_f1, delta_macro_f1=b.macro_f1 - a.macro_f1,
                     recall_A=a.recall_event, recall_B=b.recall_event,
                     precision_A=a.precision_event, precision_B=b.precision_event,
                     update_ms_A=a.update_ms_per_event, update_ms_B=b.update_ms_per_event,
                     update_cost_ratio_B_over_A=b.update_ms_per_event / a.update_ms_per_event))
ab = pd.DataFrame(rows)
ab.to_csv(f"{OUT_T}/analysis_{CITY}_A_vs_B.csv", index=False)
print(ab.round(4).to_string(index=False))

# -------------------------------------------------------------------------
# 3. BOOTSTRAP POR BLOCOS (incerteza amostral aproximada do Macro-F1 e das diferenças)
# -------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
series = {(r, m): pred[r][m].values for r in ("A", "B") for m in MODELS}


def boot_idx():
    nb = int(np.ceil(n / BLOCK_H))
    starts = rng.integers(0, n - BLOCK_H + 1, size=nb)
    return (starts[:, None] + np.arange(BLOCK_H)).ravel()[:n]


full = {k: macro_f1(y, v) for k, v in series.items()}
full_pers = macro_f1(y, pers)
boots = {k: np.empty(N_BOOT) for k in series}
boots_pers = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = boot_idx(); yb = y[idx]
    boots_pers[b] = macro_f1(yb, pers[idx])
    for k, v in series.items():
        boots[k][b] = macro_f1(yb, v[idx])

ci = lambda arr: np.percentile(arr, [2.5, 97.5])
brow = []
for m in MODELS:
    for r in ("A", "B"):
        lo, hi = ci(boots[(r, m)])
        brow.append(dict(comparison="Macro-F1", model=m, rep=r, estimate=full[(r, m)], ci_low=lo, ci_high=hi))
    lo, hi = ci(boots[("B", m)] - boots[("A", m)])
    brow.append(dict(comparison="B minus A", model=m, rep="-", estimate=full[("B", m)] - full[("A", m)], ci_low=lo, ci_high=hi))
    for r in ("A", "B"):
        lo, hi = ci(boots[(r, m)] - boots_pers)
        brow.append(dict(comparison="model minus persistence", model=m, rep=r,
                         estimate=full[(r, m)] - full_pers, ci_low=lo, ci_high=hi))
lo, hi = ci(boots_pers)
brow.append(dict(comparison="Macro-F1", model="Baseline: persistência", rep="-", estimate=full_pers, ci_low=lo, ci_high=hi))
boot_df = pd.DataFrame(brow)
boot_df.to_csv(f"{OUT_T}/analysis_{CITY}_bootstrap.csv", index=False)
print("\n", boot_df.round(4).to_string(index=False))

# -------------------------------------------------------------------------
# 4. DESEMPENHO AO LONGO DO TEMPO (Macro-F1 mensal)
# -------------------------------------------------------------------------
month = pred["A"].time.dt.to_period("M").astype(str).values
mrows = []
for r in ("A", "B"):
    for mo in np.unique(month):
        sel = month == mo
        for m in MODELS:
            mrows.append(dict(rep=r, month=mo, model=m, macro_f1=macro_f1(y[sel], series[(r, m)][sel])))
        mrows.append(dict(rep=r, month=mo, model="Baseline: persistência", macro_f1=macro_f1(y[sel], pers[sel])))
monthly = pd.DataFrame(mrows)
monthly.to_csv(f"{OUT_T}/analysis_{CITY}_monthly_macro_f1.csv", index=False)

# -------------------------------------------------------------------------
# 5. FIGURAS
# -------------------------------------------------------------------------
x = np.arange(len(MODELS)); w = 0.38
labels = [SHORT[m] for m in MODELS]

# Fig 6: A vs B (Macro-F1 e Recall do evento) com baselines
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (col, t) in zip(ax, [("macro_f1", "Macro-F1"), ("recall_event", "Recall (classe evento)")]):
    for i, r in enumerate(("A", "B")):
        vals = [met[(met.model == m) & (met.rep == r)][col].iloc[0] for m in MODELS]
        a.bar(x + (i - 0.5) * w, vals, w, label=f"Representação {r}")
    a.axhline(pers_met[col], color="r", ls="--", lw=1, label="persistência (y_t-1)")
    a.axhline(maj_met[col], color="gray", ls=":", lw=1, label="classe maioritária")
    a.set_xticks(x); a.set_xticklabels(labels); a.set_title(t); a.set_ylim(0, 1)
ax[0].legend(fontsize=8)
fig.suptitle(f"{CITY.title()} - Representação A vs B")
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig6_A_vs_B.png", dpi=150); plt.close(fig)

# Fig 7: matrizes de confusão (normalizadas por classe real, com contagens)
fig, ax = plt.subplots(2, 5, figsize=(16, 6.5))
for i, r in enumerate(("A", "B")):
    for j, m in enumerate(MODELS):
        s = met[(met.model == m) & (met.rep == r)].iloc[0]
        cm = np.array([[s.TN, s.FP], [s.FN, s.TP]], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax[i, j].imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for a_ in range(2):
            for b_ in range(2):
                ax[i, j].text(b_, a_, f"{int(cm[a_, b_])}\n({norm[a_, b_]:.0%})", ha="center", va="center",
                              color="white" if norm[a_, b_] > 0.5 else "black", fontsize=9)
        ax[i, j].set_xticks([0, 1]); ax[i, j].set_yticks([0, 1])
        ax[i, j].set_xticklabels(["prev. 0", "prev. 1"]); ax[i, j].set_yticklabels(["real 0", "real 1"])
        ax[i, j].set_title(f"{SHORT[m]} ({r})", fontsize=10)
fig.suptitle(f"{CITY.title()} - matrizes de confusão (percentagem por classe real)")
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig7_confusion.png", dpi=150); plt.close(fig)

# Fig 8: custo vs desempenho
fig, a = plt.subplots(figsize=(7, 5))
for r, mk in (("A", "o"), ("B", "s")):
    for m in MODELS:
        s = met[(met.model == m) & (met.rep == r)].iloc[0]
        cost = s.predict_ms_per_event + s.update_ms_per_event
        a.scatter(cost, s.macro_f1, marker=mk, s=45)
        a.annotate(f"{SHORT[m]} {r}", (cost, s.macro_f1), fontsize=7, xytext=(3, 3), textcoords="offset points")
a.axhline(pers_met.macro_f1, color="r", ls="--", lw=1, label="persistência (y_t-1)")
a.set_xscale("log"); a.set_xlabel("Custo por evento: predição + update (ms, escala log)"); a.set_ylabel("Macro-F1")
a.set_title(f"{CITY.title()} - custo vs desempenho (círculo = A, quadrado = B)"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig8_cost_performance.png", dpi=150); plt.close(fig)

# Fig 9: Macro-F1 mensal
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for a, r in zip(ax, ("A", "B")):
    for m in MODELS + ["Baseline: persistência"]:
        d = monthly[(monthly.rep == r) & (monthly.model == m)]
        a.plot(d.month, d.macro_f1, marker="o", ms=3, lw=1.5 if "Baseline" in m else 1,
               ls="--" if "Baseline" in m else "-", label=SHORT.get(m, "persistência"))
    a.set_ylabel(f"Macro-F1 (Repr. {r})")
ax[0].legend(fontsize=7, ncol=3)
plt.setp(ax[1].get_xticklabels(), rotation=60, fontsize=7)
fig.suptitle(f"{CITY.title()} - Macro-F1 mensal na avaliação")
fig.tight_layout(); fig.savefig(f"{OUT_F}/{CITY}_fig9_monthly_macro_f1.png", dpi=150); plt.close(fig)

print("\nTabelas em results/analysis_*.csv | figuras em results/figures/")