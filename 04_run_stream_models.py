"""
PA36 - Avaliação prequential em Évora (T6)
Representações A e B x 5 modelos obrigatórios. Ciclo: PREDICT -> EVALUATE -> LEARN.
Lê data/processed_evora.csv, results/feature_sets.json e results/frozen_thresholds.json.

NOTA: não foi executado no ambiente onde foi escrito. Valida primeiro com um subconjunto
(ex.: SUBSET = 3000) antes de correr tudo, sobretudo por causa do ARF (lento).
"""
import json
import os
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from river import ensemble, forest, linear_model, naive_bayes, preprocessing, tree

# -------------------------------------------------------------------------
# 1. CONFIGURAÇÃO (combinar com a parceira: mesmos valores nas duas cidades)
# -------------------------------------------------------------------------
CITY = "evora"
SEED = 42
N_MODELS = 5                 # tamanho dos ensembles (Bagging e ARF)
LEARN_ON_WARMUP = True       # os modelos aprendem no warm-up, sem serem avaliados
SUBSET = None                # ex.: 3000 para um teste rápido (limita o nº de linhas)
TARGET = "target_T6"
os.makedirs("results", exist_ok=True)

frozen = json.load(open("results/frozen_thresholds.json"))
WARMUP_END = pd.Timestamp(frozen["warmup_end"])
FEATURE_SETS = json.load(open("results/feature_sets.json"))   # {"A": [...], "B": [...]}

df = pd.read_csv(f"data/processed_{CITY}.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
if SUBSET:
    df = df.iloc[:SUBSET]
is_warm = (df["time"] <= WARMUP_END).values
y_all = df[TARGET].astype(int).values
print(f"{CITY}: {len(df)} linhas | warm-up={is_warm.sum()} | avaliação={(~is_warm).sum()}")


# -------------------------------------------------------------------------
# 2. MODELOS (instâncias NOVAS para cada representação)
# -------------------------------------------------------------------------
def make_models():
    return {
        "Gaussian Naive Bayes": naive_bayes.GaussianNB(),
        "Online Logistic Regression": preprocessing.StandardScaler() | linear_model.LogisticRegression(),
        "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
        "Online Bagging": ensemble.BaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=N_MODELS, seed=SEED),
        "Adaptive Random Forest": forest.ARFClassifier(n_models=N_MODELS, seed=SEED),
    }


# -------------------------------------------------------------------------
# 3. MÉTRICAS a partir da matriz de confusão (classe positiva = 1 = evento)
# -------------------------------------------------------------------------
def summarize(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    div = lambda a, b: a / b if b else 0.0
    prec1, rec1 = div(tp, tp + fp), div(tp, tp + fn)
    prec0, rec0 = div(tn, tn + fn), div(tn, tn + fp)
    f1_1 = div(2 * prec1 * rec1, prec1 + rec1)
    f1_0 = div(2 * prec0 * rec0, prec0 + rec0)
    return dict(accuracy=(tp + tn) / len(y), macro_f1=(f1_0 + f1_1) / 2,
                precision_event=prec1, recall_event=rec1, TN=tn, FP=fp, FN=fn, TP=tp)


# -------------------------------------------------------------------------
# 4. LOOP PREQUENTIAL
# -------------------------------------------------------------------------
rows = []
for rep, cols in FEATURE_SETS.items():
    X = df[cols].values.astype(float)
    models = make_models()
    preds = {m: [] for m in models}
    t_pred = {m: 0 for m in models}
    t_upd = {m: 0 for m in models}
    print(f"\n=== Representação {rep} ({len(cols)} features) ===")

    for i in range(len(df)):
        x = dict(zip(cols, X[i]))
        y = int(y_all[i])
        for name, model in models.items():
            if is_warm[i]:
                if LEARN_ON_WARMUP:
                    model.learn_one(x, y)            # só aprende; não é avaliado
                continue
            t0 = time.perf_counter_ns()
            y_hat = model.predict_one(x)             # PREDICT (sem ver y_t)
            t1 = time.perf_counter_ns()
            y_hat = 0 if y_hat is None else int(y_hat)
            preds[name].append(y_hat)                # EVALUATE (guardado antes de aprender)
            t2 = time.perf_counter_ns()
            model.learn_one(x, y)                    # LEARN
            t3 = time.perf_counter_ns()
            t_pred[name] += t1 - t0
            t_upd[name] += t3 - t2
        if i % 5000 == 0:
            print(f"  linha {i}/{len(df)}")

    y_eval = y_all[~is_warm]
    n_eval = len(y_eval)
    for name in models:
        rows.append(dict(city=CITY, rep=rep, model=name, n_eval=n_eval, **summarize(y_eval, preds[name]),
                         predict_ms_per_event=t_pred[name] / n_eval / 1e6,
                         update_ms_per_event=t_upd[name] / n_eval / 1e6))

    # Baselines de referência (não substituem os 5 modelos obrigatórios)
    idx_eval = np.where(~is_warm)[0]
    majority = int(y_all[is_warm].mean() >= 0.5)                       # classe maioritária do warm-up
    persistence = y_all[idx_eval - 1]                                   # y_{t-1}: disponível em t
    for bname, bp in {"Baseline: classe maioritária": np.full(n_eval, majority),
                      "Baseline: persistência (y_t-1)": persistence}.items():
        rows.append(dict(city=CITY, rep=rep, model=bname, n_eval=n_eval, **summarize(y_eval, bp),
                         predict_ms_per_event=0.0, update_ms_per_event=0.0))

    # Previsões guardadas para análises posteriores (por mês, janela rolante, etc.)
    out = pd.DataFrame({"time": df.loc[~is_warm, "time"].values, "y_true": y_eval})
    for name in models:
        out[name] = preds[name]
    out.to_csv(f"results/predictions_{CITY}_rep{rep}.csv", index=False)

res = pd.DataFrame(rows)
res.to_csv(f"results/metrics_{CITY}.csv", index=False)
print("\n", res.round(4).to_string(index=False))

# -------------------------------------------------------------------------
# 5. CONFIGURAÇÃO / AUDITORIA
# -------------------------------------------------------------------------
cfg = dict(city=CITY, seed=SEED, n_models=N_MODELS, learn_on_warmup=LEARN_ON_WARMUP,
           warmup_end=str(WARMUP_END), tau_W_kmh=frozen["tau_W_kmh"], rows=len(df),
           python=platform.python_version(), river=version("river"),
           pandas=version("pandas"), numpy=version("numpy"),
           models={k: repr(v) for k, v in make_models().items()}, feature_sets=FEATURE_SETS)
json.dump(cfg, open(f"results/configuration_{CITY}.json", "w"), indent=2)
print("\nGuardado: metrics, predictions e configuration em results/")