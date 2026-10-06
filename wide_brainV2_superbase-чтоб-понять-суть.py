#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wide_brainV2_superbase.py — самая короткая суть: ОДНА пара, график в конце.
Фичи/цель/модель переиспользуем из QWEN6 (чтобы не дублировать математику)."""
import importlib.util
import pathlib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ---- КОНФИГ ----
SYM, TF, NOPT, NFWD, SIGMA = "EURUSD", "H4", 1000, 300, 1.5
# ------------------

HERE = pathlib.Path(__file__).parent
_qspec = importlib.util.spec_from_file_location("q6", HERE / "система-в-единм-файле-QWEN6.py")
q6 = importlib.util.module_from_spec(_qspec)
_qspec.loader.exec_module(q6)

# данные (offline csv) -> фичи QWEN6
df = pd.read_csv(HERE / f"data/{SYM}_{TF}.csv", parse_dates=["time"]).sort_values("time").reset_index(drop=True)
fs = q6.FeatureStore(df)
fs.calc_features()
X = fs.features[q6.get_feature_cols(fs.features)].values.astype(float)
L = len(X)
close = df["close"].values[-L:]                                  # выровнять по фичам

# цель: знак наклона сглаженной цены  (y = (deriv > 0))
y = (np.gradient(gaussian_filter1d(close, SIGMA, mode="reflect")) > 0).astype(int)
ret = close[1:] / close[:-1] - 1.0                               # доходность бара

# обучение на [0,NOPT), сигнал на всех барах
Xs = StandardScaler().fit(X[:NOPT]).transform(X)
model = q6.train_model(Xs[:NOPT], y[:NOPT])
sig = (model.predict_proba(Xs)[:, 1] > 0.5).astype(int) * 2 - 1  # +1 / -1

# эквити на форварде: решение на баре t, реализация t->t+1
eq = [1.0]
for t in range(NOPT, NOPT + NFWD):
    eq.append(eq[-1] * (1 + sig[t] * ret[t]))
eq = np.array(eq)

dd = float((eq / np.maximum.accumulate(eq) - 1).min())
print(f"{SYM} {TF}: доход={eq[-1]-1:.2%}  просадка={dd:.2%}")
plt.plot(eq, color="green")
plt.title(f"{SYM} {TF} — equity (V2-super-base)")
plt.xlabel("Бар (поток)"); plt.ylabel("Equity")
plt.tight_layout(); plt.savefig(HERE / "equity_superbase.png", dpi=120)
print("график: equity_superbase.png")
