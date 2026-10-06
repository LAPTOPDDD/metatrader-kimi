#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wide_brainV2.1.py — «широкий мозг» V2.1 (АБЛЯЦИЯ: только база).
Убраны Stage 2 (кросс-активные «головы») и мета-контроллер риска.
Остаётся:
  Stage 1 — своя модель каждой пары на своих мульти-ТФ фичах.
  Портфель — равновесный, scale=1 (без риск-контроллера).
Это ровно та «Base»-линия, с которой сравнивали V2.
Все настройки — в блоке КОНФИГ в начале. График эквити строится в конце.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import importlib.util

# ======================================================================
# ============================ К О Н Ф И Г ==============================
# ======================================================================
SIGMA        = 1.5     # сглаживание цели: наклон сглаженной цены (0.5..3.0)
BASE_TF      = "M15"   # базовый таймфрейм решений
HIGHER       = ["M30", "H1", "H4"]   # старшие ТФ как контекст
BARS_TRAIN   = 1000   # размер ОБУЧАЮЩЕГО окна (старт walk-forward)
FOLD         = 90     # длина ОДНОГО тестового окна (шаг прокрутки)
N_FOLDS      = 10    # сколько окон гнать (None = пока хватает данных)
NORM         = True    # нормировать фичи по паре на train-статах
PAIRS        = None    # None = все пары из data/manifest.json
                      #   или, например: ["EURUSD", "GBPUSD", "USDJPY"]
# ======================================================================
# ====================== К О Н Е Ц   К О Н Ф И Г =======================


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
Q6_PATH = HERE / "система-в-единм-файле-QWEN6.py"

spec = importlib.util.spec_from_file_location("q6", str(Q6_PATH))
q6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q6)


def load_pair_tf(sym, tf):
    df = pd.read_csv(DATA_DIR / f"{sym}_{tf}.csv", parse_dates=["time"])
    return df.sort_values("time").reset_index(drop=True)


def make_target(close, sigma):
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(close, sigma, mode="reflect")
    deriv = np.gradient(smoothed)
    return (deriv > 0).astype(int)


def pair_features(sym):
    raw = load_pair_tf(sym, BASE_TF)
    fs = q6.FeatureStore(raw)
    fs.calc_features()
    fM15 = fs.features[q6.get_feature_cols(fs.features)].values.astype(float)
    L = len(fM15)
    offset = len(raw) - L
    close = raw["close"].values[offset:].astype(float)
    t_base = raw["time"].values[offset:]
    for tf in HIGHER:
        raw2 = load_pair_tf(sym, tf)
        fs2 = q6.FeatureStore(raw2)
        fs2.calc_features()
        f2 = fs2.features[q6.get_feature_cols(fs2.features)].values.astype(float)
        off2 = len(raw2) - len(f2)
        t2 = raw2["time"].values[off2:]
        pos = np.searchsorted(t2, t_base, side="right") - 1
        pos = np.clip(pos, 0, len(t2) - 1)
        fM15 = np.concatenate([fM15, f2[pos]], axis=1)
    return fM15, close, t_base


def metrics(eq):
    ret = eq[-1] - 1.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return ret, dd


def build_data(pairs_filter=None):
    man = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    avail = sorted({m["symbol"] for m in man})
    pairs = [s for s in avail if pairs_filter is None or s in pairs_filter]
    if pairs_filter is not None and len(pairs) < len(pairs_filter):
        print(f"  ! нет данных для: {set(pairs_filter) - set(pairs)}")
    feats, targets, returns = {}, {}, {}
    for sym in pairs:
        X, close, _ = pair_features(sym)
        feats[sym] = X[:-1]
        targets[sym] = make_target(close, SIGMA)[:-1]
        returns[sym] = close[1:] / close[:-1] - 1.0
    M = min(len(feats[s]) for s in pairs)
    for s in pairs:
        feats[s] = feats[s][:M]
        targets[s] = targets[s][:M]
        returns[s] = returns[s][:M]
    return pairs, feats, targets, returns, M


def run_fold(pairs, feats, targets, returns, tr_end, te_end):
    """ТОЛЬКО БАЗА: Stage-1 на своих фичах, равновесный портфель, scale=1.
    Возвращает (pr, sig_test)."""
    proba1 = {}
    for s in pairs:
        m = q6.train_model(feats[s][:tr_end], targets[s][:tr_end])
        proba1[s] = m.predict_proba(feats[s])[:, 1]
    proba1_mat = np.column_stack([proba1[s] for s in pairs])
    sig = (proba1_mat > 0.5).astype(float) * 2 - 1          # (M, P)
    sig_test = sig[tr_end:te_end]
    ret_test = np.column_stack([returns[s][tr_end:te_end] for s in pairs])
    pr = []
    for t in range(len(sig_test)):
        pr.append(float(np.mean(sig_test[t] * ret_test[t])))
    return np.array(pr), sig_test


def main():
    print("=== КОНФИГ (только база: без Stage2, без мета) ===")
    print(f"  SIGMA={SIGMA} BASE_TF={BASE_TF} HIGHER={HIGHER}")
    print(f"  BARS_TRAIN={BARS_TRAIN} FOLD={FOLD} N_FOLDS={N_FOLDS} NORM={NORM} PAIRS={PAIRS or 'все'}")

    pairs, feats, targets, returns, M = build_data(PAIRS)
    print(f"Пары ({len(pairs)}): {pairs}  | баров: {M}")

    tr_end = BARS_TRAIN
    pr_all = []
    print("\n--- По окнам (форвард, честно) ---")
    k = 0
    while tr_end + FOLD <= M and (N_FOLDS is None or k < N_FOLDS):
        te_end = tr_end + FOLD
        feats_n = {}
        for s in pairs:
            if NORM:
                sc = StandardScaler().fit(feats[s][:tr_end])
                feats_n[s] = sc.transform(feats[s])
            else:
                feats_n[s] = feats[s]
        pr, sig_test = run_fold(pairs, feats_n, targets, returns, tr_end, te_end)
        pr_all.append(pr)
        r = (np.prod(1 + pr) - 1) * 100
        print(f"Fold {k}: train[:{tr_end}) test[{tr_end},{te_end})  Base={r:6.1f}%")
        k += 1
        tr_end += FOLD

    pr = np.concatenate(pr_all)
    eq = np.cumprod(1 + pr)
    ret, dd = metrics(eq)
    print("\n=== ИТОГО walk-forward (только база, M15) ===")
    print(f"Base (Stage-1, без мета) : доход={ret*100:6.1f}%  maxDD={dd*100:6.1f}%")
    pd.DataFrame({"bar": np.arange(len(eq)), "eq_base": eq}).to_csv(HERE / "wide_equity_base.csv", index=False)
    print("Эквити сохранено в wide_equity_base.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(eq, label="Base (Stage-1, без мета)", color="green", lw=1.6)
        ax.set_title("Портфельная эквити — широкий мозг V2.1 (только база, walk-forward, M15)")
        ax.set_xlabel("Бар (цепочка окон)"); ax.set_ylabel("Equity")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(HERE / "wide_equity_base.png", dpi=120)
        print("График сохранён в wide_equity_base.png")
    except Exception as e:
        print("Предупреждение: не удалось построить PNG:", e)


if __name__ == "__main__":
    main()
