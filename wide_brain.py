#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wide_brain.py — «широкий мозг»: одна модель видит все пары и все ТФ.
Архитектура (честная, без утечки в будущее):
  Stage 1 — базовая модель каждой пары на своих мульти-ТФ фичах (q6-совместимых).
  Stage 2 — «головы» по паре: к своим фичам добавляем ЗАПАЗДЫВАЮЩИЕ
            вероятности ВСЕХ пар (широкий контекст рынка) -> уточнённый сигнал.
  Meta    — контроллер риска (РЕЖИМНЫЙ): предсказывает профит портфеля на
            горизонте H баров вперёд и выдаёт scale [0,1] (risk-off в шторм).
Портфельная симуляция (walk-forward, расширяющееся окно обучения):
  решение на баре t по закрытию, реализация на закрытии t->t+1.
Данные — оффлайн из data/*.csv (см. download_data.py).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import importlib.util

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
Q6_PATH = HERE / "система-в-единм-файле-QWEN6.py"

spec = importlib.util.spec_from_file_location("q6", str(Q6_PATH))
q6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q6)

SIGMA = 1.5
BASE_TF = "M15"
HIGHER = ["M30", "H1", "H4"]
BARS_TRAIN = 10000
BARS_TEST = 3000


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
        f2_aligned = f2[pos]
        fM15 = np.concatenate([fM15, f2_aligned], axis=1)
    return fM15, close, t_base


def metrics(eq):
    ret = eq[-1] - 1.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return ret, dd


def build_data():
    """Загрузка дата-свита (offline data/*.csv) -> (pairs, feats, targets, returns, M)."""
    man = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    pairs = sorted({m["symbol"] for m in man})
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


def run_fold(pairs, feats, targets, returns, tr_end, te_end, H=10):
    """Один шаг walk-forward: обучение на [0,tr_end), тест [tr_end,te_end).
    Возвращает потоки доходностей по барам (wide, base) и средний scale."""
    # ---- Stage 1: собственные мульти-ТФ фичи ----
    proba1 = {}
    for s in pairs:
        m = q6.train_model(feats[s][:tr_end], targets[s][:tr_end])
        proba1[s] = m.predict_proba(feats[s])[:, 1]

    # ---- Stage 2: головы с широким контекстом (запазд. вероятности всех пар) ----
    lag = 1
    proba1_mat = np.column_stack(
        [np.concatenate([np.full(lag, 0.5), proba1[s][:-lag]]) for s in pairs])
    proba2 = {}
    for s in pairs:
        X2 = np.column_stack([feats[s], proba1_mat])
        m = q6.train_model(X2[:tr_end], targets[s][:tr_end])
        proba2[s] = m.predict_proba(X2)[:, 1]

    proba2_mat = np.column_stack([proba2[s] for s in pairs])      # (M, P)
    sig = (proba2_mat > 0.5).astype(float) * 2 - 1               # (M, P) сигналы -1/1
    port_ret = (sig * np.column_stack([returns[s] for s in pairs])).mean(axis=1)

    # ---- Meta-контроллер риска (РЕЖИМНЫЙ) ----
    # фичи (всё causal — только до бара t):
    roll = pd.Series(port_ret).rolling(50).std().fillna(port_ret.std()).values
    recent = pd.Series(port_ret).rolling(20).sum().fillna(0.0).values
    eq_proxy = np.cumprod(1 + port_ret)
    dd = (np.maximum.accumulate(eq_proxy) - eq_proxy) / np.maximum.accumulate(eq_proxy)
    meta_X = np.column_stack([proba2_mat, proba2_mat.mean(axis=1),
                              proba2_mat.std(axis=1), roll, recent, dd])
    # метка-учитель: профит портфеля на горизонте H баров вперёд (>=0 -> в рынке)
    cum = np.concatenate([[0.0], np.cumsum(port_ret)])
    fwd = np.zeros(len(port_ret))
    for t in range(len(port_ret)):
        e = min(t + H, len(port_ret))
        fwd[t] = cum[e] - cum[t + 1]          # сумма доходностей t+1 .. t+H
    meta_y = (fwd > 0).astype(int)
    meta_Xm = meta_X[:-H]
    meta_ym = meta_y[:-H]
    mmeta = q6.train_model(meta_Xm[:tr_end - H], meta_ym[:tr_end - H])
    meta_scale = mmeta.predict_proba(meta_Xm)[:, 1]                # (M-H,) P(форвард-профит>0)
    # РЕЖИМНЫЙ risk-off: при P<=0.5 -> плоская, при уверенном вверх -> полный риск
    meta_scale = np.clip(2.0 * (meta_scale - 0.5), 0.0, 1.0)
    scale_full = np.ones(len(port_ret))
    scale_full[1:1 + len(meta_scale)] = meta_scale                # scale[t] предсказан на t-1

    # ---- Портфельная симуляция на тесте ----
    sig_test = sig[tr_end:te_end]
    scale_test = scale_full[tr_end:te_end]
    ret_test = np.column_stack([returns[s][tr_end:te_end] for s in pairs])
    pr_w, pr_b = [], []
    for t in range(len(sig_test)):
        pr = float(np.mean(sig_test[t] * ret_test[t]))
        pr_w.append(pr * scale_test[t])
        pr_b.append(pr)
    return np.array(pr_w), np.array(pr_b), scale_test, sig_test


def main():
    pairs, feats, targets, returns, M = build_data()
    print(f"Пары ({len(pairs)}): {pairs}")
    for s in pairs:
        print(f"  {s}: баров={len(feats[s])}")

    H = 10
    FOLD = 900
    tr_end = BARS_TRAIN
    print(f"Walk-forward: обучение [0,tr_end) расширяющееся, шаг теста={FOLD} баров, "
          f"горизонт мета H={H}")

    pr_w_all, pr_b_all = [], []
    print("\n--- По окнам (форвард, честно) ---")
    k = 0
    while tr_end + FOLD <= M:
        te_end = tr_end + FOLD
        # нормализация по паре на актуальном train-окне (expanding)
        feats_n = {}
        for s in pairs:
            sc = StandardScaler().fit(feats[s][:tr_end])
            feats_n[s] = sc.transform(feats[s])
        pr_w, pr_b, scale_test, sig_test = run_fold(pairs, feats_n, targets, returns, tr_end, te_end, H)
        sc = float(np.mean(scale_test))
        pr_w_all.append(pr_w)
        pr_b_all.append(pr_b)
        rw = (np.prod(1 + pr_w) - 1) * 100
        rb = (np.prod(1 + pr_b) - 1) * 100
        print(f"Fold {k}: train[:{tr_end}) test[{tr_end},{te_end})  "
              f"Wide={rw:6.1f}%  Base={rb:6.1f}%  scale={sc:.2f}")
        k += 1
        tr_end += FOLD

    pr_w = np.concatenate(pr_w_all)
    pr_b = np.concatenate(pr_b_all)
    eq_w = np.cumprod(1 + pr_w)
    eq_b = np.cumprod(1 + pr_b)
    rw, dw = metrics(eq_w)
    rb, db = metrics(eq_b)
    wins = sum(1 for a, b in zip(pr_w_all, pr_b_all)
               if np.prod(1 + a) > np.prod(1 + b))
    print("\n=== ИТОГО walk-forward (цепочка окон, M15) ===")
    print(f"Wide (мультиголовы + режимная мета) : доход={rw*100:6.1f}%  maxDD={dw*100:6.1f}%")
    print(f"Base (равновес без мета)            : доход={rb*100:6.1f}%  maxDD={db*100:6.1f}%")
    print(f"Окон, где Wide > Base: {wins}/{len(pr_w_all)}")
    pd.DataFrame({"bar": np.arange(len(eq_w)),
                  "eq_wide": eq_w,
                  "eq_base": eq_b}).to_csv(HERE / "wide_equity.csv", index=False)
    print("Эквити сохранено в wide_equity.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(eq_w, label="Wide (мультиголовы + режимная мета)", color="green", lw=1.6)
        ax.plot(eq_b, label="Base (равновес без мета)", color="gray", lw=1.2, alpha=0.85)
        ax.set_title("Портфельная эквити — широкий мозг (walk-forward, M15)")
        ax.set_xlabel("Бар (цепочка окон)"); ax.set_ylabel("Equity")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(HERE / "wide_equity.png", dpi=120)
    except Exception as e:
        print("Предупреждение: не удалось построить PNG:", e)


if __name__ == "__main__":
    main()
