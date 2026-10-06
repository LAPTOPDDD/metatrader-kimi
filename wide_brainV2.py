#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wide_brainV2.py — «широкий мозг» V2.
Все настройки вынесены в блок КОНФИГ в начале скрипта — крутите и перезапускайте.
В конце строится график эквити (wide_equity.png) и печатается сводка.

Архитектура (честная, без утечки в будущее):
  Stage 1 — базовая модель каждой пары на своих мульти-ТФ фичах (q6-совместимых).
  Stage 2 — «головы» по паре: к своим фичам добавляются ЗАПАЗДЫВАЮЩИЕ
            вероятности ВСЕХ пар (широкий контекст рынка) -> уточнённый сигнал.
  Meta    — контроллер риска: предсказывает профит портфеля на горизонте H баров
            вперёд и выдаёт scale [0,1] (risk-off в шторм).
Портфельная симуляция — walk-forward (расширяющееся окно обучения), бар за баром.
Данные — оффлайн из data/*.csv (см. download_data.py).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import importlib.util

# ======================================================================
# ============================ К О Н Ф И Г ==============================
# ======================================================================
# Играйте здесь. После правки — просто перезапустите скрипт.
SIGMA        = 1.5     # сглаживание цели: наклон сглаженной цены (0.5..3.0)
BASE_TF      = "M15"   # базовый таймфрейм, на котором принимаются решения
HIGHER       = ["M30", "H1", "H4"]   # старшие ТФ, подшиваемые как контекст
BARS_TRAIN   = 1000   # размер ОБУЧАЮЩЕГО окна (с чего начинается walk-forward)
FOLD         = 100     # длина ОДНОГО тестового окна (шаг прокрутки)
N_FOLDS      = 10    # сколько окон гнать (None = пока хватает данных)
H_META       = 10      # горизонт мета-контроллера: баров вперёд для метки
META_MODE    = "riskoff"  # "riskoff": scale=clip(2*(P-0.5),0,1)  |  "linear": scale=P
NORM         = True    # нормализовать фичи по паре на train-статах
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
    """Загрузка дата-свита -> (pairs, feats, targets, returns, M)."""
    man = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    avail = sorted({m["symbol"] for m in man})
    pairs = [s for s in avail if pairs_filter is None or s in pairs_filter]
    if pairs_filter is not None and len(pairs) < len(pairs_filter):
        missing = set(pairs_filter) - set(pairs)
        print(f"  ! нет данных для: {missing}")
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


def run_fold(pairs, feats, targets, returns, tr_end, te_end, H=H_META, meta_mode=META_MODE):
    """Один шаг walk-forward. Возвращает (pr_w, pr_b, scale_test, sig_test)."""
    proba1 = {}
    for s in pairs:
        m = q6.train_model(feats[s][:tr_end], targets[s][:tr_end])
        proba1[s] = m.predict_proba(feats[s])[:, 1]

    lag = 1
    proba1_mat = np.column_stack(
        [np.concatenate([np.full(lag, 0.5), proba1[s][:-lag]]) for s in pairs])
    proba2 = {}
    for s in pairs:
        X2 = np.column_stack([feats[s], proba1_mat])
        m = q6.train_model(X2[:tr_end], targets[s][:tr_end])
        proba2[s] = m.predict_proba(X2)[:, 1]

    proba2_mat = np.column_stack([proba2[s] for s in pairs])
    sig = (proba2_mat > 0.5).astype(float) * 2 - 1
    port_ret = (sig * np.column_stack([returns[s] for s in pairs])).mean(axis=1)

    # ---- Meta-контроллер риска (РЕЖИМНЫЙ) ----
    roll = pd.Series(port_ret).rolling(50).std().fillna(port_ret.std()).values
    recent = pd.Series(port_ret).rolling(20).sum().fillna(0.0).values
    eq_proxy = np.cumprod(1 + port_ret)
    dd = (np.maximum.accumulate(eq_proxy) - eq_proxy) / np.maximum.accumulate(eq_proxy)
    meta_X = np.column_stack([proba2_mat, proba2_mat.mean(axis=1),
                              proba2_mat.std(axis=1), roll, recent, dd])
    cum = np.concatenate([[0.0], np.cumsum(port_ret)])
    fwd = np.zeros(len(port_ret))
    for t in range(len(port_ret)):
        e = min(t + H, len(port_ret))
        fwd[t] = cum[e] - cum[t + 1]
    meta_y = (fwd > 0).astype(int)
    meta_Xm = meta_X[:-H]
    meta_ym = meta_y[:-H]
    mmeta = q6.train_model(meta_Xm[:tr_end - H], meta_ym[:tr_end - H])
    meta_scale = mmeta.predict_proba(meta_Xm)[:, 1]
    if meta_mode == "linear":
        pass  # scale = P(вверх)
    else:  # riskoff: при P<=0.5 -> плоская, при уверенном вверх -> полный риск
        meta_scale = np.clip(2.0 * (meta_scale - 0.5), 0.0, 1.0)
    scale_full = np.ones(len(port_ret))
    scale_full[1:1 + len(meta_scale)] = meta_scale

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
    print("=== КОНФИГ ===")
    print(f"  SIGMA={SIGMA} BASE_TF={BASE_TF} HIGHER={HIGHER}")
    print(f"  BARS_TRAIN={BARS_TRAIN} FOLD={FOLD} N_FOLDS={N_FOLDS} H_META={H_META}")
    print(f"  META_MODE={META_MODE} NORM={NORM} PAIRS={PAIRS or 'все'}")

    pairs, feats, targets, returns, M = build_data(PAIRS)
    print(f"Пары ({len(pairs)}): {pairs}  | баров: {M}")

    tr_end = BARS_TRAIN
    pr_w_all, pr_b_all = [], []
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
        pr_w, pr_b, scale_test, sig_test = run_fold(pairs, feats_n, targets, returns, tr_end, te_end)
        pr_w_all.append(pr_w)
        pr_b_all.append(pr_b)
        rw = (np.prod(1 + pr_w) - 1) * 100
        rb = (np.prod(1 + pr_b) - 1) * 100
        print(f"Fold {k}: train[:{tr_end}) test[{tr_end},{te_end})  "
              f"Wide={rw:6.1f}%  Base={rb:6.1f}%  scale={float(np.mean(scale_test)):.2f}")
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
    print(f"Wide (мультиголовы + мета:{META_MODE}) : доход={rw*100:6.1f}%  maxDD={dw*100:6.1f}%")
    print(f"Base (равновес без мета)              : доход={rb*100:6.1f}%  maxDD={db*100:6.1f}%")
    print(f"Окон, где Wide > Base: {wins}/{len(pr_w_all)}")
    pd.DataFrame({"bar": np.arange(len(eq_w)),
                  "eq_wide": eq_w,
                  "eq_base": eq_b}).to_csv(HERE / "wide_equity.csv", index=False)
    print("Эквити сохранено в wide_equity.csv")

    # ---- ГРАФИК ЭКВИТИ (в конце) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(eq_w, label=f"Wide (мета:{META_MODE})", color="green", lw=1.6)
        ax.plot(eq_b, label="Base (без мета)", color="gray", lw=1.2, alpha=0.85)
        ax.set_title("Портфельная эквити — широкий мозг V2 (walk-forward, M15)")
        ax.set_xlabel("Бар (цепочка окон)"); ax.set_ylabel("Equity")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(HERE / "wide_equity.png", dpi=120)
        print("График сохранён в wide_equity.png")
    except Exception as e:
        print("Предупреждение: не удалось построить PNG:", e)


if __name__ == "__main__":
    main()
