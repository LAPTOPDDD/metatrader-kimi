#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crypto_backtest.py — исследование: тот же "wide-brain V2.1" (база) на КРИПТЕ.
Берём историю из подключённого MT5-терминала (крипта торгуется 24/7),
считаем пер-парный Stage-1 + равновесный портфель в walk-forward, строим эквити.
Живых ордеров НЕ шлём — только оценка.
"""
import datetime
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

import importlib.util
from pathlib import Path
import MetaTrader5 as mt5

HERE = Path(__file__).parent

# переиспользуем MT5-хелперы и "мозг" из живого бота
_qspec = importlib.util.spec_from_file_location("bot", HERE / "mt5_live_bot.py")
bot = importlib.util.module_from_spec(_qspec)
_qspec.loader.exec_module(bot)

# ============================ К О Н Ф И Г ================================
SYMBOLS     = bot.CRYPTO_PAIRS          # крипто-пресет из бота
BASE_TF     = "M15"
HIGHER      = ["M30", "H1", "H4"]
SIGMA       = 1.5
N_BARS      = 15000     # сколько M15-баров истории берём (24/7 => ~156 дней)
BARS_TRAIN  = 10000     # обучение на старте
FOLD        = 900       # шаг walk-forward
NORM        = True
# ========================================================================

TF_MIN = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}


def main():
    print("=" * 60)
    print(" CRYPTO BACKTEST — wide-brain V2.1 (база) на крипте")
    print("=" * 60)
    if not bot.connect():
        return

    pairs = []
    feats, closes, rets, ys = {}, {}, {}, {}
    for s in SYMBOLS:
        f = bot.build_pair_features(s, N_BARS, HIGHER)
        if f is None:
            print(f"  ! {s}: нет данных — пропуск")
            continue
        X, close, _ = f
        if X.shape[0] < BARS_TRAIN + 600:
            print(f"  ! {s}: мало баров ({X.shape[0]}) — пропуск")
            continue
        # 3-классовый target с учётом спреда
        si = mt5.symbol_info(s)
        spread_cost = (si.spread * si.point) if (si and si.point) else 0.0
        grad = np.gradient(gaussian_filter1d(close, SIGMA, mode="reflect"))
        y = np.where(grad > spread_cost, 2, np.where(grad < -spread_cost, 0, 1)).astype(np.int8)
        ret = close[1:] / close[:-1] - 1.0
        feats[s] = X
        closes[s] = close
        rets[s] = ret
        ys[s] = y
        pairs.append(s)
    if not pairs:
        print("Нет годных пар. Выход.")
        return
    print(f"Пары ({len(pairs)}): {pairs}")

    # выровнять по минимальной длине
    M = min(len(feats[s]) for s in pairs)
    for s in pairs:
        feats[s] = feats[s][:M]
        rets[s] = rets[s][:M - 1]
        ys[s] = ys[s][:M]
    print(f"Баров (выровнено): {M}")

    # честные транзакционные издержки: спред при каждом входе/развороте
    cost_pairs = {}
    for s in pairs:
        si = mt5.symbol_info(s)
        if si is None or si.point == 0 or si.bid == 0:
            cost_pairs[s] = 0.0
        else:
            cost_pairs[s] = si.spread * si.point / si.bid

    pr_all = []
    tr_end = BARS_TRAIN
    k = 0
    pos_prev = {s: 0 for s in pairs}
    while tr_end + FOLD <= M - 1:
        te_end = tr_end + FOLD
        sig_pairs = {}
        for s in pairs:
            X = feats[s]
            sc = StandardScaler().fit(X[:tr_end]) if NORM else None
            Xt = sc.transform(X) if NORM else X
            model = bot.train_model_multi(Xt[:tr_end], ys[s][:tr_end])
            proba = model.predict_proba(Xt)          # (N, 3) [short, hold, long]
            want = np.argmax(proba, axis=1)          # 0=short, 1=hold, 2=long
            side = np.where(want == 2, 1, np.where(want == 0, -1, 0))
            sig_pairs[s] = side[tr_end:te_end]
        fold = []
        for t in range(te_end - tr_end):
            idx = tr_end + t
            contribs = []
            for s in pairs:
                side = sig_pairs[s][t]
                if side == 0:                    # hold — держим предыдущую позицию
                    c = pos_prev[s] * rets[s][idx]
                else:
                    c = side * rets[s][idx]
                    if side != pos_prev[s]:      # вход или разворот — платим спред
                        c -= cost_pairs[s]
                    pos_prev[s] = side
                contribs.append(c)
            fold.append(float(np.mean(contribs)))
        pr_all.extend(fold)
        print(f"Fold {k}: train[:{tr_end}) test[{tr_end},{te_end})  доход={ (np.prod(1+np.array(fold))-1)*100:6.1f}%")
        tr_end += FOLD
        k += 1

    pr = np.array(pr_all)
    eq = np.cumprod(1 + pr)
    ret_total = eq[-1] - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    print("\n=== ИТОГО (крипта, walk-forward, M15) ===")
    print(f"Портфель (Stage-1, без мета): доход={ret_total*100:6.1f}%  maxDD={dd*100:6.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(eq, color="orange", lw=1.6, label="Crypto portfolio (база)")
        ax.set_title("Портфельная эквити — wide-brain на КРИПТЕ (walk-forward, M15)")
        ax.set_xlabel("Бар (цепочка окон)"); ax.set_ylabel("Equity")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(HERE / "crypto_equity.png", dpi=120)
        print("График: crypto_equity.png")
    except Exception as e:
        print("Предупреждение: не удалось построить PNG:", e)


if __name__ == "__main__":
    main()
