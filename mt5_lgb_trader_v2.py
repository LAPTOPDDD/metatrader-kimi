#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt5_lgb_trader_v2.py — LightGBM Trader (Fixed & Production Ready)
Архитектура:
1. --dataset : Сбор истории 3 мес, M15. Нормализованные фичи. Target с учетом спреда.
2. --train   : Обучение LGBM. Хронологический сплит. Метрика AUC.
3. --live    : Торговля. SL/TP по ATR. Риск-менеджмент. Fallback filling.
"""
import sys, time, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# ===================== КОНФИГ =====================
SYMBOLS = ["BITCOIN","ETHEREUM","SOLANA","RIPPLE","LITECOIN","CARDANO",
           "POLKADOT","POLYGON","TRON","UNISWAP","AVALANCHE","BNB",
           "DOGECOIN","STELLAR","CHAINLINK"]
TIMEFRAME = mt5.TIMEFRAME_M15
MONTHS = 3
BARS = 90 * 24 * 4  # ~90 дней по 4 свечи в час

CSV_DATA   = Path("train_data_v2.csv")
CSV_TRADES = Path("trade_log_v2.csv")
MODEL_FILE = Path("lgb_model_v2.txt")

MAGIC = 20260831
LOT = 0.01
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.0
PROBA_BUY = 0.65
PROBA_SELL = 0.35

# Риск-менеджмент
MAX_POSITIONS = 5
DAILY_LOSS_LIMIT = -50.0
SPREAD_ATR_MAX = 0.5  # если спред > 0.5 ATR — не торгуем (слишком дорого)
# ===================================================

# ---------- ИНДИКАТОРЫ ----------
def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(com=n-1, adjust=False).mean()
    ma_dn = dn.ewm(com=n-1, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-9)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_features(df, si):
    """Считает все фичи. НОРМАЛИЗОВАНО через ATR."""
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["RSI14"] = rsi(df["close"], 14)
    df["RSI7"] = rsi(df["close"], 7)
    df["ATR14"] = atr(df, 14)
    
    # --- НОРМАЛИЗАЦИЯ ---
    df["dist_ma20"]  = (df["close"] - df["MA20"]) / (df["ATR14"] + 1e-9)
    df["dist_ma50"]  = (df["close"] - df["MA50"]) / (df["ATR14"] + 1e-9)
    df["ma20_slope"] = (df["MA20"] - df["MA20"].shift(5)) / (df["ATR14"] + 1e-9)
    df["atr_pct"]    = df["ATR14"] / df["close"]
    
    # Спред (везде в валюте, затем в ATR)
    spread_price = df.get("spread", si.spread) * si.point
    df["spread_atr"] = spread_price / (df["ATR14"] + 1e-9)
    
    # Объём
    df["vol_ratio"] = df["tick_volume"] / df["tick_volume"].rolling(20).mean()
    
    # Гэп
    df["gap_atr"] = (df["open"] - df["close"].shift(1)) / (df["ATR14"] + 1e-9)
    
    # Свеча
    df["bar_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
    
    # Доходность
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    
    # Время
    df["hour"] = pd.to_datetime(df["time"], unit="s").dt.hour
    df["dow"]  = pd.to_datetime(df["time"], unit="s").dt.dayofweek
    
    return df

def create_target(df):
    """Target с 'мёртвой зоной' (учитывает спред)."""
    spread_price = df["spread_atr"] * df["ATR14"]  # восстанавливаем валюту
    fwd = df["close"].shift(-4) - df["close"]
    cost = spread_price
    MIN_PROFIT = cost * 1.5
    
    df["target"] = np.where(fwd > MIN_PROFIT, 1,
                   np.where(fwd < -MIN_PROFIT, 0, np.nan))
    return df

# ---------- ЭТАП 1: ДАТАСЕТ ----------
def create_dataset():
    if not mt5.initialize():
        print(f"MT5 init fail: {mt5.last_error()}")
        return
    
    all_dfs = []
    for sym in SYMBOLS:
        print(f"Загрузка {sym}...")
        mt5.symbol_select(sym, True)
        si = mt5.symbol_info(sym)
        if si is None: continue
        
        rates = mt5.copy_rates_from_pos(sym, TIMEFRAME, 0, BARS)
        if rates is None or len(rates) == 0:
            print(f"  {sym}: нет данных"); continue
        
        df = pd.DataFrame(rates)
        df = build_features(df, si)
        df = create_target(df)
        df["symbol"] = sym
        df = df.dropna()
        all_dfs.append(df)
        print(f"  {sym}: {len(df)} строк готово")
    
    mt5.shutdown()
    
    if not all_dfs:
        print("Нет данных ни по одному символу!")
        return
    
    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(CSV_DATA, index=False)
    print(f"\nДатасет сохранён: {CSV_DATA} ({len(result)} строк)")

# ---------- ЭТАП 2: ОБУЧЕНИЕ ----------
def train():
    if not CSV_DATA.exists():
        print("Сначала запусти --dataset")
        return
    
    df = pd.read_csv(CSV_DATA)
    
    # Фичи (только нормализованные!)
    FEATS = ["dist_ma20","dist_ma50","ma20_slope","atr_pct","RSI14","RSI7",
             "spread_atr","vol_ratio","gap_atr","bar_range","close_pos",
             "ret_1","ret_3","hour","dow"]
    
    X = df[FEATS].fillna(0)
    y = df["target"]
    
    # Хронологический сплит (80/20)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"Обучение: {len(X_train)} строк, Тест: {len(X_test)} строк")
    print(f"Баланс классов: {y_train.mean():.2%} / {y_test.mean():.2%}")
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=100,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    # Прогноз
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"\n=== РЕЗУЛЬТАТ ===")
    print(f"AUC: {auc:.4f}")
    if auc < 0.52:
        print("⚠️ AUC < 0.52 — модель не видит паттерна. НЕ ЗАПУСКАЙ LIVE!")
    elif auc > 0.60:
        print("⚠️ AUC > 0.60 — возможна утечка данных (look-ahead bias). Проверь код!")
    else:
        print("✅ Модель обучена успешно. Можно запускать live.")
    
    model.booster_.save_model(str(MODEL_FILE))
    print(f"Модель сохранена: {MODEL_FILE}")

# ---------- ЭТАП 3: LIVE ----------
def get_live_features(sym, si):
    rates = mt5.copy_rates_from_pos(sym, TIMEFRAME, 0, 100)
    if rates is None or len(rates) < 60: return None
    
    df = pd.DataFrame(rates)
    df = build_features(df, si)
    if df["ATR14"].isna().iloc[-1]: return None
    
    row = df.iloc[-1]
    FEATS = ["dist_ma20","dist_ma50","ma20_slope","atr_pct","RSI14","RSI7",
             "spread_atr","vol_ratio","gap_atr","bar_range","close_pos",
             "ret_1","ret_3","hour","dow"]
    
    vals = [row.get(c, 0) if pd.notna(row.get(c, 0)) else 0 for c in FEATS]
    x = pd.DataFrame([vals], columns=FEATS)
    
    return x, row["ATR14"], row["close"], row["spread_atr"]

def trade_loop():
    if not mt5.initialize():
        print(f"MT5 fail: {mt5.last_error()}")
        return
    
    if not MODEL_FILE.exists():
        print("Сначала обучи модель --train")
        return
    
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    
    # Лог сделок
    if not CSV_TRADES.exists():
        pd.DataFrame(columns=["time","symbol","action","proba","price","sl","tp","result"]).to_csv(CSV_TRADES, index=False)
    
    print(f"=== LIVE START ===")
    print(f"Пороги: BUY > {PROBA_BUY}, SELL < {PROBA_SELL}")
    print(f"Риск: max {MAX_POSITIONS} поз, лимит {DAILY_LOSS_LIMIT}$")
    
    daily_start_equity = mt5.account_info().equity
    
    while True:
        # Проверка дневного лимита
        ai = mt5.account_info()
        if ai.equity - daily_start_equity < DAILY_LOSS_LIMIT:
            print(f"ДОСТИГНУТ ЛИМИТ ПРОСАДКИ! Закрываю всё.")
            for p in (mt5.positions_get() or []):
                if p.magic == MAGIC:
                    close_pos(p)
            break
        
        # Подсчёт открытых позиций
        positions = [p for p in (mt5.positions_get() or []) if p.magic == MAGIC]
        if len(positions) >= MAX_POSITIONS:
            time.sleep(15)
            continue
        
        for sym in SYMBOLS:
            if len([p for p in positions if p.symbol == sym]) > 0:
                continue  # уже есть поза
            
            si = mt5.symbol_info(sym)
            if si is None: continue
            mt5.symbol_select(sym, True)
            
            res = get_live_features(sym, si)
            if res is None: continue
            x, atr_val, close, spread_atr = res
            
            # Фильтр спреда
            if spread_atr > SPREAD_ATR_MAX:
                continue
            
            try:
                proba = booster.predict(x)[0]
            except Exception as e:
                continue
            
            action = None
            if proba > PROBA_BUY: action = "BUY"
            elif proba < PROBA_SELL: action = "SELL"
            
            if action:
                send_order(sym, action, proba, atr_val, si)
        
        time.sleep(15)

def send_order(sym, action, proba, atr_val, si):
    tick = mt5.symbol_info_tick(sym)
    if tick is None: return
    
    price = tick.ask if action == "BUY" else tick.bid
    
    # Защита SL/TP от минимальных стопов брокера
    stops = max(si.trade_stops_level, si.trade_freeze_level) + 2
    min_dist = stops * si.point
    
    sl_dist = max(atr_val * ATR_SL_MULT, min_dist)
    tp_dist = max(atr_val * ATR_TP_MULT, min_dist)
    
    sl = price - sl_dist if action == "BUY" else price + sl_dist
    tp = price + tp_dist if action == "BUY" else price - tp_dist
    
    otype = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    lot = max(si.volume_min, LOT)
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": lot,
        "type": otype,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 30,
        "magic": MAGIC,
        "comment": f"AI {proba:.2f}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    
    # Fallback filling modes
    for fm in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
        req["type_filling"] = fm
        r = mt5.order_send(req)
        if r and r.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"{datetime.datetime.now():%H:%M:%S} ✅ {sym} {action} @ {price:.5f} | proba={proba:.2f} | SL={sl:.5f} TP={tp:.5f}")
            log_trade(sym, action, proba, price, sl, tp)
            return
    
    if r:
        print(f"{datetime.datetime.now():%H:%M:%S} ❌ {sym} {action} FAIL retcode={r.retcode}")

def close_pos(pos):
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None: return
    price = tick.bid if pos.type == 0 else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "position": pos.ticket,
        "price": price,
        "deviation": 30,
        "magic": MAGIC,
        "comment": "Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    for fm in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC):
        req["type_filling"] = fm
        r = mt5.order_send(req)
        if r and r.retcode == mt5.TRADE_RETCODE_DONE: break

def log_trade(sym, action, proba, price, sl, tp):
    row = pd.DataFrame([{
        "time": datetime.datetime.now(),
        "symbol": sym,
        "action": action,
        "proba": round(proba, 2),
        "price": price,
        "sl": sl,
        "tp": tp,
        "result": ""
    }])
    row.to_csv(CSV_TRADES, mode="a", header=False, index=False)

# ===================== MAIN =====================
if __name__ == "__main__":
    if "--dataset" in sys.argv:
        create_dataset()
    elif "--train" in sys.argv:
        train()
    elif "--live" in sys.argv:
        trade_loop()
    else:
        print("Использование:")
        print("  python mt5_lgb_trader_v2.py --dataset  # сбор данных")
        print("  python mt5_lgb_trader_v2.py --train    # обучение")
        print("  python mt5_lgb_trader_v2.py --live     # торговля")