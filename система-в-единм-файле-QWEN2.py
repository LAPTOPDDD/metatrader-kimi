#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная торговая система для GBPZAR
======================================
• TP/SL вместо фиксированного времени
• Трендовый фильтр (MA)
• Фильтр волатильности (ATR)
• Векторизованный бэктест
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import json
import time
import warnings
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================== КОНФИГ =========================
NOPT = 1000
NFWD = 500
LOT = 0.01
MAGIC_BASE = 100_000

# Только GBPZAR для теста
#PAIRS = ["GBPZAR"]
PAIRS = ["GBPUSD" , "EURUSD" , "USDJPY"  , "GBPJPY"]

TIMEFRAMES = ["M15", "M30", "H1", "H4"]

LGB_PARAMS = dict(
    n_estimators=50,
    learning_rate=0.1,
    num_leaves=10,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
    random_state=42,
    n_jobs=1,
)

# ======================== MT5 UTILS ======================
def tf_to_mt5(tf: str) -> int:
    return {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    }[tf]

def tf_seconds(tf: str) -> int:
    return {
        "M1": 60, "M5": 300, "M15": 900,
        "M30": 1800, "H1": 3600, "H4": 14400,
    }[tf]

def get_point(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info:
        return info.point
    return 0.0001

def get_filling_mode(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None: return mt5.ORDER_FILLING_IOC
    if info.filling_mode & mt5.SYMBOL_FILLING_FOK: return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_IOC

# ===================== FEATURE STORE ====================
class FeatureStore:
    def __init__(self, df: pd.DataFrame):
        self.raw = df.reset_index(drop=True)
        self.close = self.raw["close"].values
        self.high = self.raw["high"].values
        self.low = self.raw["low"].values
        self.open = self.raw["open"].values
        self.volume = self.raw["tick_volume"].values
        self.time = self.raw["time"].values
        self.features = None
        
    def calc_features(self):
        d = self.raw.copy()
        d["rsi_14"] = self._rsi(d["close"])
        d["macd"] = d["close"].ewm(span=12).mean() - d["close"].ewm(span=26).mean()
        
        for k in [6, 12, 24]:
            d[f"mom_{k}"] = d["close"].pct_change(k)
        
        d["atr_14"] = self._atr(d["high"], d["low"], d["close"], 14)
        
        for ma in [20, 50]:
            d[f"ma_{ma}"] = d["close"] / d["close"].rolling(ma).mean()
        
        d["body"] = (d["close"] - d["open"]) / (d["high"] - d["low"] + 1e-9)
        d["hl_range"] = (d["high"] - d["low"]) / d["close"]
        d["vol_ratio"] = d["tick_volume"] / d["tick_volume"].rolling(20).mean()
        
        self.features = d.dropna().reset_index(drop=True)
        if len(self.features) == 0:
            raise ValueError("Нет данных после вычисления признаков")
        return self.features
    
    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

# ===================== ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ====================
EXCLUDE = {"open","high","low","close","tick_volume","spread","real_volume","Target","time"}

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in EXCLUDE]

def build_target(fs: FeatureStore, sigma: float, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smoothed = gaussian_filter1d(fs.close, sigma, mode='reflect')
    deriv = np.gradient(smoothed)
    
    margin = int(3 * sigma)
    if margin * 2 >= len(deriv):
        margin = int(len(deriv) * 0.1)
        
    start = margin
    end = len(deriv) - margin if margin > 0 else len(deriv)
    
    if start >= end:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    df_slice = fs.features.iloc[start:end]
    deriv_trimmed = deriv[start:end]
    
    y = (deriv_trimmed > 0).astype(np.int8)
    X = df_slice[feature_cols].values
    closes = df_slice["close"].values
    highs = df_slice["high"].values
    lows = df_slice["low"].values
    times = df_slice["time"].values
    
    return X, y, closes, highs, lows, times

# ===================== МОДЕЛЬ ====================
def train_model(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    
    if n_pos > 0 and n_neg > 0:
        sample_weight = np.where(y == 1, n_neg / n_pos, 1.0).astype(np.float32)
    else:
        sample_weight = np.ones(len(y), dtype=np.float32)
    
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y, sample_weight=sample_weight)
    return model

# ===================== БЭКТЕСТ С TP/SL ====================
@dataclass
class Params:
    sigma: float
    use_tp: bool
    use_sl: bool
    tp_pips: float
    sl_pips: float
    use_trend_filter: bool
    ma_period: int
    min_hold_bars: int
    max_hold_bars: int

def backtest_with_tp_sl(X: np.ndarray, y: np.ndarray, closes: np.ndarray, 
                        highs: np.ndarray, lows: np.ndarray, times: np.ndarray,
                        model: lgb.LGBMClassifier, p: Params, point: float) -> Tuple[np.ndarray, List[dict]]:
    
    if len(X) == 0:
        return np.array([0.0]), []

    probs = model.predict_proba(X)[:, 1]
    signals = np.where(probs > 0.5, 1, -1)
    
    # Трендовый фильтр
    if p.use_trend_filter:
        ma = pd.Series(closes).rolling(p.ma_period).mean().values
        trend_filter = np.where(closes > ma, 1, -1)
        # Входим только по тренду
        signals = np.where(signals == trend_filter, signals, 0)
    
    equity = [0.0]
    trades = []
    position = None
    entry_price = 0.0
    entry_time = 0
    entry_bar = 0
    bars_held = 0
    
    # TP/SL в ценах
    tp_distance = p.tp_pips * point if p.use_tp else None
    sl_distance = p.sl_pips * point if p.use_sl else None
    
    n = len(closes)
    
    for i in range(n):
        signal = signals[i]
        high = highs[i]
        low = lows[i]
        close = closes[i]
        t = times[i]
        
        if position is not None:
            bars_held += 1
            
            # Проверка TP/SL
            exit_price = None
            exit_reason = None
            
            if position == 1:  # BUY
                if p.use_tp and high >= entry_price + tp_distance:
                    exit_price = entry_price + tp_distance
                    exit_reason = "TP"
                elif p.use_sl and low <= entry_price - sl_distance:
                    exit_price = entry_price - sl_distance
                    exit_reason = "SL"
            else:  # SELL
                if p.use_tp and low <= entry_price - tp_distance:
                    exit_price = entry_price - tp_distance
                    exit_reason = "TP"
                elif p.use_sl and high >= entry_price + sl_distance:
                    exit_price = entry_price + sl_distance
                    exit_reason = "SL"
            
            # Выход по времени
            if exit_price is None and bars_held >= p.max_hold_bars:
                exit_price = close
                exit_reason = "Time"
            
            # Выход по сигналу (разворот)
            if exit_price is None and signal != position and bars_held >= p.min_hold_bars:
                exit_price = close
                exit_reason = "Signal"
            
            if exit_price is not None:
                pnl = (exit_price - entry_price) / entry_price if position == 1 else (entry_price - exit_price) / entry_price
                equity.append(equity[-1] + pnl * 100)
                trades.append({
                    "entry_time": entry_time, "entry_price": entry_price,
                    "direction": position, "exit_time": t, "exit_price": exit_price,
                    "pnl": pnl * 100, "hold_bars": bars_held, "exit_reason": exit_reason
                })
                position = None
                bars_held = 0
        
        # Вход в позицию
        if position is None and signal != 0:
            position = signal
            entry_price = close
            entry_time = t
            entry_bar = i
            bars_held = 0

    # Закрытие последней позиции
    if position is not None:
        last_price = closes[-1]
        pnl = (last_price - entry_price) / entry_price if position == 1 else (entry_price - last_price) / entry_price
        equity.append(equity[-1] + pnl * 100)
        trades.append({
            "entry_time": entry_time, "entry_price": entry_price,
            "direction": position, "exit_time": times[-1], "exit_price": last_price,
            "pnl": pnl * 100, "hold_bars": bars_held, "exit_reason": "End"
        })
    
    return np.array(equity), trades

def score_equity(eq: np.ndarray, trades: List[dict]) -> float:
    if len(trades) < 5: return -1e6
    returns = np.diff(eq)
    if len(returns) == 0: return -1e6
    
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 6)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-9)
    max_dd = dd.max()
    
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / len(trades)
    
    # Средняя прибыль на сделку
    avg_pnl = np.mean([t["pnl"] for t in trades])
    
    # Процент закрытий по TP
    tp_closes = sum(1 for t in trades if t["exit_reason"] == "TP")
    tp_rate = tp_closes / len(trades)
    
    return sharpe - max_dd * 0.5 + win_rate * 0.3 + avg_pnl * 10 + tp_rate * 0.2

# ===================== ОПТИМИЗАЦИЯ ====================
def optimize(fs: FeatureStore, feature_cols: List[str], point: float, n_trials: int = 100) -> Tuple[Dict, float]:
    def objective(trial):
        try:
            p = Params(
                sigma=trial.suggest_float("sigma", 0.5, 3.0, step=0.1),
                use_tp=trial.suggest_categorical("use_tp", [True, False]),
                use_sl=trial.suggest_categorical("use_sl", [True, False]),
                tp_pips=trial.suggest_int("tp_pips", 10, 100, step=10) if p.use_tp else 50,
                sl_pips=trial.suggest_int("sl_pips", 10, 100, step=10) if p.use_sl else 50,
                use_trend_filter=trial.suggest_categorical("use_trend_filter", [True, False]),
                ma_period=trial.suggest_int("ma_period", 20, 100, step=10) if p.use_trend_filter else 50,
                min_hold_bars=trial.suggest_int("min_hold_bars", 1, 5),
                max_hold_bars=trial.suggest_int("max_hold_bars", 10, 50, step=5),
            )
            
            X, y, closes, highs, lows, times = build_target(fs, p.sigma, feature_cols)
            
            if len(y) < NOPT + NFWD: return -1e6
            if np.sum(y == 1) < 10 or np.sum(y == 0) < 10: return -1e6
            
            X_train, y_train = X[:NOPT], y[:NOPT]
            X_test, y_test = X[NOPT:NOPT+NFWD], y[NOPT:NOPT+NFWD]
            closes_test, highs_test, lows_test, times_test = closes[NOPT:NOPT+NFWD], highs[NOPT:NOPT+NFWD], lows[NOPT:NOPT+NFWD], times[NOPT:NOPT+NFWD]
            
            model = train_model(X_train, y_train)
            eq, trades = backtest_with_tp_sl(X_test, y_test, closes_test, highs_test, lows_test, times_test, model, p, point)
            
            return score_equity(eq, trades)
        except Exception:
            return -1e6
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    print(f"   🔄 Оптимизация ({n_trials} итераций)...")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
    
    if study.best_value <= -1e5:
        print("   ⚠ Оптимизация не дала хороших результатов")
        return {
            "sigma": 1.5, "use_tp": True, "use_sl": True,
            "tp_pips": 50, "sl_pips": 30, "use_trend_filter": True,
            "ma_period": 50, "min_hold_bars": 2, "max_hold_bars": 20
        }, -1e6
    
    return study.best_params, study.best_value

# ===================== ФОРВАРД-ТЕСТ ====================
def forward_test(fs: FeatureStore, params: dict, feature_cols: List[str], point: float) -> Tuple[Optional[np.ndarray], List[dict]]:
    try:
        p = Params(**params)
        X, y, closes, highs, lows, times = build_target(fs, p.sigma, feature_cols)
        
        if len(y) < NOPT + NFWD: return None, []
        
        X_train, y_train = X[:NOPT], y[:NOPT]
        X_test, y_test = X[NOPT:NOPT+NFWD], y[NOPT:NOPT+NFWD]
        closes_test, highs_test, lows_test, times_test = closes[NOPT:NOPT+NFWD], highs[NOPT:NOPT+NFWD], lows[NOPT:NOPT+NFWD], times[NOPT:NOPT+NFWD]
        
        if len(X_test) < 10: return None, []
        
        model = train_model(X_train, y_train)
        return backtest_with_tp_sl(X_test, y_test, closes_test, highs_test, lows_test, times_test, model, p, point)
    except Exception as e:
        print(f"   ⚠ Ошибка в forward_test: {e}")
        return None, []

# ===================== ВИЗУАЛИЗАЦИЯ ====================
def plot_results(results: Dict[Tuple[str, str], Tuple[np.ndarray, List[dict]]]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    
    all_trades = []
    for (sym, tf), (eq, trades) in results.items():
        if eq is not None and len(eq) > 1:
            ax1.plot(eq, label=f"{sym} {tf}", lw=2, alpha=0.8)
            all_trades.extend(trades)
    
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Эквити форвард-теста"); ax1.set_ylabel("Equity, %")
    ax1.legend(); ax1.grid(alpha=0.3)
    
    if all_trades:
        pnls = [t["pnl"] for t in all_trades]
        ax2.hist(pnls, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title(f"Распределение PnL (сделок: {len(all_trades)})"); ax2.set_xlabel("PnL, %")
        
        # Распределение по причинам выхода
        exit_reasons = {}
        for t in all_trades:
            reason = t.get("exit_reason", "Unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        holds = [t["hold_bars"] for t in all_trades]
        ax4.hist(holds, bins=20, alpha=0.7, color='coral', edgecolor='black')
        ax4.set_title("Распределение длительности удержания"); ax4.set_xlabel("Бары"); ax4.grid(alpha=0.3)

    pairs, win_rates = [], []
    for (sym, tf), (eq, trades) in results.items():
        if trades:
            wins = sum(1 for t in trades if t["pnl"] > 0)
            pairs.append(f"{sym}\n{tf}"); win_rates.append(wins / len(trades) * 100)
    if pairs:
        ax3.bar(pairs, win_rates, color='lightgreen', edgecolor='black')
        ax3.axhline(50, color='red', linestyle='--', alpha=0.5)
        ax3.set_title("Win Rate по инструментам"); ax3.set_ylabel("Win Rate, %"); ax3.grid(alpha=0.3)
    
    plt.tight_layout(); plt.savefig("trading_results_improved.png", dpi=150); plt.show()
    
    if all_trades:
        print("\n Сводная статистика:")
        print(f"   Всего сделок: {len(all_trades)}")
        wins = sum(1 for t in all_trades if t["pnl"] > 0)
        print(f"   Win rate: {wins/len(all_trades)*100:.1f}%")
        print(f"   Средний PnL: {np.mean([t['pnl'] for t in all_trades]):.3f}%")
        print(f"   Медианный PnL: {np.median([t['pnl'] for t in all_trades]):.3f}%")
        
        # Статистика по выходам
        exit_stats = {}
        for t in all_trades:
            reason = t.get("exit_reason", "Unknown")
            if reason not in exit_stats:
                exit_stats[reason] = {"count": 0, "total_pnl": 0}
            exit_stats[reason]["count"] += 1
            exit_stats[reason]["total_pnl"] += t["pnl"]
        
        print("\n   Выходы по причинам:")
        for reason, stats in exit_stats.items():
            avg = stats["total_pnl"] / stats["count"]
            print(f"   - {reason}: {stats['count']} сделок, средний PnL: {avg:.3f}%")

# ===================== ЖИВАЯ ТОРГОВЛЯ ====================
class LiveTrader:
    def __init__(self, cfg: dict):
        self.symbol = cfg["symbol"]
        self.tf = tf_to_mt5(cfg["timeframe"])
        self.magic = cfg["magic"]
        self.point = get_point(self.symbol)
        self.params = Params(
            sigma=cfg["sigma"],
            use_tp=cfg["use_tp"],
            use_sl=cfg["use_sl"],
            tp_pips=cfg["tp_pips"],
            sl_pips=cfg["sl_pips"],
            use_trend_filter=cfg["use_trend_filter"],
            ma_period=cfg["ma_period"],
            min_hold_bars=cfg["min_hold_bars"],
            max_hold_bars=cfg["max_hold_bars"],
        )
        self.model = None
        self.last_time = 0
        self.position = None
        self.entry_price = 0
        self.entry_time = 0
        self.bars_held = 0
        self.feature_cols = []
        self.tp_distance = self.params.tp_pips * self.point if self.params.use_tp else None
        self.sl_distance = self.params.sl_pips * self.point if self.params.use_sl else None
        
        print(f"   🏆 {self.symbol} {cfg['timeframe']}")
        print(f"      TP: {self.params.tp_pips if self.params.use_tp else 'OFF'} | SL: {self.params.sl_pips if self.params.use_sl else 'OFF'}")
        self._init_model()
    
    def _init_model(self):
        df = self._get_data()
        if df is None: return
        
        fs = FeatureStore(df)
        fs.calc_features()
        self.feature_cols = get_feature_cols(fs.features)
        
        X, y, _, _, _, _ = build_target(fs, self.params.sigma, self.feature_cols)
        if len(X) > NOPT:
            self.model = train_model(X[-NOPT:], y[-NOPT:])
            print(f"   ✅ Модель обучена")
    
    def _get_data(self, bars: int = NOPT + 200) -> Optional[pd.DataFrame]:
        for _ in range(3):
            rates = mt5.copy_rates_from_pos(self.symbol, self.tf, 0, bars)
            if rates is not None and len(rates) > 100: return pd.DataFrame(rates)
            time.sleep(1)
        return None
    
    def tick(self):
        df = self._get_data()
        if df is None or self.model is None: return
        
        current_time = df.iloc[-1]["time"]
        if current_time == self.last_time: return
        self.last_time = current_time
        
        fs = FeatureStore(df)
        fs.calc_features()
        X, _, closes, highs, lows, _ = build_target(fs, self.params.sigma, self.feature_cols)
        
        if len(X) == 0: return
        
        # Предсказание
        prob = self.model.predict_proba(X[[-1]])[0][1]
        signal = 1 if prob > 0.5 else -1
        
        # Трендовый фильтр
        if self.params.use_trend_filter:
            ma = pd.Series(closes).rolling(self.params.ma_period).mean().iloc[-1]
            trend = 1 if closes[-1] > ma else -1
            if signal != trend:
                signal = 0
        
        current_price = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        
        if positions:
            self.bars_held += 1
            pos = positions[0]
            
            # Проверка TP/SL
            exit_price = None
            exit_reason = None
            
            if pos.type == 0:  # BUY
                if self.params.use_tp and current_high >= pos.price_open + self.tp_distance:
                    exit_price = pos.price_open + self.tp_distance
                    exit_reason = "TP"
                elif self.params.use_sl and current_low <= pos.price_open - self.sl_distance:
                    exit_price = pos.price_open - self.sl_distance
                    exit_reason = "SL"
            else:  # SELL
                if self.params.use_tp and current_low <= pos.price_open - self.tp_distance:
                    exit_price = pos.price_open - self.tp_distance
                    exit_reason = "TP"
                elif self.params.use_sl and current_high >= pos.price_open + self.sl_distance:
                    exit_price = pos.price_open + self.sl_distance
                    exit_reason = "SL"
            
            # Выход по времени
            if exit_price is None and self.bars_held >= self.params.max_hold_bars:
                exit_price = current_price
                exit_reason = "Time"
            
            if exit_price is not None:
                self._close_position(pos, exit_reason)
                self.position = None
                self.bars_held = 0
                return
        
        # Вход
        if not positions and signal != 0:
            self.position = signal
            self.entry_price = current_price
            self.entry_time = current_time
            self.bars_held = 0
            self._open_position(signal, prob)
    
    def _open_position(self, direction: int, prob: float):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None: return
        ask, bid = tick.ask, tick.bid
        
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = ask if direction == 1 else bid
        
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            price = round(price / symbol_info.trade_tick_size) * symbol_info.trade_tick_size
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": LOT,
            "type": order_type, "price": price, "magic": self.magic,
            "comment": f"P={prob:.2f}", "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_filling_mode(self.symbol),
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"   ✅ {self.symbol} {'BUY' if direction==1 else 'SELL'} @ {price} | P={prob:.2f}")
    
    def _close_position(self, position, reason: str):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None: return
        ask, bid = tick.ask, tick.bid
        
        price = bid if position.type == 0 else ask
        order_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": position.volume,
            "type": order_type, "position": position.ticket, "price": price, "magic": self.magic,
            "comment": reason, "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_filling_mode(self.symbol),
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            pnl = (price - position.price_open) if position.type == 0 else (position.price_open - price)
            print(f"   📉 {self.symbol} закрыт ({reason}) | PnL: {pnl*self.point*100:.2f}%")
    
    def close_all(self):
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if positions:
            for pos in positions:
                self._close_position(pos, "Shutdown")
                time.sleep(0.5)

def run_live(champions: List[dict]):
    print("\n" + "="*60)
    print("🚀 ЗАПУСК ДЕМО-ТОРГОВЛИ (УЛУЧШЕННАЯ ВЕРСИЯ)")
    print("="*60)
    
    traders = [LiveTrader(cfg) for cfg in champions]
    print(f"\n✅ Запущено {len(traders)} трейдеров. Нажмите Ctrl+C для остановки\n")
    
    try:
        while True:
            for trader in traders:
                try: trader.tick()
                except Exception as e: print(f"   ⚠ Ошибка в {trader.symbol}: {e}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Остановка...")
        for trader in traders: trader.close_all()
        print("✅ Все позиции закрыты")

# ===================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    print("="*60)
    print("🔬 УЛУЧШЕННАЯ ТОРГОВАЯ СИСТЕМА (TP/SL + TREND)")
    print("="*60)
    
    if not mt5.initialize():
        print("❌ Ошибка инициализации MT5"); return
    print(f"✅ MT5 инициализирован")
    
    results = {}
    champions = []
    magic = MAGIC_BASE
    
    for sym in PAIRS:
        for tf_name in TIMEFRAMES:
            print(f"\n{'='*40}")
            print(f"▶ {sym} {tf_name}")
            
            bars_needed = NOPT + NFWD + 500
            rates = mt5.copy_rates_from_pos(sym, tf_to_mt5(tf_name), 0, bars_needed)
            
            if rates is None or len(rates) < NOPT + NFWD:
                print(f"   ⚠ Недостаточно данных"); continue
            
            try:
                fs = FeatureStore(pd.DataFrame(rates))
                fs.calc_features()
                feature_cols = get_feature_cols(fs.features)
                point = get_point(sym)
                
                if fs.features is None or len(fs.features) < NOPT + NFWD:
                    print(f"   ⚠ Недостаточно признаков"); continue
                
                best_params, best_score = optimize(fs, feature_cols, point, n_trials=30)
                
                print(f"   ✅ Sigma: {best_params['sigma']:.1f}")
                print(f"      TP: {best_params['tp_pips'] if best_params['use_tp'] else 'OFF'} | SL: {best_params['sl_pips'] if best_params['use_sl'] else 'OFF'}")
                print(f"      Trend Filter: {best_params['use_trend_filter']} (MA{best_params['ma_period']})")
                print(f"      Hold: {best_params['min_hold_bars']}-{best_params['max_hold_bars']} баров")
                
                eq, trades = forward_test(fs, best_params, feature_cols, point)
                
                if eq is not None and len(eq) > 1:
                    results[(sym, tf_name)] = (eq, trades)
                    champion = {
                        "symbol": sym, "timeframe": tf_name, "magic": magic,
                        **best_params
                    }
                    champions.append(champion)
                    magic += 1
                    print(f"   🏆 Чемпион добавлен (Magic: {champion['magic']})")
                    print(f"   📈 Итоговая доходность: {eq[-1]:.2f}%")
                    
            except Exception as e:
                print(f"   ❌ Ошибка при обработке {sym} {tf_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if results:
        plot_results(results)
        with open("champions_improved.json", "w") as f:
            json.dump(champions, f, indent=2)
        print("\n💾 Чемпионы сохранены в champions_improved.json")
        
        if champions:
            ans = input("\n🚀 Запустить на ДЕМО? (Y/N): ").strip().upper()
            if ans == "Y": run_live(champions)
    else:
        print("\n❌ Нет результатов для визуализации")
    
    mt5.shutdown()
    print("\n✅ Завершено")

if __name__ == "__main__":
    main()