#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенная эволюционная торговая система (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)
==================================================================
• Векторизованные вычисления (NumPy вместо Pandas в циклах)
• Параллельная оптимизация Optuna (n_jobs=-1)
• LightGBM обучается на чистых numpy-массивах
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import json
import time
import sys
import io
import warnings
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional

# Принудительно переключаем вывод на UTF-8, чтобы эмодзи не ломали запуск в консоли Windows (cp1251)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings('ignore')
# Отключаем варнинги Optuna для чистого вывода
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================== КОНФИГ =========================
NOPT = 10000          # Баров для обучения
NFWD = 500           # Баров для форвард-теста
LOT = 0.01
MAGIC_BASE = 100_000

#PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD" , "USDCHF"  ]

PAIRS = [
    "GBPZAR"
]

TIMEFRAMES = ["M15", "M30", "H1" , "H4"]

# ВАЖНО: n_jobs=1, чтобы не было конфликтов с многопоточностью Optuna
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
    return 0.01 if "JPY" in symbol else 0.0001

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
        
        d["atr_14"] = (d["high"] - d["low"]).rolling(14).mean() / d["close"]
        
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

# ===================== ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ====================
EXCLUDE = {"open","high","low","close","tick_volume","spread","real_volume","Target","time"}

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in EXCLUDE]

def build_target(fs: FeatureStore, sigma: float, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает готовые NumPy массивы: X, y, closes, times.
    Это убирает необходимость работать с Pandas внутри циклов оптимизации.
    """
    smoothed = gaussian_filter1d(fs.close, sigma, mode='reflect')
    deriv = np.gradient(smoothed)
    
    margin = int(3 * sigma)
    if margin * 2 >= len(deriv):
        margin = int(len(deriv) * 0.1)
        
    start = margin
    end = len(deriv) - margin if margin > 0 else len(deriv)
    
    if start >= end:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Срез БЕЗ .copy()
    df_slice = fs.features.iloc[start:end]
    deriv_trimmed = deriv[start:end]
    
    # Целевая переменная (используем int8 для экономии памяти)
    y = (deriv_trimmed > 0).astype(np.int8)
    
    # Признаки сразу в NumPy
    X = df_slice[feature_cols].values
    closes = df_slice["close"].values
    times = df_slice["time"].values
    
    return X, y, closes, times

# ===================== МОДЕЛЬ ====================
def train_model(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    """Обучает модель на чистых numpy-массивах"""
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    
    if n_pos > 0 and n_neg > 0:
        sample_weight = np.where(y == 1, n_neg / n_pos, 1.0).astype(np.float32)
    else:
        sample_weight = np.ones(len(y), dtype=np.float32)
    
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y, sample_weight=sample_weight)
    return model

# ===================== БЭКТЕСТ ====================
@dataclass
class Params:
    sigma: float
    hold: int

def backtest(X: np.ndarray, y: np.ndarray, closes: np.ndarray, times: np.ndarray,
             model: lgb.LGBMClassifier, p: Params) -> Tuple[np.ndarray, List[dict]]:
    """
    Векторизованный бэктест. 
    Предсказание делается ОДИН РАЗ на весь массив.
    """
    if len(X) == 0:
        return np.array([0.0]), []

    # 1. ВЕКТОРИЗОВАННОЕ ПРЕДСКАЗАНИЕ (Мгновенно)
    probs = model.predict_proba(X)[:, 1]
    signals = np.where(probs > 0.5, 1, -1)
    
    equity = [0.0]
    trades = []
    position = None
    entry_price = 0.0
    entry_time = 0
    bars_held = 0
    
    n = len(closes)
    # 2. Цикл по чистым NumPy массивам (Очень быстро)
    for i in range(n):
        signal = signals[i]
        price = closes[i]
        t = times[i]
        
        if position is not None:
            bars_held += 1
            if bars_held >= p.hold:
                pnl = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
                equity.append(equity[-1] + pnl * 100)
                trades.append({
                    "entry_time": entry_time, "entry_price": entry_price,
                    "direction": position, "exit_time": t, "exit_price": price,
                    "pnl": pnl * 100, "hold_bars": bars_held
                })
                position = None
                bars_held = 0
        
        if position is None:
            position = signal
            entry_price = price
            entry_time = t
            bars_held = 0

    if position is not None:
        last_price = closes[-1]
        pnl = (last_price - entry_price) / entry_price if position == 1 else (entry_price - last_price) / entry_price
        equity.append(equity[-1] + pnl * 100)
        trades.append({
            "entry_time": entry_time, "entry_price": entry_price,
            "direction": position, "exit_time": times[-1], "exit_price": last_price,
            "pnl": pnl * 100, "hold_bars": bars_held
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
    
    return sharpe - max_dd * 0.5 + win_rate * 0.5

# ===================== ОПТИМИЗАЦИЯ ====================
def optimize(fs: FeatureStore, feature_cols: List[str], n_trials: int = 100) -> Tuple[Dict, float]:
    def objective(trial):
        try:
            sigma = trial.suggest_float("sigma", 0.5, 3.0, step=0.1)
            hold = trial.suggest_int("hold", 2, 20)
            
            X, y, closes, times = build_target(fs, sigma, feature_cols)
            
            if len(y) < NOPT + NFWD: return -1e6
            if np.sum(y == 1) < 10 or np.sum(y == 0) < 10: return -1e6
            
            X_train, y_train = X[:NOPT], y[:NOPT]
            X_test, y_test, closes_test, times_test = X[NOPT:NOPT+NFWD], y[NOPT:NOPT+NFWD], closes[NOPT:NOPT+NFWD], times[NOPT:NOPT+NFWD]
            
            model = train_model(X_train, y_train)
            p = Params(sigma=sigma, hold=hold)
            eq, trades = backtest(X_test, y_test, closes_test, times_test, model, p)
            
            return score_equity(eq, trades)
        except Exception:
            return -1e6
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    print(f"   🔄 Оптимизация ({n_trials} итераций, n_jobs=-1)...")
    # n_jobs=-1 включает параллельные вычисления на все ядра CPU
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
    
    if study.best_value <= -1e5:
        print("   ⚠ Оптимизация не дала хороших результатов")
        return {"sigma": 1.5, "hold": 10}, -1e6
    
    return study.best_params, study.best_value

# ===================== ФОРВАРД-ТЕСТ ====================
def forward_test(fs: FeatureStore, params: dict, feature_cols: List[str]) -> Tuple[Optional[np.ndarray], List[dict]]:
    try:
        p = Params(**params)
        X, y, closes, times = build_target(fs, p.sigma, feature_cols)
        
        if len(y) < NOPT + NFWD: return None, []
        
        X_train, y_train = X[:NOPT], y[:NOPT]
        X_test, y_test, closes_test, times_test = X[NOPT:NOPT+NFWD], y[NOPT:NOPT+NFWD], closes[NOPT:NOPT+NFWD], times[NOPT:NOPT+NFWD]
        
        if len(X_test) < 10: return None, []
        
        model = train_model(X_train, y_train)
        return backtest(X_test, y_test, closes_test, times_test, model, p)
    except Exception as e:
        print(f"   ⚠ Ошибка в forward_test: {e}")
        return None, []

# ===================== ВИЗУАЛИЗАЦИЯ ====================
# (Оставлена без изменений, так как работает с готовыми результатами и не влияет на скорость оптимизации)
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
        ax2.hist(pnls, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title(f"Распределение PnL (сделок: {len(all_trades)})"); ax2.set_xlabel("PnL, %")
        
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
    
    plt.tight_layout(); plt.savefig("trading_results.png", dpi=150); plt.show()
    
    if all_trades:
        print("\n📊 Сводная статистика:")
        print(f"   Всего сделок: {len(all_trades)}")
        wins = sum(1 for t in all_trades if t["pnl"] > 0)
        print(f"   Win rate: {wins/len(all_trades)*100:.1f}%")
        print(f"   Средний PnL: {np.mean([t['pnl'] for t in all_trades]):.2f}%")

# ===================== ЖИВАЯ ТОРГОВЛЯ ====================
class LiveTrader:
    def __init__(self, cfg: dict):
        self.symbol = cfg["symbol"]
        self.tf = tf_to_mt5(cfg["timeframe"])
        self.magic = cfg["magic"]
        self.params = Params(sigma=cfg["sigma"], hold=cfg["hold"])
        self.model = None
        self.last_time = 0
        self.bars_held = 0
        self.position = None
        self.feature_cols = []
        
        print(f"   🏆 {self.symbol} {cfg['timeframe']} | Hold: {cfg['hold']} bars")
        self._init_model()
    
    def _init_model(self):
        df = self._get_data()
        if df is None: return
        
        fs = FeatureStore(df)
        fs.calc_features()
        self.feature_cols = get_feature_cols(fs.features)
        
        X, y, _, _ = build_target(fs, self.params.sigma, self.feature_cols)
        if len(X) > NOPT:
            self.model = train_model(X[-NOPT:], y[-NOPT:])
            print(f"   ✅ Модель обучена для {self.symbol}")
    
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
        X, _, closes, _ = build_target(fs, self.params.sigma, self.feature_cols)
        
        if len(X) == 0: return
        
        # Предсказание только для последней строки
        prob = self.model.predict_proba(X[[-1]])[0][1]
        signal = 1 if prob > 0.5 else -1
        
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        
        if positions:
            self.bars_held += 1
            if self.bars_held >= self.params.hold:
                self._close_position(positions[0], "Hold")
                self.position = None
                self.bars_held = 0
                return
        
        if not positions:
            self.position = signal
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
            print(f"   ✅ {self.symbol} {'BUY' if direction==1 else 'SELL'} | P={prob:.2f}")
    
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
            print(f"   📉 Закрыт {self.symbol} ({reason})")
    
    def close_all(self):
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if positions:
            for pos in positions:
                self._close_position(pos, "Shutdown")
                time.sleep(0.5)

def run_live(champions: List[dict]):
    print("\n" + "="*60)
    print("🚀 ЗАПУСК ДЕМО-ТОРГОВЛИ")
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
    print("🔬 УПРОЩЕННАЯ ЭВОЛЮЦИОННАЯ ТОРГОВАЯ СИСТЕМА (v2.0 OPTIMIZED)")
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
                
                if fs.features is None or len(fs.features) < NOPT + NFWD:
                    print(f"   ⚠ Недостаточно признаков"); continue
                
                best_params, best_score = optimize(fs, feature_cols, n_trials=20) # Увеличил до 20 для теста
                
                if best_score <= -1e5:
                    best_params = {"sigma": 1.5, "hold": 10}
                
                print(f"   ✅ Параметры: sigma={best_params['sigma']:.1f}, hold={best_params['hold']}")
                
                eq, trades = forward_test(fs, best_params, feature_cols)
                
                if eq is not None and len(eq) > 1:
                    results[(sym, tf_name)] = (eq, trades)
                    champion = {"symbol": sym, "timeframe": tf_name, "magic": magic, **best_params}
                    champions.append(champion)
                    magic += 1
                    print(f"   🏆 Чемпион добавлен (Magic: {champion['magic']})")
                    
            except Exception as e:
                print(f"   ❌ Ошибка при обработке {sym} {tf_name}: {e}")
                continue
    
    if results:
        plot_results(results)
        with open("champions_simple.json", "w") as f:
            json.dump(champions, f, indent=2)
        print("\n💾 Чемпионы сохранены в champions_simple.json")
        
        if champions:
            ans = input("\n🚀 Запустить на ДЕМО? (Y/N): ").strip().upper()
            if ans == "Y": run_live(champions)
    else:
        print("\n❌ Нет результатов для визуализации")
    
    mt5.shutdown()
    print("\n✅ Завершено")

if __name__ == "__main__":
    main()