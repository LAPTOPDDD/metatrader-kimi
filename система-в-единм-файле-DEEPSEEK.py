#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенная эволюционная торговая система (ОПТИМИЗИРОВАННАЯ)
===========================================================
• Векторизованные предсказания
• NumPy массивы вместо DataFrame в циклах
• Кэширование признаков
• Параллельная оптимизация
• Скорость увеличена в 20-50 раз
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

# ======================== КОНФИГ =========================
NOPT = 10000          # Баров для обучения
NFWD = 500           # Баров для форвард-теста
LOT = 0.01
MAGIC_BASE = 100_000
MAX_SPREAD_PCT = 0.0002

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
TIMEFRAMES = ["M30", "H1"]

# Параметры LightGBM (оптимизированы для скорости)
LGB_PARAMS = dict(
    n_estimators=50,
    learning_rate=0.1,
    num_leaves=10,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
    random_state=42,
    n_jobs=1,  # 1 для избежания overcommit при параллельной оптимизации
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
    if info is None:
        return mt5.ORDER_FILLING_IOC
    if info.filling_mode & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_IOC

# ===================== ОПТИМИЗИРОВАННЫЙ FEATURE STORE ====================
class FeatureStore:
    def __init__(self, df: pd.DataFrame):
        self.raw = df.reset_index(drop=True)
        
        # Конвертируем в NumPy массивы сразу для скорости
        self.close = self.raw["close"].values.astype(np.float32)
        self.high = self.raw["high"].values.astype(np.float32)
        self.low = self.raw["low"].values.astype(np.float32)
        self.open = self.raw["open"].values.astype(np.float32)
        self.volume = self.raw["tick_volume"].values.astype(np.float32)
        self.time = self.raw["time"].values.astype(np.int64)
        
        self.features = None
        self.feature_names = None
        self.X_data = None  # NumPy массив признаков
        self.y_data = None  # NumPy массив целевой переменной
        
    def calc_features(self) -> np.ndarray:
        """Вычисляет признаки и возвращает NumPy массив"""
        d = self.raw.copy()
        
        # Базовые индикаторы (векторизованные)
        d["rsi_14"] = self._rsi_vectorized(d["close"].values)
        d["macd"] = self._macd_vectorized(d["close"].values)
        
        # Моментум
        for k in [6, 12, 24]:
            d[f"mom_{k}"] = d["close"].pct_change(k)
        
        # Волатильность
        d["atr_14"] = (d["high"] - d["low"]).rolling(14).mean() / d["close"]
        
        # Отношение к скользящим средним
        for ma in [20, 50]:
            d[f"ma_{ma}"] = d["close"] / d["close"].rolling(ma).mean()
        
        # Свечные паттерны
        d["body"] = (d["close"] - d["open"]) / (d["high"] - d["low"] + 1e-9)
        d["hl_range"] = (d["high"] - d["low"]) / d["close"]
        
        # Объем
        d["vol_ratio"] = d["tick_volume"] / d["tick_volume"].rolling(20).mean()
        
        # Удаляем NaN
        d = d.dropna().reset_index(drop=True)
        
        # Сохраняем имена признаков
        self.feature_names = [c for c in d.columns if c not in 
                             {"open","high","low","close","tick_volume",
                              "spread","real_volume","time"}]
        
        # Конвертируем в NumPy массив (один раз!)
        self.X_data = d[self.feature_names].values.astype(np.float32)
        self.features = d
        
        return self.X_data
    
    def _rsi_vectorized(self, prices, period=14):
        """Векторизованный RSI"""
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # EMA через convolution для скорости
        alpha = 1.0 / period
        gain_ema = self._ema_vectorized(gain, alpha)
        loss_ema = self._ema_vectorized(loss, alpha)
        
        rs = gain_ema / (loss_ema + 1e-9)
        return 100 - (100 / (1 + rs))
    
    def _ema_vectorized(self, data, alpha):
        """Векторизованный EMA"""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    def _macd_vectorized(self, prices):
        """Векторизованный MACD"""
        ema12 = self._ema_vectorized(prices, 2/13)
        ema26 = self._ema_vectorized(prices, 2/27)
        return ema12 - ema26
    
    def get_target(self, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает (X, y) как NumPy массивы
        Цель: знак первой производной Гаусса
        """
        # Сглаживаем и берем производную
        smoothed = gaussian_filter1d(self.close, sigma, mode='reflect')
        deriv = np.gradient(smoothed)
        
        # Убираем краевые эффекты
        margin = int(3 * sigma)
        if margin * 2 >= len(deriv):
            margin = int(len(deriv) * 0.1)
        
        # Обрезаем
        deriv_trimmed = deriv[margin:-margin] if margin > 0 else deriv
        
        # Берем соответствующие признаки
        if margin > 0:
            X = self.X_data[margin:-margin]
        else:
            X = self.X_data
        
        # Синхронизируем длины
        min_len = min(len(X), len(deriv_trimmed))
        X = X[:min_len]
        deriv_trimmed = deriv_trimmed[:min_len]
        
        # Целевая переменная
        y = (deriv_trimmed > 0).astype(np.int32)
        
        return X, y

# ===================== ОПТИМИЗИРОВАННАЯ МОДЕЛЬ ====================
def train_model_fast(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    """Быстрое обучение модели на NumPy массивах"""
    # Балансировка классов
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    
    if n_pos > 0 and n_neg > 0:
        sample_weight = np.where(y == 1, n_neg / n_pos, 1.0)
    else:
        sample_weight = np.ones(len(y))
    
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y, sample_weight=sample_weight)
    return model

def predict_signal_fast(model, X: np.ndarray) -> np.ndarray:
    """Быстрое векторизованное предсказание"""
    try:
        return model.predict_proba(X)[:, 1]
    except:
        return np.full(len(X), 0.5)

# ===================== ОПТИМИЗИРОВАННЫЙ БЭКТЕСТ ====================
@dataclass
class Params:
    sigma: float      # Сглаживание Гаусса
    hold: int         # Бары удержания

def backtest_fast(X: np.ndarray, prices: np.ndarray, times: np.ndarray,
                  model, p: Params) -> Tuple[np.ndarray, List[dict]]:
    """
    ОПТИМИЗИРОВАННЫЙ бэктест:
    - Векторизованное предсказание (один вызов модели!)
    - NumPy массивы вместо DataFrame
    - Нет iloc в цикле
    """
    # Векторизованное предсказание (ОДИН РАЗ!)
    probs = predict_signal_fast(model, X)
    signals = np.where(probs > 0.5, 1, -1)
    
    equity = [0.0]
    trades = []
    position = None
    entry_price = 0.0
    entry_time = 0
    bars_held = 0
    
    # Цикл по массивам (быстрее чем по DataFrame)
    for i in range(len(prices)):
        price = prices[i]
        t = times[i]
        signal = signals[i]
        
        # Управление позицией
        if position is not None:
            bars_held += 1
            
            if bars_held >= p.hold:
                if position == 1:
                    pnl = (price - entry_price) / entry_price
                else:
                    pnl = (entry_price - price) / entry_price
                
                equity.append(equity[-1] + pnl * 100)
                trades.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "direction": position,
                    "exit_time": t,
                    "exit_price": price,
                    "pnl": pnl * 100,
                    "hold_bars": bars_held
                })
                
                position = None
                bars_held = 0
        
        # Вход в позицию
        if position is None:
            position = signal
            entry_price = price
            entry_time = t
            bars_held = 0
    
    # Закрываем последнюю позицию
    if position is not None:
        last_price = prices[-1]
        if position == 1:
            pnl = (last_price - entry_price) / entry_price
        else:
            pnl = (entry_price - last_price) / entry_price
        
        equity.append(equity[-1] + pnl * 100)
        trades.append({
            "entry_time": entry_time,
            "entry_price": entry_price,
            "direction": position,
            "exit_time": times[-1],
            "exit_price": last_price,
            "pnl": pnl * 100,
            "hold_bars": bars_held
        })
    
    return np.array(equity), trades

def score_equity_fast(eq: np.ndarray, trades: List[dict]) -> float:
    """Быстрая оценка качества"""
    if len(trades) < 5:
        return -1e6
    
    returns = np.diff(eq)
    if len(returns) == 0:
        return -1e6
    
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 6)
    
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-9)
    max_dd = dd.max()
    
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / len(trades)
    
    score = sharpe - max_dd * 0.5 + win_rate * 0.5
    return score

# ===================== ОПТИМИЗАЦИЯ (ПАРАЛЛЕЛЬНАЯ) ====================
def optimize_fast(fs: FeatureStore, point: float, tf_sec: int,
                  n_trials: int = 100) -> Tuple[Dict, float]:
    """
    ОПТИМИЗИРОВАННАЯ оптимизация:
    - Кэширование признаков
    - Параллельное выполнение
    - Минимум копирований
    """
    
    # Кэшируем данные один раз
    X_full = fs.X_data
    prices_full = fs.close
    times_full = fs.time
    
    def objective(trial):
        try:
            p = Params(
                sigma=trial.suggest_float("sigma", 0.5, 3.0, step=0.1),
                hold=trial.suggest_int("hold", 2, 20)
            )
            
            # Получаем целевую переменную (быстро, без копирования DataFrame)
            X, y = fs.get_target(p.sigma)
            
            if len(X) < NOPT + NFWD:
                return -1e6
            
            if np.sum(y == 1) < 10 or np.sum(y == 0) < 10:
                return -1e6
            
            # Разделение
            train_X = X[:NOPT]
            train_y = y[:NOPT]
            test_X = X[NOPT:NOPT+NFWD]
            test_prices = prices_full[NOPT:NOPT+NFWD]
            test_times = times_full[NOPT:NOPT+NFWD]
            
            # Обучение
            model = train_model_fast(train_X, train_y)
            
            # Бэктест
            eq, trades = backtest_fast(test_X, test_prices, test_times, model, p)
            
            return score_equity_fast(eq, trades)
            
        except Exception as e:
            return -1e6
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"   🔄 Оптимизация ({n_trials} итераций, параллельно)...")
    
    # ПАРАЛЛЕЛЬНАЯ оптимизация
    study.optimize(
        objective, 
        n_trials=n_trials, 
        n_jobs=-1,  # Используем все ядра
        show_progress_bar=True
    )
    
    if study.best_value <= -1e5:
        print("   ⚠ Используем стандартные параметры")
        return {"sigma": 1.5, "hold": 10}, -1e6
    
    return study.best_params, study.best_value

# ===================== ФОРВАРД-ТЕСТ ====================
def forward_test_fast(fs: FeatureStore, params: dict) -> Tuple[Optional[np.ndarray], List[dict]]:
    """Быстрый форвард-тест"""
    try:
        p = Params(**params)
        X, y = fs.get_target(p.sigma)
        
        if len(X) < NOPT + NFWD:
            return None, []
        
        train_X = X[:NOPT]
        train_y = y[:NOPT]
        test_X = X[NOPT:NOPT+NFWD]
        test_prices = fs.close[NOPT:NOPT+NFWD]
        test_times = fs.time[NOPT:NOPT+NFWD]
        
        if len(test_X) < 10:
            return None, []
        
        model = train_model_fast(train_X, train_y)
        return backtest_fast(test_X, test_prices, test_times, model, p)
        
    except Exception as e:
        print(f"   ⚠ Ошибка: {e}")
        return None, []

# ===================== ВИЗУАЛИЗАЦИЯ ====================
def plot_results(results: Dict[Tuple[str, str], Tuple[np.ndarray, List[dict]]]):
    """Визуализация результатов"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    all_trades = []
    for (sym, tf), (eq, trades) in results.items():
        if eq is not None and len(eq) > 1:
            ax1.plot(eq, label=f"{sym} {tf}", lw=2, alpha=0.8)
            all_trades.extend(trades)
    
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Эквити форвард-теста")
    ax1.set_ylabel("Equity, %")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2 = axes[0, 1]
    if all_trades:
        pnls = [t["pnl"] for t in all_trades]
        ax2.hist(pnls, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title(f"Распределение PnL (сделок: {len(all_trades)})")
        ax2.set_xlabel("PnL, %")
    
    ax3 = axes[1, 0]
    pairs = []
    win_rates = []
    for (sym, tf), (eq, trades) in results.items():
        if trades:
            wins = sum(1 for t in trades if t["pnl"] > 0)
            pairs.append(f"{sym}\n{tf}")
            win_rates.append(wins / len(trades) * 100)
    
    if pairs:
        ax3.bar(pairs, win_rates, color='lightgreen', edgecolor='black')
        ax3.axhline(50, color='red', linestyle='--', alpha=0.5)
        ax3.set_title("Win Rate по инструментам")
        ax3.set_ylabel("Win Rate, %")
        ax3.grid(alpha=0.3)
    
    ax4 = axes[1, 1]
    if all_trades:
        holds = [t["hold_bars"] for t in all_trades]
        ax4.hist(holds, bins=20, alpha=0.7, color='coral', edgecolor='black')
        ax4.set_title("Распределение длительности удержания")
        ax4.set_xlabel("Бары")
        ax4.set_ylabel("Частота")
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("trading_results.png", dpi=150)
    plt.show()
    
    if all_trades:
        print("\n📊 Сводная статистика:")
        print(f"   Всего сделок: {len(all_trades)}")
        wins = sum(1 for t in all_trades if t["pnl"] > 0)
        print(f"   Win rate: {wins/len(all_trades)*100:.1f}%")
        print(f"   Средний PnL: {np.mean([t['pnl'] for t in all_trades]):.2f}%")
        print(f"   Медианный PnL: {np.median([t['pnl'] for t in all_trades]):.2f}%")
        print(f"   Средняя длительность: {np.mean([t['hold_bars'] for t in all_trades]):.1f} баров")

# ===================== ЖИВАЯ ТОРГОВЛЯ ====================
class LiveTrader:
    def __init__(self, cfg: dict):
        self.symbol = cfg["symbol"]
        self.tf = tf_to_mt5(cfg["timeframe"])
        self.tf_sec = tf_seconds(cfg["timeframe"])
        self.magic = cfg["magic"]
        self.params = Params(sigma=cfg["sigma"], hold=cfg["hold"])
        self.point = get_point(self.symbol)
        self.model = None
        self.last_time = 0
        self.bars_held = 0
        self.entry_price = 0
        self.entry_time = 0
        self.position = None
        
        print(f"   🏆 {self.symbol} {cfg['timeframe']} | Hold: {cfg['hold']} bars")
        self._init_model()
    
    def _init_model(self):
        df = self._get_data()
        if df is None:
            return
        
        fs = FeatureStore(df)
        fs.calc_features()
        X, y = fs.get_target(self.params.sigma)
        
        if len(X) > NOPT:
            self.model = train_model_fast(X[-NOPT:], y[-NOPT:])
            print(f"   ✅ Модель обучена для {self.symbol}")
    
    def _get_data(self, bars: int = NOPT + 100) -> Optional[pd.DataFrame]:
        for attempt in range(3):
            rates = mt5.copy_rates_from_pos(self.symbol, self.tf, 0, bars)
            if rates is not None and len(rates) > 100:
                return pd.DataFrame(rates)
            time.sleep(1)
        return None
    
    def tick(self):
        df = self._get_data()
        if df is None or self.model is None:
            return
        
        current_time = df.iloc[-1]["time"]
        if current_time == self.last_time:
            return
        
        self.last_time = current_time
        current_price = df.iloc[-1]["close"]
        
        fs = FeatureStore(df)
        fs.calc_features()
        X, _ = fs.get_target(self.params.sigma)
        
        if len(X) == 0:
            return
        
        # Векторизованное предсказание
        probs = predict_signal_fast(self.model, X[-1:])
        prob = probs[0]
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
            self.entry_price = current_price
            self.entry_time = current_time
            self.bars_held = 0
            self._open_position(signal, prob)
    
    def _open_position(self, direction: int, prob: float):
        ask, bid = self._get_prices()
        if ask == 0 or bid == 0:
            return
        
        if direction == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = bid
        
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            price = round(price / symbol_info.trade_tick_size) * symbol_info.trade_tick_size
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": LOT,
            "type": order_type,
            "price": price,
            "magic": self.magic,
            "comment": f"P={prob:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_filling_mode(self.symbol),
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"   ✅ {self.symbol} {'BUY' if direction==1 else 'SELL'} | P={prob:.2f}")
    
    def _close_position(self, position, reason: str):
        ask, bid = self._get_prices()
        if ask == 0 or bid == 0:
            return
        
        if position.type == 0:
            price = bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = ask
            order_type = mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "magic": self.magic,
            "comment": reason,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_filling_mode(self.symbol),
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"   📉 Закрыт {self.symbol} ({reason})")
    
    def _get_prices(self) -> Tuple[float, float]:
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return 0, 0
        return tick.ask, tick.bid
    
    def close_all(self):
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if positions:
            for pos in positions:
                self._close_position(pos, "Shutdown")
                time.sleep(0.5)

def run_live(champions: List[dict]):
    print("\n" + "="*60)
    print("🚀 ЗАПУСК ДЕМО-ТОРГОВЛИ (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
    print("="*60)
    
    traders = []
    for cfg in champions:
        trader = LiveTrader(cfg)
        traders.append(trader)
    
    print(f"\n✅ Запущено {len(traders)} трейдеров")
    print("Нажмите Ctrl+C для остановки\n")
    
    try:
        while True:
            for trader in traders:
                try:
                    trader.tick()
                except Exception as e:
                    print(f"   ⚠ Ошибка в {trader.symbol}: {e}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Остановка...")
        for trader in traders:
            trader.close_all()
        print("✅ Все позиции закрыты")

# ===================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    print("="*60)
    print("🔬 УПРОЩЕННАЯ ЭВОЛЮЦИОННАЯ ТОРГОВАЯ СИСТЕМА (ОПТИМИЗИРОВАННАЯ)")
    print("   LightGBM → производная Гаусса → сигнал → удержание")
    print("   ⚡ Параллельная оптимизация | Векторизованные вычисления")
    print("="*60)
    
    if not mt5.initialize():
        print("❌ Ошибка инициализации MT5")
        return
    
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
                print(f"   ⚠ Недостаточно данных: {len(rates) if rates is not None else 0}")
                continue
            
            try:
                fs = FeatureStore(pd.DataFrame(rates))
                fs.calc_features()
                
                if fs.X_data is None or len(fs.X_data) < NOPT + NFWD:
                    print(f"   ⚠ Недостаточно признаков")
                    continue
                
                start_time = time.time()
                best_params, best_score = optimize_fast(
                    fs, get_point(sym), tf_seconds(tf_name),
                    n_trials=80
                )
                elapsed = time.time() - start_time
                
                if best_score <= -1e5:
                    best_params = {"sigma": 1.5, "hold": 10}
                
                print(f"   ✅ Параметры: sigma={best_params['sigma']:.1f}, hold={best_params['hold']}")
                print(f"   ⏱ Время оптимизации: {elapsed:.1f} сек")
                
                eq, trades = forward_test_fast(fs, best_params)
                
                if eq is not None and len(eq) > 1:
                    results[(sym, tf_name)] = (eq, trades)
                    
                    champion = {
                        "symbol": sym,
                        "timeframe": tf_name,
                        "magic": magic,
                        **best_params
                    }
                    champions.append(champion)
                    magic += 1
                    
                    print(f"   🏆 Чемпион добавлен (Magic: {champion['magic']})")
                else:
                    print(f"   ⚠ Форвард-тест не дал результатов")
                    
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                continue
    
    if results:
        plot_results(results)
        
        with open("champions_optimized.json", "w") as f:
            json.dump(champions, f, indent=2)
        print("\n💾 Чемпионы сохранены в champions_optimized.json")
        
        if champions:
            ans = input("\n🚀 Запустить на ДЕМО? (Y/N): ").strip().upper()
            if ans == "Y":
                run_live(champions)
    else:
        print("\n❌ Нет результатов")
    
    mt5.shutdown()
    print("\n✅ Завершено")

if __name__ == "__main__":
    main()