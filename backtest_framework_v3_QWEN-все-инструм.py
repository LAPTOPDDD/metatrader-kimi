# -*- coding: utf-8 -*-
"""
Martingale Global Market Backtester v5 (ALL ASSETS)
====================================================
Тестирует мартингейл на ВСЕХ доступных инструментах: Форекс, Крипта, Металлы, Индексы.
Использует только реальные данные из MT5.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ Ошибка: Установи библиотеку: pip install MetaTrader5")
    sys.exit(1)

# ============================================================================
# СПИСОК ИНСТРУМЕНТОВ (ГЛОБАЛЬНЫЙ)
# ============================================================================
GLOBAL_SYMBOLS = [
    # --- Основные Форекс пары ---
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY", "CADJPY", "CHFJPY",
    "EURAUD", "GBPAUD", "EURCAD", "GBPCAD", "AUDCAD", "AUDCHF", "AUDNZD",
    "NZDJPY", "NZDCAD", "NZDCHF", "GBPNZD", "EURNZD", "CADCHF",
    
    # --- Металлы и Сырье ---
    "XAUUSD", "XAGUSD", "XAUEUR", 
    "USOIL", "UKOIL", "NGAS", # Нефть и Газ (названия могут отличаться у брокеров)
    
    # --- Криптовалюты (Топ) ---
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "ADAUSD", 
    "DOTUSD", "MATICUSD", "LTCUSD", "LINKUSD", "AVAXUSD", "UNIUSD",
    
    # --- Индексы (CFD) ---
    "US30", "SPX500", "NAS100", "GER40", "UK100", "JPN225", "AUS200"
]

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: GLOBAL_SYMBOLS)
    timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "H1"])
    
    start_lot: float = 0.01
    multiplier: float = 1.5
    max_positions: int = 10
    start_balance: float = 10000.0
    
    bars_count: int = 3000  # Берем последние 3000 баров для скорости

@dataclass
class CycleResult:
    symbol: str
    timeframe: str
    n_cycles: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    max_dd: float = 0.0
    safety_score: float = 0.0

# ============================================================================
# ЯДРО РАСЧЕТОВ
# ============================================================================
def calculate_max_drawdown(equity_curve: List[float]) -> float:
    if not equity_curve: return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > max_dd: max_dd = dd
    return max_dd

def simulate_cycle(prices: List[float], cfg: Config) -> tuple:
    positions = []
    realized_pnl = 0.0
    equity_curve = []
    
    # Определяем размер контракта динамически
    # Для крипты часто 1, для форекса 100000. 
    # В MT5 это можно узнать через symbol_info, но для скорости возьмем среднее:
    # Если цена > 1000 (крипта/индексы), контракт скорее всего 1 или 10.
    # Если цена < 10 (форекс), контракт 100000.
    
    contract_size = 100000 # Стандарт для форекса

    for i in range(min(cfg.max_positions, len(prices) - 1)):
        if i == 0:
            direction = 1
            vol = cfg.start_lot
        else:
            direction = -positions[-1][2]
            sum_vol = sum(v for v, _, _ in positions)
            vol = round(sum_vol * cfg.multiplier, 2)

        entry = prices[i]
        exit_price = prices[i+1]
        
        # Адаптация под разные классы активов
        if entry > 500: contract_size = 1 # Крипта/Индексы
        elif entry > 10: contract_size = 100 # Золото/Серебро (часто 100 унций)
        else: contract_size = 100000 # Форекс

        pos_pnl = (exit_price - entry) * direction * vol * contract_size
        positions.append((vol, entry, direction))
        realized_pnl += pos_pnl

        unrealized = 0.0
        for v, e, d in positions:
            unrealized += (exit_price - e) * d * v * contract_size
            
        current_eq = cfg.start_balance + realized_pnl + unrealized
        equity_curve.append(current_eq)

        if (realized_pnl + unrealized) > 0:
            break

    return realized_pnl, equity_curve

def run_backtest(prices: List[float], symbol: str, tf: str, cfg: Config) -> CycleResult:
    res = CycleResult(symbol=symbol, timeframe=tf)
    if len(prices) < 15: return res

    idx = 0
    total_profit = 0.0
    wins = 0
    cycles = 0
    global_equity = [cfg.start_balance]

    while idx + cfg.max_positions + 1 < len(prices):
        chunk = prices[idx : idx + cfg.max_positions + 2]
        pnl, eq_curve = simulate_cycle(chunk, cfg)
        
        cycles += 1
        total_profit += pnl
        if pnl > 0: wins += 1
        
        if eq_curve: global_equity.extend(eq_curve)
        idx += 1 

    res.n_cycles = cycles
    res.total_profit = total_profit
    res.win_rate = (wins / cycles * 100) if cycles > 0 else 0
    res.max_dd = calculate_max_drawdown(global_equity)
    res.safety_score = (total_profit / res.max_dd) if res.max_dd > 0 else 0
    return res

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("🌍 GLOBAL MARKET MARTINGALE TESTER (MT5 REAL DATA)")
    print("="*70)
    
    if not mt5.initialize():
        print("❌ MT5 не запущен или нет авторизации.")
        return

    cfg = Config()
    print(f"📊 Инструментов: {len(cfg.symbols)} | ТФ: {cfg.timeframes} | Баров: {cfg.bars_count}")
    print("⚠️ Убедись, что все символы активны в Обзоре рынка MT5!\n")

    results = []
    total_tests = len(cfg.symbols) * len(cfg.timeframes)
    current_test = 0

    for symbol in cfg.symbols:
        if not mt5.symbol_select(symbol, True):
            continue # Пропускаем, если брокер не дает этот символ

        for tf in cfg.timeframes:
            current_test += 1
            print(f"[{current_test}/{total_tests}] {symbol:<10} {tf:<4} ...", end=" ")
            
            rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{tf}"), 0, cfg.bars_count)
            if rates is None or len(rates) == 0:
                print("Нет данных")
                continue
                
            prices = [float(r['close']) for r in rates]
            res = run_backtest(prices, symbol, tf, cfg)
            results.append(res)
            
            status = "✅" if res.safety_score > 1 else "⚠️"
            print(f"Win:{res.win_rate:.1f}% DD:{res.max_dd:.0f}$ Score:{res.safety_score:.3f} {status}")

    mt5.shutdown()

    # Сортировка и вывод топа
    results.sort(key=lambda x: x.safety_score, reverse=True)
    
    print("\n" + "="*80)
    print("🏆 ТОП-10 САМЫХ БЕЗОПАСНЫХ АКТИВОВ ДЛЯ МАРТИНГЕЙЛА")
    print("="*80)
    print(f"{'#':<3} {'Symbol':<10} {'TF':<5} {'Cycles':>7} {'Win%':>6} {'Profit':>10} {'MaxDD':>10} {'Score':>8}")
    print("-"*80)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:<3} {r.symbol:<10} {r.timeframe:<5} {r.n_cycles:>7} {r.win_rate:>5.1f}% "
              f"{r.total_profit:>10.2f} {r.max_dd:>10.2f} {r.safety_score:>8.4f}")

    # Сохранение
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_global")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "global_results.json"), "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)
        
    print(f"\n💾 Полный отчет сохранен в папку: {out_dir}")

if __name__ == "__main__":
    main()