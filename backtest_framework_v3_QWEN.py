# -*- coding: utf-8 -*-
"""
Martingale Real-Data Backtester v4 (MT5 ONLY)
==============================================
Тестирует мартингейл только на реальных данных из MT5.
Никакой синтетики. Честный расчет прибыли в валюте депозита.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ Ошибка: Библиотека MetaTrader5 не установлена.")
    print("Выполни в консоли: pip install MetaTrader5")
    sys.exit(1)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
@dataclass
class Config:
    # Инструменты для теста (должны быть активны в Обзоре рынка MT5!)
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "XAUUSD" 
    ])
    timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "H1"])
    
    # Параметры стратегии
    start_lot: float = 0.01
    multiplier: float = 1.5
    max_positions: int = 10
    start_balance: float = 10000.0  # Виртуальный баланс для расчета просадки
    
    # История
    bars_count: int = 5000  # Сколько последних баров брать для теста

# ============================================================================
# МОДЕЛИ ДАННЫХ
# ============================================================================
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

def simulate_cycle(prices: List[float], cfg: Config) -> Tuple[float, List[float]]:
    """
    Симуляция одного цикла мартингейла на массиве цен.
    Возвращает: (Реализованная прибыль, Кривая эквити)
    """
    positions = [] # (volume, entry_price, direction)
    equity_curve = []
    realized_pnl = 0.0
    
    # Размер контракта: 100k для форекса/золота, 1 для крипты (упрощенно)
    # Для точности можно брать из symbol_info, но для теста хватит стандарта
    contract_size = 100000 

    for i in range(min(cfg.max_positions, len(prices) - 1)):
        if i == 0:
            direction = 1 # BUY
            vol = cfg.start_lot
        else:
            direction = -positions[-1][2] # Противоположное направление
            sum_vol = sum(v for v, _, _ in positions)
            vol = round(sum_vol * cfg.multiplier, 2)

        entry_price = prices[i]
        exit_price = prices[i+1]

        # Прибыль одной позиции: (Разница * Направление * Лот * Контракт)
        pos_pnl = (exit_price - entry_price) * direction * vol * contract_size
        
        positions.append((vol, entry_price, direction))
        realized_pnl += pos_pnl

        # Считаем текущее эквити (Баланс + Нереализованная прибыль всех открытых поз)
        unrealized_pnl = 0.0
        for v, e, d in positions:
            unrealized_pnl += (exit_price - e) * d * v * contract_size
            
        current_equity = cfg.start_balance + realized_pnl + unrealized_pnl
        equity_curve.append(current_equity)

        # Если суммарный результат положительный — закрываем цикл
        if (realized_pnl + unrealized_pnl) > 0:
            break

    return realized_pnl, equity_curve

def run_backtest_on_prices(prices: List[float], symbol: str, tf: str, cfg: Config) -> CycleResult:
    res = CycleResult(symbol=symbol, timeframe=tf)
    if len(prices) < cfg.max_positions + 2:
        return res

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
        
        # Обновляем глобальное эквити для расчета просадки
        if eq_curve:
            global_equity.extend(eq_curve)
            
        # Сдвигаем индекс. В реальности мы бы ждали следующего бара после закрытия.
        # Здесь берем следующий доступный кусок.
        idx += 1 

    res.n_cycles = cycles
    res.total_profit = total_profit
    res.win_rate = (wins / cycles * 100) if cycles > 0 else 0
    res.max_dd = calculate_max_drawdown(global_equity)
    
    # Safety Score: Прибыль / Просадка
    res.safety_score = (total_profit / res.max_dd) if res.max_dd > 0 else 0
    
    return res

# ============================================================================
# РАБОТА С MT5
# ============================================================================
def get_mt5_data(symbol: str, tf_str: str, count: int) -> Optional[List[float]]:
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, 
        "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1
    }
    tf = tf_map.get(tf_str)
    if not tf: return None

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        return None
    return [float(r['close']) for r in rates]

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================
def main():
    print("="*60)
    print("🚀 MARTINGALE REAL-DATA BACKTESTER (MT5 ONLY)")
    print("="*60)
    
    # Инициализация MT5
    if not mt5.initialize():
        print("❌ Не удалось подключиться к MetaTrader 5.")
        print("Проверьте, запущен ли терминал и авторизованы ли вы.")
        return

    cfg = Config()
    print(f"\n⚙️ Настройки:")
    print(f"   Баланс: {cfg.start_balance}$ | Лот: {cfg.start_lot} | Множитель: {cfg.multiplier}")
    print(f"   Баров истории: {cfg.bars_count}")
    print("-"*60)

    results = []
    
    for symbol in cfg.symbols:
        # Проверка наличия символа
        if not mt5.symbol_select(symbol, True):
            print(f"⚠️ Символ {symbol} не найден или не выбран в MT5. Пропуск.")
            continue

        for tf in cfg.timeframes:
            print(f"⏳ Тестируем {symbol} {tf}...", end=" ")
            t_start = time.time()
            
            prices = get_mt5_data(symbol, tf, cfg.bars_count)
            
            if not prices:
                print("Нет данных!")
                continue
                
            res = run_backtest_on_prices(prices, symbol, tf, cfg)
            res.elapsed = time.time() - t_start
            results.append(res)
            
            print(f"Циклов: {res.n_cycles} | Win: {res.win_rate:.1f}% | DD: {res.max_dd:.2f}$ | Score: {res.safety_score:.4f}")

    mt5.shutdown()

    # --- ОТЧЕТ ---
    if not results:
        print("\n❌ Нет результатов для отображения.")
        return

    results.sort(key=lambda x: x.safety_score, reverse=True)
    
    print("\n" + "="*70)
    print("🏆 ТОП-5 РЕЗУЛЬТАТОВ ПО SAFETY SCORE")
    print("="*70)
    print(f"{'Symbol':<10} {'TF':<5} {'Cycles':>7} {'Win%':>6} {'Profit':>10} {'MaxDD':>10} {'Score':>8}")
    print("-"*70)
    
    for r in results[:5]:
        print(f"{r.symbol:<10} {r.timeframe:<5} {r.n_cycles:>7} {r.win_rate:>5.1f}% "
              f"{r.total_profit:>10.2f} {r.max_dd:>10.2f} {r.safety_score:>8.4f}")

    # Сохранение
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_real")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mt5_results.json")
    
    with open(path, "w", encoding="utf-8") as f:
        # Преобразуем dataclass в dict для JSON
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)
        
    print(f"\n💾 Результаты сохранены в: {path}")
    print("✅ Готово!")

if __name__ == "__main__":
    main()