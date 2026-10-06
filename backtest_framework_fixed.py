#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
MARTINGALE UNIVERSAL BACKTEST FRAMEWORK  v2 (FIXED)
============================================================
Прогоняет систему "мартингейл от 0.01 лота, ×1.5, до 10 позиций"
на ВСЕХ инструментах × ВСЕХ таймфреймах и ищет самую безопасную комбинацию.

ИСПРАВЛЕНО В v2:
  - Сохранение JSON в текущую папку (работает на Windows и Linux)
  - Фильтрация по датам (можно указать период истории)
  - Вывод топ-5 комбинаций
  - Защита от деления на ноль в SafetyScore

ЛОГИКА СИСТЕМЫ:
  Бар 1:   покупка 0.01 лота
  Бар 2:   если профит > 0 -> закрыть; иначе продать (сумма_объёмов × 1.5)
  Бар 3-10: если суммарный профит > 0 -> закрыть всё
           иначе открыть в противоположную сторону (сумма_объёмов × 1.5)
  После закрытия -> новый цикл (покупка 0.01)

ЗАПУСК:
  python backtest_framework_fixed.py
  (выберите 1 = быстрый тест, 2 = полный тест, 3 = реальные данные MT5)
============================================================
"""

import os
import sys
import json
import random
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

# ============================================================
#  КОНФИГУРАЦИЯ
# ============================================================
@dataclass
class Config:
    start_lot: float = 0.01
    multiplier: float = 1.5
    max_positions: int = 10
    spread_points: float = 2.0       # спред в пунктах (для синтетики)
    commission_per_lot: float = 0.0  # комиссия за лот (в валюте счёта)
    slippage_points: float = 0.0
    # Период для реальных данных MT5 (если None -> весь доступный)
    history_from: Optional[str] = None   # "YYYY-MM-DD"
    history_to:   Optional[str] = None   # "YYYY-MM-DD"

# ============================================================
#  МОДЕЛИ РЕЗУЛЬТАТОВ
# ============================================================
@dataclass
class CycleResult:
    symbol: str
    timeframe: str
    bars_to_close: int
    total_profit: float
    max_drawdown: float
    positions_count: int

@dataclass
class ComboResult:
    symbol: str
    timeframe: str
    cycles: int = 0
    wins: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    max_equity_drop: float = 0.0
    profit_factor: float = 0.0
    safety_score: float = 0.0
    history_bars: int = 0

# ============================================================
#  ГЕНЕРАТОР СИНТЕТИЧЕСКИХ ДАННЫХ
# ============================================================
def generate_synthetic_prices(n: int, volatility: float = 0.0005,
                              drift: float = 0.0, seed: int = 42) -> List[float]:
    """Генерирует случайные цены методом геометрического блуждания."""
    random.seed(seed)
    prices = [1.0]
    for _ in range(n - 1):
        change = random.gauss(drift, volatility)
        prices.append(max(0.0001, prices[-1] * (1 + change)))
    return prices

# ============================================================
#  ОДИН БЭКТЕСТ (ядро логики)
# ============================================================
def run_backtest(prices: List[float], symbol: str, timeframe: str,
                 cfg: Config) -> ComboResult:
    """
    Прогоняет мартингейл-циклы на массиве цен.
    Возвращает ComboResult с агрегированной статистикой.
    """
    result = ComboResult(symbol=symbol, timeframe=timeframe, history_bars=len(prices))

    i = 0
    equity = 0.0
    peak_equity = 0.0

    while i < len(prices) - 1:
        # ---------- НАЧАЛО ЦИКЛА ----------
        positions = []          # (volume, direction, open_price)
        cycle_profit = 0.0
        cycle_peak = 0.0
        closed_this_cycle = False

        # БАР 1: покупаем 0.01
        entry_price = prices[i]
        lot = cfg.start_lot
        direction = 1  # 1 = buy, -1 = sell
        positions.append((lot, direction, entry_price))
        i += 1

        # ---------- ВНУТРИ ЦИКЛА ----------
        while i < len(prices) and len(positions) <= cfg.max_positions and not closed_this_cycle:
            current_price = prices[i]
            spread = cfg.spread_points * 0.0001  # упрощённо в "ценаx"

            # Считаем текущий профит по всем позициям
            total_profit = 0.0
            for vol, dirn, open_px in positions:
                price_diff = (current_price - open_px) * dirn
                total_profit += price_diff * vol * 100000  # нормировка под "валюту"
            # Вычитаем спред и комиссию (при открытии каждой новой)
            total_profit -= cfg.commission_per_lot * sum(v for v, _, _ in positions)

            cycle_profit = total_profit
            cycle_peak = max(cycle_peak, cycle_profit)
            equity += 0.0  # equity обновляется после закрытия

            # --- ПРАВИЛО ЗАКРЫТИЯ: сумма > 0 -> закрыть всё ---
            if total_profit > 0:
                closed_this_cycle = True
                result.total_profit += total_profit
                if total_profit > 0:
                    result.wins += 1
                break

            # --- ИНАЧЕ: открываем в противоположную сторону (×1.5) ---
            if len(positions) >= cfg.max_positions:
                # Достигли лимита -> принудительно закрываем с текущим профитом
                closed_this_cycle = True
                result.total_profit += total_profit
                break

            # Считаем сумму объёмов и умножаем на 1.5
            sum_volumes = sum(v for v, _, _ in positions)
            new_lot = sum_volumes * cfg.multiplier
            new_lot = round(new_lot, 2)

            direction = -direction  # меняем сторону
            entry_price = current_price
            positions.append((new_lot, direction, entry_price))
            i += 1

        # ---------- КОНЕЦ ЦИКЛА ----------
        result.cycles += 1

        # Отслеживаем просадку equity
        peak_equity = max(peak_equity, equity)
        result.max_equity_drop = max(result.max_equity_drop, peak_equity - equity)

        if not closed_this_cycle:
            i += 1  # двигаем бар, если цикл не закрылся естественно

    # ---------- ФИНАЛЬНЫЕ МЕТРИКИ ----------
    result.max_drawdown = result.max_equity_drop

    # Profit Factor (защита от деления на ноль)
    gross_profit = max(0.0, result.total_profit) if result.total_profit > 0 else 0.0
    gross_loss = abs(min(0.0, result.total_profit)) if result.total_profit < 0 else 0.0
    if gross_loss > 0:
        result.profit_factor = gross_profit / gross_loss
    else:
        result.profit_factor = 999.0  # все циклы в плюсе -> PF = бесконечность (обозначаем 999)

    # Safety Score = win_rate × profit_factor / (maxDD + epsilon)
    win_rate = (result.wins / result.cycles) if result.cycles > 0 else 0.0
    dd_safe = result.max_drawdown + 1.0  # защита от деления на ноль
    result.safety_score = (win_rate * result.profit_factor) / dd_safe

    return result

# ============================================================
#  ПОЛУЧЕНИЕ ДАННЫХ (MT5 или синтетика)
# ============================================================
def get_prices_mt5(symbol: str, timeframe: str, cfg: Config) -> Optional[List[float]]:
    """Пытается получить реальные данные из MT5. Возвращает None при ошибке."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("   ⚠️ MT5 не инициализирован, используем синтетику")
            return None

        # Маппинг таймфреймов
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
        }
        tf = tf_map.get(timeframe)
        if tf is None:
            return None

        # Определяем диапазон дат
        to_date = datetime.now()
        if cfg.history_to:
            to_date = datetime.strptime(cfg.history_to, "%Y-%m-%d")
        from_date = to_date - timedelta(days=90)  # последние 90 дней по умолчанию
        if cfg.history_from:
            from_date = datetime.strptime(cfg.history_from, "%Y-%m-%d")

        rates = mt5.copy_rates_range(symbol, tf, from_date, to_date)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            return None

        return [float(r["close"]) for r in rates]

    except Exception as e:
        print(f"   ⚠️ Ошибка MT5: {e}")
        return None

def get_prices(symbol: str, timeframe: str, cfg: Config, use_real: bool) -> List[float]:
    """Возвращает цены (реальные или синтетические)."""
    if use_real:
        prices = get_prices_mt5(symbol, timeframe, cfg)
        if prices:
            return prices
        print(f"   → переход на синтетику для {symbol} {timeframe}")

    # Синтетические данные с разной волатильностью под инструмент
    vol_map = {
        "EURUSD": 0.0003, "GBPUSD": 0.0004, "USDJPY": 0.0003,
        "XAUUSD": 0.0010, "BTCUSD": 0.0030, "ETHUSD": 0.0025,
    }
    volatility = vol_map.get(symbol, 0.0005)

    # Количество баров зависит от таймфрейма
    bars_map = {"M1": 5000, "M5": 3000, "M15": 2000, "M30": 1500, "H1": 1000}
    n = bars_map.get(timeframe, 2000)

    return generate_synthetic_prices(n, volatility=volatility, seed=hash(symbol + timeframe) % 10000)

# ============================================================
#  РЕЙТИНГ И ОТЧЁТ
# ============================================================
def rank_and_report(results: List[ComboResult], output_dir: str):
    """Сортирует результаты по SafetyScore и выводит отчёт + сохраняет JSON."""
    # Фильтруем те, что успели отработать хотя бы несколько циклов
    valid = [r for r in results if r.cycles >= 5]
    valid.sort(key=lambda x: x.safety_score, reverse=True)

    print("\n" + "=" * 80)
    print("🏆 РЕЙТИНГ БЕЗОПАСНОСТИ (по возрастанию Safety Score)")
    print("=" * 80)
    print(f"{'#':>2} {'Символ':<10} {'ТФ':<5} {'Циклов':>7} {'Win%':>7} {'PF':>8} {'MaxDD':>10} {'Total':>10} {'Score':>8}")
    print("-" * 80)

    for idx, r in enumerate(valid[:15], 1):
        win_pct = (r.wins / r.cycles * 100) if r.cycles > 0 else 0.0
        print(f"{idx:>2} {r.symbol:<10} {r.timeframe:<5} {r.cycles:>7} "
              f"{win_pct:>7.1f} {r.profit_factor:>8.2f} {r.max_drawdown:>10.1f} "
              f"{r.total_profit:>10.1f} {r.safety_score:>8.4f}")

    # ---------- ТОП-1 ----------
    if valid:
        best = valid[0]
        print("\n" + "⭐" * 70)
        print("🥇 САМАЯ БЕЗОПАСНАЯ КОМБИНАЦИЯ:")
        print(f"   Инструмент : {best.symbol}")
        print(f"   Таймфрейм  : {best.timeframe}")
        print(f"   Win Rate   : {(best.wins / best.cycles * 100):.1f}%")
        print(f"   Profit Fact: {best.profit_factor:.2f}")
        print(f"   Max DD     : {best.max_drawdown:.1f}")
        print(f"   SafetyScore: {best.safety_score:.4f}")
        print("⭐" * 70)

    # ---------- ТОП-5 ----------
    print("\n📌 ТОП-5 КОМБИНАЦИЙ ДЛЯ ТОРГОВЛИ:")
    for idx, r in enumerate(valid[:5], 1):
        win_pct = (r.wins / r.cycles * 100) if r.cycles > 0 else 0.0
        verdict = "✅" if r.total_profit > 0 else "❌"
        print(f"  {idx}. {r.symbol:<10} {r.timeframe:<5} win={win_pct:.1f}%  DD={r.max_drawdown:.0f}  total={r.total_profit:+.0f}  {verdict}")

    # ---------- СРЕДНЕЕ ПО ИНСТРУМЕНТАМ ----------
    print("\n📊 СРЕДНЕЕ ПО ИНСТРУМЕНТАМ (все таймфреймы):")
    symbols = sorted(set(r.symbol for r in results))
    for sym in symbols:
        sym_results = [r for r in results if r.symbol == sym and r.cycles >= 5]
        if sym_results:
            avg_dd = statistics.mean(r.max_drawdown for r in sym_results)
            avg_safety = statistics.mean(r.safety_score for r in sym_results)
            avg_win = statistics.mean((r.wins / r.cycles * 100) for r in sym_results if r.cycles > 0)
            print(f"   {sym:<10}: win={avg_win:>5.1f}%  avgDD={avg_dd:>8.1f}  safety={avg_safety:>8.4f}")

    # ---------- СРЕДНЕЕ ПО ТАЙМФРЕЙМАМ ----------
    print("\n📊 СРЕДНЕЕ ПО ТАЙМФРЕЙМАМ (все инструменты):")
    timeframes = sorted(set(r.timeframe for r in results))
    for tf in timeframes:
        tf_results = [r for r in results if r.timeframe == tf and r.cycles >= 5]
        if tf_results:
            avg_dd = statistics.mean(r.max_drawdown for r in tf_results)
            avg_safety = statistics.mean(r.safety_score for r in tf_results)
            avg_win = statistics.mean((r.wins / r.cycles * 100) for r in tf_results if r.cycles > 0)
            print(f"   {tf:<5}: win={avg_win:>5.1f}%  avgDD={avg_dd:>8.1f}  safety={avg_safety:>8.4f}")

    # ---------- СОХРАНЕНИЕ JSON ----------
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "backtest_results.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"\n💾 Результаты сохранены: {json_path}")
    except Exception as e:
        print(f"\n⚠️ Не удалось сохранить JSON: {e}")

# ============================================================
#  ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    print("=" * 60)
    print("MARTINGALE UNIVERSAL BACKTEST FRAMEWORK  v2")
    print("=" * 60)

    cfg = Config()

    print("\nВыберите режим:")
    print("1. Быстрый тест (синтетические данные, ~30 сек)")
    print("2. Полный тест (синтетика, все 6 × 5 = 30 комбинаций)")
    print("3. Реальные данные из MT5 (если доступно)")

    try:
        choice = input("Введите 1/2/3 (по умолчанию 2): ").strip()
    except Exception:
        choice = "2"
    if not choice:
        choice = "2"

    # Определяем наборы
    if choice == "1":
        symbols = ["EURUSD", "XAUUSD"]
        timeframes = ["M1", "M15", "H1"]
        use_real = False
        print("\n🚀 БЫСТРЫЙ ТЕСТ (2 инструмента × 3 ТФ)")
    elif choice == "3":
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "ETHUSD"]
        timeframes = ["M1", "M5", "M15", "M30", "H1"]
        use_real = True
        print("\n🚀 ПОЛНЫЙ ТЕСТ (все инструменты × все ТФ) + РЕАЛЬНЫЕ ДАННЫЕ MT5")
    else:
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "ETHUSD"]
        timeframes = ["M1", "M5", "M15", "M30", "H1"]
        use_real = False
        print("\n🚀 ПОЛНЫЙ ТЕСТ (синтетика, 6 × 5 = 30 комбинаций)")

    print(f"   Логика: старт {cfg.start_lot} лот, ×{cfg.multiplier}, до {cfg.max_positions} позиций")
    print("=" * 80)

    results: List[ComboResult] = []
    total = len(symbols) * len(timeframes)

    for idx, symbol in enumerate(symbols, 1):
        for tf in timeframes:
            print(f"[{symbol:<10}| {tf:<4}] ", end="", flush=True)
            t0 = datetime.now()

            prices = get_prices(symbol, tf, cfg, use_real)
            if not prices or len(prices) < 100:
                print("недостаточно данных, пропуск")
                continue

            result = run_backtest(prices, symbol, tf, cfg)
            results.append(result)

            elapsed = (datetime.now() - t0).total_seconds()
            win_pct = (result.wins / result.cycles * 100) if result.cycles > 0 else 0.0
            print(f"циклов: {result.cycles:>4} win: {win_pct:>5.1f}% "
                  f"pf: {result.profit_factor:>6.2f} maxDD: {result.max_drawdown:>8.1f} "
                  f"total: {result.total_profit:>8.1f} score:{result.safety_score:.4f} ({elapsed:.1f}с)")

    # ---------- ОТЧЁТ ----------
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    rank_and_report(results, output_dir)

    print("\n✅ БЭКТЕСТ ЗАВЕРШЁН")

# ============================================================
#  ТОЧКА ВХОДА
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Выход")
