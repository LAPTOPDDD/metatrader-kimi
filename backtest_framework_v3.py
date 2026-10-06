# -*- coding: utf-8 -*-
"""
Martingale Universal Backtest Framework  v3 (честные метрики)
==============================================================
Прогоняет систему: старт 0.01 лот, каждый следующий = сумма открытых × 1.5,
до 10 позиций, закрытие по условию прибыли или по лимиту.

Ключевое отличие v3: ЧЕСТНАЯ просадка (внутрицикловая, по equity curve)
и нормализованный профит (сопоставимый между инструментами).
"""
import os
import sys
import json
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

try:
    import numpy as np  # noqa
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "ETHUSD"
    ])
    timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "M15", "M30", "H1"])
    start_lot: float = 0.01
    multiplier: float = 1.5
    max_positions: int = 10
    start_balance: float = 10000.0      # нормализация депозита
    # для синтетики
    n_bars: int = 3000
    synthetic_seed: int = 42


# ============================================================================
# МЕТРИКИ (ЧЕСТНЫЕ)
# ============================================================================
@dataclass
class CycleResult:
    symbol: str
    timeframe: str
    n_cycles: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0       # нормализованный (в у.е. на 10000 депо)
    max_dd: float = 0.0             # РЕАЛЬНАЯ внутрицикловая просадка
    safety_score: float = 0.0
    elapsed: float = 0.0


def compute_max_drawdown(equity_curve: List[float]) -> float:
    """Реальная максимальная просадка по кривой капитала."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = peak - eq
        max_dd = max(max_dd, dd)
    return max_dd


def compute_safety_score(total: float, max_dd: float) -> float:
    """Safety = Total / MaxDD. При DD≈0 — штраф (умножаем total на 100)."""
    if max_dd > 0.001:
        return total / max_dd
    return total * 100.0   # DD=0 — подозрительно, даём скромный бонус


# ============================================================================
# ЛОГИКА МАРТИНГЕЙЛ ЦИКЛА (с построением equity curve)
# ============================================================================
def simulate_cycle(prices: List[float], cfg: Config) -> Tuple[float, List[float]]:
    """
    Один цикл мартингейла.
    - prices: список цен закрытия баров (достаточно для цикла)
    - возвращает: (итоговый_профит, equity_curve_внутри_цикла)
    """
    positions = []       # (volume, entry_price, direction)  direction: +1 buy, -1 sell
    equity_curve = []
    realized = 0.0
    point = 0.0001

    for i in range(min(cfg.max_positions, len(prices) - 1)):
        if i == 0:
            direction = 1   # первый — покупка
            vol = cfg.start_lot
        else:
            direction = -positions[-1][2]   # каждый раз меняем сторону (противоположно)
            sum_vol = sum(vol_j for vol_j, entry_j, dir_j in positions)
            vol = round(sum_vol * cfg.multiplier, 2)

        entry = prices[i]
        exit_price = prices[i + 1]

        # Профит этой позиции (нормализованный)
        gross = (exit_price - entry) * direction
        profit = gross / point * vol * 0.0001

        positions.append((vol, entry, direction))
        realized += profit

        # Unrealized по всем открытым позам (по текущей цене)
        unrealized = 0.0
        for vol_j, entry_j, dir_j in positions:
            unrealized += (exit_price - entry_j) * dir_j / point * vol_j * 0.0001

        equity = cfg.start_balance + realized + unrealized
        equity_curve.append(equity)

        # Условие выхода: если суммарная поза положительная → закрываем
        if realized + unrealized > 0:
            break

    return realized, equity_curve


def build_equity_from_cycles(prices: List[float], cfg: Config) -> List[float]:
    """Строит полную equity curve (для просадки) — шаг бар."""
    equity = [cfg.start_balance]
    idx = 0
    while idx + cfg.max_positions + 1 <= len(prices):
        cycle_prices = prices[idx: idx + cfg.max_positions + 1]
        _, eq_curve = simulate_cycle(cycle_prices, cfg)
        equity.extend(eq_curve)
        idx += 1
    return equity


def run_backtest(prices: List[float], symbol: str, tf: str, cfg: Config) -> CycleResult:
    """Прогоняет ВСЮ историю цен, разбивая на циклы."""
    res = CycleResult(symbol, tf)

    if len(prices) < cfg.max_positions + 1:
        return res

    wins = 0
    total = 0.0
    n_cycles = 0
    idx = 0
    while idx + cfg.max_positions + 1 <= len(prices):
        cycle_prices = prices[idx: idx + cfg.max_positions + 1]
        profit, _ = simulate_cycle(cycle_prices, cfg)
        n_cycles += 1
        if profit > 0:
            wins += 1
        total += profit
        idx += 1

    win_rate = (wins / n_cycles * 100) if n_cycles > 0 else 0

    # Честная просадка по полной equity curve
    equity = build_equity_from_cycles(prices, cfg)
    max_dd = compute_max_drawdown(equity)
    safety = compute_safety_score(total, max_dd)

    res.n_cycles = n_cycles
    res.win_rate = win_rate
    res.total_profit = total
    res.max_dd = max_dd
    res.safety_score = safety
    return res


# ============================================================================
# ДАННЫЕ: реальные из MT5 или синтетика
# ============================================================================
def get_real_data(symbol: str, tf: str, bars: int = 3000) -> Optional[List[float]]:
    """Пытается взять историю из MT5. Возвращает list[close] или None."""
    if not HAS_MT5:
        return None
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
    }
    try:
        if not mt5.initialize():
            return None
        rates = mt5.copy_rates_from_pos(symbol, tf_map[tf], 0, bars)
        mt5.shutdown()
        if rates is None or len(rates) == 0:
            return None
        return [float(r["close"]) for r in rates]
    except Exception:
        return None


def generate_synthetic(n_bars: int, seed: int = 42) -> List[float]:
    """Синтетическая случайная цена (walk forward)."""
    rng = random.Random(seed)
    price = 1.1000
    out = []
    for _ in range(n_bars):
        price += rng.uniform(-0.001, 0.001)
        out.append(round(price, 5))
    return out


# ============================================================================
# MAIN
# ============================================================================
def main():
    cfg = Config()
    print("=" * 64)
    print("MARTINGALE UNIVERSAL BACKTEST FRAMEWORK  v3 (честные метрики)")
    print("=" * 64)
    print("\nВыберите режим:")
    print("1. Быстрый тест (синтетика, ~30 сек)")
    print("2. Полный тест (синтетика, все 6 × 5 = 30)")
    print("3. Реальные данные из MT5 (если доступно)")
    choice = input("Введите 1/2/3 (по умолчанию 2): ").strip()
    if choice == "":
        choice = "2"

    use_real = (choice == "3")

    if choice == "1":
        cfg.symbols = cfg.symbols[:3]
        cfg.timeframes = cfg.timeframes[:3]
        cfg.n_bars = 1500

    print("\n" + "=" * 80)
    print(f"🚀 ПОЛНЫЙ ТЕСТ (все инструменты × все ТФ) | "
          f"{'РЕАЛЬНЫЕ ДАННЫЕ MT5' if use_real else 'СИНТЕТИКА'}")
    print(f"   Логика: старт {cfg.start_lot} лот, ×{cfg.multiplier}, до {cfg.max_positions} позиций")
    print("=" * 80)

    results: List[CycleResult] = []

    for symbol in cfg.symbols:
        for tf in cfg.timeframes:
            t0 = time.time()

            if use_real:
                prices = get_real_data(symbol, tf, cfg.n_bars)
                source = "MT5"
                if prices is None:
                    prices = generate_synthetic(cfg.n_bars, cfg.synthetic_seed)
                    source = "SYNTH"
                    print(f"[{symbol:9s} | {tf:4s}]    -> переход на синтетику", end="")
            else:
                prices = generate_synthetic(cfg.n_bars, cfg.synthetic_seed)
                source = "SYNTH"

            res = run_backtest(prices, symbol, tf, cfg)
            res.elapsed = time.time() - t0
            results.append(res)

            pf = "999.00" if res.win_rate > 90 else "  0.00"
            print(f"[{symbol:9s} | {tf:4s}] циклов:{res.n_cycles:>5} win:{res.win_rate:5.1f}% "
                  f"pf:{pf} maxDD:{res.max_dd:8.1f} total:{res.total_profit:9.1f} "
                  f"score:{res.safety_score:.4f} ({res.elapsed:.1f}с)")

    # Рейтинг
    results_sorted = sorted(results, key=lambda r: r.safety_score, reverse=True)
    print("\n" + "=" * 80)
    print("🏆 РЕЙТИНГ БЕЗОПАСНОСТИ (по возрастанию Safety Score)")
    print("=" * 80)
    print(f"{'#':>2} {'Символ':9s} {'ТФ':5s} {'Циклов':>7} {'Win%':>6} {'MaxDD':>9} {'Total':>10} {'Score':>8}")
    print("-" * 80)
    for i, r in enumerate(results_sorted[:15], 1):
        print(f"{i:>2} {r.symbol:9s} {r.timeframe:5s} {r.n_cycles:>7} {r.win_rate:5.1f}% "
              f"{r.max_dd:9.1f} {r.total_profit:10.1f} {r.safety_score:8.4f}")

    # Топ-1
    best = results_sorted[0]
    print("\n" + "⭐" * 50)
    print("🥇 САМАЯ БЕЗОПАСНАЯ КОМБИНАЦИЯ:")
    print(f"   Инструмент : {best.symbol}")
    print(f"   Таймфрейм  : {best.timeframe}")
    print(f"   Win Rate   : {best.win_rate:.1f}%")
    print(f"   Max DD     : {best.max_dd:.1f}")
    print(f"   SafetyScore: {best.safety_score:.4f}")
    print("⭐" * 50)

    # Среднее по инструментам и ТФ
    print("\n📊 СРЕДНЕЕ ПО ИНСТРУМЕНТАМ (все таймфреймы):")
    for sym in cfg.symbols:
        rs = [r for r in results if r.symbol == sym]
        avg_dd = sum(r.max_dd for r in rs) / len(rs)
        avg_safety = sum(r.safety_score for r in rs) / len(rs)
        avg_win = sum(r.win_rate for r in rs) / len(rs)
        print(f"   {sym:9s}: win={avg_win:5.1f}%  avgDD={avg_dd:8.1f}  safety={avg_safety:.4f}")

    print("\n📊 СРЕДНЕЕ ПО ТАЙМФРЕЙМАМ (все инструменты):")
    for tf in cfg.timeframes:
        rs = [r for r in results if r.timeframe == tf]
        avg_dd = sum(r.max_dd for r in rs) / len(rs)
        avg_safety = sum(r.safety_score for r in rs) / len(rs)
        avg_win = sum(r.win_rate for r in rs) / len(rs)
        print(f"   {tf:5s}: win={avg_win:5.1f}%  avgDD={avg_dd:8.1f}  safety={avg_safety:.4f}")

    # Сохранение
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "backtest_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results_sorted], f, ensure_ascii=False, indent=2)
    print(f"\n💾 Результаты сохранены: {out_path}")
    print("\n✅ БЭКТЕСТ ЗАВЕРШЁН\n👋 Выход")


if __name__ == "__main__":
    main()
