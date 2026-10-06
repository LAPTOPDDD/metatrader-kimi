# -*- coding: utf-8 -*-
"""
Martingale Global Market Backtester v7 (ROBUST contract_size)
=================================================================
Исправление ошибки AttributeError: 'SymbolInfo' object has no attribute
'contract_size'.

Теперь размер контракта определяется универсально:
  1) явные override-значения для металлов/крипты/индексов
  2) поля MT5 (contract_size / trade_contract_size через hasattr)
  3) fallback по цене, если брокер ничего не дал
"""
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ Ошибка: Установи библиотеку: pip install MetaTrader5")
    sys.exit(1)


# ============================================================================
# СПИСОК ИНСТРУМЕНТОВ (ГЛОБАЛЬНЫЙ)
# ============================================================================
GLOBAL_SYMBOLS = [
    # Форекс
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY", "CADJPY", "CHFJPY",
    "EURAUD", "GBPAUD", "EURCAD", "GBPCAD", "AUDCAD", "AUDCHF", "AUDNZD",
    "NZDJPY", "NZDCAD", "NZDCHF", "GBPNZD", "EURNZD", "CADCHF",
    # Металлы и сырьё
    "XAUUSD", "XAGUSD", "XAUEUR", "USOIL", "UKOIL", "NGAS",
    # Криптовалюты
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "ADAUSD",
    "DOTUSD", "MATICUSD", "LTCUSD", "LINKUSD", "AVAXUSD", "UNIUSD",
    # Индексы
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

    bars_count: int = 3000


@dataclass
class CycleResult:
    symbol: str = ""
    timeframe: str = ""
    n_cycles: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    max_dd: float = 0.0
    safety_score: float = 0.0
    contract_size: float = 0.0


# ============================================================================
# УНИВЕРСАЛЬНЫЙ РАЗМЕР КОНТРАКТА
# ============================================================================
def get_contract_size(symbol: str) -> float:
    """
    Универсально получаем размер контракта для символа.
    Порядок: явные override -> поля MT5 (contract_size / trade_contract_size) -> fallback по цене.
    """
    sym = symbol.upper()

    # 1) Явные override (если брокер криво отдаёт данные)
    overrides = {
        "XAU": 100,    # золото (XAUUSD, XAUEUR)
        "XAG": 5000,   # серебро
        "BTC": 1,
        "ETH": 1,
        "SOL": 1,
        "XRP": 1,
        "DOGE": 1,
        "ADA": 1,
        "US30": 1, "SPX": 1, "NAS": 1, "GER": 1, "UK100": 1, "JPN": 1, "AUS200": 1,
    }
    for key, val in overrides.items():
        if key in sym:
            return val

    # 2) Пробуем взять из MT5 (разные версии называют поле по-разному)
    if mt5:
        info = mt5.symbol_info(symbol)
        if info is not None:
            for attr in ("contract_size", "trade_contract_size"):
                if hasattr(info, attr):
                    try:
                        v = getattr(info, attr)
                        if v and v > 0:
                            return float(v)
                    except Exception:
                        pass

    # 3) Fallback: угадываем по текущей цене
    price = 0.0
    if mt5:
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None:
            price = tick.ask if tick.ask > 0 else (tick.bid if tick.bid > 0 else 0.0)
    if price == 0.0:
        # если даже цену не взяли — дефолт форекс
        return 100000.0

    if price > 500:
        return 1.0     # крипта / индексы
    if price > 10:
        return 100.0   # золото / серебро
    return 100000.0    # форекс


# ============================================================================
# ЯДРО РАСЧЁТОВ
# ============================================================================
def calculate_max_drawdown(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return max_dd


def simulate_cycle(
    prices: List[float], cfg: Config, contract_size: float = 100000.0
) -> Tuple[float, List[float]]:
    positions = []
    realized_pnl = 0.0
    equity_curve = []

    for i in range(min(cfg.max_positions, len(prices) - 1)):
        if i == 0:
            direction = 1
            vol = cfg.start_lot
        else:
            direction = -positions[-1][2]
            sum_vol = sum(v for v, _, _ in positions)
            vol = round(sum_vol * cfg.multiplier, 2)

        entry = prices[i]
        exit_price = prices[i + 1]

        # ✅ Правильная формула: contract_size приходит извне
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


def run_backtest(
    prices: List[float], symbol: str, tf: str, cfg: Config, contract_size: float = 100000.0
) -> CycleResult:
    res = CycleResult(symbol=symbol, timeframe=tf, contract_size=contract_size)
    if len(prices) < 15:
        return res

    idx = 0
    total_profit = 0.0
    wins = 0
    cycles = 0
    global_equity = [cfg.start_balance]

    while idx + cfg.max_positions + 1 < len(prices):
        chunk = prices[idx: idx + cfg.max_positions + 2]
        pnl, eq_curve = simulate_cycle(chunk, cfg, contract_size)

        cycles += 1
        total_profit += pnl
        if pnl > 0:
            wins += 1

        if eq_curve:
            global_equity.extend(eq_curve)
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
    print("=" * 70)
    print("🌍 GLOBAL MARKET MARTINGALE TESTER v7 (MT5 REAL DATA, ROBUST)")
    print("=" * 70)

    if not mt5.initialize():
        print("❌ MT5 не запущен или нет авторизации.")
        return

    cfg = Config()
    print(f"📊 Инструментов: {len(cfg.symbols)} | ТФ: {cfg.timeframes} | Баров: {cfg.bars_count}")
    print("⚠️ Убедись, что все символы активны в Обзоре рынка MT5!\n")

    # Мини-тест резолвера (опционально, можно удалить)
    print("🔍 Проверка размеров контрактов:")
    for s in ["EURUSD", "XAUUSD", "XAGUSD", "BTCUSD", "US30", "NAS100"]:
        print(f"   {s:<8}: contract_size = {get_contract_size(s)}")
    print("")

    results = []
    total_tests = len(cfg.symbols) * len(cfg.timeframes)
    current_test = 0

    for symbol in cfg.symbols:
        if not mt5.symbol_select(symbol, True):
            continue

        # ✅ Универсальное получение contract_size
        contract_size = get_contract_size(symbol)

        for tf in cfg.timeframes:
            current_test += 1
            print(f"[{current_test}/{total_tests}] {symbol:<10} {tf:<4} (contract={contract_size}) ...", end=" ")

            rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{tf}"), 0, cfg.bars_count)
            if rates is None or len(rates) == 0:
                print("Нет данных")
                continue

            prices = [float(r['close']) for r in rates]
            res = run_backtest(prices, symbol, tf, cfg, contract_size)
            results.append(res)

            status = "✅" if res.safety_score > 1 else "⚠️"
            print(f"Win:{res.win_rate:.1f}% DD:{res.max_dd:.0f}$ Score:{res.safety_score:.3f} {status}")

    mt5.shutdown()

    # Сортировка и вывод топа
    results.sort(key=lambda x: x.safety_score, reverse=True)

    print("\n" + "=" * 80)
    print("🏆 ТОП-10 САМЫХ БЕЗОПАСНЫХ АКТИВОВ ДЛЯ МАРТИНГЕЙЛА")
    print("=" * 80)
    print(f"{'#':<3} {'Symbol':<10} {'TF':<5} {'Cycles':>7} {'Win%':>6} {'Profit':>10} {'MaxDD':>10} {'Score':>8}")
    print("-" * 80)

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