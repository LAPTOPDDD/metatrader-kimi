"""
Быстрый тест фреймворка (без input) — автоматическая проверка логики.
3 инструмента x 3 таймфрейма на синтетической истории.
"""
import statistics
from dataclasses import dataclass
from typing import List, Dict
import random


START_LOT = 0.01
MULTIPLIER = 1.5
MAX_POSITIONS = 10
SPREAD_PIPS = 1.5

SYMBOLS = ["EURUSD", "XAUUSD", "BTCUSD"]
TIMEFRAMES = {"M1": 1, "M15": 15, "H1": 60}
PIP_SIZE = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
            "XAUUSD": 0.1, "BTCUSD": 1.0, "ETHUSD": 0.1}


@dataclass
class TradeMetrics:
    symbol: str
    timeframe: str
    total_cycles: int = 0
    winning_cycles: int = 0
    losing_cycles: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    max_loss_cycle: float = 0.0
    avg_cycle_profit: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    safety_score: float = 0.0

    def compute_derived(self):
        self.win_rate = (self.winning_cycles / self.total_cycles * 100) if self.total_cycles else 0
        gross_profit = max(self.total_profit, 0)
        gross_loss = abs(min(self.total_profit, 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0


def generate_synthetic_bars(symbol, n=3000, seed=42):
    rng = random.Random(seed)
    pip = PIP_SIZE.get(symbol, 0.0001)
    vol_map = {"EURUSD": 5.0, "GBPUSD": 6.5, "USDJPY": 4.0,
               "XAUUSD": 12.0, "BTCUSD": 40.0, "ETHUSD": 35.0}
    vol_pips = vol_map.get(symbol, 5.0)
    base_price = {"EURUSD": 1.08, "XAUUSD": 2400.0, "BTCUSD": 60000.0}.get(symbol, 1.0)
    bars = []
    price = base_price
    for _ in range(n):
        price += rng.gauss(0, vol_pips * pip)
        bars.append({"open": price, "high": price, "low": price,
                     "close": price + rng.gauss(0, vol_pips * pip * 0.3)})
    return bars


def next_lot_size(open_positions):
    if not open_positions:
        return START_LOT
    lot = sum(p["volume"] for p in open_positions) * MULTIPLIER
    lot = round(lot / 0.01) * 0.01
    return max(lot, START_LOT)


def calc_profit(positions, current_price, pip):
    total = 0.0
    for p in positions:
        total += (current_price - p["entry"]) * p["direction"] / pip * p["volume"]
    return total


def simulate_bot(bars, symbol):
    pip = PIP_SIZE.get(symbol, 0.0001)
    cycle_profits = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    i = 0
    while i < len(bars) - 1:
        open_positions = [{"volume": next_lot_size([]), "direction": 1, "entry": bars[i]["open"]}]
        cycle_closed = False
        step = 1
        for step in range(1, MAX_POSITIONS + 1):
            if i + step >= len(bars):
                break
            cp = bars[i + step]["close"]
            spread = SPREAD_PIPS * pip * sum(p["volume"] for p in open_positions)
            if step == 1:
                profit = calc_profit(open_positions, cp, pip) - spread
                if profit > 0:
                    cycle_profits.append(profit); equity += profit; cycle_closed = True; break
                open_positions.append({"volume": next_lot_size(open_positions), "direction": -1, "entry": cp})
            else:
                profit = calc_profit(open_positions, cp, pip) - spread
                if profit > 0:
                    cycle_profits.append(profit); equity += profit; cycle_closed = True; break
                open_positions.append({"volume": next_lot_size(open_positions),
                                       "direction": -open_positions[-1]["direction"], "entry": cp})
        if not cycle_closed:
            fp = bars[min(i + MAX_POSITIONS, len(bars) - 1)]["close"]
            profit = calc_profit(open_positions, fp, pip) - SPREAD_PIPS * pip * sum(p["volume"] for p in open_positions)
            cycle_profits.append(profit); equity += profit
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        i += max(1, step)
    return cycle_profits, max_dd


def aggregate_bars(bars, factor):
    if factor <= 1:
        return bars
    out = []
    for i in range(0, len(bars), factor):
        chunk = bars[i:i + factor]
        if not chunk:
            continue
        out.append({"open": chunk[0]["open"], "high": max(c["high"] for c in chunk),
                    "low": min(c["low"] for c in chunk), "close": chunk[-1]["close"]})
    return out


def run_test():
    print("=" * 70)
    print("QUICK BACKTEST: 3 инструмента x 3 таймфрейма (синтетика)")
    print("=" * 70)
    results = []
    for symbol in SYMBOLS:
        base = generate_synthetic_bars(symbol, n=3000, seed=hash(symbol) % 1000)
        for tf_name, tf_min in TIMEFRAMES.items():
            bars = aggregate_bars(base, tf_min)
            profits, max_dd = simulate_bot(bars, symbol)
            m = TradeMetrics(symbol=symbol, timeframe=tf_name)
            m.total_cycles = len(profits)
            m.winning_cycles = sum(1 for p in profits if p > 0)
            m.losing_cycles = sum(1 for p in profits if p <= 0)
            m.total_profit = sum(profits)
            m.max_drawdown = max_dd
            m.max_loss_cycle = min(profits) if profits else 0
            m.avg_cycle_profit = statistics.mean(profits) if profits else 0
            m.compute_derived()
            m.safety_score = (m.win_rate / 100) * (1 + min(m.profit_factor, 10)) / (1 + abs(m.max_drawdown))
            results.append(m)
            print(f"[{symbol:8s}|{tf_name:4s}] циклов:{m.total_cycles:4d} win:{m.win_rate:5.1f}% "
                  f"pf:{m.profit_factor:5.2f} DD:{m.max_drawdown:8.1f} total:{m.total_profit:8.1f} score:{m.safety_score:.4f}")

    print("\n" + "=" * 70)
    print("TOP-5 BEZOPASNYH KOMBINATSII (po safety_score):")
    print("=" * 70)
    ranked = sorted(results, key=lambda x: -x.safety_score)
    print(f"{'#':>2} {'Symbol':10s} {'TF':6s} {'Win%':>7s} {'PF':>7s} {'MaxDD':>10s} {'Score':>8s}")
    print("-" * 60)
    for idx, m in enumerate(ranked[:5], 1):
        print(f"{idx:2d} {m.symbol:10s} {m.timeframe:6s} {m.win_rate:6.1f}% {m.profit_factor:6.2f} "
              f"{m.max_drawdown:9.1f} {m.safety_score:7.4f}")

    best = ranked[0]
    print(f"\n★ SAMAYA BEZOPASNAYA: {best.symbol} @ {best.timeframe}")
    print(f"  Win Rate: {best.win_rate:.1f}% | PF: {best.profit_factor:.2f} | MaxDD: {best.max_drawdown:.1f}")

    print("\n[PO INSTRUMENTAM]")
    by_sym: Dict[str, list] = {}
    for m in results:
        by_sym.setdefault(m.symbol, []).append(m)
    for sym, lst in sorted(by_sym.items()):
        print(f"  {sym:10s}: win={statistics.mean(x.win_rate for x in lst):5.1f}%  avgDD={statistics.mean(x.max_drawdown for x in lst):8.1f}")

    print("\n[PO TAYMFREYMAM]")
    by_tf: Dict[str, list] = {}
    for m in results:
        by_tf.setdefault(m.timeframe, []).append(m)
    for tf in ["M1", "M15", "H1"]:
        if tf in by_tf:
            lst = by_tf[tf]
            print(f"  {tf:5s}: win={statistics.mean(x.win_rate for x in lst):5.1f}%  avgDD={statistics.mean(x.max_drawdown for x in lst):8.1f}")

    return results


if __name__ == "__main__":
    run_test()
