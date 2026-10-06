"""
Симулятор стратегии v4 (без трейлинга, новая формула лотов).

Логика:
  - Бар 1: покупка initial_lot (0.01)
  - На каждом новом баре: если суммарная прибыль > 0 -> закрываем всё, новый цикл
    иначе -> открываем противоположную позицию объёмом = сумма_объёмов_открытых * 1.5
  - Лимит 10 позиций -> закрываем всё, новый цикл

Запуск: python simulator.py
"""

import random
from dataclasses import dataclass


@dataclass
class Config:
    symbol: str = "EURUSD"
    initial_lot: float = 0.01
    lot_multiplier: float = 1.5
    lot_step: float = 0.01
    max_positions: int = 10


class Simulator:
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.cycle = 0
        self.total_result = 0.0

    # ---- основная логика (повторяет бота) ----
    def run(self, num_bars=200, seed=42):
        random.seed(seed)
        self.cycle = 0
        self.total_result = 0.0

        # price walk
        price = 1.08000
        positions = []  # (direction, volume, open_price)

        for bar in range(1, num_bars + 1):
            # случайное движение цены
            price += random.uniform(-0.00030, 0.00030)

            if not positions:
                # новый цикл: покупка
                self.cycle += 1
                positions.append(("buy", self.cfg.initial_lot, price))
                print(f"[Bar {bar}] Цикл #{self.cycle} СТАРТ: BUY {self.cfg.initial_lot:.2f} @ {price:.5f}")
                continue

            # считаем суммарную прибыль (упрощённо: каждая позиция = (price - open)*volume*10000 пипсов * 10 $/пип)
            total_profit = 0.0
            for direction, volume, open_price in positions:
                pip_gain = (price - open_price) if direction == "buy" else (open_price - price)
                # 1 лот на 1 пип EURUSD ~ $10, но для наглядности возьмём $1 за 0.01 лота на 1 пип
                total_profit += pip_gain * volume * 100000  # примерно $ на 1 пип * 10000...

            total_profit = round(total_profit, 2)

            if total_profit > 0:
                print(f"[Bar {bar}] Профит {total_profit:+.2f} > 0 -> CLOSE ALL (цикл #{self.cycle} завершён)")
                self.total_result += total_profit
                positions = []
                continue

            if len(positions) >= self.cfg.max_positions:
                print(f"[Bar {bar}] Достигнут лимит {self.cfg.max_positions} -> CLOSE ALL (профит {total_profit:+.2f})")
                self.total_result += total_profit
                positions = []
                continue

            # открываем противоположную позицию: объём = сумма объёмов * 1.5
            sum_volume = sum(v for _, v, _ in positions)
            next_lot = round(sum_volume * self.cfg.lot_multiplier / self.cfg.lot_step) * self.cfg.lot_step
            next_lot = max(self.cfg.lot_step, round(next_lot, 2))
            last_direction = positions[-1][0]
            direction = "sell" if last_direction == "buy" else "buy"
            positions.append((direction, next_lot, price))
            print(f"[Bar {bar}] Профит {total_profit:+.2f} <= 0 -> {direction.upper()} {next_lot:.2f} "
                  f"(sum_vol={sum_volume:.2f} * {self.cfg.lot_multiplier})")

        print("\n" + "=" * 55)
        print(f"ИТОГО циклов: {self.cycle}, Общий результат: {self.total_result:+.2f}")
        print("Бот продолжает работу (не останавливается)." )


if __name__ == "__main__":
    sim = Simulator()
    sim.run(num_bars=60)
