# -*- coding: utf-8 -*-
"""
Martingale Safe Bot v2 (FIXED: numpy array checks)
=================================================
Исправлена ошибка: "The truth value of an array... is ambiguous"
Теперь все проверки mt5.positions_get / symbol_info_tick делаются через len().
"""
import os
import json
import time
import datetime
from dataclasses import dataclass, field
from typing import List, Optional

import MetaTrader5 as mt5
import numpy as np


# ============================================================================
# НАСТРОЙКИ
# ============================================================================
@dataclass
class Settings:
    pairs: List[dict] = field(default_factory=lambda: [
        {"symbol": "AUDCHF", "timeframe": mt5.TIMEFRAME_H1, "tf_name": "H1"},
        {"symbol": "NAS100", "timeframe": mt5.TIMEFRAME_M5, "tf_name": "M5"},
    ])

    start_lot: float = 0.01
    multiplier: float = 1.5
    max_positions: int = 10

    # Риск-менеджмент
    max_lot: float = 1.0
    max_cycle_dd_pct: float = 30.0
    daily_loss_pct: float = 10.0
    pause_after_loss_bars: int = 10

    magic: int = 260901
    comment: str = "SafeMarti"
    sleep_sec: int = 5
    status_file: str = "bot_status.json"


# ============================================================================
# УНИВЕРСАЛЬНЫЙ РАЗМЕР КОНТРАКТА
# ============================================================================
def get_contract_size(symbol: str) -> float:
    sym = symbol.upper()
    overrides = {
        "XAU": 100, "XAG": 5000,
        "BTC": 1, "ETH": 1, "SOL": 1, "XRP": 1, "DOGE": 1, "ADA": 1,
        "US30": 1, "SPX": 1, "NAS": 1, "GER": 1, "UK100": 1, "JPN": 1, "AUS200": 1,
    }
    for key, val in overrides.items():
        if key in sym:
            return val

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

    tick = mt5.symbol_info_tick(symbol)
    price = 0.0
    if tick is not None:
        price = tick.ask if tick.ask > 0 else (tick.bid if tick.bid > 0 else 0.0)
    if price == 0.0:
        return 100000.0
    if price > 500:
        return 1.0
    if price > 10:
        return 100.0
    return 100000.0


# ============================================================================
# СОСТОЯНИЕ ЦИКЛА
# ============================================================================
@dataclass
class CycleState:
    symbol: str = ""
    positions: List[int] = field(default_factory=list)
    total_volume: float = 0.0
    opened_bars: int = 0
    last_loss_time: float = 0.0

    def reset(self):
        self.positions.clear()
        self.total_volume = 0.0
        self.opened_bars = 0


# ============================================================================
# БОТ
# ============================================================================
class SafeMartingaleBot:
    def __init__(self, settings: Settings):
        self.cfg = settings
        self.contracts = {}
        self.cycles = {}
        self.day_start_equity = 0.0
        self.day_str = ""
        self._init_mt5()
        self._init_contracts()
        self._init_cycles()

    def _init_mt5(self):
        if not mt5.initialize():
            raise RuntimeError("❌ MT5 не запущен / нет авторизации")
        print("✅ MT5 подключен")

    def _init_contracts(self):
        for p in self.cfg.pairs:
            sym = p["symbol"]
            if not mt5.symbol_select(sym, True):
                print(f"⚠️ {sym} не активен у брокера, пропускаем")
                continue
            self.contracts[sym] = get_contract_size(sym)
            print(f"   {sym}: contract_size = {self.contracts[sym]}")

    def _init_cycles(self):
        for p in self.cfg.pairs:
            self.cycles[p["symbol"]] = CycleState(symbol=p["symbol"])

    def _account_equity(self) -> float:
        info = mt5.account_info()
        if info is None:
            return 0.0
        return float(info.equity)

    def _daily_loss_exceeded(self) -> bool:
        if self.day_start_equity <= 0:
            return False
        limit = self.day_start_equity * (self.cfg.daily_loss_pct / 100.0)
        return (self.day_start_equity - self._account_equity()) >= limit

    def _open_buy(self, symbol: str, volume: float) -> Optional[int]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        price = float(tick.ask)
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(volume, 2),
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 20,
            "magic": self.cfg.magic,
            "comment": self.cfg.comment,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        res = mt5.order_send(req)
        if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
            return int(res.order)
        rc = res.retcode if res is not None else "no response"
        print(f"   ❌ Не удалось открыть {symbol} BUY: {rc}")
        return None

    def _close_position(self, symbol: str, ticket: int) -> bool:
        pos = mt5.positions_get(ticket=ticket)
        if pos is None or len(pos) == 0:
            return True  # уже закрыта
        p = pos[0]
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = float(tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask)
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(p.volume),
            "type": close_type,
            "position": int(ticket),
            "price": price,
            "deviation": 20,
            "magic": self.cfg.magic,
            "comment": self.cfg.comment,
        }
        res = mt5.order_send(req)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE

    def _close_all(self, symbol: str, state: CycleState):
        for ticket in list(state.positions):
            self._close_position(symbol, ticket)
        state.reset()

    def _cycle_profit(self, symbol: str, state: CycleState) -> float:
        profit = 0.0
        for ticket in list(state.positions):
            pos = mt5.positions_get(ticket=ticket)
            if pos is None or len(pos) == 0:
                state.positions.remove(ticket)
                continue
            profit += float(pos[0].profit)
        return profit

    def _process_pair(self, pair: dict):
        symbol = pair["symbol"]
        tf = pair["timeframe"]
        state = self.cycles[symbol]

        # пауза после убытка
        if state.last_loss_time > 0:
            if time.time() < state.last_loss_time + (self.cfg.pause_after_loss_bars * 60):
                return
            state.last_loss_time = 0

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 3)
        if rates is None or len(rates) < 3:
            return
        # rates — numpy array; берём скаляр через .item() или индексацию
        last_close = float(rates[-2]["close"])

        if not state.positions:
            # стартуем цикл раз в бар (простейшая логика — можно заменить)
            self._start_cycle(symbol, state)
            return

        self._check_close(symbol, state)

    def _start_cycle(self, symbol: str, state: CycleState):
        lot = self.cfg.start_lot
        ticket = self._open_buy(symbol, lot)
        if ticket is not None:
            state.positions.append(ticket)
            state.total_volume = lot

    def _check_close(self, symbol: str, state: CycleState):
        profit = self._cycle_profit(symbol, state)
        equity = self._account_equity()

        # защита от DD
        if equity > 0 and profit <= -(equity * (self.cfg.max_cycle_dd_pct / 100.0)):
            print(f"   🛑 {symbol}: защита DD ({profit:.2f}$) — закрываем")
            self._close_all(symbol, state)
            state.last_loss_time = time.time()
            return

        if profit > 0:
            self._close_all(symbol, state)

    def run(self):
        print("🚀 Бот запущен. Ctrl+C для остановки.\n")
        self.day_start_equity = self._account_equity()
        self.day_str = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            while True:
                now = datetime.datetime.now().strftime("%Y-%m-%d")
                if now != self.day_str:
                    self.day_str = now
                    self.day_start_equity = self._account_equity()
                    print("🔄 Новый день — дневной лимит сброшен")

                if self._daily_loss_exceeded():
                    print("🛑 Дневной лимит убытка достигнут — пауза до завтра")
                    time.sleep(60)
                    continue

                for pair in self.cfg.pairs:
                    try:
                        self._process_pair(pair)
                    except Exception as e:
                        print(f"   ⚠️ Ошибка в {pair['symbol']}: {e}")

                self._save_status()
                time.sleep(self.cfg.sleep_sec)

        except KeyboardInterrupt:
            print("\n🛑 Остановка бота...")
        finally:
            for sym, state in self.cycles.items():
                if state.positions:
                    self._close_all(sym, state)
            mt5.shutdown()
            print("✅ MT5 отключен")

    def _save_status(self):
        data = {
            "time": datetime.datetime.now().isoformat(),
            "equity": self._account_equity(),
            "day_start_equity": self.day_start_equity,
            "pairs": {
                sym: {
                    "open_positions": len(state.positions),
                    "total_volume": state.total_volume,
                }
                for sym, state in self.cycles.items()
            },
        }
        try:
            with open(self.cfg.status_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def main():
    cfg = Settings()
    bot = SafeMartingaleBot(cfg)
    bot.run()


if __name__ == "__main__":
    main()