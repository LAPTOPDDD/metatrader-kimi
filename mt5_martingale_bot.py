"""
MT5 Martingale Bot v4
========================
Стратегия (одна минута, EURUSD):
  - Рабочий интервал: M1
  - С приходом нового бара открываем позицию (покупка) — САМЫЙ МИНИМАЛЬНЫЙ лот (0.01)
  - Далее на КАЖДОМ новом баре:
      * если суммарная прибыль по всем открытым позициям > 0 -> закрываем ВСЕ и начинаем цикл заново
      * если суммарная прибыль <= 0 -> открываем позицию в ПРОТИВОПОЛОЖНУЮ сторону
        объёмом: (сумма объёмов уже открытых позиций) * 1.5
  - Максимум 10 одновременно открытых позиций. Когда достигнут лимит -> закрываем все,
    цикл начинается заново.

Изменения в v4:
  - УБРАН трейлинг-стоп
  - Новая формула лота: lot = sum(уже открытых объёмов) * 1.5
  - Старт с 0.01 лота
  - Бот НЕ останавливается при ошибках / после закрытия позиций — работает бесконечно

Зависимости: pip install MetaTrader5
"""

import time
import logging
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: не установлен MetaTrader5. Выполните: pip install MetaTrader5")
    sys.exit(1)


# ============================================================================
#  КОНФИГУРАЦИЯ
# ============================================================================
@dataclass
class Config:
    symbol: str = "EURUSD"               # Торгуемый инструмент
    timeframe: int = mt5.TIMEFRAME_M1    # Минутный график
    magic_number: int = 24092001         # Уникальный magic для наших ордеров
    max_positions: int = 10              # Максимум одновременных позиций в цикле
    initial_lot: float = 0.01            # Стартовый лот
    lot_multiplier: float = 1.5          # Множитель: следующий лот = сумма_объёмов * 1.5
    lot_step: float = 0.01               # Шаг нормализации лота (округление)
    lot_min: float = 0.01                # Минимальный лот
    lot_max: float = 100.0               # Максимальный лот
    slippage_points: int = 30            # Допустимый проскальзывание (пунктов)
    comment: str = "MMv4"                # Комментарий к ордерам
    tick_check_interval: float = 0.5      # Интервал проверки тика (сек)
    bar_wait_timeout: float = 70.0        # Таймаут ожидания нового бара (сек)
    retry_on_error: bool = True           # Продолжать работу при ошибках
    max_consecutive_errors: int = 5       # После стольких ошибок подряд -> пауза


# ============================================================================
#  ЛОГИРОВАНИЕ
# ============================================================================
def setup_logger(level=logging.INFO):
    logger = logging.getLogger("mt5bot")
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ============================================================================
#  БЕЗОПАСНЫЙ ДОСТУП К ПОЛЯМ TradePosition
# ============================================================================
def safe_get(obj, attr, default=0.0):
    """Безопасное получение атрибута (защита от отсутствия полей в некоторых версиях MT5)."""
    return getattr(obj, attr, default)


def get_position_profit(pos) -> float:
    """Полная прибыль по позиции: profit + swap (+ commission, если доступно)."""
    profit = safe_get(pos, "profit", 0.0)
    swap = safe_get(pos, "swap", 0.0)
    commission = safe_get(pos, "commission", 0.0)
    return profit + swap + commission


# ============================================================================
#  ОСНОВНОЙ БОТ
# ============================================================================
class MartingaleBot:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self.logger = setup_logger()
        self._running = False
        self._consecutive_errors = 0
        self.cycle_count = 0
        self.total_result = 0.0
        self._symbol_info_cache = None

        # Состояние текущего цикла
        self.cycle_positions: List[int] = []   # ticket-ы открытых позиций цикла
        self.cycle_direction = None            # "buy" | "sell" — направление ПОСЛЕДНЕЙ открытой позиции
        self.cycle_start_time: Optional[datetime] = None

    # ---------- жизненный цикл ----------
    def start(self):
        self._running = True
        self.logger.info("=" * 60)
        self.logger.info("MT5 Martingale Bot v4 запуск...")
        self.logger.info(
            "symbol=%s TF=M1 initial_lot=%s multiplier=%s max_positions=%d",
            self.cfg.symbol, self.cfg.initial_lot, self.cfg.lot_multiplier, self.cfg.max_positions,
        )

        if not self._init_mt5():
            self.logger.error("Не удалось инициализировать MT5. Выход.")
            return

        try:
            self.run()
        except KeyboardInterrupt:
            self.logger.info("Получен сигнал остановки (Ctrl+C)")
        except Exception as e:
            self.logger.exception("Критическая ошибка: %s", e)
        finally:
            self.shutdown()

    def shutdown(self):
        self._running = False
        self.logger.info("Завершение работы бота...")
        self.logger.info("Всего циклов: %d, Общий результат: %.2f", self.cycle_count, self.total_result)
        # Позиции НЕ закрываем — просто отключаемся от MT5, пусть остаются открытыми
        if mt5.terminal_info() is not None:
            mt5.shutdown()
            self.logger.info("MT5 shutdown")
        self.logger.info("Бот остановлен")

    # ---------- инициализация MT5 ----------
    def _init_mt5(self) -> bool:
        if not mt5.initialize():
            self.logger.error("mt5.initialize() failed, error=%s", mt5.last_error())
            return False

        # Выбираем символ (может потребоваться добавление в Market Watch)
        if not mt5.symbol_select(self.cfg.symbol, True):
            self.logger.error("Не удалось выбрать символ %s", self.cfg.symbol)
            return False

        info = mt5.symbol_info(self.cfg.symbol)
        if info is None:
            self.logger.error("symbol_info(%s) вернул None", self.cfg.symbol)
            return False
        self._symbol_info_cache = info
        self.logger.info("Symbol info: %s, point=%s, digits=%d, min_lot=%s",
                         self.cfg.symbol, info.point, info.digits, info.volume_min)
        return True

    # ---------- основной цикл ----------
    def run(self):
        self.logger.info("Бот начал работу. Ожидаем новый бар...")
        last_bar_time = self._get_current_bar_time()

        while self._running:
            try:
                # Ждём появления НОВОГО бара
                new_bar_time = self._wait_new_bar(last_bar_time)
                if new_bar_time is None:
                    continue  # таймаут / остановка

                last_bar_time = new_bar_time
                self._consecutive_errors = 0

                # ---- логика на новом баре ----
                if not self.cycle_positions:
                    # Цикл пуст -> начинаем новый цикл: ПОКУПКА 0.01
                    self._start_new_cycle(direction="buy", lot=self.cfg.initial_lot)
                else:
                    # У нас уже есть открытые позиции -> проверяем прибыль
                    total_profit = self._get_total_profit()

                    if total_profit > 0:
                        # В плюсе -> закрываем всё, начинаем цикл заново
                        self._close_all_and_reset(total_profit)
                    else:
                        # В минусе или ноль -> открываем противоположную позицию
                        if len(self.cycle_positions) >= self.cfg.max_positions:
                            # Достигли лимита -> закрываем всё, новый цикл
                            self.logger.warning(
                                "Достигнут лимит позиций (%d). Закрываем все.", self.cfg.max_positions,
                            )
                            self._close_all_and_reset(total_profit)
                        else:
                            next_lot = self._calculate_next_lot()
                            # Противоположная сторона относительно ПОСЛЕДНЕЙ открытой позиции
                            direction = "sell" if self.cycle_direction == "buy" else "buy"
                            self._open_position(direction, next_lot)

            except Exception as e:
                self._handle_cycle_error(e)

    # ---------- обработка ошибок (НЕ останавливаем бота) ----------
    def _handle_cycle_error(self, e: Exception):
        self._consecutive_errors += 1
        self.logger.exception("Ошибка в цикле: %s", e)
        if not self._running:
            return
        if self._consecutive_errors >= self.cfg.max_consecutive_errors:
            self.logger.warning("Слишком много ошибок подряд (%d), пауза 10 сек",
                                self._consecutive_errors)
            time.sleep(10)
            self._consecutive_errors = 0
        else:
            self.logger.info("Продолжаем работу после ошибки...")
            time.sleep(3)

    # ---------- управление циклом ----------
    def _start_new_cycle(self, direction: str, lot: float):
        self.cycle_count += 1
        self.cycle_start_time = datetime.now()
        self.logger.info("===== НОВЫЙ ЦИКЛ #%d =====", self.cycle_count)
        self._open_position(direction, lot)

    def _close_all_and_reset(self, total_profit: float):
        self.logger.info("Закрываем все позиции цикла (суммарный профит=%.2f) ...", total_profit)
        for ticket in list(self.cycle_positions):
            self._close_position(ticket)
        self.total_result += total_profit
        self.logger.info("Цикл #%d завершён. Итог цикла: %.2f, Общий результат: %.2f",
                         self.cycle_count, total_profit, self.total_result)
        # Сброс состояния -> на следующем баре начнётся новый цикл (БЕЗ остановки)
        self._reset_cycle_state()

    def _reset_cycle_state(self):
        self.cycle_positions = []
        self.cycle_direction = None
        self.cycle_start_time = None

    # ---------- расчёт лота по новой формуле ----------
    def _calculate_next_lot(self) -> float:
        """
        Следующий объём = (сумма объёмов уже открытых позиций) * multiplier,
        после чего округляется до lot_step и ограничивается min/max.
        """
        sum_volume = self._get_open_volumes_sum()
        raw = sum_volume * self.cfg.lot_multiplier
        normalized = self._normalize_lot(raw)
        self.logger.info("Расчёт лота: sum_open_volumes=%.4f * %.2f = %.4f -> normalized=%.2f",
                         sum_volume, self.cfg.lot_multiplier, raw, normalized)
        return normalized

    def _get_open_volumes_sum(self) -> float:
        """Сумма объёмов (лотов) открытых позиций ТЕКУЩЕГО цикла (по нашим ticket-ам)."""
        positions = self._get_our_positions_objects()
        return sum(safe_get(p, "volume", 0.0) for p in positions)

    def _normalize_lot(self, lot: float) -> float:
        step = self.cfg.lot_step
        # Округление вверх до ближайшего шага (чтобы лот не был меньше нужного)
        normalized = round(lot / step) * step
        if normalized < step:
            normalized = step
        normalized = max(self.cfg.lot_min, min(self.cfg.lot_max, normalized))
        # Ограничение точности (2 знака для лотов обычно достаточно)
        return round(normalized, 2)

    # ---------- работа с позициями ----------
    def _open_position(self, direction: str, lot: float):
        lot = self._normalize_lot(lot)
        info = self._symbol_info_cache or mt5.symbol_info(self.cfg.symbol)

        if direction == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.cfg.symbol).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.cfg.symbol).bid

        slippage = self.cfg.slippage_points * info.point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "slippage": slippage,
            "magic": self.cfg.magic_number,
            "comment": self.cfg.comment,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.retcode if result else mt5.last_error()
            self.logger.error("Ошибка открытия %s %.2f: retcode=%s, comment=%s",
                              direction, lot, err, getattr(result, "comment", ""))
            # Пробуем другой тип заполнения
            request["type_filling"] = mt5.ORDER_FILLING_IOC
            result2 = mt5.order_send(request)
            if result2 is None or result2.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error("Повторная попытка тоже не удалась")
                return
            result = result2

        ticket = result.order
        self.cycle_positions.append(ticket)
        self.cycle_direction = direction
        self.logger.info("ОТКРЫТА %s ПОЗИЦИЯ: ticket=%d, volume=%.2f, price=%.5f, bar=%s",
                         direction.upper(), ticket, lot, result.price, self._get_current_bar_time())
        self._log_cycle_status()

    def _close_position(self, ticket: int):
        positions = self._get_all_positions()
        for p in positions:
            if safe_get(p, "ticket", None) == ticket:
                profit = get_position_profit(p)
                close_req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": ticket,
                    "symbol": self.cfg.symbol,
                    "volume": safe_get(p, "volume", 0.0),
                    "type": mt5.ORDER_TYPE_SELL if safe_get(p, "type", 0) == mt5.POSITION_TYPE_BUY
                           else mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(self.cfg.symbol).bid
                            if safe_get(p, "type", 0) == mt5.POSITION_TYPE_BUY
                            else mt5.symbol_info_tick(self.cfg.symbol).ask,
                    "magic": self.cfg.magic_number,
                    "comment": "close",
                }
                res = mt5.order_send(close_req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info("Position closed: ticket=%d, profit=%.2f", ticket, profit)
                else:
                    self.logger.error("Ошибка закрытия ticket=%d: %s", ticket,
                                      res.retcode if res else mt5.last_error())
                return
        self.logger.warning("Позиция ticket=%d не найдена среди открытых", ticket)

    # ---------- получение позиций ----------
    def _get_all_positions(self):
        try:
            positions = mt5.positions_get(symbol=self.cfg.symbol)
        except Exception:
            positions = None
        return positions if positions is not None else []

    def _get_our_positions_objects(self):
        """Наши позиции текущего цикла (по magic + ticket-ам цикла)."""
        all_pos = self._get_all_positions()
        our = [p for p in all_pos
               if safe_get(p, "magic", None) == self.cfg.magic_number
               and safe_get(p, "ticket", None) in self.cycle_positions]
        return our

    def _get_total_profit(self) -> float:
        """Суммарная прибыль по ВСЕМ позициям символа (с учётом всех открытых)."""
        # В стратегии "суммарная позиция" — берём все наши открытые по символу
        positions = self._get_all_positions()
        our_positions = [p for p in positions
                         if safe_get(p, "magic", None) == self.cfg.magic_number]
        if not our_positions:
            return 0.0
        return sum(get_position_profit(p) for p in our_positions)

    # ---------- ожидание нового бара ----------
    def _get_current_bar_time(self) -> Optional[datetime]:
        rates = mt5.copy_rates_from_pos(self.cfg.symbol, self.cfg.timeframe, 0, 1)
        if not rates:
            return None
        return datetime.fromtimestamp(rates[0]["time"])

    def _wait_new_bar(self, current_bar_time: Optional[datetime]) -> Optional[datetime]:
        start = time.time()
        while self._running:
            now = time.time()
            if now - start > self.cfg.bar_wait_timeout:
                self.logger.debug("Таймаут ожидания нового бара")
                return current_bar_time  # продолжаем с тем же временем
            bar_time = self._get_current_bar_time()
            if bar_time and bar_time != current_bar_time:
                return bar_time
            time.sleep(0.2)
        return None

    # ---------- статус ----------
    def _log_cycle_status(self):
        total_profit = self._get_total_profit()
        sum_vol = self._get_open_volumes_sum()
        self.logger.info(
            "СТАТУС: позиций=%d, sum_volume=%.2f, total_profit=%.2f, direction=%s",
            len(self.cycle_positions), sum_vol, total_profit, self.cycle_direction,
        )


# ============================================================================
#  ТОЧКА ВХОДА
# ============================================================================
def main():
    config = Config(
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_M1,
        magic_number=24092001,
        max_positions=10,
        initial_lot=0.01,
        lot_multiplier=1.5,
        lot_step=0.01,
        lot_min=0.01,
        lot_max=100.0,
    )
    bot = MartingaleBot(config)
    bot.start()


if __name__ == "__main__":
    main()
