"""
MetaTrader 5 Martingale Bot - ИСПРАВЛЕННАЯ ВЕРСИЯ
=================================================
Стратегия:
- Торгуем на EURUSD, таймфрейм M1
- С приходом нового бара покупаем 1 лот (одна позиция)
- Если на следующем баре профит > 0 — закрываем
- Если профит < 0 — открываем в противоположную сторону с удвоенным лотом
- Продолжаем до 10 одновременно открытых позиций, потом закрываем всё
- После закрытия цикл НЕ останавливается, а начинается заново (бесконечно)
- Добавлен тиковый трейлинг-стоп (проверка каждые 0.5 сек)

ИСПРАВЛЕНИЯ:
- Безопасный доступ к полям TradePosition (через getattr, commission может отсутствовать)
- Бот НЕ останавливается при ошибках — цикл продолжается
- Кэширование symbol info для ускорения

Автор: AI Assistant
"""

import time
import logging
import signal
import sys
from datetime import datetime
from typing import List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Ошибка: не установлен MetaTrader5. Установите: pip install MetaTrader5")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "symbol": "EURUSD",
    "timeframe": mt5.TIMEFRAME_M1,
    "magic_number": 123456,
    "max_positions": 10,           # Максимум одновременно открытых позиций
    "use_fixed_lot": True,         # True = лоты 1,2,3,4...  False = мартингейл 1,2,4,8...
    "base_lot": 1.0,               # Базовый лот (если use_fixed_lot=False, это начальный лот)
    "deviation": 20,               # Допустимое отклонение цены (в пунктах)
    "slippage": 5,                 # Проскальзывание (в пунктах)

    # Тиковый трейлинг-стоп (проверяется каждые tick_check_interval секунд)
    "trailing_enabled": True,
    "trailing_start_profit": 3.0,  # После какого профита в пипсах активируется трейлинг
    "trailing_distance": 2.0,      # На сколько пипсов откат от пика → закрытие (1/2/3 пипса)
    "tick_check_interval": 0.5,    # Интервал проверки тиков (секунды)

    # Защита от ошибок
    "max_consecutive_errors": 5,    # Максимум последовательных ошибок перед паузой
    "error_pause_seconds": 10,      # Пауза при серии ошибок
}

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ БЕЗОПАСНОГО ДОСТУПА К ПОЛЯМ MT5
# ═══════════════════════════════════════════════════════════════

def safe_get(obj, attr, default=0.0):
    """Безопасное получение атрибута объекта с дефолтным значением."""
    return getattr(obj, attr, default)


def get_position_profit(pos) -> float:
    """
    Безопасно получает общую прибыль по позиции.
    Учитывает: profit + swap + commission (если доступно).
    """
    profit = safe_get(pos, 'profit', 0.0)
    swap = safe_get(pos, 'swap', 0.0)
    # commission может отсутствовать в некоторых версиях MT5 / типах счетов
    commission = safe_get(pos, 'commission', 0.0)
    return profit + swap + commission


def get_position_type(pos) -> str:
    """Безопасно получает тип позиции (buy/sell)."""
    pos_type = safe_get(pos, 'type', mt5.POSITION_TYPE_BUY)
    return "buy" if pos_type == mt5.POSITION_TYPE_BUY else "sell"


def get_position_volume(pos) -> float:
    """Безопасно получает объём позиции."""
    return safe_get(pos, 'volume', 0.0)


# ═══════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС БОТА
# ═══════════════════════════════════════════════════════════════

class MartingaleBot:
    """Мартингейл бот с тиковым трейлинг-стопом и бесконечным циклом."""

    def __init__(self, config: dict = None):
        self.config = config or CONFIG
        self.symbol = self.config["symbol"]
        self.timeframe = self.config["timeframe"]
        self.magic = self.config["magic_number"]

        # Состояние цикла
        self.cycle_count = 0
        self.total_profit = 0.0
        self._running = True
        self._symbol_info = None

        # Трейлинг-стоп
        self._peak_profit = 0.0
        self._trailing_active = False

        # Счётчик ошибок для защиты
        self._consecutive_errors = 0

        # Обработчик сигналов (Ctrl+C)
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)

    # ─────────────────────────────────────────────────────────
    # УПРАВЛЕНИЕ ЖИЗНЕННЫМ ЦИКЛОМ
    # ─────────────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        """Обработка сигнала остановки (Ctrl+C)."""
        logger.info("Получен сигнал остановки (%s). Завершаем работу...", signum)
        self._running = False

    def _reset_cycle(self):
        """Сброс состояния цикла для начала нового (БЕЗ остановки бота)."""
        self._peak_profit = 0.0
        self._trailing_active = False
        self._consecutive_errors = 0
        logger.info("─" * 60)
        logger.info("🔄 Цикл сброшен. Готов к новому циклу.")
        logger.info("─" * 60)

    # ─────────────────────────────────────────────────────────
    # ПОДКЛЮЧЕНИЕ К MT5
    # ─────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Инициализация и подключение к MT5."""
        logger.info("Подключение к MetaTrader 5...")

        if not mt5.initialize():
            logger.error("Не удалось инициализировать MT5. Код ошибки: %s", mt5.last_error())
            return False

        # Проверяем авторизацию
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Не удалось получить информацию о счёте. Возможно, вы не авторизованы.")
            return False

        logger.info("✅ Подключено к MT5")
        logger.info("   Счёт: %s | Баланс: %.2f %s",
                    safe_get(account_info, 'login', 'N/A'),
                    safe_get(account_info, 'balance', 0.0),
                    safe_get(account_info, 'currency', 'USD'))

        # Выбираем символ
        if not mt5.symbol_select(self.symbol, True):
            logger.error("Не удалось выбрать символ %s", self.symbol)
            return False

        # Кэшируем информацию о символе
        self._symbol_info = mt5.symbol_info(self.symbol)
        if self._symbol_info is None:
            logger.error("Не удалось получить info для символа %s", self.symbol)
            return False

        logger.info("   Символ: %s | Точность: %s знаков | Пункт: %s",
                    self.symbol,
                    safe_get(self._symbol_info, 'digits', 'N/A'),
                    safe_get(self._symbol_info, 'point', 'N/A'))

        return True

    def disconnect(self):
        """Завершение работы с MT5."""
        logger.info("MT5 shutdown")
        mt5.shutdown()

    # ─────────────────────────────────────────────────────────
    # РАБОТА С ПОЗИЦИЯМИ
    # ─────────────────────────────────────────────────────────

    def _get_my_positions(self) -> List:
        """Получить все открытые позиции бота по данному символу и магику."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        return [p for p in positions if safe_get(p, 'magic', -1) == self.magic]

    def _get_total_profit(self) -> float:
        """
        Общая прибыль по всем позициям бота (с защитой от ошибок).
        Возвращает 0.0 если позиций нет или произошла ошибка.
        """
        try:
            positions = self._get_my_positions()
            if not positions:
                return 0.0
            return sum(get_position_profit(p) for p in positions)
        except Exception as e:
            logger.debug("Ошибка получения профита: %s", e)
            return 0.0

    def _get_position_count(self) -> int:
        """Количество открытых позиций бота."""
        return len(self._get_my_positions())

    def _get_position_point_value(self) -> float:
        """Стоимость одного пункта для расчёта пипсов."""
        if self._symbol_info is None:
            self._symbol_info = mt5.symbol_info(self.symbol)
        point = safe_get(self._symbol_info, 'point', 0.0001)
        digits = safe_get(self._symbol_info, 'digits', 5)
        # Для 5-значных котировок: 1 пипс = 10 пунктов
        if digits >= 4:
            return point * 10
        return point

    # ─────────────────────────────────────────────────────────
    # ТОРГОВЫЕ ОПЕРАЦИИ
    # ─────────────────────────────────────────────────────────

    def _calculate_lot(self, position_index: int) -> float:
        """
        Расчёт лота для позиции.
        position_index — номер позиции в цикле (1, 2, 3, ...).
        """
        if self.config["use_fixed_lot"]:
            # Фиксированные лоты: 1, 2, 3, 4...
            return self.config["base_lot"] * position_index
        else:
            # Классический мартингейл: 1, 2, 4, 8...
            return self.config["base_lot"] * (2 ** (position_index - 1))

    def _open_position(self, position_index: int) -> bool:
        """
        Открыть позицию нужного типа.
        Чётные позиции — sell, нечётные — buy (попеременно).
        """
        lot = self._calculate_lot(position_index)
        lot = round(lot, 2)

        # Определяем тип: первая позиция = buy, дальше чередуем
        # (позиция 1: buy, 2: sell, 3: buy, 4: sell ...)
        if position_index % 2 == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = safe_get(mt5.symbol_info_tick(self.symbol), 'ask', 0.0)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = safe_get(mt5.symbol_info_tick(self.symbol), 'bid', 0.0)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": self.config["deviation"],
            "magic": self.magic,
            "comment": f"Martingale_{position_index}",
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        # Пытаемся отправить ордер
        result = mt5.order_send(request)

        if result is None:
            logger.error("❌ order_send вернул None. Ошибка: %s", mt5.last_error())
            return False

        if safe_get(result, 'retcode', -1) != mt5.TRADE_RETCODE_DONE:
            logger.error("❌ Ошибка открытия позиции #%s: retcode=%s, comment=%s",
                         position_index,
                         safe_get(result, 'retcode', 'N/A'),
                         safe_get(result, 'comment', 'N/A'))
            return False

        deal_type = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
        logger.info("✅ Открыта позиция #%s: %s, лот=%.2f, цена=%.5f, deal=%s",
                    position_index, deal_type, lot, price,
                    safe_get(result, 'deal', 'N/A'))
        return True

    def _close_all_positions(self, reason: str = "") -> bool:
        """
        Закрыть все открытые позиции бота.
        Возвращает True если все позиции закрыты успешно.
        """
        positions = self._get_my_positions()
        if not positions:
            logger.info("ℹ️ Нет открытых позиций для закрытия")
            return True

        logger.info("🔒 Закрытие всех позиций (%s). Причина: %s", len(positions), reason)

        success = True
        for pos in positions:
            ticket = safe_get(pos, 'ticket', 0)
            pos_type = get_position_type(pos)
            volume = get_position_volume(pos)

            close_type = mt5.ORDER_TYPE_SELL if pos_type == "buy" else mt5.ORDER_TYPE_BUY
            price = safe_get(mt5.symbol_info_tick(self.symbol),
                             'bid' if close_type == mt5.ORDER_TYPE_SELL else 'ask', 0.0)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": self.config["deviation"],
                "magic": self.magic,
                "comment": "Close_" + reason,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "type_time": mt5.ORDER_TIME_GTC,
            }

            result = mt5.order_send(request)
            if result is None or safe_get(result, 'retcode', -1) != mt5.TRADE_RETCODE_DONE:
                err_code = safe_get(result, 'retcode', 'N/A') if result else mt5.last_error()
                logger.error("❌ Ошибка закрытия позиции %s: %s", ticket, err_code)
                success = False
            else:
                profit = get_position_profit(pos)
                logger.info("   ✅ Закрыта позиция %s, профит=%.2f",
                            ticket, profit)

        # Небольшая пауза для обработки ордеров
        time.sleep(0.5)
        return success

    # ─────────────────────────────────────────────────────────
    # ТИКОВЫЙ ТРЕЙЛИНГ-СТОП
    # ─────────────────────────────────────────────────────────

    def _update_trailing_stop(self):
        """
        Обновление тикового трейлинг-стопа.
        Вызывается в цикле проверки тиков.
        При срабатывании закрывает все позиции.
        """
        if not self.config["trailing_enabled"]:
            return

        positions = self._get_my_positions()
        if not positions:
            self._trailing_active = False
            self._peak_profit = 0.0
            return

        total_profit = self._get_total_profit()
        point_value = self._get_position_point_value()

        # Переводим профит в пипсы (приблизительно, через стоимость пункта)
        # Для точного расчёта в пипсах используем point_value
        profit_in_pips = total_profit / (point_value * 10) if point_value > 0 else 0

        # Активация трейлинга
        if not self._trailing_active:
            if profit_in_pips >= self.config["trailing_start_profit"]:
                self._trailing_active = True
                self._peak_profit = profit_in_pips
                logger.info("🎯 Трейлинг АКТИВИРОВАН. Профит: %.2f пипсов",
                            profit_in_pips)
        else:
            # Обновляем пик
            if profit_in_pips > self._peak_profit:
                self._peak_profit = profit_in_pips

            # Проверяем откат
            drawdown = self._peak_profit - profit_in_pips
            if drawdown >= self.config["trailing_distance"]:
                logger.info("🚨 ТРЕЙЛИНГ СРАБОТАЛ! Пик: %.2f, текущий: %.2f, откат: %.2f пипсов",
                            self._peak_profit, profit_in_pips, drawdown)
                if self._close_all_positions("trailing_stop"):
                    # После закрытия — сразу НОВЫЙ цикл (без остановки)
                    self._reset_cycle()

    def _wait_for_tick_events(self, timeout: float = None) -> bool:
        """
        Ожидание событий тика с периодической проверкой трейлинг-стопа.
        Возвращает False если нужно остановиться.
        """
        start_time = time.time()
        while self._running:
            if timeout and (time.time() - start_time) >= timeout:
                return True

            # Проверяем трейлинг-стоп
            self._update_trailing_stop()

            # Проверяем, не закрылись ли все позиции (например, по трейлингу)
            if self._get_position_count() == 0 and self._trailing_active:
                # Трейлинг сработал и всё закрыл — цикл уже сброшен в _update_trailing_stop
                pass

            time.sleep(self.config["tick_check_interval"])

        return False

    # ─────────────────────────────────────────────────────────
    # ОЖИДАНИЕ НОВОГО БАРА
    # ─────────────────────────────────────────────────────────

    def _wait_for_new_bar(self) -> Optional[int]:
        """
        Ожидание формирования нового бара.
        Возвращает timestamp нового бара или None при остановке.
        """
        # Получаем последний бар
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 1)
        if rates is None or len(rates) == 0:
            logger.warning("⚠️ Не удалось получить данные баров")
            time.sleep(1)
            return None

        last_bar_time = rates[0][0]  # time (Unix timestamp)
        logger.debug("Текущий бар: %s",
                     datetime.fromtimestamp(last_bar_time).strftime('%Y-%m-%d %H:%M:%S'))

        while self._running:
            time.sleep(1)  # Проверяем каждую секунду
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 1)
            if rates is None or len(rates) == 0:
                continue

            current_bar_time = rates[0][0]
            if current_bar_time > last_bar_time:
                logger.debug("📊 Новый бар: %s",
                             datetime.fromtimestamp(current_bar_time).strftime('%Y-%m-%d %H:%M:%S'))
                return current_bar_time

        return None

    # ─────────────────────────────────────────────────────────
    # ОСНОВНОЙ ЦИКЛ ТОРГОВЛИ
    # ─────────────────────────────────────────────────────────

    def _run_cycle(self) -> bool:
        """
        Выполнение одного торгового цикла.
        Возвращает True если цикл завершён нормально.
        """
        self.cycle_count += 1
        logger.info("")
        logger.info("=" * 60)
        logger.info("🚀 ЦИКЛ #%s — СТАРТ", self.cycle_count)
        logger.info("=" * 60)

        position_index = 0

        # ШАГ 1: Покупаем 1 лот (позиция #1)
        logger.info("📍 Шаг 1: Покупаем 1 лот")
        if not self._open_position(1):
            self._consecutive_errors += 1
            logger.error("Не удалось открыть первую позицию")
            return False
        position_index = 1

        # Основной цикл наращивания позиций
        while self._running:
            # Ждём новый бар
            new_bar = self._wait_for_new_bar()
            if new_bar is None and not self._running:
                break

            if not self._running:
                break

            # Проверяем текущие позиции (может быть несколько из-за мартингейла)
            positions = self._get_my_positions()
            if not positions:
                # Все позиции закрылись (например, по трейлингу или с положительным профитом)
                logger.info("✅ Все позиции закрыты. Завершаем цикл.")
                break

            total_profit = self._get_total_profit()
            logger.info("📊 Бар закрыт. Позиций: %s, Общий профит: %.2f",
                        len(positions), total_profit)

            # ПРАВИЛО: Если суммарная позиция положительная → закрываем всё
            if total_profit > 0:
                logger.info("✅ Профит положительный (%.2f). Закрываем всё!", total_profit)
                self._close_all_positions("profit_positive")
                break

            # ПРАВИЛО: Если отрицательная — открываем в противоположную сторону
            position_index += 1

            if position_index > self.config["max_positions"]:
                logger.warning("⚠️ Достигнут лимит позиций (%s). Закрываем всё!",
                               self.config["max_positions"])
                self._close_all_positions("max_positions_reached")
                break

            logger.info("📍 Шаг %s: Открываем позицию #%s (противоположная сторона, лот x%s)",
                        position_index, position_index, position_index)

            if not self._open_position(position_index):
                self._consecutive_errors += 1
                logger.error("Не удалось открыть позицию #%s", position_index)
                # Пытаемся закрыть то, что есть
                self._close_all_positions("open_error")
                break

            # Активируем трейлинг после открытия новой позиции
            self._trailing_active = False  # сбросим, пусть активируется заново по профиту
            self._peak_profit = 0.0

        # Записываем результат цикла
        cycle_profit = self._get_total_profit()
        self.total_profit += cycle_profit
        logger.info("🏁 Цикл #%s завершён. Результат цикла: %.2f, Общий результат: %.2f",
                    self.cycle_count, cycle_profit, self.total_profit)

        # Сбрасываем цикл для следующего (БЕЗ остановки)
        self._reset_cycle()
        return True

    # ─────────────────────────────────────────────────────────
    # ПУБЛИЧНЫЙ МЕТОД ЗАПУСКА
    # ─────────────────────────────────────────────────────────

    def run(self):
        """Главный цикл работы бота (бесконечный)."""
        logger.info("=" * 60)
        logger.info("🤖 MARTINGALE BOT ЗАПУЩЕН")
        logger.info("   Символ: %s | Таймфрейм: M1", self.symbol)
        logger.info("   Макс. позиций: %s", self.config["max_positions"])
        logger.info("   Трейлинг: %s (старт: +%.1f пипсов, дистанция: %.1f)",
                    "ВКЛ" if self.config["trailing_enabled"] else "ВЫКЛ",
                    self.config["trailing_start_profit"],
                    self.config["trailing_distance"])
        logger.info("   Режим лотов: %s",
                   "Фиксированные (1,2,3...)" if self.config["use_fixed_lot"]
                   else "Мартингейл (1,2,4...)")
        logger.info("=" * 60)
        logger.info("⏳ Ожидание нового бара для начала торговли...")
        logger.info("   (Нажмите Ctrl+C для остановки)")

        try:
            while self._running:
                try:
                    # Запускаем новый цикл
                    self._run_cycle()

                    # Защита от серии ошибок
                    if self._consecutive_errors >= self.config["max_consecutive_errors"]:
                        logger.warning("⚠️ Слишком много ошибок подряд (%s). Пауза %ss...",
                                       self._consecutive_errors,
                                       self.config["error_pause_seconds"])
                        time.sleep(self.config["error_pause_seconds"])
                        self._consecutive_errors = 0

                    # Небольшая пауза между циклами
                    if self._running:
                        logger.info("⏳ Ожидание следующего цикла...")
                        time.sleep(2)

                except Exception as e:
                    self._consecutive_errors += 1
                    logger.exception("❌ Ошибка в цикле: %s", e)
                    # ВАЖНО: НЕ останавливаемся, продолжаем работу
                    if self._running:
                        logger.info("🔄 Продолжаем работу после ошибки...")
                        time.sleep(3)

        except KeyboardInterrupt:
            logger.info("Получена команда остановки (Ctrl+C)")
        finally:
            # При выходе НЕ закрываем позиции автоматически — пусть остаются открытыми
            # (закомментируйте следующие 2 строки, если нужно автозакрытие при выходе)
            # logger.info("Закрываем все открытые позиции...")
            # self._close_all_positions("shutdown")

            logger.info("📊 ИТОГО:")
            logger.info("   Всего циклов: %s", self.cycle_count)
            logger.info("   Общий результат: %.2f", self.total_profit)

            self.disconnect()
            logger.info("🛑 Бот остановлен")


# ═══════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════

def main():
    """Запуск бота."""
    # Можно переопределить настройки здесь
    custom_config = {
        **CONFIG,
        # "trailing_enabled": True,      # ВКЛ тиковый трейлинг
        # "trailing_start_profit": 3.0,  # Активация после +3 пипсов
        # "trailing_distance": 2.0,      # Закрытие при откате на 2 пипса (можно 1 или 3)
        # "use_fixed_lot": True,         # True = 1,2,3,4...  False = 1,2,4,8...
        # "max_positions": 10,
    }

    bot = MartingaleBot(custom_config)

    if not bot.connect():
        logger.error("❌ Не удалось подключиться к MT5. Проверьте, что терминал запущен.")
        sys.exit(1)

    try:
        bot.run()
    except Exception as e:
        logger.exception("❌ Критическая ошибка: %s", e)
    finally:
        bot.disconnect()


if __name__ == "__main__":
    main()
