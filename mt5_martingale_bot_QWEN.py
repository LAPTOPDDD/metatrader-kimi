import MetaTrader5 as mt5
import pandas as pd
import time
import datetime

# --- НАСТРОЙКИ ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
MAX_LEVELS = 10  # Максимум 10 позиций
INITIAL_LOT = 0.01  # Минимальный лот
MAGIC_NUMBER = 123456  # Идентификатор бота, чтобы не путать с ручными сделками

# Глобальные переменные для отслеживания состояния
current_level = 0
last_bar_time = 0
is_trading_cycle_active = False

def initialize_mt5():
    """Инициализация подключения к MT5"""
    if not mt5.initialize():
        print("Ошибка инициализации MT5")
        return False
    print(f"MT5 подключен. Версия: {mt5.version()}")
    return True

def get_last_bar_time():
    """Получает время открытия последнего закрытого бара"""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 2)
    if rates is None or len(rates) < 2:
        return None
    # Берем время предпоследнего бара (последний еще формируется)
    return rates[-2]['time']

def get_open_positions():
    """Получает все открытые позиции по символу с нашим MAGIC_NUMBER"""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    # Фильтруем только наши позиции
    my_positions = [pos for pos in positions if pos.magic == MAGIC_NUMBER]
    return my_positions

def calculate_total_profit(positions):
    """Считает суммарную прибыль всех открытых позиций"""
    total_profit = sum(pos.profit for pos in positions)
    return total_profit

def get_current_lot(level):
    """Рассчитывает лот для текущего уровня (Мартингейл)"""
    if level == 0:
        return INITIAL_LOT
    else:
        # Удвоение: 1, 2, 4, 8... но у нас логика чуть другая по ТЗ
        # ТЗ: 1-й бар: 1 лот. 2-й бар (если минус): удвоенный (2). 3-й бар (если минус): утроенный относительно предыдущего? 
        # В ТЗ сказано: "удвоенным лотом", потом "утроенным лотом". 
        # Обычно под этим понимают умножение на 2 каждый раз. 
        # Давай сделаем классический мартингейл: lot * 2^(level)
        # Но если строго по ТЗ: 
        # Уровень 0: 1 лот
        # Уровень 1: 2 лота
        # Уровень 2: 3 лота? Или 4? "Утроенным" может означать x3 от начального или x3 от предыдущего.
        # Давай сделаем стандартный множитель 2 для надежности математики, но если хочешь жестче - измени multiplier.
        
        multiplier = 2 
        return round(INITIAL_LOT * (multiplier ** level), 2)

def open_position(direction, lot_size):
    """Открывает позицию"""
    point = mt5.symbol_info(SYMBOL).point
    price = mt5.symbol_info_tick(SYMBOL).ask if direction == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": direction,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": f"Martingale Level {current_level}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Ошибка открытия позиции: {result.comment}")
        return False
    print(f"Открыта позиция: {'BUY' if direction == mt5.ORDER_TYPE_BUY else 'SELL'}, Лот: {lot_size}")
    return True

def close_all_positions():
    """Закрывает все позиции нашего бота"""
    positions = get_open_positions()
    for pos in positions:
        point = mt5.symbol_info(SYMBOL).point
        price = mt5.symbol_info_tick(SYMBOL).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": price,
            "position": pos.ticket,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Close Cycle",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Ошибка закрытия позиции {pos.ticket}: {result.comment}")
        else:
            print(f"Закрыта позиция {pos.ticket}")

def determine_direction():
    """Определяет направление первой сделки. Можно заменить на любую логику."""
    # Для простоты: случайное направление или по последнему тику
    # В реальном боте здесь будет твоя стратегия входа
    import random
    return mt5.ORDER_TYPE_BUY if random.choice([True, False]) else mt5.ORDER_TYPE_SELL

def main_loop():
    global current_level, last_bar_time, is_trading_cycle_active
    
    print("Бот запущен. Ожидание нового бара...")
    
    while True:
        # Обновляем данные
        current_bar_time = get_last_bar_time()
        
        if current_bar_time is None:
            time.sleep(1)
            continue
            
        # Проверяем, пришел ли новый бар
        if current_bar_time != last_bar_time:
            last_bar_time = current_bar_time
            print(f"\n--- Новый бар в {datetime.datetime.fromtimestamp(current_bar_time)} ---")
            
            positions = get_open_positions()
            num_positions = len(positions)
            
            # СЦЕНАРИЙ 1: Нет открытых позиций -> Начинаем цикл
            if num_positions == 0:
                is_trading_cycle_active = True
                current_level = 0
                direction = determine_direction() # Здесь твоя логика входа
                lot = get_current_lot(current_level)
                open_position(direction, lot)
                
            # СЦЕНАРИЙ 2: Есть открытые позиции -> Проверяем условия выхода/усреднения
            elif is_trading_cycle_active:
                total_profit = calculate_total_profit(positions)
                
                # Условие 1: Суммарная позиция положительна -> Закрываем все
                if total_profit > 0:
                    print(f"Суммарная прибыль положительна: {total_profit:.2f}. Закрываем все.")
                    close_all_positions()
                    is_trading_cycle_active = False
                    current_level = 0
                    
                # Условие 2: Достигли лимита позиций (10) -> Закрываем все принудительно
                elif num_positions >= MAX_LEVELS:
                    print(f"Достигнут лимит позиций ({MAX_LEVELS}). Закрываем все.")
                    close_all_positions()
                    is_trading_cycle_active = False
                    current_level = 0
                    
                # Условие 3: Позиция отрицательная -> Открываем противоположную с увеличенным лотом
                else:
                    # Определяем направление последней позиции, чтобы открыть противоположную
                    last_pos = positions[-1] # Берем последнюю открытую
                    opposite_direction = mt5.ORDER_TYPE_SELL if last_pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    
                    current_level += 1
                    new_lot = get_current_lot(current_level)
                    
                    print(f"Позиция отрицательна. Открываем уровень {current_level}, лот {new_lot}, направление {'BUY' if opposite_direction == mt5.ORDER_TYPE_BUY else 'SELL'}")
                    open_position(opposite_direction, new_lot)
        
        time.sleep(1) # Пауза чтобы не грузить процессор

if __name__ == "__main__":
    if initialize_mt5():
        try:
            main_loop()
        except KeyboardInterrupt:
            print("Бот остановлен пользователем.")
            close_all_positions()
            mt5.shutdown()
    else:
        print("Не удалось подключиться к MT5")