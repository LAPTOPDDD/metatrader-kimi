import MetaTrader5 as mt5
import time
import datetime

# --- НАСТРОЙКИ БОТА ---
TRADING_PAIRS = {
    "AUDCHF": mt5.TIMEFRAME_H1,
    "NAS100": mt5.TIMEFRAME_M5,
    "XAGUSD": mt5.TIMEFRAME_M1,
    "US30":   mt5.TIMEFRAME_M1,
    "AUDCAD": mt5.TIMEFRAME_H1
}

INITIAL_LOT = 0.01
MULTIPLIER = 1.5
MAX_LEVELS = 10
MAGIC_BASE = 777000
DRAWDOWN_LIMIT_PERCENT = 50 

last_bar_times = {}
cycle_levels = {} 

def get_filling_mode(symbol):
    """Автоматически определяет поддерживаемый режим заполнения ордера"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return mt5.ORDER_FILLING_FOK
    
    filling_modes = symbol_info.filling_modes
    
    # Приоритет: FOK -> IOC -> RETURN
    if filling_modes & mt5.SYMBOL_FILLING_FOK == mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    elif filling_modes & mt5.SYMBOL_FILLING_IOC == mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

def get_contract_size(symbol):
    info = mt5.symbol_info(symbol)
    if info and hasattr(info, 'trade_contract_size'):
        return float(info.trade_contract_size)
    overrides = {"XAG": 5000, "XAU": 100, "US30": 1, "NAS": 1, "GER": 1}
    for key, val in overrides.items():
        if key in symbol: return val
    return 100000 

def get_current_lot(symbol, level):
    lot = INITIAL_LOT * (MULTIPLIER ** level)
    info = mt5.symbol_info(symbol)
    if info:
        lot = max(info.volume_min, min(lot, info.volume_max))
        lot = round(lot, 2)
    return lot

def get_open_positions_count(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None: return 0
    return len([p for p in positions if p.magic == magic])

def calculate_total_profit(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if not positions: return 0.0
    total = 0.0
    for pos in positions:
        if pos.magic == magic:
            total += pos.profit
    return total

def close_all_for_symbol(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if not positions: return
    for pos in positions:
        if pos.magic == magic:
            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if pos.type == 0 else tick.ask
            filling_type = get_filling_mode(symbol) # <-- Исправление здесь
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": price,
                "position": pos.ticket,
                "magic": magic,
                "deviation": 20,
                "type_filling": filling_type, # <-- Используем динамический режим
            }
            res = mt5.order_send(request)
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"⚠️ Ошибка закрытия {symbol}: {res.comment}")

def open_new_position(symbol, timeframe, level, magic):
    lot = get_current_lot(symbol, level)
    direction = mt5.ORDER_TYPE_BUY if level % 2 == 0 else mt5.ORDER_TYPE_SELL
    
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if direction == mt5.ORDER_TYPE_BUY else tick.bid
    filling_type = get_filling_mode(symbol) # <-- Исправление здесь
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": direction,
        "price": price,
        "magic": magic,
        "deviation": 20,
        "comment": f"Martingale L{level}",
        "type_filling": filling_type, # <-- Используем динамический режим
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Ошибка открытия {symbol}: {result.comment} (FillMode: {filling_type})")
    else:
        fill_name = {mt5.ORDER_FILLING_FOK: "FOK", mt5.ORDER_FILLING_IOC: "IOC", mt5.ORDER_FILLING_RETURN: "RETURN"}
        print(f"✅ {symbol} Открыт уровень {level}, Лот: {lot}, {'BUY' if direction==0 else 'SELL'} (Режим: {fill_name.get(filling_type)})")

def check_drawdown_limit(initial_balance):
    account_info = mt5.account_info()
    if not account_info: return True
    current_equity = account_info.equity
    loss_percent = ((initial_balance - current_equity) / initial_balance) * 100
    if loss_percent >= DRAWDOWN_LIMIT_PERCENT:
        print(f"🚨 КРИТИЧЕСКАЯ ПРОСАДКА {loss_percent:.1f}%! Закрытие всех позиций.")
        for symbol in TRADING_PAIRS:
            close_all_for_symbol(symbol, MAGIC_BASE + list(TRADING_PAIRS.keys()).index(symbol))
        return True
    return False

def main():
    if not mt5.initialize():
        print("❌ MT5 не подключен")
        return

    print("🚀 ЗАПУСК БОТА 'ЗОЛОТАЯ ПЯТЕРКА' v2 (FIXED FILLING)")
    account_info = mt5.account_info()
    initial_balance = account_info.balance
    print(f"💰 Баланс: {initial_balance}$ | Плечо: 1:{account_info.leverage}")

    for symbol in TRADING_PAIRS:
        last_bar_times[symbol] = 0
        cycle_levels[symbol] = 0
        mt5.symbol_select(symbol, True)

    try:
        while True:
            if check_drawdown_limit(initial_balance):
                break

            for symbol, tf in TRADING_PAIRS.items():
                magic = MAGIC_BASE + list(TRADING_PAIRS.keys()).index(symbol)
                
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, 2)
                if rates is None or len(rates) < 2: continue
                current_bar_time = rates[-2]['time']

                if current_bar_time != last_bar_times.get(symbol, 0):
                    last_bar_times[symbol] = current_bar_time
                    
                    pos_count = get_open_positions_count(symbol, magic)
                    total_profit = calculate_total_profit(symbol, magic)
                    
                    if pos_count > 0:
                        if total_profit > 0:
                            print(f"💰 {symbol} Цель достигнута ({total_profit:.2f}$). Закрытие цикла.")
                            close_all_for_symbol(symbol, magic)
                            cycle_levels[symbol] = 0
                        elif pos_count >= MAX_LEVELS:
                            print(f"⚠️ {symbol} Лимит уровней ({MAX_LEVELS}). Принудительное закрытие.")
                            close_all_for_symbol(symbol, magic)
                            cycle_levels[symbol] = 0
                        else:
                            cycle_levels[symbol] += 1
                            open_new_position(symbol, tf, cycle_levels[symbol], magic)
                    else:
                        cycle_levels[symbol] = 0
                        open_new_position(symbol, tf, 0, magic)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен пользователем")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()