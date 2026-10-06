#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt5_crypto_bot.py — крипто-обёртка над mt5_live_bot.py.
Дефолты уже под крипту: PAIRS = CRYPTO_PAIRS, сессия 24/7, живые ордера ВКЛ.
Запуск (живой, демо):  & .venv\\Scripts\\python.exe mt5_crypto_bot.py
Только симуляция:       & .venv\\Scripts\\python.exe mt5_crypto_bot.py --dry
Один проход и выход:    & .venv\\Scripts\\python.exe mt5_crypto_bot.py --once
Логика/модель/фичи переиспользуются из mt5_live_bot.py (файл не дублируется).
"""
import sys
from pathlib import Path
import importlib.util

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("bot", HERE / "mt5_live_bot.py")
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)

# --- крипто-дефолты (перебиваем значения из mt5_live_bot.py) ---
bot.PAIRS = list(bot.CRYPTO_PAIRS)
bot.SESSION_START = (0, 0)
bot.SESSION_END = (23, 59)
bot.LIVE_TRADING = "--dry" not in sys.argv   # обычный запуск = реальные ордера; --dry = симуляция

bot.main()
