#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_test_v2.py — быстрый тест исправленного фреймворка (без MT5).
Проверяет: сохранение JSON, топ-5, фильтрацию по датам, защиту от деления на ноль.
"""
import os
import sys
import subprocess

# Убеждаемся, что модуль запускается из своей папки
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Мокаем MT5, чтобы принудительно использовать синтетику
sys.modules["MetaTrader5"] = None

# Патчим input(), чтобы автоматически выбрать режим 1 (быстрый тест)
import builtins
builtins.input = lambda prompt="": "1"

# Импортируем и запускаем main()
import backtest_framework_fixed as btf

if __name__ == "__main__":
    btf.main()
