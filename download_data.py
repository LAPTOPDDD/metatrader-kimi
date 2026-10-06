#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
download_data.py — оффлайн-загрузка исторических котировок из MetaTrader5.

Скачивает бары для заданного списка пар и таймфреймов и сохраняет в
data/<SYMBOL>_<TF>.csv плюс data/manifest.json с метаданными. Дальше система/
дашборд могут грузить их оффлайн, без живого терминала. Скрипт идемпотентен:
при повторном запуске просто перезаписывает файлы свежими данными.

Запуск (в .venv, при запущенном и авторизованном терминале MT5):
    & .venv\Scripts\python.exe download_data.py
"""
import sys
import json
import time
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

# Что качаем: 10 пар x 4 таймфрейма (легко расширяется — допишите в списки)
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "AUDUSD",
         "USDCHF", "NZDUSD", "USDCAD", "EURJPY", "AUDJPY"]
TIMEFRAMES = ["M15", "M30", "H1", "H4"]
BARS = 15000  # баров на каждую пару/таймфрейм (последние N)

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def main():
    if not mt5.initialize():
        print("ERROR: не удалось инициализировать MT5. Запущен ли терминал и авторизован ли счёт?")
        return 1
    try:
        manifest = []
        total = len(PAIRS) * len(TIMEFRAMES)
        done = 0
        t0 = time.time()
        for sym in PAIRS:
            # символ может быть не в обзоре — пытаемся включить
            if mt5.symbol_info(sym) is None:
                mt5.symbol_select(sym, True)
            if mt5.symbol_info(sym) is None:
                print(f"  ! {sym}: символ недоступен, пропускаем")
                continue
            for tf in TIMEFRAMES:
                rates = mt5.copy_rates_from_pos(sym, TF_MAP[tf], 0, BARS)
                if rates is None or len(rates) == 0:
                    print(f"  ! {sym} {tf}: нет данных")
                    continue
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                out = DATA_DIR / f"{sym}_{tf}.csv"
                df.to_csv(out, index=False)
                manifest.append({
                    "symbol": sym, "timeframe": tf, "bars": int(len(df)),
                    "start": df["time"].iloc[0].isoformat(),
                    "end": df["time"].iloc[-1].isoformat(),
                    "file": out.name,
                })
                done += 1
                print(f"  + {sym} {tf}: {len(df)} баров  "
                      f"[{df['time'].iloc[0].date()} .. {df['time'].iloc[-1].date()}]  ({done}/{total})")
        (DATA_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nГотово: {done} файлов за {time.time() - t0:.1f}s. Каталог: {DATA_DIR}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
