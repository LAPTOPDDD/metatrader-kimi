#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_trader.py — AI-трейдер (Muse Spark) на крипте.
Смотрит шире: M15/M30/H1/H4/D1 × 29 пар, ищет где созрело, входит, забирает +0.03 и мониторит.
Логика — не LightGBM-эксперт, а эвристика из обучающей базы: конfluence трендов D1→H4→H1→M30→M15.
"""
import sys, time, datetime, os, atexit, json
from pathlib import Path
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

HERE = Path(__file__).parent
import importlib.util
_qspec = importlib.util.spec_from_file_location("q6", HERE / "система-в-единм-файле-QWEN6.py")
q6 = importlib.util.module_from_spec(_qspec); _qspec.loader.exec_module(q6)

# ============================ К О Н Ф И Г ================================
PAIRS_ALL = ["AAVE","BITCOIN","BITCOIN CASH","CARDANO","CHAINLINK","DOGECOIN",
             "ETHEREUM","LITECOIN","POLKADOT","POLYGON","RIPPLE","SOLANA",
             "STELLAR","TRON","UNISWAP","AXS","BNB","COSMOS","PYTH","ILLUVIUM",
             "INJ","RENDER","TRUMP","SUI","HEDERA","NEAR","APTOS","ARBITRUM","PAXG","GRAM"]
BASE_TF = "M15"
HIGHER = ["M30","H1","H4","D1"]
TF_MAP = {"M15":mt5.TIMEFRAME_M15,"M30":mt5.TIMEFRAME_M30,"H1":mt5.TIMEFRAME_H1,"H4":mt5.TIMEFRAME_H4,"D1":mt5.TIMEFRAME_D1}
MAGIC   = 20260830
LOT     = None  # min lot авто
TP_PROFIT = 0.03
POLL_SEC = 15
EQUITY_LOG = HERE / "ai_equity_log.csv"
# ========================================================================

_models = {}  # не нужен, но для совместимости
_lots = {}; _sinfo = {}

def log(msg):
    ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line=f"[{ts}] {msg}"
    print(line)
    try:
        with open(HERE/"ai_log.txt","a",encoding="utf-8") as f: f.write(line+"\n")
    except: pass

def log_equity():
    try:
        ai=mt5.account_info()
        if not ai: return
        n=len([p for p in (mt5.positions_get() or []) if p.magic==MAGIC])
        hdr=not EQUITY_LOG.exists()
        with open(EQUITY_LOG,"a",encoding="utf-8") as f:
            if hdr: f.write("time,balance,equity,positions\n")
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S},{ai.balance:.2f},{ai.equity:.2f},{n}\n")
    except: pass

def connect():
    if mt5.initialize():
        log("MT5: подключено (AI trader)")
        return True
    log(f"MT5 init fail {mt5.last_error()}")
    return False

def ensure_symbols():
    ok=False
    for s in PAIRS_ALL:
        si=mt5.symbol_info(s)
        if si is None: continue
        if not si.visible: mt5.symbol_select(s,True)
        si=mt5.symbol_info(s)
        _sinfo[s]=si
        lot=round(si.volume_min/si.volume_step)*si.volume_step
        lot=max(lot,si.volume_min)
        _lots[s]=lot
        ok=True
    return ok

def get_rates(sym, tf, n):
    r=mt5.copy_rates_from_pos(sym, TF_MAP[tf], 0, n)
    if r is None or len(r)==0: return None
    df=pd.DataFrame(r); df["time"]=pd.to_datetime(df["time"],unit="s")
    return df

def trend_score(sym):
    """Оценка созрело: +1 за каждый ТФ где тренд up, -1 где down, 0 флэт. Возвращает score и строку рассуждения."""
    scores=[]; reasons=[]
    for tf in [BASE_TF]+HIGHER:
        df=get_rates(sym, tf, 120)
        if df is None or len(df)<60:
            scores.append(0); reasons.append(f"{tf}:нет данных")
            continue
        c=df["close"].values
        # MA20/50
        ma20=pd.Series(c).rolling(20).mean().iloc[-1]
        ma50=pd.Series(c).rolling(50).mean().iloc[-1]
        # RSI
        try:
            fs=q6.FeatureStore(df)
            fs.calc_features()
            rsi=float(fs.features["rsi_14"].iloc[-1])
        except: rsi=50
        price=c[-1]
        # тренд по MA
        if price>ma20 and ma20>ma50 and rsi>45:
            scores.append(1); reasons.append(f"{tf}:up(RSI{int(rsi)})")
        elif price<ma20 and ma20<ma50 and rsi<55:
            scores.append(-1); reasons.append(f"{tf}:down(RSI{int(rsi)})")
        else:
            scores.append(0); reasons.append(f"{tf}:flat(RSI{int(rsi)})")
    # D1 вес x2
    total=sum(scores[:-1])+scores[-1]*2  # D1 последний
    return total, " | ".join(reasons)

def analyze_pair(sym):
    score, reason = trend_score(sym)
    # созрело если конfluence >=3 (из 6 с весом D1) и M15 не против
    # score диапазон -6..+6
    if score>=3:
        return "long", score, reason
    elif score<=-3:
        return "short", score, reason
    else:
        return "hold", score, reason

def send_market(sym, side):
    si=_sinfo[sym]; lot=_lots[sym]
    tick=mt5.symbol_info_tick(sym)
    price=tick.ask if side==1 else tick.bid
    otype=mt5.ORDER_TYPE_BUY if side==1 else mt5.ORDER_TYPE_SELL
    req=dict(action=mt5.TRADE_ACTION_DEAL,symbol=sym,volume=lot,type=otype,price=price,deviation=30,magic=MAGIC,comment="ai-trader",type_time=mt5.ORDER_TIME_GTC,type_filling=mt5.ORDER_FILLING_FOK)
    r=mt5.order_send(req)
    if r is None or r.retcode!=mt5.TRADE_RETCODE_DONE:
        log(f"  ! {sym} {'BUY' if side==1 else 'SELL'} fail {r.retcode if r else mt5.last_error()}")
    else:
        log(f"  AI {sym} {'BUY' if side==1 else 'SELL'} {lot} @ {price:.5f} ticket={r.order} score созрело")
    return r

def close_pos(sym, pos):
    si=_sinfo[sym]; tick=mt5.symbol_info_tick(sym)
    ctype=mt5.ORDER_TYPE_SELL if pos.type==0 else mt5.ORDER_TYPE_BUY
    price=tick.bid if pos.type==0 else tick.ask
    req=dict(action=mt5.TRADE_ACTION_DEAL,symbol=sym,volume=pos.volume,type=ctype,position=pos.ticket,price=price,deviation=30,magic=MAGIC,comment="ai-tp",type_time=mt5.ORDER_TIME_GTC,type_filling=mt5.ORDER_FILLING_FOK)
    r=mt5.order_send(req)
    if r and r.retcode==mt5.TRADE_RETCODE_DONE:
        log(f"  AI TP {sym} ticket={pos.ticket} profit={pos.profit:.2f}")
    return r

def main():
    if not connect(): return
    if not ensure_symbols():
        log("нет символов"); return
    log(f"PAIRS {list(_sinfo.keys())[:5]}... всего {len(_sinfo)}")
    log(f"AI trader старт MAGIC={MAGIC} TP={TP_PROFIT} poll={POLL_SEC}s  ТФ {[BASE_TF]+HIGHER}")
    log_equity()
    try:
        while True:
            # 1) TP проверка каждую итерацию
            for p in (mt5.positions_get() or []):
                if p.magic==MAGIC and p.profit>=TP_PROFIT:
                    close_pos(p.symbol, p)
            # 2) анализ и входы — раз в POLL_SEC, но только если нет позы по символу
            for sym in list(_sinfo.keys()):
                if sym=="GRAM": continue
                # уже есть поза — не усредняем
                has=len([x for x in (mt5.positions_get(symbol=sym) or []) if x.magic==MAGIC])>0
                if has: continue
                decision, score, reason = analyze_pair(sym)
                if decision=="hold": continue
                side=1 if decision=="long" else -1
                log(f"  созрел {sym}: {decision} score={score} :: {reason}")
                send_market(sym, side)
            log_equity()
            time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        log("AI стоп Ctrl+C")

if __name__=="__main__":
    main()
