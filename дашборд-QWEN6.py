#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ДАШБОРД СИСТЕМЫ QWEN6 — потоковая симуляция (бар за баром)
==========================================================
Переиспользует логику система-в-единм-файле-QWEN6.py:
  FeatureStore, calc_features, build_target, train_model,
  Params, backtest_advanced (те же правила входа/выхода, что в LiveTrader).

Поведение: обучаем модель на NOPT барах, затем подаём NFWD новых баров
по одному (слайдер / авто-проигрывание) и видим, как система открывает/
закрывает сделки и растёт эквити.
"""

import sys
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import MetaTrader5 as mt5

HERE = Path(__file__).parent
Q6_PATH = HERE / "система-в-единм-файле-QWEN6.py"

st.set_page_config(page_title="QWEN6 Live Sim", layout="wide")
st.title("📡 Симуляция системы QWEN6 (поток баров)")

@st.cache_resource
def load_q6():
    spec = importlib.util.spec_from_file_location("q6", str(Q6_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

q6 = load_q6()

# ======================== КОНФИГУРАЦИИ (ПРЕСЕТЫ) ========================
# Пресет выставляет ВСЕ параметры системы. Выбирайте из выпадающего списка,
# чтобы мгновенно увидеть эквити под эти настройки. Можно докрутить вручную.
ALL_KEYS = ["sigma", "use_dynamic_tp_sl", "tp_pips", "sl_pips", "use_trailing",
            "trailing_pips", "use_trend_filter", "ma_period", "min_hold_bars",
            "max_hold_bars", "use_atr_filter", "atr_multiplier",
            "dynamic_tp_atr", "dynamic_sl_atr"]

DEFAULTS = dict(sigma=1.5, use_dynamic_tp_sl=False, tp_pips=30, sl_pips=20,
                use_trailing=True, trailing_pips=30, use_trend_filter=True, ma_period=50,
                min_hold_bars=3, max_hold_bars=50, use_atr_filter=False,
                atr_multiplier=0.7, dynamic_tp_atr=2.0, dynamic_sl_atr=1.0)

CONFIGS = {
    # Агрессивный: больше сигналов, нет фильтров, трейлинг вкл, быстрые сделки
    "Агрессивный": dict(sigma=2.0, use_dynamic_tp_sl=False, tp_pips=30, sl_pips=20,
                        use_trailing=True, trailing_pips=15, use_trend_filter=False, ma_period=50,
                        min_hold_bars=1, max_hold_bars=20, use_atr_filter=False, atr_multiplier=1.5,
                        dynamic_tp_atr=2.0, dynamic_sl_atr=1.0),
    # Консервативный: только уверенные сигналы, фильтры вкл, крупные TP/SL
    "Консервативный": dict(sigma=3.0, use_dynamic_tp_sl=False, tp_pips=60, sl_pips=40,
                           use_trailing=False, trailing_pips=30, use_trend_filter=True, ma_period=50,
                           min_hold_bars=3, max_hold_bars=40, use_atr_filter=True, atr_multiplier=2.5,
                           dynamic_tp_atr=2.0, dynamic_sl_atr=1.0),
    # Много сделок: минимум фильтров, тесные TP/SL, быстрое закрытие
    "Много сделок": dict(sigma=2.0, use_dynamic_tp_sl=False, tp_pips=15, sl_pips=10,
                         use_trailing=False, trailing_pips=15, use_trend_filter=False, ma_period=50,
                         min_hold_bars=1, max_hold_bars=5, use_atr_filter=False, atr_multiplier=0.7,
                         dynamic_tp_atr=2.0, dynamic_sl_atr=1.0),
    # Мало сделок: жёсткие фильтры, динамический TP/SL, долгие удержания
    "Мало сделок": dict(sigma=3.5, use_dynamic_tp_sl=True, tp_pips=100, sl_pips=60,
                        use_trailing=True, trailing_pips=30, use_trend_filter=True, ma_period=100,
                        min_hold_bars=10, max_hold_bars=80, use_atr_filter=True, atr_multiplier=3.0,
                        dynamic_tp_atr=2.0, dynamic_sl_atr=1.0),
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

def apply_src(src):
    for k in ALL_KEYS:
        if k in src:
            st.session_state[k] = src[k]

sel = None

# ======================== БОКОВАЯ ПАНЕЛЬ ========================
st.sidebar.header("⚙️ Данные")
SYMBOL = st.sidebar.text_input("Символ", "EURUSD")
TIMEFRAME = st.sidebar.selectbox("Таймфрейм", ["M15", "M30", "H1", "H4"], index=2)
NOPT = st.sidebar.number_input("Баров обучения (NOPT)", 200, 20000, 1000, 100)
NFWD = st.sidebar.number_input("Баров потока (NFWD)", 50, 5000, 500, 50)

# Загрузка чемпиона из JSON (результат честной оптимизации QWEN6)
champions = []
cpath = HERE / "champions_ultimate.json"
if cpath.exists():
    try:
        champions = json.loads(cpath.read_text(encoding="utf-8"))
    except Exception:
        champions = []

def on_config_change():
    ch = st.session_state.get("config_sel")
    if ch in CONFIGS:
        apply_src(CONFIGS[ch])
    elif ch == "🏆 Чемпион":
        apply_src(sel if sel else (champions[0] if champions else {}))

config_options = ["— свои параметры —", "Агрессивный", "Консервативный", "Много сделок", "Мало сделок"]
if champions:
    opts = [f"{c['symbol']} {c['timeframe']}" for c in champions]
    def on_champ_change():
        ch = st.session_state.get("champ_choice")
        if ch and ch != "— нет —":
            c = champions[opts.index(ch)]
            st.session_state["config_sel"] = "🏆 Чемпион"
            apply_src(c)
    st.sidebar.selectbox("Загрузить чемпиона", ["— нет —"] + opts, key="champ_choice", on_change=on_champ_change)
    if st.session_state.get("champ_choice", "— нет —") != "— нет —":
        sel = champions[opts.index(st.session_state["champ_choice"])]
    config_options.append("🏆 Чемпион")

st.sidebar.selectbox("Конфигурация (пресет)", config_options, key="config_sel", on_change=on_config_change)
st.sidebar.caption("Пресет выставляет все параметры ниже. Можно докрутить вручную — пересчёт мгновенный.")

st.sidebar.header("🎯 Параметры входа/выхода")
sigma = st.sidebar.slider("sigma (Гаусс)", 0.5, 3.0, step=0.1, key="sigma")
use_dynamic = st.sidebar.checkbox("Динамический TP/SL (ATR)", key="use_dynamic_tp_sl")
tp_pips = float(st.sidebar.number_input("TP (пипсы)", 1, 500, key="tp_pips"))
sl_pips = float(st.sidebar.number_input("SL (пипсы)", 1, 500, key="sl_pips"))
use_trailing = st.sidebar.checkbox("Трейлинг-стоп", key="use_trailing")
trailing_pips = float(st.sidebar.number_input("Трейлинг (пипсы)", 5, 200, key="trailing_pips"))
use_trend_filter = st.sidebar.checkbox("Тренд-фильтр (MA)", key="use_trend_filter")
ma_period = int(st.sidebar.number_input("MA период", 10, 200, key="ma_period"))
min_hold_bars = int(st.sidebar.number_input("Мин. баров", 1, 50, key="min_hold_bars"))
max_hold_bars = int(st.sidebar.number_input("Макс. баров", 5, 300, key="max_hold_bars"))
use_atr_filter = st.sidebar.checkbox("ATR-фильтр", key="use_atr_filter")
atr_multiplier = float(st.sidebar.number_input("ATR множитель", 0.3, 2.0, step=0.1, key="atr_multiplier"))
dynamic_tp_atr = float(st.sidebar.number_input("TP (x ATR)", 0.5, 5.0, step=0.5, key="dynamic_tp_atr"))
dynamic_sl_atr = float(st.sidebar.number_input("SL (x ATR)", 0.5, 5.0, step=0.5, key="dynamic_sl_atr"))

params = dict(
    sigma=sigma,
    tp_pips=tp_pips,
    sl_pips=sl_pips,
    use_trailing=use_trailing,
    trailing_pips=trailing_pips,
    use_trend_filter=use_trend_filter,
    ma_period=ma_period,
    min_hold_bars=min_hold_bars,
    max_hold_bars=max_hold_bars,
    use_atr_filter=use_atr_filter,
    atr_multiplier=atr_multiplier,
    use_dynamic_tp_sl=use_dynamic,
    dynamic_tp_atr=dynamic_tp_atr,
    dynamic_sl_atr=dynamic_sl_atr,
)

# ======================== ЗАГРУЗКА И ОБУЧЕНИЕ ========================
if not mt5.initialize():
    st.error("❌ Ошибка инициализации MT5")
    st.stop()

bars_needed = NOPT + NFWD + 500
rates = mt5.copy_rates_from_pos(SYMBOL, q6.tf_to_mt5(TIMEFRAME), 0, bars_needed)
if rates is None or len(rates) < NOPT + NFWD:
    st.error(f"❌ Недостаточно данных: {len(rates) if rates is not None else 0}")
    st.stop()

df = pd.DataFrame(rates)
fs = q6.FeatureStore(df)
fs.calc_features()
feature_cols = q6.get_feature_cols(fs.features)

X, y, closes, highs, lows, times = q6.build_target(fs, sigma, feature_cols)
if X is None or len(y) < NOPT + NFWD:
    st.error("❌ Ошибка построения целевой переменной")
    st.stop()

X_train, y_train = X[:NOPT], y[:NOPT]
f_start, f_end = NOPT, NOPT + NFWD
X_test = X[f_start:f_end]
y_test = y[f_start:f_end]
closes_test = closes[f_start:f_end]
highs_test = highs[f_start:f_end]
lows_test = lows[f_start:f_end]
times_test = times[f_start:f_end]

info = mt5.symbol_info(SYMBOL)
point = info.point if info else 0.0001
pip_size = point * 10
spread = (info.spread if info else 0) * point

atr_full = (pd.Series(highs).rolling(14).mean() - pd.Series(lows).rolling(14).mean()).values
atr_test = atr_full[f_start:f_end]

p = q6.Params(**params)
model = q6.train_model(X_train, y_train)
eq, trades = q6.backtest_advanced(X_test, y_test, closes_test, highs_test, lows_test, times_test,
                                  model, p, point, spread, pip_size, atr_test)
mt5.shutdown()

if eq is None or len(eq) < 2:
    st.error("❌ Пустой результат симуляции")
    st.stop()

# ======================== МЕТРИКИ ========================
wins = sum(1 for t in trades if t["pnl"] > 0)
win_rate = wins / len(trades) if trades else 0
total_pnl = eq[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Сделок", len(trades))
col2.metric("Win Rate", f"{win_rate*100:.1f}%")
col3.metric("Итоговый PnL", f"{total_pnl:.2f}%")
col4.metric("Баров в потоке", len(eq))

# ======================== ПОШАГОВОЕ ПРОИГРЫВАНИЕ ========================
if "idx" not in st.session_state:
    st.session_state.idx = len(eq) - 1

auto = st.sidebar.checkbox("▶▶ Авто-проигрывание")
idx = st.sidebar.slider("Показать до бара", 0, len(eq) - 1, st.session_state.idx)
st.session_state.idx = idx

if auto and idx < len(eq) - 1:
    st.session_state.idx = idx + 1
    time.sleep(0.04)
    st.rerun()

# Текущее состояние на баре idx
t_now = times_test[idx]
open_trade = next((t for t in trades if t["entry_time"] <= t_now < t["exit_time"]), None)

state_cols = st.columns(3)
state_cols[0].metric("Текущий бар", idx)
if open_trade:
    direction = "BUY 🟢" if open_trade["direction"] == 1 else "SELL 🔴"
    state_cols[1].metric("Позиция", direction)
    state_cols[2].metric("Цена входа", f"{open_trade['entry_price']:.5f}")
else:
    state_cols[1].metric("Позиция", "— нет —")
    state_cols[2].metric("Цена", f"{closes_test[idx]:.5f}")

# ======================== ГРАФИКИ ========================
st.subheader("📈 Цена и сделки (поток новых баров)")
entry_idx = [int(np.argmin(np.abs(times_test - t["entry_time"]))) for t in trades]
exit_idx = [int(np.argmin(np.abs(times_test - t["exit_time"]))) for t in trades]

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(idx + 1)), y=closes_test[:idx + 1],
                         mode='lines', name='Close', line=dict(color='black', width=1)))
for i, t in enumerate(trades):
    ei, xi = entry_idx[i], exit_idx[i]
    if ei > idx:
        continue
    color = 'green' if t["direction"] == 1 else 'red'
    fig.add_trace(go.Scatter(x=[ei], y=[t["entry_price"]], mode='markers',
                             marker=dict(color=color, size=10, symbol='triangle-up' if t["direction"] == 1 else 'triangle-down'),
                             name='Вход' if i == 0 else None, showlegend=(i == 0)))
    if xi <= idx:
        fig.add_trace(go.Scatter(x=[xi], y=[t["exit_price"]], mode='markers',
                                 marker=dict(color='blue', size=9, symbol='x'),
                                 name='Выход' if i == 0 else None, showlegend=(i == 0)))
fig.update_layout(height=400, template="plotly_white", xaxis_title="Бар (поток)", yaxis_title="Цена")

st.subheader("📊 Эквити")
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=list(range(idx + 1)), y=eq[:idx + 1],
                            mode='lines', name='Equity', line=dict(color='green', width=2)))
fig_eq.update_layout(height=300, template="plotly_white", xaxis_title="Бар", yaxis_title="Equity, %")

st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig_eq, use_container_width=True)

# ======================== ТАБЛИЦА СДЕЛОК ========================
if trades:
    st.subheader("📋 Сделки (закрытые до текущего бара)")
    rows = []
    for i, t in enumerate(trades):
        if exit_idx[i] > idx:
            continue
        rows.append({
            "№": i + 1,
            "Направление": "BUY" if t["direction"] == 1 else "SELL",
            "Вход": f"{t['entry_price']:.5f}",
            "Выход": f"{t['exit_price']:.5f}",
            "Причина": t.get("exit_reason", ""),
            "PnL, %": f"{t['pnl']:.2f}",
            "Баров": t["hold_bars"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
