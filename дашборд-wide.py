#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
дашборд-wide.py — дашборд «широкого мозга».
Берёт оффлайн-дата-свит (data/*.csv), обучается на отрезке [0, NOPT),
затем подаёт бары [NOPT, ...) ПО ОДНОМУ (те, что модель не видела) и
строит портфельную эквити в процессе — имитация торговли на демо.
Эквити считается ТЕМ ЖЕ кодом, что wide_brain.py (run_fold).
"""
import importlib.util
import pathlib
import time
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

HERE = pathlib.Path(__file__).parent
WIDE = HERE / "wide_brain.py"
spec = importlib.util.spec_from_file_location("wide_brain", str(WIDE))
wide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wide)

st.set_page_config(page_title="Wide Brain Live Sim", layout="wide")
st.title("🧠 Широкий мозг — поток баров (имитация демо)")

# ======================== БОКОВАЯ ПАНЕЛЬ ========================
st.sidebar.header("⚙️ Параметры")
NOPT = st.sidebar.number_input("Баров обучения (NOPT)", 2000, 14000, 10000, 500)
NFWD = st.sidebar.number_input("Баров потока (NFWD)", 200, 5000, 3000, 100)
H = st.sidebar.number_input("Горизонт мета H", 3, 30, 10, 1)


@st.cache_data(show_spinner="Обучение широкого мозга…")
def compute_stream(NOPT, NFWD, H):
    pairs, feats, targets, returns, M = wide.build_data()
    if NOPT + NFWD > M:
        return None
    feats_n = {}
    for s in pairs:
        sc = StandardScaler().fit(feats[s][:NOPT])
        feats_n[s] = sc.transform(feats[s])
    pr_w, pr_b, scale_test, sig_test = wide.run_fold(
        pairs, feats_n, targets, returns, NOPT, NOPT + NFWD, H)
    return dict(pairs=pairs,
                eq_w=np.cumprod(1 + pr_w),
                eq_b=np.cumprod(1 + pr_b),
                scale=scale_test,
                sig=sig_test,
                NOPT=NOPT)


res = compute_stream(NOPT, NFWD, H)
if res is None:
    st.error("❌ Недостаточно данных: NOPT + NFWD > доступно в data/*.csv")
    st.stop()

pairs = res["pairs"]
eq_w = res["eq_w"]
eq_b = res["eq_b"]
scale = res["scale"]
sig = res["sig"]
T = len(eq_w)

# ======================== ПОШАГОВОЕ ПРОИГРЫВАНИЕ ========================
if "idx" not in st.session_state:
    st.session_state.idx = T - 1

auto = st.sidebar.checkbox("▶▶ Авто-проигрывание")
idx = st.sidebar.slider("Показать до бара", 0, T - 1, st.session_state.idx)
st.session_state.idx = idx
if auto and idx < T - 1:
    st.session_state.idx = idx + 1
    time.sleep(0.02)
    st.rerun()

i = idx
tot_w = eq_w[i] - 1.0
tot_b = eq_b[i] - 1.0
dd_w = float((eq_w[:i + 1] / np.maximum.accumulate(eq_w[:i + 1]) - 1).min())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Эквити Wide", f"{tot_w * 100:.2f}%")
c2.metric("Эквити Base", f"{tot_b * 100:.2f}%")
c3.metric("Просадка Wide", f"{dd_w * 100:.2f}%")
c4.metric("Мета scale", f"{scale[i]:.2f}")

# ======================== ГРАФИК ЭКВИТИ ========================
st.subheader("📈 Портфельная эквити (поток новых баров)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(i + 1)), y=eq_w[:i + 1],
                         name="Wide (мультиголовы + мета)", line=dict(color="green", width=2)))
fig.add_trace(go.Scatter(x=list(range(i + 1)), y=eq_b[:i + 1],
                         name="Base (без мета)", line=dict(color="gray", width=1.5, dash="dot")))
fig.update_layout(height=420, template="plotly_white",
                  xaxis_title="Бар (поток)", yaxis_title="Equity")
st.plotly_chart(fig, use_container_width=True)

# ======================== СИГНАЛЫ ПО ПАРАМ ========================
st.subheader("📡 Сигналы по парам на текущем баре")
rows = []
for j, s in enumerate(pairs):
    d = int(sig[i, j])
    rows.append({"Пара": s, "Сигнал": "BUY 🟢" if d == 1 else ("SELL 🔴" if d == -1 else "— нет —")})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.caption(f"Обучено на [0,{NOPT}). Дальше подаются бары [NOPT, {NOPT + T}), "
           f"которые модель НЕ видела. Эквити считается тем же кодом, что wide_brain.py "
           f"-> совпадает с оффлайн-запуском. Бар потока {i + 1}/{T}.")
