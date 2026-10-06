#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ДАШБОРД v3.0: СРАВНЕНИЕ ПОДХОДОВ
=================================
A. Бинарная классификация (sign производной)
B. Мягкие метки (нормализованная производная)
C. Forward return (будущее изменение цены)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

NOPT = 1000
NFWD = 150
SYMBOL = "EURUSD"
TIMEFRAME = "H1"
HORIZON = 5  # Для forward return: предсказываем на 5 баров вперёд

def tf_to_mt5(tf: str) -> int:
    return {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    }[tf]

class FeatureStore:
    def __init__(self, df: pd.DataFrame):
        self.raw = df.reset_index(drop=True)
        self.close = self.raw["close"].values
        self.high = self.raw["high"].values
        self.low = self.raw["low"].values
        self.open = self.raw["open"].values
        self.volume = self.raw["tick_volume"].values
        self.time = self.raw["time"].values
        self.features = None
        
    def calc_features(self):
        d = self.raw.copy()
        d["rsi_14"] = self._rsi(d["close"])
        d["macd"] = d["close"].ewm(span=12).mean() - d["close"].ewm(span=26).mean()
        for k in [6, 12, 24]:
            d[f"mom_{k}"] = d["close"].pct_change(k)
        d["atr_14"] = self._atr(d["high"], d["low"], d["close"], 14)
        for ma in [20, 50]:
            d[f"ma_{ma}"] = d["close"] / d["close"].rolling(ma).mean()
        d["body"] = (d["close"] - d["open"]) / (d["high"] - d["low"] + 1e-9)
        d["hl_range"] = (d["high"] - d["low"]) / d["close"]
        d["vol_ratio"] = d["tick_volume"] / d["tick_volume"].rolling(20).mean()
        self.features = d.dropna().reset_index(drop=True)
        return self.features
    
    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

EXCLUDE = {"open","high","low","close","tick_volume","spread","real_volume","time"}

def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE]

def main():
    st.set_page_config(page_title="GAUSSIAN ENSEMBLE v3.0", layout="wide")
    
    st.title("🔬 Сравнение подходов к предсказанию")
    st.markdown(f"**{SYMBOL} {TIMEFRAME}** | Обучение: {NOPT} | Тест: {NFWD}")
    
    @st.cache_data(ttl=300)
    def load_data():
        if not mt5.initialize():
            return None
        rates = mt5.copy_rates_from_pos(SYMBOL, tf_to_mt5(TIMEFRAME), 0, NOPT + NFWD + 500)
        return pd.DataFrame(rates) if rates is not None else None
    
    df = load_data()
    if df is None:
        st.error("❌ Ошибка загрузки данных")
        st.stop()
    
    fs = FeatureStore(df)
    fs.calc_features()
    feature_cols = get_feature_cols(fs.features)
    
    # ======================== ПАРАМЕТРЫ ========================
    st.sidebar.header("⚙️ Параметры")
    sigma = st.sidebar.slider("sigma (Гаусс)", 0.5, 3.0, 1.5, 0.1)
    n_estimators = st.sidebar.slider("n_estimators", 50, 300, 100, 50)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)
    num_leaves = st.sidebar.slider("num_leaves", 5, 50, 15, 5)
    
    # ======================== ПОДГОТОВКА ДАННЫХ ========================
    # A. Бинарная классификация
    smoothed = gaussian_filter1d(fs.close, sigma, mode='reflect')
    deriv = np.gradient(smoothed)
    margin = int(3 * sigma)
    if margin * 2 >= len(deriv):
        margin = int(len(deriv) * 0.1)
    
    start = margin
    end = len(deriv) - margin
    
    df_slice = fs.features.iloc[start:end]
    X_full = df_slice[feature_cols].values
    closes_full = df_slice["close"].values
    
    # Target A: бинарный знак производной
    y_binary = (deriv[start:end] > 0).astype(int)
    
    # Target B: нормализованная производная (мягкие метки)
    deriv_trim = deriv[start:end]
    y_soft = (deriv_trim - deriv_trim.min()) / (deriv_trim.max() - deriv_trim.min() + 1e-9)
    
    # Target C: forward return (изменение цены через HORIZON баров)
    # Сдвигаем цену вперёд на HORIZON
    forward_returns = np.zeros(len(closes_full))
    for i in range(len(closes_full) - HORIZON):
        forward_returns[i] = (closes_full[i + HORIZON] - closes_full[i]) / closes_full[i]
    y_forward = forward_returns
    
    # Разделение на train/test
    X_train = X_full[:NOPT]
    X_test = X_full[NOPT:NOPT+NFWD]
    closes_test = closes_full[NOPT:NOPT+NFWD]
    
    y_train_A = y_binary[:NOPT]
    y_test_A = y_binary[NOPT:NOPT+NFWD]
    
    y_train_B = y_soft[:NOPT]
    y_test_B = y_soft[NOPT:NOPT+NFWD]
    
    y_train_C = y_forward[:NOPT]
    y_test_C = y_forward[NOPT:NOPT+NFWD]
    
    # Нормализация признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ======================== ОБУЧЕНИЕ ТРЁХ МОДЕЛЕЙ ========================
    lgb_params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': 1,
    }
    
    # Модель A: Classifier (бинарная)
    model_A = lgb.LGBMClassifier(**lgb_params)
    model_A.fit(X_train_scaled, y_train_A)
    prob_A = model_A.predict_proba(X_test_scaled)[:, 1]
    pred_A = (prob_A > 0.5).astype(int)
    
    # Модель B: Classifier (мягкие метки)
    model_B = lgb.LGBMClassifier(**lgb_params)
    model_B.fit(X_train_scaled, y_train_B)
    prob_B = model_B.predict_proba(X_test_scaled)[:, 1]
    pred_B = (prob_B > 0.5).astype(int)
    
    # Модель C: Regressor (forward return)
    model_C = lgb.LGBMRegressor(**lgb_params)
    model_C.fit(X_train_scaled, y_train_C)
    pred_C = model_C.predict(X_test_scaled)
    
    # ======================== МЕТРИКИ ========================
    def calc_sign_match(y_true, y_pred):
        """% совпадения знаков"""
        return np.mean(np.sign(y_true - 0.5) == np.sign(y_pred - 0.5)) * 100
    
    def calc_equity(signals, closes):
        """Симуляция эквити по сигналам"""
        equity = [0.0]
        for i in range(1, len(signals)):
            if signals[i] != signals[i-1]:  # Смена сигнала
                if signals[i] == 1:
                    pnl = (closes[i] - closes[i-1]) / closes[i-1] * 100
                else:
                    pnl = (closes[i-1] - closes[i]) / closes[i-1] * 100
                equity.append(equity[-1] + pnl)
            else:
                equity.append(equity[-1])
        return np.array(equity)
    
    acc_A = np.mean(pred_A == y_test_A) * 100
    acc_B = np.mean(pred_B == y_test_A) * 100  # Сравниваем с бинарным target
    sign_match_C = np.mean(np.sign(pred_C) == np.sign(y_test_C)) * 100
    
    equity_A = calc_equity(pred_A, closes_test)
    equity_B = calc_equity(pred_B, closes_test)
    equity_C = calc_equity((pred_C > 0).astype(int), closes_test)
    
    # ======================== ВИЗУАЛИЗАЦИЯ ========================
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f"A. Бинарная классификация (Accuracy: {acc_A:.1f}%)",
            f"B. Мягкие метки (Accuracy: {acc_B:.1f}%)",
            f"C. Forward Return H={HORIZON} (Sign match: {sign_match_C:.1f}%)",
        ]
    )
    
    # A. Бинарная
    fig.add_trace(go.Scatter(
        x=list(range(NFWD)), y=y_test_A * 0.8,
        mode='markers', name='Target (0/1)',
        marker=dict(color='blue', size=6, opacity=0.5),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(NFWD)), y=prob_A,
        mode='lines', name='P(BUY)',
        line=dict(color='red', width=2)
    ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
    
    # B. Мягкие метки
    fig.add_trace(go.Scatter(
        x=list(range(NFWD)), y=y_test_B,
        mode='lines', name='Soft target',
        line=dict(color='blue', width=1.5, opacity=0.6),
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(NFWD)), y=prob_B,
        mode='lines', name='P(BUY)',
        line=dict(color='red', width=2)
    ), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)
    
    # C. Forward return
    fig.add_trace(go.Scatter(
        x=list(range(NFWD)), y=y_test_C * 10000,  # В пипсах
        mode='lines', name='Real return',
        line=dict(color='blue', width=1.5, opacity=0.6),
        showlegend=False
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(NFWD)), y=pred_C * 10000,
        mode='lines', name='Predicted',
        line=dict(color='red', width=2)
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        height=900,
        template="plotly_white",
        hovermode='x unified',
        title="Сравнение трёх подходов"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ======================== ЭКВИТИ ========================
    st.subheader("💰 Эквити по трём подходам")
    
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=list(range(NFWD)), y=equity_A,
        mode='lines', name=f'A. Binary (PnL: {equity_A[-1]:.2f}%)',
        line=dict(color='blue', width=2)
    ))
    fig_eq.add_trace(go.Scatter(
        x=list(range(NFWD)), y=equity_B,
        mode='lines', name=f'B. Soft (PnL: {equity_B[-1]:.2f}%)',
        line=dict(color='green', width=2)
    ))
    fig_eq.add_trace(go.Scatter(
        x=list(range(NFWD)), y=equity_C,
        mode='lines', name=f'C. Forward H={HORIZON} (PnL: {equity_C[-1]:.2f}%)',
        line=dict(color='red', width=2)
    ))
    fig_eq.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_eq.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title="Бар",
        yaxis_title="Equity, %"
    )
    st.plotly_chart(fig_eq, use_container_width=True)
    
    # ======================== ВЫВОД ========================
    st.markdown("""
    ---
    ### 🎯 Какой подход выбрать?
    
    | Подход | Когда использовать |
    |--------|-------------------|
    | **A. Бинарная** | Простая, стабильная, работает всегда |
    | **B. Мягкие метки** | Когда нужна "уверенность" сигнала |
    | **C. Forward Return** | Прямая оптимизация под прибыль |
    
    ### 💡 Рекомендация:
    - **Для старта**: используй **A** (бинарную) — она уже показала +30%
    - **Для улучшения**: попробуй **C** с разными горизонтами (H=3, 5, 10)
    - **B** — компромисс, но редко даёт преимущество
    """)

if __name__ == "__main__":
    main()