#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ИССЛЕДОВАТЕЛЬСКИЙ ДАШБОРД GAUSSIAN ENSEMBLE
=============================================
• N=1000 баров обучения (скрыто)
• K=150 тестовых баров на дашборде
• 3 графика: производная, бинарная, аппроксиматор
• Слайдеры параметров LightGBM
• Мгновенное переобучение при изменении
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
import warnings
import time

warnings.filterwarnings('ignore')

# ======================== КОНФИГ =========================
NOPT = 10000  # Баров для обучения
NFWD = 150   # Тестовых баров на дашборде
SYMBOL = "EURUSD"
TIMEFRAME = "H1"

# ======================== MT5 UTILS ======================
def tf_to_mt5(tf: str) -> int:
    return {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    }[tf]

# ======================== FEATURE STORE ====================
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
        if len(self.features) == 0:
            raise ValueError("Нет данных после вычисления признаков")
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

# ======================== ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ====================
EXCLUDE = {"open","high","low","close","tick_volume","spread","real_volume","Target","time"}

def get_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in EXCLUDE]

def build_target(fs: FeatureStore, sigma: float, feature_cols: list):
    smoothed = gaussian_filter1d(fs.close, sigma, mode='reflect')
    deriv = np.gradient(smoothed)
    
    margin = int(3 * sigma)
    if margin * 2 >= len(deriv):
        margin = int(len(deriv) * 0.1)
        
    start = margin
    end = len(deriv) - margin if margin > 0 else len(deriv)
    
    if start >= end:
        return None, None, None, None
    
    df_slice = fs.features.iloc[start:end]
    deriv_trimmed = deriv[start:end]
    
    y = (deriv_trimmed > 0).astype(np.int8)
    X = df_slice[feature_cols].values
    closes = df_slice["close"].values
    times = df_slice["time"].values
    
    return X, y, closes, times

# ======================== МОДЕЛЬ ====================
def train_model(X: np.ndarray, y: np.ndarray, params: dict) -> lgb.LGBMClassifier:
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    
    if n_pos > 0 and n_neg > 0:
        sample_weight = np.where(y == 1, n_neg / n_pos, 1.0).astype(np.float32)
    else:
        sample_weight = np.ones(len(y), dtype=np.float32)
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, sample_weight=sample_weight)
    return model

# ======================== ДАШБОРД ====================
def main():
    st.set_page_config(page_title="GAUSSIAN ENSEMBLE Dashboard", layout="wide")
    
    st.title("🎛️ Исследовательский дашборд GAUSSIAN ENSEMBLE")
    st.markdown(f"**Пара:** {SYMBOL} | **Таймфрейм:** {TIMEFRAME} | **Обучение:** {NOPT} баров | **Тест:** {NFWD} баров")
    
    # Загрузка данных
    @st.cache_data(ttl=300)  # Кэш на 5 минут
    def load_data():
        if not mt5.initialize():
            st.error("❌ Ошибка инициализации MT5")
            return None
        
        bars_needed = NOPT + NFWD + 500
        rates = mt5.copy_rates_from_pos(SYMBOL, tf_to_mt5(TIMEFRAME), 0, bars_needed)
        
        if rates is None or len(rates) < NOPT + NFWD:
            st.error(f"❌ Недостаточно данных: {len(rates) if rates is not None else 0}")
            return None
        
        return pd.DataFrame(rates)
    
    with st.spinner("Загрузка данных из MT5..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    st.success(f"✅ Загружено {len(df)} баров")
    
    # Вычисление признаков
    fs = FeatureStore(df)
    fs.calc_features()
    feature_cols = get_feature_cols(fs.features)
    
    # ======================== БОКОВАЯ ПАНЕЛЬ ========================
    st.sidebar.header("⚙️ Параметры LightGBM")
    
    # Слайдеры
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 50, 10)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)
    num_leaves = st.sidebar.slider("num_leaves", 5, 50, 10, 5)
    min_child_samples = st.sidebar.slider("min_child_samples", 5, 50, 20, 5)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.1)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.1)
    
    st.sidebar.header(" Параметры Гаусса")
    sigma = st.sidebar.slider("sigma (сглаживание)", 0.5, 3.0, 1.5, 0.1)
    
    # Сбор параметров
    lgb_params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': 1,
    }
    
    # ======================== ОБУЧЕНИЕ И ПРОГНОЗ ========================
    with st.spinner("Обучение модели..."):
        # Разделение на train/test
        X, y, closes, times = build_target(fs, sigma, feature_cols)
        
        if X is None:
            st.error("❌ Ошибка построения целевой переменной")
            st.stop()
        
        # Разделение
        X_train = X[:NOPT]
        y_train = y[:NOPT]
        X_test = X[NOPT:NOPT+NFWD]
        y_test = y[NOPT:NOPT+NFWD]
        closes_test = closes[NOPT:NOPT+NFWD]
        times_test = times[NOPT:NOPT+NFWD]
        
        # Обучение
        model = train_model(X_train, y_train, lgb_params)
        
        # Прогноз на тесте
        probs = model.predict_proba(X_test)[:, 1]
        predictions = (probs > 0.5).astype(int)
    
    # ======================== МЕТРИКИ ========================
    accuracy = np.mean(predictions == y_test)
    win_rate = np.sum(predictions == y_test) / len(y_test)
    
    # PnL симуляция
    pnl = np.zeros(len(y_test))
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            pnl[i] = 0.001  # +10 пипсов
        else:
            pnl[i] = -0.001  # -10 пипсов
    
    equity = np.cumsum(pnl) * 100  # В процентах
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy*100:.1f}%")
    col2.metric("Win Rate", f"{win_rate*100:.1f}%")
    col3.metric("Total PnL", f"{equity[-1]:.2f}%")
    col4.metric("Sharpe", f"{equity.mean() / (equity.std() + 1e-9) * np.sqrt(252):.2f}")
    
    # ======================== ГРАФИКИ ========================
    st.subheader("📈 Тестовые данные (последние 150 баров)")
    
    # Создаем фигуру с 3 графиками на одной оси
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # 1. Первая производная от конволюции
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=np.gradient(gaussian_filter1d(closes_test, sigma))[3*int(sigma):-3*int(sigma)] if int(3*sigma) < len(closes_test) else np.zeros(len(y_test)),
            mode='lines',
            name='1-я производная',
            line=dict(color='blue', width=2),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 2. Бинарная форма (целевая переменная)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=y_test * 0.5,  # Масштабируем для наглядности
            mode='markers',
            name='Бинарная цель (0/1)',
            marker=dict(color='green', size=8, symbol='square'),
            opacity=0.5
        ),
        row=1, col=1
    )
    
    # 3. Выход аппроксиматора (вероятность)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=probs,
            mode='lines',
            name='Аппроксиматор (P)',
            line=dict(color='red', width=2),
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Линия порога 0.5
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Порог 0.5")
    
    fig.update_layout(
        title="Сравнение: производная vs бинарная цель vs аппроксиматор",
        xaxis_title="Бар (тестовая выборка)",
        yaxis_title="Значение",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ======================== ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ========================
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Эквити (симуляция)")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=list(range(len(equity))),
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='green', width=2)
        ))
        fig_equity.update_layout(
            xaxis_title="Бар",
            yaxis_title="Equity, %",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Распределение вероятностей")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=probs,
            nbinsx=30,
            name='Вероятности',
            marker_color='blue',
            opacity=0.7
        ))
        fig_hist.update_layout(
            xaxis_title="Вероятность (P)",
            yaxis_title="Частота",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # ======================== ТАБЛИЦА ПАРАМЕТРОВ ========================
    st.subheader("📋 Текущие параметры модели")
    
    params_df = pd.DataFrame([
        {"Параметр": "n_estimators", "Значение": n_estimators},
        {"Параметр": "learning_rate", "Значение": learning_rate},
        {"Параметр": "num_leaves", "Значение": num_leaves},
        {"Параметр": "min_child_samples", "Значение": min_child_samples},
        {"Параметр": "subsample", "Значение": subsample},
        {"Параметр": "colsample_bytree", "Значение": colsample_bytree},
        {"Параметр": "sigma", "Значение": sigma},
    ])
    
    st.dataframe(params_df, use_container_width=True)
    
    # ======================== ИНФОРМАЦИЯ ========================
    st.markdown("""
    ---
    ### 💡 Как использовать дашборд:
    1. **Изменяйте слайдеры** слева — модель переобучается автоматически
    2. **Смотрите на графики** — как меняется аппроксимация производной
    3. **Оптимизируйте метрики** — Accuracy, Win Rate, PnL
    4. **Найдите лучшие параметры** — и используйте их в основной системе
    
    ### 🎯 Цели:
    - Accuracy > 60%
    - Win Rate > 55%
    - Положительный PnL
    - Стабильная эквити
    """)

if __name__ == "__main__":
    main()