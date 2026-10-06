#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ИССЛЕДОВАТЕЛЬСКИЙ ДАШБОРД v2.0
================================
• Регрессия аналоговой производной (не бинарная!)
• 2 графика: реальная производная vs аппроксимация
• LightGBM Regressor вместо Classifier
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import warnings
import time

warnings.filterwarnings('ignore')

# ======================== КОНФИГ =========================
NOPT = 1000  # Баров для обучения
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

# ======================== ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (АНАЛОГОВАЯ) ====================
EXCLUDE = {"open","high","low","close","tick_volume","spread","real_volume","Target","time"}

def get_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in EXCLUDE]

def build_target_analog(fs: FeatureStore, sigma: float, feature_cols: list):
    """
    Возвращает АНАЛОГОВУЮ производную (не бинарную!)
    """
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
    deriv_trimmed = deriv[start:end]  # АНАЛОГОВАЯ, не бинарная!
    
    X = df_slice[feature_cols].values
    closes = df_slice["close"].values
    times = df_slice["time"].values
    
    return X, deriv_trimmed, closes, times

# ======================== МОДЕЛЬ (РЕГРЕССОР) ====================
def train_regressor(X: np.ndarray, y: np.ndarray, params: dict) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model

# ======================== МЕТРИКИ РЕГРЕССИИ ====================
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Считаем метрики для регрессии
    """
    # MSE, MAE
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-9))
    
    # Корреляция Пирсона
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Знаковое совпадение (как в бинарной версии)
    sign_match = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation,
        'sign_match': sign_match,
    }

# ======================== СИМУЛЯЦИЯ ТОРГОВЛИ ====================
def simulate_trading(y_pred: np.ndarray, closes: np.ndarray, spread: float = 0.0001):
    """
    Симуляция торговли на основе предсказанной производной
    """
    equity = [0.0]
    position = 0
    entry_price = 0
    bars_held = 0
    max_hold = 10
    
    for i in range(len(y_pred)):
        signal = np.sign(y_pred[i])  # +1 или -1
        price = closes[i]
        
        if position != 0:
            bars_held += 1
            
            # Выход по смене сигнала или времени
            if signal != position or bars_held >= max_hold:
                if position == 1:
                    pnl = (price - entry_price) / entry_price
                else:
                    pnl = (entry_price - price) / entry_price
                
                equity.append(equity[-1] + pnl * 100)
                position = 0
                bars_held = 0
        
        # Вход
        if position == 0 and signal != 0:
            position = signal
            entry_price = price
            bars_held = 0
    
    # Закрытие последней позиции
    if position != 0:
        last_price = closes[-1]
        if position == 1:
            pnl = (last_price - entry_price) / entry_price
        else:
            pnl = (entry_price - last_price) / entry_price
        equity.append(equity[-1] + pnl * 100)
    
    return np.array(equity)

# ======================== ДАШБОРД ====================
def main():
    st.set_page_config(page_title="GAUSSIAN ENSEMBLE - Регрессия", layout="wide")
    
    st.title("️ Дашборд: Регрессия аналоговой производной")
    st.markdown(f"**Пара:** {SYMBOL} | **Таймфрейм:** {TIMEFRAME} | **Обучение:** {NOPT} баров | **Тест:** {NFWD} баров")
    st.info("💡 Аппроксимируем НЕ бинарную (0/1), а АНАЛОГОВУЮ производную Гаусса")
    
    # Загрузка данных
    @st.cache_data(ttl=300)
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
    st.sidebar.header("⚙️ Параметры LightGBM Regressor")
    
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 50, 10)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)
    num_leaves = st.sidebar.slider("num_leaves", 5, 50, 10, 5)
    min_child_samples = st.sidebar.slider("min_child_samples", 5, 50, 20, 5)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.1)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.1)
    max_depth = st.sidebar.slider("max_depth", 3, 15, 5, 1)
    
    st.sidebar.header(" Параметры Гаусса")
    sigma = st.sidebar.slider("sigma (сглаживание)", 0.5, 3.0, 1.5, 0.1)
    
    # Сбор параметров для регрессора
    lgb_params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'max_depth': max_depth,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': 1,
    }
    
    # ======================== ОБУЧЕНИЕ И ПРОГНОЗ ========================
    with st.spinner("Обучение регрессора..."):
        # Построение аналоговой целевой переменной
        X, y_true, closes, times = build_target_analog(fs, sigma, feature_cols)
        
        if X is None:
            st.error("❌ Ошибка построения целевой переменной")
            st.stop()
        
        # Разделение на train/test
        X_train = X[:NOPT]
        y_train = y_true[:NOPT]
        X_test = X[NOPT:NOPT+NFWD]
        y_test = y_true[NOPT:NOPT+NFWD]
        closes_test = closes[NOPT:NOPT+NFWD]
        times_test = times[NOPT:NOPT+NFWD]
        
        # Нормализация target (важно для регрессии!)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Нормализация признаков
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Обучение регрессора
        model = train_regressor(X_train_scaled, y_train_scaled, lgb_params)
        
        # Прогноз на тесте (в нормализованном виде)
        y_pred_scaled = model.predict(X_test_scaled)
        
        # Обратное преобразование в исходный масштаб
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # ======================== МЕТРИКИ ========================
    metrics = calc_metrics(y_test, y_pred)
    equity = simulate_trading(y_pred, closes_test)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Корреляция", f"{metrics['correlation']:.3f}")
    col2.metric("R² Score", f"{metrics['r2']:.3f}")
    col3.metric("Знак совпал", f"{metrics['sign_match']*100:.1f}%")
    col4.metric("MAE", f"{metrics['mae']:.2e}")
    col5.metric("PnL", f"{equity[-1]:.2f}%")
    
    # ======================== ГЛАВНЫЙ ГРАФИК: 2 ЛИНИИ ========================
    st.subheader("📈 Аналоговая производная vs Аппроксимация")
    
    fig = go.Figure()
    
    # 1. Реальная производная (синяя)
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))),
        y=y_test,
        mode='lines',
        name='Реальная производная',
        line=dict(color='blue', width=2),
        opacity=0.8
    ))
    
    # 2. Предсказание модели (красная)
    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred))),
        y=y_pred,
        mode='lines',
        name='Аппроксимация (LightGBM)',
        line=dict(color='red', width=2),
        opacity=0.8
    ))
    
    # Нулевая линия
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    
    fig.update_layout(
        title=f"Сравнение: реальная производная (σ={sigma}) vs предсказание модели",
        xaxis_title="Бар (тестовая выборка, последние 150 баров)",
        yaxis_title="Значение производной",
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ======================== ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ========================
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Ошибка аппроксимации")
        error = y_test - y_pred
        
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(
            x=list(range(len(error))),
            y=error,
            mode='markers',
            name='Ошибка (real - pred)',
            marker=dict(color='orange', size=6, opacity=0.6)
        ))
        fig_error.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
        fig_error.update_layout(
            xaxis_title="Бар",
            yaxis_title="Ошибка",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_error, use_container_width=True)
    
    with col_right:
        st.subheader(" Scatter: реальная vs предсказанная")
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Точки',
            marker=dict(color='blue', size=6, opacity=0.5)
        ))
        
        # Идеальная линия y=x
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Идеальная (y=x)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_scatter.update_layout(
            xaxis_title="Реальная производная",
            yaxis_title="Предсказанная",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ======================== ЭКВИТИ ========================
    st.subheader("💰 Симуляция торговли (знак производной → направление)")
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=list(range(len(equity))),
        y=equity,
        mode='lines',
        name='Equity',
        line=dict(color='green', width=2)
    ))
    fig_equity.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    fig_equity.update_layout(
        xaxis_title="Бар",
        yaxis_title="Equity, %",
        height=300,
        template="plotly_white"
    )
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # ======================== ТАБЛИЦА ПАРАМЕТРОВ ========================
    st.subheader("📋 Текущие параметры")
    
    params_df = pd.DataFrame([
        {"Параметр": "n_estimators", "Значение": n_estimators},
        {"Параметр": "learning_rate", "Значение": learning_rate},
        {"Параметр": "num_leaves", "Значение": num_leaves},
        {"Параметр": "min_child_samples", "Значение": min_child_samples},
        {"Параметр": "subsample", "Значение": subsample},
        {"Параметр": "colsample_bytree", "Значение": colsample_bytree},
        {"Параметр": "max_depth", "Значение": max_depth},
        {"Параметр": "sigma", "Значение": sigma},
    ])
    
    st.dataframe(params_df, use_container_width=True)
    
    # ======================== ИНФОРМАЦИЯ ========================
    st.markdown("""
    ---
    ### 💡 Ключевое отличие от классификации:
    
    | Классификация (v1) | Регрессия (v2) |
    |-------------------|----------------|
    | Target: 0 или 1 | Target: непрерывное значение |
    | Модель: LGBMClassifier | Модель: LGBMRegressor |
    | Метрика: Accuracy | Метрика: Корреляция, R² |
    | Выход: вероятность | Выход: число |
    
    ### 🎯 Что смотреть:
    1. **Корреляция > 0.5** — модель ловит направление
    2. **Знак совпал > 60%** — можно торговать по знаку
    3. **R² > 0.3** — модель объясняет дисперсию
    4. **PnL > 0** — стратегия прибыльна
    
    ### ⚙️ Как использовать:
    - Двигай слайдеры → модель переобучается → графики обновляются
    - Найди параметры с максимальной корреляцией
    - Используй их в основной торговой системе
    """)

if __name__ == "__main__":
    main()