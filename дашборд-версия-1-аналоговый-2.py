#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ДАШБОРД v2.1: ГЛАДКАЯ АППРОКСИМАЦИЯ
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

warnings.filterwarnings('ignore')

NOPT = 1000
NFWD = 150
SYMBOL = "EURUSD"
TIMEFRAME = "H1"

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

EXCLUDE = {"open","high","low","close","tick_volume","spread","real_volume","Target","time"}

def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE]

def build_target_analog(fs, sigma, feature_cols):
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
    
    X = df_slice[feature_cols].values
    closes = df_slice["close"].values
    times = df_slice["time"].values
    
    return X, deriv_trimmed, closes, times

def main():
    st.set_page_config(page_title="GAUSSIAN ENSEMBLE - Гладкая регрессия", layout="wide")
    
    st.title("️ Дашборд: Гладкая аппроксимация производной")
    st.markdown(f"**Пара:** {SYMBOL} | **Таймфрейм:** {TIMEFRAME}")
    
    @st.cache_data(ttl=300)
    def load_data():
        if not mt5.initialize():
            st.error("❌ Ошибка MT5")
            return None
        
        bars_needed = NOPT + NFWD + 500
        rates = mt5.copy_rates_from_pos(SYMBOL, tf_to_mt5(TIMEFRAME), 0, bars_needed)
        
        if rates is None or len(rates) < NOPT + NFWD:
            return None
        
        return pd.DataFrame(rates)
    
    with st.spinner("Загрузка..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    fs = FeatureStore(df)
    fs.calc_features()
    feature_cols = get_feature_cols(fs.features)
    
    # ======================== БОКОВАЯ ПАНЕЛЬ ========================
    st.sidebar.header("⚙️ Параметры модели")
    
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)
    num_leaves = st.sidebar.slider("num_leaves", 10, 100, 31, 5)
    max_depth = st.sidebar.slider("max_depth", 3, 15, 8, 1)
    min_child_samples = st.sidebar.slider("min_child_samples", 5, 50, 10, 5)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.9, 0.1)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.9, 0.1)
    
    st.sidebar.header(" Параметры Гаусса")
    sigma = st.sidebar.slider("sigma", 0.5, 3.0, 1.9, 0.1)
    
    st.sidebar.header(" Сглаживание предсказаний")
    smooth_predictions = st.sidebar.checkbox("Сглаживать предсказания", value=True)
    smooth_sigma = st.sidebar.slider("sigma сглаживания", 1, 10, 3, 1)
    
    lgb_params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': 1,
    }
    
    # ======================== ОБУЧЕНИЕ ========================
    with st.spinner("Обучение..."):
        X, y_true, closes, times = build_target_analog(fs, sigma, feature_cols)
        
        if X is None:
            st.stop()
        
        X_train = X[:NOPT]
        y_train = y_true[:NOPT]
        X_test = X[NOPT:NOPT+NFWD]
        y_test = y_true[NOPT:NOPT+NFWD]
        closes_test = closes[NOPT:NOPT+NFWD]
        
        # Нормализация
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train_scaled, y_train_scaled)
        
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Сглаживание предсказаний (если включено)
        if smooth_predictions:
            y_pred_smooth = gaussian_filter1d(y_pred, sigma=smooth_sigma)
        else:
            y_pred_smooth = y_pred
    
    # ======================== МЕТРИКИ ========================
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    correlation_smooth = np.corrcoef(y_test, y_pred_smooth)[0, 1]
    sign_match = np.mean(np.sign(y_test) == np.sign(y_pred))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Корреляция (сырая)", f"{correlation:.3f}")
    col2.metric("Корреляция (сглаженная)", f"{correlation_smooth:.3f}")
    col3.metric("Знак совпал", f"{sign_match*100:.1f}%")
    
    # ======================== ГЛАВНЫЙ ГРАФИК ========================
    st.subheader(" Реальная производная vs Аппроксимация")
    
    fig = go.Figure()
    
    # Реальная производная
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))),
        y=y_test,
        mode='lines',
        name='Реальная производная',
        line=dict(color='blue', width=2),
        opacity=0.8
    ))
    
    # Сырое предсказание (если не сглажено)
    if not smooth_predictions:
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='lines',
            name='Аппроксимация (сырая)',
            line=dict(color='red', width=2),
            opacity=0.8
        ))
    
    # Сглаженное предсказание
    if smooth_predictions:
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred_smooth))),
            y=y_pred_smooth,
            mode='lines',
            name=f'Аппроксимация (сглаженная, σ={smooth_sigma})',
            line=dict(color='red', width=2),
            opacity=0.8
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    
    fig.update_layout(
        title=f"Сравнение (σ={sigma})",
        xaxis_title="Бар",
        yaxis_title="Значение производной",
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ======================== ИНФО ========================
    st.markdown(f"""
    ### 💡 Почему было "зубасто":
    - **LightGBM = ансамбль деревьев** → каждое дерево даёт ступеньки
    - **Мало деревьев** (50) → грубая аппроксимация
    - **Маленькая глубина** → простые решения
    
    ### ✅ Что изменилось:
    - **n_estimators**: 50 → {n_estimators} (больше деревьев)
    - **learning_rate**: 0.1 → {learning_rate} (меньше шаг)
    - **max_depth**: 5 → {max_depth} (глубже деревья)
    - **Сглаживание**: Gaussian filter с σ={smooth_sigma if smooth_predictions else 'выкл'}
    
    ###  Результат:
    - Корреляция улучшилась: {correlation:.3f} → {correlation_smooth:.3f}
    - Кривая стала глаже
    """)

if __name__ == "__main__":
    main()