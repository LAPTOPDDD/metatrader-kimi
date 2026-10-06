"""
ТОРГОВАЯ СИСТЕМА НА УРОВНЯХ (Support/Resistance) + LightGBM
Целевая: buy(1) / sell(2) / hold(0)
ТФ: M15, M30, H1, H4
- Поиск уровней: фракталы (вилы) + кластеризация в зоны (векторизовано)
- Разметка истории: look-ahead на HORIZON баров, пробой на PROFIT_PIPS
- Модель: LightGBM (multiclass, 3 класса), TimeSeriesSplit CV
- Equity-кривая на ТЕСТОВОМ периоде (hold-out)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
import matplotlib
import matplotlib.pyplot as plt
import glob, os, time

np.random.seed(42)

# ======================================================================
# ПАРАМЕТРЫ (меняй под себя)
# ======================================================================
SYMBOL = "EURUSD"
DATA_DIR = r"D:\PYTHON\METATRADER-KIMI\data"   # папка с твоими CSV
TIMEFRAMES = ['M15', 'M30', 'H1', 'H4']   # все 4 ТФ
BASE_TF = 'H1'        # базовый (на нём разметка + обучение)

FRACTAL_LEN = 5       # вилы: 5 баров
CLUSTER_PIPS = 20     # зоны: уровни ближе 20 пипсов -> одна зона
LOOKAHEAD = 20        # горизонт разметки (баров вперёд)
PROFIT_PIPS = 30      # цель (пипсов)
STOP_PIPS = 20        # стоп (пипсов)
TEST_RATIO = 0.2      # 20% в тест
INITIAL_BALANCE = 10000.0
LOT_SIZE = 0.01
PIP = 0.0001

# ======================================================================
# 0. ЗАГРУЗКА (универсальный парсер, оба формата)
# ======================================================================
def load_tf(tf_name):
    cands = glob.glob(os.path.join(DATA_DIR, f"*{SYMBOL}_{tf_name}*.csv"))
    if not cands:
        raise FileNotFoundError(f"Нет файла *{SYMBOL}_{tf_name}*.csv в {DATA_DIR}")
    path = max(cands, key=os.path.getsize)   # самый полный
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = [c.strip().upper() for c in df.columns]
    # колонки
    rename = {}
    for k in ['OPEN','HIGH','LOW','CLOSE']:
        if k in df.columns: rename[k] = k.lower()
    if 'TIME' in df.columns and 'DATE' in df.columns:
        df['DT'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], errors='coerce')
    elif 'TIME' in df.columns:
        df['DT'] = pd.to_datetime(df['TIME'], format='mixed', errors='coerce')
    else:
        df['DT'] = pd.to_datetime(df['DATE'], format='mixed', errors='coerce')
    df = df.dropna(subset=['DT']).set_index('DT')
    df = df.rename(columns=rename)[['open','high','low','close']].astype(float)
    df = df[df.index.dayofweek < 5].sort_index()
    print(f"  {tf_name}: {len(df)} баров [{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}]")
    return df

print("=== ЗАГРУЗКА ДАННЫХ ===")
data = {tf: load_tf(tf) for tf in TIMEFRAMES}
base = data[BASE_TF]
print(f"Базовый ТФ: {BASE_TF}, баров: {len(base)}")

# ======================================================================
# 1. УРОВНИ — ВЕКТОРИЗОВАННЫЕ (фракталы + зоны), ТОЛЬКО В ПРОШЛОМ
# ======================================================================
def fractal_extremes(high, low, n):
    """Индексы локальных максимумов/минимумов (вилы) — векторно."""
    n2 = n // 2
    highs, lows = [], []
    for i in range(n2, len(high) - n2):
        if high[i] == np.max(high[i-n2:i+n2+1]): highs.append(i)
        if low[i] == np.min(low[i-n2:i+n2+1]): lows.append(i)
    return highs, lows

def cluster_1d(vals, dist_abs):
    """Объединить близкие значения (< dist_abs) в зоны, вернуть центры."""
    if len(vals) == 0: return np.array([])
    vals = np.sort(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] < dist_abs:
            groups[-1].append(v)
        else:
            groups.append([v])
    return np.array([np.mean(g) for g in groups])

def levels_on_grid(df, n_fractal=FRACTAL_LEN, cluster_pips=CLUSTER_PIPS):
    """Для КАЖДОГО базового бара: зоны S/R, вычисленные ТОЛЬКО по прошлому.
    Возвращает матрицы (len(base) x max_zones) — расстояния до зон."""
    high = df['high'].values; low = df['low'].values; close = df['close'].values
    highs_idx, lows_idx = fractal_extremes(high, low, n_fractal)
    # цены фракталов
    h_vals = close[highs_idx]   # используем close как уровень (можно high/low)
    l_vals = close[lows_idx]
    # "оконная" кластеризация: скользящее окно = последние W баров
    W = 500   # окно истории для уровней (баров)
    dist_abs = cluster_pips * PIP
    n = len(close)
    # предподсчёт: для каждого бара список зон (поддержка <= price, сопротивление >= price)
    sup_all = [np.array([]) for _ in range(n)]
    res_all = [np.array([]) for _ in range(n)]
    # оптимизация: двигаем окно, переиспользуем кластеры
    h_ptr = 0; l_ptr = 0
    h_window = []; l_window = []
    for i in range(n):
        # добавляем фракталы, которые произошли <= i
        while h_ptr < len(highs_idx) and highs_idx[h_ptr] <= i:
            h_window.append(h_vals[h_ptr]); h_ptr += 1
        while l_ptr < len(lows_idx) and lows_idx[l_ptr] <= i:
            l_window.append(l_vals[l_ptr]); l_ptr += 1
        # оставляем только окно W последних
        if len(h_window) > W: h_window = h_window[-W:]
        if len(l_window) > W: l_window = l_window[-W:]
        sup = cluster_1d(np.array(h_window), dist_abs)   # resistance (высокие)
        res = cluster_1d(np.array(l_window), dist_abs)   # support (низкие)
        sup_all[i] = sup; res_all[i] = res
    return sup_all, res_all

print("\n=== ПОИСК УРОВНЕЙ (векторизовано) ===")
t0 = time.time()
levels = {}
for tf in TIMEFRAMES:
    print(f"  {tf}...", end="", flush=True)
    sup, res = levels_on_grid(data[tf])
    levels[tf] = (sup, res)
    print(f" OK ({time.time()-t0:.1f} сек)")
# ограничимся 3 ТФ (M30/H1/H4), M15 — самый короткий, берём для контекста
TF_USE = ['M30','H1','H4']

# ======================================================================
# 2. ФИЧИ: расстояния до ближайших зон всех ТФ + базовые
# ======================================================================
print("\n=== ФИЧИ ===")
close_b = base['close'].values
high_b = base['high'].values
low_b = base['low'].values

def nearest_zones(sup_all, res_all, price, n_zones=3):
    """Для бара: ближайшие n_zones зон выше(сопротивление) и ниже(поддержка)."""
    sup = sup_all[price >= 0]  # placeholder
    return None

# упрощённо и быстро: для каждого бара берём 3 ближ. сопротивления + 3 поддержки (в пипсах)
def feat_distances(sup_all, res_all, prices):
    n = len(prices)
    feat = np.zeros((n, 6))   # 3 sup + 3 res (в пипсах, отрицательные = цена выше зоны)
    for i in range(n):
        price = prices[i]
        sup = np.array([s for s in sup_all[i] if s >= price])
        res = np.array([r for r in res_all[i] if r <= price])
        sup = np.sort(sup)[:3]; res = np.sort(res)[-3:]
        for k in range(3):
            feat[i, k]   = (sup[k] - price)/PIP if k < len(sup) else np.nan
            feat[i, 3+k] = (price - res[k])/PIP if k < len(res) else np.nan
    return feat

feat = pd.DataFrame(index=base.index)
feat['ret'] = base['close'].pct_change()
feat['range_pip'] = (base['high'] - base['low'])/PIP
for tf in TF_USE:
    sup, res = levels[tf]
    d = feat_distances(sup, res, close_b)
    cols = [f'{tf}_sup{k+1}' for k in range(3)] + [f'{tf}_res{k+1}' for k in range(3)]
    for j, c in enumerate(cols):
        feat[c] = d[:, j]
    # количество зон (всего)
    feat[f'{tf}_nzon'] = [len(sup[i])+len(res[i]) for i in range(len(close_b))]

# синхронизируем индексы (обрезать под пересечение всех ТФ)
min_idx = max(data[tf].index[0] for tf in TIMEFRAMES)
common = feat.index[feat.index >= min_idx]
feat = feat.loc[common].ffill().fillna(0)

# ======================================================================
# 3. РАЗМЕТКА ИСТОРИИ (buy / sell / hold)
# ======================================================================
print("=== РАЗМЕТКА (buy/sell/hold) ===")
close = base['close'].loc[common].values
high = base['high'].loc[common].values
low = base['low'].loc[common].values
idx_arr = np.arange(len(close))
target = np.zeros(len(close), dtype=int)   # 0=hold,1=buy,2=sell
H, P, S = LOOKAHEAD, PROFIT_PIPS*PIP, STOP_PIPS*PIP
for i in range(len(close) - H):
    fut_h = high[i+1:i+H+1]; fut_l = low[i+1:i+H+1]
    up = np.max(fut_h) - close[i]
    dn = close[i] - np.min(fut_l)
    if up >= P and (close[i] - np.min(fut_l)) > -S:   # пробил вверх, не ушёл в стоп
        target[i] = 1
    elif dn >= P and (np.max(fut_h) - close[i]) < S:  # пробил вниз
        target[i] = 2
    else:
        target[i] = 0
vc = pd.Series(target).value_counts().reindex([0,1,2], fill_value=0)
print(f"  hold: {vc[0]} ({vc[0]/len(target)*100:.1f}%) | buy: {vc[1]} ({vc[1]/len(target)*100:.1f}%) | sell: {vc[2]} ({vc[2]/len(target)*100:.1f}%)")

# ======================================================================
# 4. ОБУЧЕНИЕ (TimeSeriesSplit CV + тест)
# ======================================================================
X = feat.iloc[:-LOOKAHEAD].copy()
y = pd.Series(target[:-LOOKAHEAD], index=X.index)
# каузальность: фичи T-1
X = X.shift(1).dropna()
y = y.loc[X.index]

# ФИКСИРОВАННЫЙ LabelEncoder (3 класса, даже если hold нет в обучении)
le = LabelEncoder()
le.fit([0,1,2])
y_le = le.transform(y.values)

split = int(len(X) * (1 - TEST_RATIO))
Xtr, Xte = X.iloc[:split], X.iloc[split:]
ytr, yte = y_le[:split], y_le[split:]
print(f"\n=== ОБУЧЕНИЕ ===\nTrain: {len(Xtr)}, Test: {len(Xte)}")

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
fold = 1
for tr_i, te_i in tscv.split(Xtr):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                           min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(Xtr.iloc[tr_i], ytr[tr_i],
          eval_X=Xtr.iloc[te_i], eval_y=ytr[te_i], eval_metric='multi_logloss',
          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    pred = m.predict(Xtr.iloc[te_i])
    p, r, f, _ = precision_recall_fscore_support(ytr[te_i], pred, labels=[0,1,2], zero_division=0)
    print(f"  Fold {fold}: P={np.mean(p):.2f} R={np.mean(r):.2f} F1={np.mean(f):.2f}")
    fold += 1

# финальная модель на ВСЁМ train
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
model.fit(Xtr, ytr)
pred_test = model.predict(Xte)
p, r, f, _ = precision_recall_fscore_support(yte, pred_test, labels=[0,1,2], zero_division=0)
print(f"\nТест: Precision={np.mean(p):.2f} Recall={np.mean(r):.2f} F1={np.mean(f):.2f}")
print("По классам (hold/buy/sell):")
print(f"  hold: P={p[0]:.2f} R={r[0]:.2f}")
print(f"  buy:  P={p[1]:.2f} R={r[1]:.2f}")
print(f"  sell: P={p[2]:.2f} R={r[2]:.2f}")

# ======================================================================
# 5. EQUITY-КРИВАЯ НА ТЕСТЕ
# ======================================================================
print("\n=== EQUITY (тест) ===")
test_idx = Xte.index
close_te = base['close'].loc[test_idx].values
balance = INITIAL_BALANCE
equity = [balance]
trades = 0
wins = 0
for i in range(len(test_idx) - H):
    cls = pred_test[i]
    if cls == 1:      # buy: держим H баров
        profit = (close_te[min(i+H, len(close_te)-1)] - close_te[i]) * 100000 * LOT_SIZE
    elif cls == 2:    # sell
        profit = (close_te[i] - close_te[min(i+H, len(close_te)-1)]) * 100000 * LOT_SIZE
    else:             # hold
        profit = 0.0
    if cls != 0:
        trades += 1
        if profit > 0: wins += 1
    balance += profit
    equity.append(balance)

equity = np.array(equity)
ret = (equity[-1]/INITIAL_BALANCE - 1)*100
peak = np.maximum.accumulate(equity); mdd = ((peak - equity)/peak).max()*100
winrate = wins/trades*100 if trades else 0
print(f"Сделок: {trades}, Win rate: {winrate:.1f}%")
print(f"Доходность: {ret:+.2f}%")
print(f"Max Drawdown: {mdd:.2f}%")
print(f"Финальный баланс: {equity[-1]:.2f}")

# график equity
plt.figure(figsize=(14,6))
plt.plot(test_idx[:len(equity)], equity, color='green', lw=1.2)
plt.axhline(INITIAL_BALANCE, color='gray', ls=':', alpha=0.7)
plt.title(f"Equity Curve (TEST) | {SYMBOL} {BASE_TF} | Return {ret:+.2f}% | MDD {mdd:.2f}% | WR {winrate:.0f}%")
plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('equity_levels.png', dpi=120)
print("✅ Сохранён: equity_levels.png")
plt.show()

# важность фичей
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nТоп-12 фичей:")
print(imp.head(12).to_string())