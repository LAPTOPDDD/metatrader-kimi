import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

from features import get_features
from backtest import run_backtest
from mt5_utils import timeframe_from_string

# ====================================================================
#                     НАСТРОЙКИ ОПТИМИЗАТОРА (УСКОРЕННЫЕ)
# ====================================================================

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "XAUUSD", "USDCHF"
]

TIMEFRAMES = ["M15", "M30", "H1", "H4"]

# УМЕНЬШЕНО для скорости оптимизации
HISTORY_TOTAL = 6000        # Было 10000 — ускоряет загрузку и обучение
BARS_FOR_BACKTEST = 300     # Было 400 — достаточно для оценки
CHAMPIONS_PER_COMBO = 3
MAGIC_BASE = 100000

# Ограничения на параметры
PARAM_BOUNDS = {
    'thr': (0.20, 0.80),
    'sigma': (0.5, 3.0),
    'stop_loss_pips': (10, 120),
    'hold_bars': (2, 20),
}

# LightGBM для оптимизации — ЛЕГКАЯ модель
LGB_PARAMS_OPT = {
    'n_estimators': 50,        # Было 100
    'learning_rate': 0.05,     # Чуть выше — сходится быстрее
    'num_leaves': 20,          # Было 31 — меньше дерево
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1,
}

def get_point(symbol):
    if 'JPY' in symbol:
        return 0.01
    elif 'XAU' in symbol or 'GOLD' in symbol:
        return 0.01
    else:
        return 0.0001

# ====================================================================
#                     УСКОРЕННЫЙ БЭКТЕСТ
# ====================================================================

def run_backtest_fast(df_feats, symbol, tf, params, history_total, bars_for_backtest, lgb_params):
    """Ускоренный бэктест с легкой моделью"""
    import lightgbm as lgb
    from scipy.ndimage import gaussian_filter1d
    from mt5_utils import get_bars_count, get_timeframe_seconds
    import MetaTrader5 as mt5

    thr = params['thr']
    sigma = params['sigma']
    stop_loss_pips = params['stop_loss_pips']
    hold_bars = params['hold_bars']

    tf_mt5 = mt5.TIMEFRAME_M15
    if tf == 'M30': tf_mt5 = mt5.TIMEFRAME_M30
    elif tf == 'H1': tf_mt5 = mt5.TIMEFRAME_H1
    elif tf == 'H4': tf_mt5 = mt5.TIMEFRAME_H4

    tf_seconds = get_timeframe_seconds(tf_mt5)
    lookahead = int(4 * sigma)

    total_bars = len(df_feats)
    test_start_idx = total_bars - bars_for_backtest - lookahead
    if test_start_idx < lookahead:
        return None

    test_start_real = test_start_idx + lookahead
    test_df = df_feats.iloc[test_start_real:].copy().reset_index(drop=True)
    if len(test_df) < 50:
        return None

    # Начальное обучение
    initial_train = df_feats.iloc[:test_start_idx + lookahead].copy()
    d = initial_train.copy()
    d['Conv_Deriv'] = gaussian_filter1d(d['close'].values, sigma, order=1)
    d['Target'] = np.where(d['Conv_Deriv'] > 0, 1, 0)
    train_data = d.iloc[:-lookahead].dropna() if len(d) > lookahead + 100 else d.dropna()

    if len(train_data) < 100:
        return None

    exclude_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 
                   'real_volume', 'Target', 'Conv_Deriv', 'time']
    feats = [c for c in train_data.columns if c not in exclude_cols]
    if len(feats) == 0:
        return None

    model = lgb.LGBMClassifier(**lgb_params)
    try:
        model.fit(train_data[feats], train_data['Target'])
    except:
        return None

    point = 0.0001
    if 'JPY' in symbol: point = 0.01
    elif 'XAU' in symbol or 'GOLD' in symbol: point = 0.01

    retrain_every = 5
    equity = [0.0]
    position = None
    entry_price = 0
    entry_time = 0
    trades = []

    for i in range(len(test_df)):
        if i % retrain_every == 0 and i > 0:
            available = df_feats.iloc[:test_start_idx + i + lookahead].copy()
            d2 = available.copy()
            d2['Conv_Deriv'] = gaussian_filter1d(d2['close'].values, sigma, order=1)
            d2['Target'] = np.where(d2['Conv_Deriv'] > 0, 1, 0)
            td = d2.iloc[:-lookahead].dropna() if len(d2) > lookahead + 100 else d2.dropna()
            if len(td) >= 100:
                try:
                    model.fit(td[feats], td['Target'])
                except:
                    pass

        row = test_df.iloc[i:i+1]
        try:
            prob = model.predict_proba(row[feats])[0][1]
        except:
            prob = 0.5

        signal = 0
        if prob > thr: signal = 1
        elif prob < (1 - thr): signal = -1

        current_price = test_df.iloc[i]['close']
        current_time = test_df.iloc[i]['time']
        sl_dist = stop_loss_pips * point

        if position is not None:
            sl_hit = False
            if position == 1 and current_price <= entry_price - sl_dist:
                sl_hit = True
            elif position == -1 and current_price >= entry_price + sl_dist:
                sl_hit = True

            bars_held = get_bars_count(current_time - entry_time, tf_seconds)
            time_hit = bars_held >= hold_bars
            reverse = (position == 1 and signal == -1) or (position == -1 and signal == 1)

            if sl_hit or time_hit or reverse:
                profit = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                profit = profit / entry_price * 100
                equity.append(equity[-1] + profit)
                reason = 'SL' if sl_hit else ('Time' if time_hit else 'Reverse')
                trades.append({'profit': profit, 'reason': reason})
                position = None

        if position is None and signal != 0:
            position = signal
            entry_price = current_price
            entry_time = current_time

    if position is not None:
        current_price = test_df.iloc[-1]['close']
        profit = (current_price - entry_price) if position == 1 else (entry_price - current_price)
        profit = profit / entry_price * 100
        equity.append(equity[-1] + profit)
        trades.append({'profit': profit, 'reason': 'End'})

    eq_arr = np.array(equity)
    total_trades = len(trades)
    win_trades = sum(1 for t in trades if t['profit'] > 0)
    total_profit = sum(t['profit'] for t in trades)

    max_dd = 0
    peak = eq_arr[0]
    for val in eq_arr:
        if val > peak: peak = val
        dd = peak - val
        if dd > max_dd: max_dd = dd

    win_rate = win_trades / total_trades if total_trades > 0 else 0
    profits = [t['profit'] for t in trades if t['profit'] > 0]
    losses = [t['profit'] for t in trades if t['profit'] < 0]
    profit_factor = sum(profits) / abs(sum(losses)) if losses else float('inf')

    returns = np.diff(eq_arr)
    sharpe = 0
    if len(returns) > 1 and np.std(returns) > 1e-9:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4)

    return {
        'equity': eq_arr,
        'total_trades': total_trades,
        'win_trades': win_trades,
        'total_profit': total_profit,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
    }

# ====================================================================
#                     OPTUNA С ПРУНИНГОМ
# ====================================================================

def optimize_with_optuna(df_feats, symbol, tf, n_trials=60, n_jobs=4):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    def objective(trial):
        params = {
            'thr': trial.suggest_float('thr', PARAM_BOUNDS['thr'][0], PARAM_BOUNDS['thr'][1], step=0.01),
            'sigma': trial.suggest_float('sigma', PARAM_BOUNDS['sigma'][0], PARAM_BOUNDS['sigma'][1], step=0.1),
            'stop_loss_pips': trial.suggest_int('stop_loss_pips', PARAM_BOUNDS['stop_loss_pips'][0], PARAM_BOUNDS['stop_loss_pips'][1], step=5),
            'hold_bars': trial.suggest_int('hold_bars', PARAM_BOUNDS['hold_bars'][0], PARAM_BOUNDS['hold_bars'][1], step=1),
        }
        res = run_backtest_fast(df_feats, symbol, tf, params, HISTORY_TOTAL, BARS_FOR_BACKTEST, LGB_PARAMS_OPT)
        if res is None:
            return -999
        score = res['sharpe'] - (res['max_drawdown'] / 10)
        return score

    sampler = TPESampler(multivariate=True, seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)

    champions = []
    seen = set()
    for t in trials:
        if len(champions) >= CHAMPIONS_PER_COMBO:
            break
        if t.value is None:
            continue
        p = t.params
        key = (round(p['thr'], 2), round(p['sigma'], 1), p['stop_loss_pips'], p['hold_bars'])
        if key in seen:
            continue
        seen.add(key)
        res = run_backtest_fast(df_feats, symbol, tf, p, HISTORY_TOTAL, BARS_FOR_BACKTEST, LGB_PARAMS_OPT)
        if res is None:
            continue
        champions.append({
            'params': p,
            'metrics': {
                'sharpe': round(res['sharpe'], 3),
                'max_drawdown': round(res['max_drawdown'], 3),
                'total_profit': round(res['total_profit'], 3),
                'win_rate': round(res['win_rate'], 3),
                'profit_factor': round(res['profit_factor'], 3),
                'total_trades': res['total_trades'],
            },
            'equity': res['equity'].tolist(),
        })
    return champions

# ====================================================================
#                     ВИЗУАЛИЗАЦИЯ
# ====================================================================

def plot_equity(champions, symbol, tf, save_dir='equity_charts'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green', 'blue', 'orange']
        for i, champ in enumerate(champions):
            eq = np.array(champ['equity'])
            p = champ['params']
            label = f"#{i+1} THR={p['thr']:.2f} SL={p['stop_loss_pips']} HB={p['hold_bars']} Sharpe={champ['metrics']['sharpe']:.2f}"
            ax.plot(eq, color=colors[i % len(colors)], lw=1.5, label=label)

        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_title(f"Эквити чемпионов | {symbol} {tf}")
        ax.set_xlabel("Бары")
        ax.set_ylabel("Эквити, %")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fname = os.path.join(save_dir, f"{symbol}_{tf}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 График сохранен: {fname}")
    except Exception as e:
        print(f"   ⚠️ Не удалось построить график: {e}")

# ====================================================================
#                     ОСНОВНОЙ ПРОЦЕСС
# ====================================================================

def main():
    print("=" * 70)
    print("  ОПТИМИЗАТОР ЧЕМПИОНОВ — УСКОРЕННАЯ ВЕРСИЯ")
    print("  Optuna + Прунинг + Кэш признаков + Легкая модель")
    print("=" * 70)

    if not mt5.initialize():
        print("❌ Ошибка инициализации MT5")
        return
    print("✅ MT5 подключен")
    print(f"🔧 Optuna: ДА | Параллельных trial'ов: 4")
    print(f"📊 История: {HISTORY_TOTAL} баров | Тест: {BARS_FOR_BACKTEST} баров")
    print(f"🌲 LightGBM: {LGB_PARAMS_OPT['n_estimators']} деревьев (в Live будет 100)")

    all_champions = []
    magic_counter = 0

    for symbol in PAIRS:
        for tf in TIMEFRAMES:
            print(f"{'='*70}")
            print(f"📈 Оптимизация: {symbol} | {tf}")
            print(f"{'='*70}")

            tf_mt5 = timeframe_from_string(tf)
            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, HISTORY_TOTAL + 500)

            if rates is None or len(rates) < 1000:
                print(f"⚠️ Недостаточно данных для {symbol} {tf}, пропускаем")
                continue

            df = pd.DataFrame(rates)
            print(f"   Загружено {len(df)} баров")

            # === КЭШ ПРИЗНАКОВ ===
            t0 = datetime.now()
            df_feats = get_features(df)
            dt = (datetime.now() - t0).total_seconds()
            print(f"   ✅ Признаков: {len(df_feats.columns) - 10} (рассчитано за {dt:.1f}с)")

            # Количество trials по ТФ (M15/M30 — больше, H1/H4 — меньше)
            n_trials = 80 if tf in ['M15', 'M30'] else 50

            champs = optimize_with_optuna(df_feats, symbol, tf, n_trials=n_trials, n_jobs=4)

            if not champs:
                print(f"   ⚠️ Чемпионы не найдены")
                continue

            plot_equity(champs, symbol, tf)

            for i, champ in enumerate(champs):
                magic = MAGIC_BASE + magic_counter
                magic_counter += 1

                record = {
                    'champion_id': f"{symbol}_{tf}_{i+1}",
                    'symbol': symbol,
                    'timeframe': tf,
                    'magic': magic,
                    'lot': 0.01,
                    'thr': champ['params']['thr'],
                    'sigma': champ['params']['sigma'],
                    'stop_loss_pips': champ['params']['stop_loss_pips'],
                    'hold_bars': champ['params']['hold_bars'],
                    'point': get_point(symbol),
                    'metrics': champ['metrics'],
                }
                all_champions.append(record)

                print(f"   🏆 Чемпион #{i+1} | Magic: {magic}")
                print(f"      THR={champ['params']['thr']:.2f} SIGMA={champ['params']['sigma']:.1f} "
                      f"SL={champ['params']['stop_loss_pips']} HB={champ['params']['hold_bars']}")
                print(f"      Sharpe={champ['metrics']['sharpe']:.2f} "
                      f"DD={champ['metrics']['max_drawdown']:.2f}% "
                      f"WinRate={champ['metrics']['win_rate']*100:.1f}% "
                      f"Trades={champ['metrics']['total_trades']}")

    output_file = 'champions.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_champions, f, indent=2, ensure_ascii=False)

    print(f"{'='*70}")
    print(f"✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"   Всего чемпионов: {len(all_champions)}")
    print(f"   Файл: {output_file}")
    print(f"   Теперь запустите: python main.py")
    print(f"{'='*70}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
