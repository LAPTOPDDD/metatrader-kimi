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
#                     НАСТРОЙКИ ОПТИМИЗАТОРА
# ====================================================================

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "XAUUSD", "USDCHF"
]

TIMEFRAMES = ["M15", "M30", "H1", "H4"]

HISTORY_TOTAL = 10000       # Баров для обучения
BARS_FOR_BACKTEST = 400     # Баров для теста
CHAMPIONS_PER_COMBO = 3     # Чемпионов на пару×ТФ
MAGIC_BASE = 100000

# Ограничения на параметры
PARAM_BOUNDS = {
    'thr': (0.20, 0.80),
    'sigma': (0.5, 3.0),
    'stop_loss_pips': (10, 120),
    'hold_bars': (2, 20),
}

# Point для разных типов пар
def get_point(symbol):
    if 'JPY' in symbol:
        return 0.01
    elif 'XAU' in symbol or 'GOLD' in symbol:
        return 0.01  # Уточните под вашего брокера (0.01 или 0.1)
    else:
        return 0.0001

# ====================================================================
#                     OPTUNA / GRID SEARCH
# ====================================================================

def has_optuna():
    try:
        import optuna
        return True
    except ImportError:
        return False

def optimize_with_optuna(df_feats, symbol, tf, n_trials=50):
    import optuna

    def objective(trial):
        params = {
            'thr': trial.suggest_float('thr', PARAM_BOUNDS['thr'][0], PARAM_BOUNDS['thr'][1], step=0.01),
            'sigma': trial.suggest_float('sigma', PARAM_BOUNDS['sigma'][0], PARAM_BOUNDS['sigma'][1], step=0.1),
            'stop_loss_pips': trial.suggest_int('stop_loss_pips', PARAM_BOUNDS['stop_loss_pips'][0], PARAM_BOUNDS['stop_loss_pips'][1], step=5),
            'hold_bars': trial.suggest_int('hold_bars', PARAM_BOUNDS['hold_bars'][0], PARAM_BOUNDS['hold_bars'][1], step=1),
        }
        res = run_backtest(df_feats, symbol, tf, params, HISTORY_TOTAL, BARS_FOR_BACKTEST)
        if res is None:
            return -999
        # Целевая функция: Sharpe - штраф за просадку
        score = res['sharpe'] - (res['max_drawdown'] / 10)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Возвращаем топ-N уникальных параметров
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
        res = run_backtest(df_feats, symbol, tf, p, HISTORY_TOTAL, BARS_FOR_BACKTEST)
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

def optimize_with_grid(df_feats, symbol, tf):
    """Fallback: простой перебор по сетке (медленно, но без зависимостей)"""
    thr_vals = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    sigma_vals = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    sl_vals = [15, 25, 35, 50, 70, 100]
    hold_vals = [3, 4, 5, 6, 8, 10, 12]

    results = []
    total = len(thr_vals) * len(sigma_vals) * len(sl_vals) * len(hold_vals)
    count = 0

    for thr in thr_vals:
        for sigma in sigma_vals:
            for sl in sl_vals:
                for hb in hold_vals:
                    count += 1
                    params = {'thr': thr, 'sigma': sigma, 'stop_loss_pips': sl, 'hold_bars': hb}
                    res = run_backtest(df_feats, symbol, tf, params, HISTORY_TOTAL, BARS_FOR_BACKTEST)
                    if res is None:
                        continue
                    score = res['sharpe'] - (res['max_drawdown'] / 10)
                    results.append({
                        'score': score,
                        'params': params,
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
                    if count % 50 == 0:
                        print(f"   Прогресс: {count}/{total}")

    results.sort(key=lambda x: x['score'], reverse=True)

    champions = []
    seen = set()
    for r in results:
        if len(champions) >= CHAMPIONS_PER_COMBO:
            break
        p = r['params']
        key = (round(p['thr'], 2), round(p['sigma'], 1), p['stop_loss_pips'], p['hold_bars'])
        if key in seen:
            continue
        seen.add(key)
        champions.append(r)
    return champions

# ====================================================================
#                     ВИЗУАЛИЗАЦИЯ
# ====================================================================

def plot_equity(champions, symbol, tf, save_dir='equity_charts'):
    """Сохраняет графики эквити чемпионов"""
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
    print("  ОПТИМИЗАТОР ЧЕМПИОНОВ — Автономная Эволюционная ТС")
    print("=" * 70)

    if not mt5.initialize():
        print("❌ Ошибка инициализации MT5")
        return
    print("✅ MT5 подключен")

    use_optuna = has_optuna()
    if use_optuna:
        print("🔧 Используется Optuna для оптимизации")
    else:
        print("🔧 Optuna не найдена — используется Grid Search (медленно)")

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

            df_feats = get_features(df)
            print(f"   Признаков: {len(df_feats.columns) - 10}")

            if use_optuna:
                n_trials = 80 if tf in ['M15', 'M30'] else 50
                champs = optimize_with_optuna(df_feats, symbol, tf, n_trials=n_trials)
            else:
                champs = optimize_with_grid(df_feats, symbol, tf)

            if not champs:
                print(f"   ⚠️ Чемпионы не найдены")
                continue

            # Построение графиков
            plot_equity(champs, symbol, tf)

            # Формирование записей
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

    # Сохранение
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
