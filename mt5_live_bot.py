#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt5_live_bot.py — live-бот для ДЕМО MetaTrader 5 (wide-brain V2.1, "база").
Для каждой из 10 пар держит позицию по сигналу per-pair модели (BUY при proba>0.5, SELL иначе).
Минимальный лот инструмента. Ретрейн каждое утро, торговля в сессии 8:00-17:00.
Модель/фичи/цель переиспользуются из QWEN6 (тот же "мозг", что в прототипе).

ЗАПУСК: терминал MT5 открыт, аккаунт демо, кнопка "Algo Trading" ВКЛ.
  & .venv\\Scripts\\python.exe mt5_live_bot.py
Перед реальной торговлей поставь LIVE_TRADING = True (по умолчанию False = только логика, без ордеров).
"""
import sys, time, datetime, json, os, atexit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

import importlib.util
import MetaTrader5 as mt5

HERE = Path(__file__).parent

# ============================ К О Н Ф И Г ================================
PAIRS       = ["AUDJPY", "AUDUSD", "EURJPY", "EURUSD", "GBPJPY",
               "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]   # 10 позиций
BASE_TF     = "M15"
HIGHER      = ["M30", "H1", "H4"]
SIGMA       = 1.5
BARS_TRAIN  = 10000      # сколько M15-баров берём в обучение (самые свежие)
N_M15       = BARS_TRAIN + 300
NORM        = True
SESSION_START = (8, 0)   # торгуем только внутри сессии
SESSION_END   = (17, 0)
RETRAIN_HOUR  = 7        # ежедневный ретрейн после этого часа
MAGIC         = 20260829
POLL_SEC      = 5
LIVE_TRADING  = False    # <<< поставь True для реальных ордеров на демо
FILLING_MODE  = mt5.ORDER_FILLING_FOK   # крипто-брокер принимает только FOK; для форекса может быть IOC
LOGIN_FILE    = HERE / "mt5_login.json"   # опц.: {"login":..,"password":..,"server":..,"path":..}
# --- защита от whipsaw и тонких инструментов ---
MIN_HOLD_BARS = 4       # не флиповать чаще чем раз в N баров
MIN_BARS     = 8000     # минимум баров истории; меньше — не торгуем (GRAM и т.п.)
TP_PROFIT   = 0.03     # V3: тейк на позу в USD (на 0.01 лота); 0 = выкл
# Крипто-пресет (имена из твоего брокера; --crypto подхватит их и режим 24/7)
CRYPTO_PAIRS   = ["AAVE", "BITCOIN", "BITCOIN CASH", "CARDANO", "CHAINLINK", "DOGECOIN",
                  "ETHEREUM", "LITECOIN", "POLKADOT", "POLYGON", "RIPPLE", "SOLANA",
                  "STELLAR", "TRON", "UNISWAP", "AVALANCHE", "AXS", "BNB", "COSMOS",
                  "PYTH", "ILLUVIUM", "INJ", "RENDER", "TRUMP", "SUI", "HEDERA", "NEAR",
                  "APTOS", "ARBITRUM", "PAXG", "GRAM"]
# ========================================================================

TF_MAP = {"M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
          "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
TF_MIN = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}

# подгрузить QWEN6 (фичи/модель/цель)
_qspec = importlib.util.spec_from_file_location("q6", HERE / "система-в-единм-файле-QWEN6.py")
q6 = importlib.util.module_from_spec(_qspec)
_qspec.loader.exec_module(q6)

_models = {}   # symbol -> (model, scaler)
_lots   = {}   # symbol -> volume (min lot)
_sinfo  = {}   # symbol -> symbol_info
_last_flip = {}  # symbol -> время (unix) последнего флипа, для MIN_HOLD_BARS


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(HERE / "bot_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


EQUITY_LOG = HERE / "equity_log.csv"   # time,balance,equity,positions — для анализа утром


def log_equity():
    try:
        ai = mt5.account_info()
        if ai is None:
            return
        npos = 0
        for s in PAIRS:
            npos += len(mt5.positions_get(symbol=s) or [])
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hdr = not EQUITY_LOG.exists()
        with open(EQUITY_LOG, "a", encoding="utf-8") as f:
            if hdr:
                f.write("time,balance,equity,positions\n")
            f.write(f"{ts},{ai.balance:.2f},{ai.equity:.2f},{npos}\n")
    except Exception:
        pass


# ======================== 3-КЛАССОВАЯ МОДЕЛЬ =============================
# 0 = short, 1 = hold, 2 = long — модель сама учитывает спред
LGB_PARAMS = dict(
    n_estimators=50, learning_rate=0.1, num_leaves=10,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    verbosity=-1, random_state=42, n_jobs=1,
)


def train_model_multi(X, y):
    """Обучение на 3 класса: short=0, hold=1, long=2."""
    classes = np.unique(y)
    n_classes = len(classes)
    counts = np.bincount(y, minlength=n_classes)
    weights = np.array([1.0 / c if c > 0 else 1.0 for c in counts])
    sample_weight = weights[y]
    model = lgb.LGBMClassifier(**LGB_PARAMS, objective="multiclass",
                               num_class=n_classes)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def connect():
    if mt5.initialize():
        log("MT5: подключено к запущенному терминалу")
        return True
    log(f"MT5 initialize() не удалось: {mt5.last_error()}")
    if LOGIN_FILE.exists():
        cfg = json.loads(LOGIN_FILE.read_text(encoding="utf-8"))
        if mt5.initialize(login=cfg.get("login"), password=cfg.get("password"),
                          server=cfg.get("server"), path=cfg.get("path")):
            log("MT5: подключено по mt5_login.json")
            return True
    log("MT5 не подключён. Откройте терминал с демо-аккаунтом.")
    return False


def ensure_symbols():
    ok = True
    for s in PAIRS:
        si = mt5.symbol_info(s)
        if si is None:
            log(f"! символ {s} недоступен на счёте — пропускаем")
            ok = False
            continue
        if not si.visible:
            mt5.symbol_select(s, True)
        si = mt5.symbol_info(s)
        _sinfo[s] = si
        lot = round(si.volume_min / si.volume_step) * si.volume_step
        lot = max(lot, si.volume_min)
        _lots[s] = lot
        log(f"  {s}: min_lot={lot} step={si.volume_step} filling={si.filling_mode}")
    return any(s in _sinfo for s in PAIRS)


def choose_filling(si):
    return FILLING_MODE


def get_rates(symbol, tf_const, count):
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def build_pair_features(symbol, n_m15, higher):
    base = get_rates(symbol, TF_MAP[BASE_TF], n_m15)
    if base is None or len(base) < 200:
        return None
    fs = q6.FeatureStore(base)
    fs.calc_features()
    fM15 = fs.features[q6.get_feature_cols(fs.features)].values.astype(float)
    L = len(fM15)
    offset = len(base) - L
    close = base["close"].values[offset:]
    t_base = base["time"].values[offset:]
    for tf in higher:
        n_h = (n_m15 * TF_MIN[BASE_TF]) // TF_MIN[tf] + 80
        h = get_rates(symbol, TF_MAP[tf], n_h)
        if h is None:
            fM15 = np.concatenate([fM15, np.zeros((L, 11))], axis=1)
            continue
        fs2 = q6.FeatureStore(h)
        fs2.calc_features()
        f2 = fs2.features[q6.get_feature_cols(fs2.features)].values.astype(float)
        off2 = len(h) - len(f2)
        t2 = h["time"].values[off2:]
        pos = np.searchsorted(t2, t_base, side="right") - 1
        pos = np.clip(pos, 0, len(t2) - 1)
        fM15 = np.concatenate([fM15, f2[pos]], axis=1)
    return fM15, close, t_base


def train_pair(symbol):
    chk = mt5.copy_rates_from_pos(symbol, TF_MAP[BASE_TF], 0, N_M15)
    if chk is None or len(chk) < MIN_BARS:
        log(f"! {symbol}: недостаточно истории ({len(chk) if chk is not None else 0} < "
            f"{MIN_BARS}) — не торгуем")
        _models.pop(symbol, None)
        return
    feats = build_pair_features(symbol, N_M15, HIGHER)
    if feats is None:
        log(f"! {symbol}: нет данных для обучения")
        return
    X, close, _ = feats
    if X.shape[0] < 200:
        log(f"! {symbol}: мало баров ({X.shape[0]})")
        return
    # --- 3-классовый target с учётом спреда ---
    si = mt5.symbol_info(symbol)
    spread_cost = (si.spread * si.point) if (si and si.point) else 0.0
    grad = np.gradient(gaussian_filter1d(close, SIGMA, mode="reflect"))
    y = np.where(grad > spread_cost, 2,
                 np.where(grad < -spread_cost, 0, 1)).astype(np.int8)
    sc = StandardScaler().fit(X) if NORM else None
    Xt = sc.transform(X) if NORM else X
    model = train_model_multi(Xt, y)
    _models[symbol] = (model, sc)
    log(f"  {symbol}: обучено {X.shape[0]} баров, фичей={X.shape[1]}, "
        f"short={np.sum(y==0)} hold={np.sum(y==1)} long={np.sum(y==2)}")


def train_all():
    log("=== Ретрейн всех пар ===")
    for s in PAIRS:
        if s in _sinfo:
            train_pair(s)


def predict_proba(symbol):
    """Возвращает массив [p_short, p_hold, p_long] или None."""
    m = _models.get(symbol)
    if m is None:
        return None
    model, sc = m
    feats = build_pair_features(symbol, N_M15, HIGHER)
    if feats is None:
        return None
    X, _, _ = feats
    Xt = sc.transform(X[-1:]) if NORM else X[-1:]
    proba = model.predict_proba(Xt)[0]   # [p_short, p_hold, p_long]
    return proba


def send_market(symbol, side, volume, comment):
    si = _sinfo[symbol]
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == 1 else tick.bid
    otype = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
           "type": otype, "price": price, "deviation": 30, "magic": MAGIC,
           "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": choose_filling(si)}
    if not LIVE_TRADING:
        log(f"  [DRY] {symbol} {'BUY' if side==1 else 'SELL'} {volume}")
        return None
    r = mt5.order_send(req)
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"  ! ордер НЕ исполнен {symbol} {'BUY' if side==1 else 'SELL'}: "
            f"{r.retcode if r else mt5.last_error()}")
    else:
        log(f"  ✓ {symbol} {'BUY' if side==1 else 'SELL'} {volume} @ {price:.5f} ticket={r.order}")
    return r


def close_position(symbol, pos):
    si = _sinfo[symbol]
    tick = mt5.symbol_info_tick(symbol)
    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    price = tick.bid if pos.type == 0 else tick.ask
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": pos.volume,
           "type": close_type, "position": pos.ticket, "price": price, "deviation": 30,
           "magic": MAGIC, "comment": "wbrain-close", "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": choose_filling(si)}
    if not LIVE_TRADING:
        log(f"  [DRY] {symbol} CLOSE ticket={pos.ticket}")
        return None
    r = mt5.order_send(req)
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"  ! закрытие НЕ исполнено {symbol}: {r.retcode if r else mt5.last_error()}")
    else:
        log(f"  ✓ {symbol} CLOSE ticket={pos.ticket}")
    return r


def reconcile(symbol, bar_time):
    lot = _lots.get(symbol)
    if lot is None:
        return
    proba = predict_proba(symbol)          # [p_short, p_hold, p_long]
    if proba is None:
        return
    want = int(np.argmax(proba))           # 0=short, 1=hold, 2=long
    pos = mt5.positions_get(symbol=symbol)
    pos = pos[0] if (pos is not None and len(pos) > 0) else None
    # V3: тейк по прибыли на позу — закрываем и ждём след. бара
    if pos is not None and TP_PROFIT > 0 and pos.profit >= TP_PROFIT:
        close_position(symbol, pos)
        log(f"  TP {symbol} profit={pos.profit:.2f} >= {TP_PROFIT:.2f} — закрыто")
        _last_flip[symbol] = bar_time
        return
    if pos is None:
        # плоская — входим если модель уверена (hold → ждём)
        if want == 1:
            return
        side = 1 if want == 2 else -1
        send_market(symbol, side, lot, "wbrain-open")
        _last_flip[symbol] = bar_time
        return
    cur_side = 1 if pos.type == 0 else -1
    if want == 1:
        return                            # hold — держим
    side = 1 if want == 2 else -1
    if side == cur_side:
        return                            # уже в нужной позиции
    # флип: защита от слишком частых разворотов (MIN_HOLD_BARS)
    last = _last_flip.get(symbol, -10**18)
    if bar_time - last < MIN_HOLD_BARS * 900:
        return
    close_position(symbol, pos)
    send_market(symbol, side, lot, "wbrain-flip")
    _last_flip[symbol] = bar_time


def check_tp_all():
    if TP_PROFIT <= 0:
        return 0
    n = 0
    for s in list(_models.keys()):
        for p in (mt5.positions_get(symbol=s) or []):
            if p.magic != MAGIC:
                continue
            if p.profit >= TP_PROFIT:
                close_position(s, p)
                log(f"  TP {s} profit={p.profit:.2f} >= {TP_PROFIT:.2f} — закрыто")
                _last_flip[s] = int(datetime.datetime.now().timestamp())
                n += 1
    return n


def manage_all():
    rates = mt5.copy_rates_from_pos(PAIRS[0], TF_MAP[BASE_TF], 0, 1)
    bar_time = int(rates[0]["time"]) if (rates is not None and len(rates)) else 0
    log("--- новый бар M15: синхронизация позиций ---")
    for s in list(_models.keys()):
        reconcile(s, bar_time)


def close_all_bot_positions():
    closed = 0
    for s in PAIRS:
        for p in (mt5.positions_get(symbol=s) or []):
            if p.magic == MAGIC:
                r = close_position(s, p)
                closed += 1
                log(f"  закрываю {s} ticket={p.ticket} -> {r.retcode if r else 'ERR'}")
    log(f"--close-all: закрыто позиций бота = {closed}")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # на Windows для уже мёртвого/чужого PID может прилететь
        # не ProcessLookupError, а другой OSError (WinError 11 и т.п.) — считаем мёртвым
        return False
    return True


def acquire_instance_lock(key):
    """Один экземпляр бота на набор пар. Блокирует запуск второго (дубль ордеров)."""
    lock = HERE / f"bot_{key}.lock"
    if lock.exists():
        old = None
        try:
            old = int(lock.read_text().strip())
        except Exception:
            old = None
        if old and _pid_alive(old):
            log(f"ЭКЗЕМПЛЯР УЖЕ ЗАПУЩЕН (pid={old}). Второй запуск запрещён "
                f"во избежание дублирования ордеров. Выход.")
            sys.exit(1)
        log("Старый lock-файл устарел (процесс не жив) — перехватываем.")
    try:
        lock.write_text(str(os.getpid()))
    except Exception:
        pass

    def _release():
        try:
            if lock.exists() and _pid_alive(int(lock.read_text().strip())):
                lock.unlink()
        except Exception:
            pass
    atexit.register(_release)


def apply_args():
    global LIVE_TRADING, PAIRS, SESSION_START, SESSION_END
    argv = sys.argv
    if "--live" in argv:
        LIVE_TRADING = True
    if "--24" in argv:                      # торговать круглосуточно (крипта)
        SESSION_START = (0, 0)
        SESSION_END = (23, 59)
    if "--symbols" in argv:
        i = argv.index("--symbols")
        if i + 1 < len(argv):
            PAIRS[:] = [s.strip().upper() for s in argv[i + 1].split(",") if s.strip()]
    if "--crypto" in argv:                    # крипто-пресет: пары + круглосуточно
        PAIRS[:] = list(CRYPTO_PAIRS)
        SESSION_START = (0, 0)
        SESSION_END = (23, 59)


def main():
    once = "--once" in sys.argv
    apply_args()
    key = "crypto" if set(PAIRS) == set(CRYPTO_PAIRS) else "forex"
    acquire_instance_lock(key)
    print("=" * 60)
    print(" WIDE-BRAIN LIVE BOT (демо MT5)  |  LIVE_TRADING =", LIVE_TRADING,
          "| once =", once)
    print("=" * 60)
    if not connect():
        return
    if "--close-all" in sys.argv:          # закрыть все позиции бота и выйти
        log("--close-all: закрываю все позиции бота...")
        close_all_bot_positions()
        log("--close-all завершён.")
        return
    if "--list" in sys.argv:               # --list [подстрока]: показать символы брокера
        i = sys.argv.index("--list")
        sub = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        allsym = mt5.symbols_get() or []
        names = sorted(s.name for s in allsym if sub.upper() in s.name.upper())
        log(f"Символы брокера (фильтр='{sub}'): {len(names)}")
        for n in names:
            print(" ", n)
        return
    ti = mt5.terminal_info()
    if ti is not None and not ti.trade_allowed:
        log("ВНИМАНИЕ: в терминале отключён 'Algo Trading' — ордера не уйдут!")
    if not ensure_symbols():
        log("Нет ни одного доступного символа. Выход.")
        return
    train_all()
    log_equity()   # стартовый снимок баланса
    if once:
        log("Режим --once: один проход управления и выход (без цикла).")
        manage_all()
        # снимок позиций
        for s in PAIRS:
            p = mt5.positions_get(symbol=s)
            if p:
                for x in p:
                    log(f"  позиция {s}: {'BUY' if x.type==0 else 'SELL'} {x.volume} ticket={x.ticket}")
            else:
                log(f"  позиция {s}: нет")
        log("--once завершён.")
        return
    last_bar = None
    last_day = datetime.datetime.now().date()
    last_equity_t = datetime.datetime.now()
    log("Бот работает. Ctrl+C — стоп (позиции не закрываются).")
    try:
        while True:
            now = datetime.datetime.now()
            if now.hour >= RETRAIN_HOUR and now.date() != last_day:
                log("Ежедневный ретрейн...")
                train_all()
                last_day = now.date()
            in_session = SESSION_START <= (now.hour, now.minute) < SESSION_END
            rates = mt5.copy_rates_from_pos(PAIRS[0], TF_MAP[BASE_TF], 0, 1)
            cur = rates[0]["time"] if (rates is not None and len(rates)) else None
            if cur != last_bar and cur is not None:
                last_bar = cur
                if in_session:
                    manage_all()
                else:
                    log("вне сессии — без действий")
            # V3: проверка тейка каждую итерацию (каждые POLL_SEC)
            check_tp_all()
            if (now - last_equity_t).total_seconds() >= 60:
                log_equity()
                last_equity_t = now
            time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        log("Остановка по Ctrl+C. Открытые позиции оставлены как есть.")


if __name__ == "__main__":
    main()
