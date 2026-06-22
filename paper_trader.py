#!/usr/bin/env python3
"""Prophet Futures — Local Paper Trading Engine (v24)
Real market data + simulated execution. Runs alongside SimNow.
"""
import sys, os, time, json, signal
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb
import pickle

# ===== CONFIG =====
SYMBOLS = {
    'lh2609': {'code': 'LH0', 'name': 'LH', 'cost': 0.0006, 'multiplier': 16,
               'stop_type': 'atr', 'stop_mult': 1.5, 'rr': 4, 'max_pos': 6, 'reverse': False},
    'jm2609': {'code': 'JM0', 'name': 'JM', 'cost': 0.0011, 'multiplier': 60,
               'stop_type': 'struct', 'struct_n': 20, 'rr': 3.5, 'max_pos': 4, 'reverse': False},
}

CAPITAL = 300000
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state.json')
VOL_LOOKBACK = 20
MAX_LEVERAGE = 3.0
MIN_LEVERAGE = 0.5
BAR_INTERVAL = 300  # 5 min scan

running = True
def signal_handler(sig, frame):
    global running; running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===== DATA =====
def fetch_history(code, days=1200):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = ak.futures_main_sina(symbol=code, start_date=start.strftime('%Y%m%d'),
                                   end_date=end.strftime('%Y%m%d'))
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except Exception as e:
        print(f"  Fetch error {code}: {e}")
        return None

def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx-window:idx+1]
    c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values
    v = w['volume'].values; oi = w['oi'].values
    f = []
    if idx >= 1: f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
    else: f.extend([0, 0])
    for lag in [1,3,5,10,20]: f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    for p in [5,10,20,60]: ma = np.mean(c[-min(p,len(c)):]); f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:])); f.append((h[-1]-l[-1])/c[-1])
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1; f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1)
    ema12 = c[-1]; ema26 = c[-1]
    for j in range(len(c)-2, -1, -1): ema12 = (2/13)*c[j] + (11/13)*ema12; ema26 = (2/27)*c[j] + (25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    dd_ = np.diff(c[-15:]); g = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
    lo = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
    f.append(100 - 100/(1+g/lo) if lo > 0 else 50)
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:]); f.append((c[-1]-ma20)/(2*bb+1e-10))
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

def calc_position_size(sym_key, df):
    cfg = SYMBOLS[sym_key]
    atr_vals = [abs(df.iloc[i]['high'] - df.iloc[i]['low'])
                for i in range(max(0, len(df)-VOL_LOOKBACK), len(df))]
    if not atr_vals: return 1
    atr = np.mean(atr_vals); price = df.iloc[-1]['close']; atr_pct = atr / price
    if atr_pct < 0.01: leverage = MAX_LEVERAGE
    elif atr_pct < 0.02: leverage = 2.0
    elif atr_pct < 0.03: leverage = 1.5
    elif atr_pct < 0.05: leverage = MIN_LEVERAGE
    else: leverage = 0
    base = cfg['max_pos'] // 2
    return max(1, int(leverage * base)) if leverage > 0 else 0

def is_trading_time(now=None):
    if now is None: now = datetime.now()
    wd = now.weekday()
    if wd >= 5: return False
    h, m = now.hour, now.minute
    t = h * 60 + m
    return (540 <= t < 615) or (630 <= t < 690) or (810 <= t < 900)  # 9:00-10:15, 10:30-11:30, 13:30-15:00

# ===== STATE =====
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'cash': CAPITAL, 'positions': {}, 'trades': [], 'equity_history': []}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2, default=str)

# ===== MAIN =====
def main():
    global running
    print("=" * 60)
    print("  Prophet v24 — Paper Trading Engine")
    print(f"  Capital: ¥{CAPITAL:,}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load models
    models = {}
    print("\nLoading models...")
    for sym_key in SYMBOLS:
        mp = os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')
        if os.path.exists(mp):
            with open(mp, 'rb') as f: models[sym_key] = pickle.load(f)
            print(f"  {sym_key}: loaded")
        else:
            print(f"  {sym_key}: missing {mp}")

    # Load state
    state = load_state()
    print(f"\nCash: ¥{state['cash']:,.0f}")
    if state['positions']:
        for k, v in state['positions'].items():
            print(f"  {k}: {v['dir']} {v['vol']}手 @ {v['entry']}")
    
    print(f"\n{'='*60}")
    print(f"  Waiting for trading hours...")
    print(f"{'='*60}\n")

    traded_today = set()
    today_str = datetime.now().strftime('%Y%m%d')

    while running:
        now = datetime.now()
        current_date = now.strftime('%Y%m%d')
        if current_date != today_str:
            traded_today.clear()
            today_str = current_date

        if not is_trading_time(now):
            if now.second == 0:
                print(f"[{now.strftime('%H:%M:%S')}] Non-trading, waiting...")
            for _ in range(min(BAR_INTERVAL, 30)):
                if not running: break
                time.sleep(1)
            continue

        print(f"\n[{now.strftime('%H:%M:%S')}] Scanning...")

        # Fetch prices + generate signals
        for sym_key, cfg in SYMBOLS.items():
            if sym_key in traded_today or sym_key in state['positions']:
                continue

            df = fetch_history(cfg['code'], 1200)
            if df is None or len(df) < 100: continue

            f = build_features(df, len(df)-1, 60)
            if f is None: continue

            model_key = sym_key
            if model_key not in models: continue

            try:
                p = models[model_key].predict_proba(f.reshape(1, -1))[0]
                prob = float(p[1])
            except: continue

            signal_dir = 'LONG' if prob > 0.5 else 'SHORT'
            confidence = prob if prob > 0.5 else (1 - prob)
            price = float(df.iloc[-1]['close'])
            pos_size = calc_position_size(sym_key, df)

            # Calculate stop (ATR or struct)
            stop_type = cfg.get('stop_type', 'struct')
            if stop_type == 'atr':
                # ATR-based stop
                atr_vals = []
                for i in range(max(0, len(df)-20), len(df)):
                    atr_vals.append(abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low'])))
                atr = np.mean(atr_vals) if atr_vals else price * 0.01
                stop_mult = cfg.get('stop_mult', 1.5)
                stop_dist = atr * stop_mult
                if signal_dir == 'LONG':
                    stop_price = price - stop_dist
                else:
                    stop_price = price + stop_dist
            else:
                # Struct stop
                struct_n = cfg.get('struct_n', 20)
                if signal_dir == 'LONG':
                    lows = [float(df.iloc[k]['low']) for k in range(max(0, len(df)-struct_n), len(df))]
                    stop_price = min(lows)
                else:
                    highs = [float(df.iloc[k]['high']) for k in range(max(0, len(df)-struct_n), len(df))]
                    stop_price = max(highs)

            rr_val = cfg['rr']
            margin = pos_size * price * cfg['multiplier'] * 0.15
            emoji = '\U0001F7E2' if signal_dir == 'LONG' else '\U0001F534'

            if pos_size == 0: continue

            print(f"  {emoji} {sym_key} {signal_dir} {pos_size}手 @ {price:.0f} "
                  f"保证金{margin/10000:.1f}万 conf={confidence:.2f} "
                  f"RR=1:{rr_val} STOP={stop_price:.0f}")

            # Execute trade
            cost_pct = cfg['cost']
            margin_used = margin
            if margin_used > state['cash']:
                print(f"    ⚠️ 保证金不足 (需{margin_used/10000:.1f}万, 可用{state['cash']/10000:.1f}万)")
                continue

            # Calculate take-profit
            if signal_dir == 'LONG':
                tp_price = price + (price - stop_price) * rr_val
            else:
                tp_price = price - (stop_price - price) * rr_val

            state['positions'][sym_key] = {
                'dir': signal_dir, 'vol': pos_size,
                'entry': price, 'entry_time': now.isoformat(),
                'stop': stop_price, 'take_profit': tp_price,
                'cost_pct': cost_pct
            }
            state['cash'] -= margin_used * cost_pct  # Commission
            traded_today.add(sym_key)
            save_state(state)
            print(f"    ✅ 开仓成功 STOP={stop_price:.0f} TP={tp_price:.0f}")

        # Check stops and take-profits for existing positions
        # Use real-time spot prices, not daily OHLC (which is stale intraday)
        for sym_key in list(state['positions'].keys()):
            pos = state['positions'][sym_key]
            cfg = SYMBOLS[sym_key]
            
            # Get real-time price
            try:
                spot = ak.futures_zh_spot(symbol=sym_key.upper(), market='DCE')
                if spot is None or len(spot) == 0:
                    spot = ak.futures_spot_em(symbol=sym_key.upper())
                if spot is not None and len(spot) > 0:
                    row = spot.iloc[0]
                    current_price = float(row['最新价'] if '最新价' in row else row['price'])
                    current_high = float(row.get('最高价', current_price))
                    current_low = float(row.get('最低价', current_price))
                else:
                    continue
            except:
                # Fallback: use daily data
                df = fetch_history(cfg['code'], 50)
                if df is None or len(df) == 0: continue
                current_price = float(df.iloc[-1]['close'])
                current_high = float(df.iloc[-1]['high'])
                current_low = float(df.iloc[-1]['low'])

            entry = pos['entry']
            stop = pos['stop']
            tp = pos['take_profit']
            d = pos['dir']
            vol = pos['vol']

            hit_stop = (d == 'LONG' and current_low <= stop) or \
                       (d == 'SHORT' and current_high >= stop)
            hit_tp = (d == 'LONG' and current_high >= tp) or \
                     (d == 'SHORT' and current_low <= tp)

            if hit_stop or hit_tp:
                exit_type = 'STOP' if hit_stop else 'TP'
                exit_price = stop if hit_stop else tp
                
                if d == 'LONG':
                    pnl_pct = (exit_price - entry) / entry - cfg['cost'] * 2
                else:
                    pnl_pct = (entry - exit_price) / entry - cfg['cost'] * 2
                
                pnl_amount = pnl_pct * vol * entry * cfg['multiplier'] * 0.15
                margin = vol * entry * cfg['multiplier'] * 0.15
                state['cash'] += margin + pnl_amount

                trade = {
                    'sym': sym_key, 'dir': d, 'entry': entry,
                    'exit': exit_price, 'exit_type': exit_type,
                    'vol': vol, 'pnl_pct': pnl_pct,
                    'pnl_amount': pnl_amount,
                    'entry_time': pos['entry_time'],
                    'exit_time': now.isoformat()
                }
                state['trades'].append(trade)
                emoji2 = '\U0001F7E2' if pnl_amount > 0 else '\U0001F534'
                print(f"  {emoji2} {exit_type} {sym_key} {d} {vol}手 "
                      f"@{exit_price:.0f} PnL={pnl_amount:+,.0f} ({pnl_pct:+.2%}) "
                      f"余额=¥{state['cash']:,.0f}")
                del state['positions'][sym_key]
                save_state(state)

        # Record equity
        total = state['cash']
        for k, p in state['positions'].items():
            cfg = SYMBOLS[k]
            total += p['vol'] * p['entry'] * cfg['multiplier'] * 0.15  # margin
        state['equity_history'].append({
            'time': now.isoformat(), 'equity': total
        })

        # Wait for next bar
        for _ in range(BAR_INTERVAL):
            if not running: break
            time.sleep(1)

    print("\nShutting down...")
    save_state(state)
    print(f"Final equity: ¥{state['cash']:,.0f}")
    print(f"Total trades: {len(state['trades'])}")

if __name__ == '__main__':
    main()
