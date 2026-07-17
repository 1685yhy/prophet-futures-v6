#!/usr/bin/env python3
"""Prophet Futures — Paper Trading Engine V31
动态止盈止损: 模型趋势逆转退出 + 移动止损 + 保本止损
LH: 紧止损(ATR×0.8) + 模型退出(prob<0.35/>0.65) + 移动止损(盈利>2ATR)
JM: 宽止损(ATR×1.8) + 模型退出(prob<0.30/>0.70) + 移动止损(盈利>3ATR)
"""
import sys, os, time, json, signal
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
from trade_logger import log_trade
import xgboost as xgb
import pickle
try:
    from feishu_send import send_alert, send_scan
except Exception as _fe:
    import sys
    print(f"[WARN] feishu_send import failed: {_fe}. Feishu alerts disabled.", file=sys.stderr, flush=True)

# ===== CONFIG =====
SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'cost': 0.0006, 'multiplier': 16,
        'max_pos': 6,
        # V31 dynamic exit params (LH: tighter, faster)
        'hard_atr': 1.5,       # V31 硬止损 ATR倍数
        'trail_atr': 2.0,      # 移动止损启动: 盈利>2ATR
        'be_atr': 1.0,         # 保本启动: 盈利>1ATR
        'trail_dist': 1.5,     # 移动止损距离 ATR倍数
        'rr': 4.0,             # 止盈风险比 (V31修复: 不应用trail_atr+trail_dist当RR)
        'model_low': 0.35,     # LONG exit: prob<0.35
        'model_high': 0.65,    # SHORT exit: prob>0.65
        'confirm_bars': 2,     # 反转确认K线数
        'min_hold': 3,         # 最小持仓K线(扫描周期)
    },
    'jm2609': {
        'code': 'JM0', 'name': 'JM', 'cost': 0.0011, 'multiplier': 60,
        'max_pos': 4,
        # V31 dynamic exit params (JM: wider, slower — higher volatility)
        'hard_atr': 1.8,       # 硬止损 ATR倍数 (JM波动大)
        'trail_atr': 3.0,      # 移动止损启动: 盈利>3ATR
        'be_atr': 2.0,         # 保本启动: 盈利>2ATR
        'trail_dist': 2.5,     # 移动止损距离
        'rr': 3.5,             # 止盈风险比 (V31修复: 不应用trail_atr+trail_dist当RR)
        'model_low': 0.30,     # LONG exit: prob<0.30
        'model_high': 0.70,    # SHORT exit: prob>0.70
        'confirm_bars': 3,     # 反转确认K线数 (JM趋势强需更多确认)
        'min_hold': 5,         # 最小持仓K线
    },
}

CAPITAL = 300000
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state_v31.json')
TRADE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_journal.log')
VOL_LOOKBACK = 20
MAX_LEVERAGE = 3.0
MIN_LEVERAGE = 0.5
BAR_INTERVAL = 60  # 1 min scan

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

def calc_atr(df, idx, period=20):
    if idx < period: return None
    vals = [abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
            for i in range(idx-period+1, idx+1)]
    return np.mean(vals)

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

def log_event(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(TRADE_LOG, 'a') as f:
        f.write(f"[{ts}] {msg}\n")

def is_trading_time(now=None):
    if now is None: now = datetime.now()
    wd = now.weekday()
    if wd >= 5: return False
    h, m = now.hour, now.minute
    t = h * 60 + m
    return (540 <= t < 615) or (630 <= t < 690) or (810 <= t < 900)

# ===== STATE =====
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'cash': CAPITAL, 'positions': {}, 'trades': [], 'equity_history': []}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2, default=str)

# ===== STARTUP REFRESH (每次唤醒自动更新止损止盈) =====
def refresh_stops_on_startup(state):
    """用最新日线数据重新计算所有持仓的止损止盈。只紧不松。"""
    import akshare as ak
    changed = False
    for sym_key in list(state['positions'].keys()):
        pos = state['positions'][sym_key]
        cfg = SYMBOLS.get(sym_key)
        if not cfg: continue
        
        try:
            df = ak.futures_main_sina(symbol=cfg['code'])
            df = df.rename(columns={'日期':'date','开盘价':'open','最高价':'high','最低价':'low','收盘价':'close'})
            for c in ['open','high','low','close']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['close']).tail(60)
            if len(df) < 20: continue
            
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            atr = df['tr'].tail(20).mean()
            price = float(df['close'].iloc[-1])
            stop_dist = atr * cfg['hard_atr']
            rr = cfg.get('rr', 3.5)
            
            old_stop = pos.get('stop', 0)
            old_tp = pos.get('take_profit', 0)
            
            if pos['dir'] == 'LONG':
                new_stop = price - stop_dist
                new_stop = max(new_stop, old_stop)  # 只紧不松
                actual_dist = price - new_stop
                new_tp = price + actual_dist * rr
            else:
                new_stop = price + stop_dist
                new_stop = min(new_stop, old_stop)  # 只紧不松
                actual_dist = new_stop - price
                new_tp = price - actual_dist * rr
            
            pos['stop'] = round(new_stop, 1)
            pos['take_profit'] = round(new_tp, 1)
            new_s = pos['stop']
            new_t = pos['take_profit']
            print(f'  [Refresh] {sym_key}: stop {old_stop}→{new_s} | tp {old_tp}→{new_t} (RR={rr})')
            changed = True
        except Exception as e:
            print(f'  [Refresh] {sym_key}: FAILED ({e})')
    
    if changed:
        save_state(state)
        print(f'  [Refresh] State saved.')
    return changed

# ===== V31 DYNAMIC EXIT =====
def check_V31_exit(sym_key, pos, models, df, now):
    """
    V31 dynamic exit logic:
    1. Model reversal: re-predict, if prob crosses threshold, exit
    2. Trailing stop: if in profit > trail_atr ATR, trail the stop
    3. Breakeven stop: if in profit > be_atr ATR, move stop to entry
    4. Hard stop: ATR × hard_atr
    Returns: (should_exit, exit_price, exit_reason) or (False, None, None)
    """
    cfg = SYMBOLS[sym_key]
    d = pos['dir']
    entry = pos['entry']
    
    # Get current price and ATR
    if df is None or len(df) < 20:
        return False, None, None
    
    price = float(df.iloc[-1]['close'])
    try:
        from realtime_data import get_realtime_quote
        rt = get_realtime_quote(sym_key)
        if rt and rt.get('price', 0) > 0: price = rt['price']
    except: pass
    atr = calc_atr(df, len(df)-1, 20)
    if atr is None or price <= 0:
        return False, None, None
    
    # Count holding bars from entry_time
    bar_count = pos.get('bar_count', 0) + 1
    pos['bar_count'] = bar_count
    
    # Re-predict with model for reversal check
    model_exit = False
    model_key = sym_key
    if model_key in models:
        feats = build_features(df, len(df)-1, 60)
        if feats is not None:
            try:
                prob = float(models[model_key].predict_proba(feats.reshape(1, -1))[0][1])
                # Store reversal counter in position
                if d == 'LONG':
                    if prob < cfg['model_low']:
                        pos['_rev_count'] = pos.get('_rev_count', 0) + 1
                    else:
                        pos['_rev_count'] = 0
                else:
                    if prob > cfg['model_high']:
                        pos['_rev_count'] = pos.get('_rev_count', 0) + 1
                    else:
                        pos['_rev_count'] = 0
                
                model_exit = (pos.get('_rev_count', 0) >= cfg['confirm_bars']) and (bar_count >= cfg['min_hold'])
            except:
                pass
    
    # Hard stop
    hard_stop_dist = atr * cfg['hard_atr']
    if d == 'LONG':
        hard_stop = price - hard_stop_dist
        
        # Trailing stop (only tighten, never loosen)
        trail_stop = pos.get('_trail_stop', price - atr * (cfg['hard_atr'] + 0.5))
        if bar_count >= cfg['min_hold'] and price > entry + atr * cfg['trail_atr']:
            new_trail = price - atr * cfg['trail_dist']
            trail_stop = max(trail_stop, new_trail)
        if bar_count >= cfg['min_hold'] and price > entry + atr * cfg['be_atr']:
            trail_stop = max(trail_stop, entry)  # Breakeven
        pos['_trail_stop'] = trail_stop
        
        effective_stop = max(hard_stop, trail_stop)
        # Also check old-style stop/tp for backward compat
        old_stop = pos.get('stop', 0)
        old_tp = pos.get('take_profit', 1e9)
        effective_stop = max(effective_stop, old_stop)
        
        exit_trigger = (price <= effective_stop) or (price >= old_tp) or model_exit
        if exit_trigger:
            if model_exit: reason = 'MODEL'
            elif price >= old_tp: reason = 'TP'
            elif trail_stop >= hard_stop and trail_stop >= old_stop: reason = 'TRAIL'
            elif old_stop > hard_stop and price <= old_stop: reason = 'STOP'
            else: reason = 'HARD'
            exit_price = max(effective_stop, old_stop) if price <= max(effective_stop, old_stop) else old_tp if price >= old_tp else price
            return True, exit_price, reason
    else:
        hard_stop = price + hard_stop_dist
        
        trail_stop = pos.get('_trail_stop')
        if trail_stop is None or trail_stop == 0:  # 0 是之前 bug 的残留值
            trail_stop = price + atr * (cfg['hard_atr'] + 0.5)
        if bar_count >= cfg['min_hold'] and price < entry - atr * cfg['trail_atr']:
            new_trail = price + atr * cfg['trail_dist']
            trail_stop = min(trail_stop, new_trail)
        if bar_count >= cfg['min_hold'] and price < entry - atr * cfg['be_atr']:
            trail_stop = min(trail_stop, entry)
        pos['_trail_stop'] = trail_stop
        
        effective_stop = min(hard_stop, trail_stop)
        old_stop = pos.get('stop', 1e9)
        old_tp = pos.get('take_profit', 0)
        effective_stop = min(effective_stop, old_stop)
        
        exit_trigger = (price >= effective_stop) or (price <= old_tp) or model_exit
        if exit_trigger:
            if model_exit: reason = 'MODEL'
            elif price <= old_tp: reason = 'TP'
            elif trail_stop <= hard_stop and trail_stop <= old_stop: reason = 'TRAIL'
            elif old_stop < hard_stop and price >= old_stop: reason = 'STOP'
            else: reason = 'HARD'
            exit_price = min(effective_stop, old_stop) if price >= min(effective_stop, old_stop) else old_tp if price <= old_tp else price
            return True, exit_price, reason
    
    return False, None, None

# ===== MAIN =====
def main():
    global running
    # PID锁
    import fcntl
    lock_fd = open('/tmp/paper_v31.lock', 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid())); lock_fd.flush()
    except (IOError, OSError):
        print('V25已在运行,退出')
        sys.exit(0)
    
    print("=" * 60)
    print("  Prophet V31 — Dynamic Stop/Take-Profit Engine")
    print(f"  Capital: ¥{CAPITAL:,}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  LH: 硬止损ATR×1.5 | 模型退出prob 0.35/0.65 | 移动止损2ATR")
    print(f"  JM: 硬止损ATR×1.8 | 模型退出prob 0.30/0.70 | 移动止损3ATR")
    print("=" * 60)

    # Load models
    models = {}
    print("\nLoading models...")
    for sym_key in SYMBOLS:
        mp = os.path.join(MODEL_DIR, sym_key+'_xgb.pkl')
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
            cfg = SYMBOLS.get(k, {})
            # Initialize V31 tracking fields
            v.setdefault('_rev_count', 0)
            if v.get('_trail_stop', 0) == 0:  # 清理旧bug残留
                if '_trail_stop' in v: del v['_trail_stop']
            v.setdefault('bar_count', 0)
            print(f"  {k}: {v['dir']} {v['vol']}手 @ {v['entry']} "
                  f"stop={v.get('stop','?')} tp={v.get('take_profit','?')}")
    
    # 每次启动自动用最新数据刷新止损止盈 (只紧不松)
    print(f"\n{'='*60}")
    print(f"  启动刷新: 用最新日线数据更新止损止盈...")
    refresh_stops_on_startup(state)
    print(f"  刷新后现金: ¥{state['cash']:,.0f}")
    if state['positions']:
        for k, v in state['positions'].items():
            stop_d = abs(v['entry'] - v['stop'])
            tp_d = abs(v['entry'] - v['take_profit'])
            rr = tp_d / stop_d if stop_d > 0 else 0
            print(f"  {k}: {v['dir']} {v['vol']}手 stop={v['stop']} tp={v['take_profit']} RR={rr:.1f}")
    
    print(f"\n{'='*60}")
    print(f"  V31 基线止盈止损已就绪")
    print(f"  Waiting for trading hours...")
    print(f"{'='*60}\n")

    # 恢复检查：启动时补执行已穿止损
    try:
        from trader_recovery import run_recovery
        run_recovery(STATE_FILE, SYMBOLS, "V31")
    except Exception as e:
        print(f"  [V31] 恢复检查跳过: {e}")

    traded_today = set()
    today_str = datetime.now().strftime('%Y%m%d')

    while running:
        now = datetime.now()
        current_date = now.strftime('%Y%m%d')
        if current_date != today_str:
            traded_today.clear()
            today_str = current_date
            # Reset bar counters for new day
            for p in state['positions'].values():
                p['bar_count'] = 0

        if not is_trading_time(now):
            if now.second == 0:
                print(f"[{now.strftime('%H:%M:%S')}] Non-trading, waiting...")
            for _ in range(min(BAR_INTERVAL, 30)):
                if not running: break
                time.sleep(1)
            continue

        print(f"\n[{now.strftime('%H:%M:%S')}] V31 Scanning...")

        # === 保证金强平检查 ===
        try:
            from margin_call import check_margin_call
            triggered, msg = check_margin_call(STATE_FILE, SYMBOLS)
            if triggered:
                print(f"  💥 {msg}")
                state = load_state()
                try:
                    from feishu_send import send_alert
                    send_alert("🚨 强平触发", msg, color="red", pin=True)
                except: pass
            elif "⚠️" in msg:
                print(f"  ⚠️ {msg}")
        except Exception as e:
            pass

        # Fetch daily data once per scan for all symbols
        daily_dfs = {}
        for sym_key in SYMBOLS:
            cfg = SYMBOLS[sym_key]
            df = fetch_history(cfg['code'], 1200)
            if df is not None and len(df) >= 100:
                daily_dfs[sym_key] = df

        # ===== CHECK EXISTING POSITIONS (V31 DYNAMIC EXIT) =====
        for sym_key in list(state['positions'].keys()):
            pos = state['positions'][sym_key]
            cfg = SYMBOLS.get(sym_key)
            if not cfg: continue
            
            df = daily_dfs.get(sym_key)
            should_exit, exit_price, reason = check_V31_exit(sym_key, pos, models, df, now)
            
            if should_exit:
                d = pos['dir']
                entry = pos['entry']
                vol = pos['vol']
                
                if d == 'LONG':
                    pnl_pct = (exit_price - entry) / entry - cfg['cost'] * 2
                else:
                    pnl_pct = (entry - exit_price) / entry - cfg['cost'] * 2
                
                pnl_amount = pnl_pct * vol * entry * cfg['multiplier'] * 0.15
                margin = vol * entry * cfg['multiplier'] * 0.15
                state['cash'] += margin + pnl_amount

                trade = {
                    'sym': sym_key, 'dir': d, 'entry': entry,
                    'exit': exit_price, 'exit_type': reason,
                    'vol': vol, 'pnl_pct': pnl_pct,
                    'pnl_amount': pnl_amount,
                    'entry_time': pos['entry_time'],
                    'exit_time': now.isoformat()
                }
                state['trades'].append(trade)
                
                emoji = '🟢' if pnl_amount > 0 else '🔴'
                dir_cn = '做多' if d == 'LONG' else '做空'
                print(f"  {emoji} [{reason}] {sym_key} {d} {vol}手 "
                      f"@{exit_price:.0f} PnL={pnl_amount:+,.0f} ({pnl_pct:+.2%}) "
                      f"余额=¥{state['cash']:,.0f}")
                log_event(f"{reason} {sym_key} {d} {vol}手 @{entry}→{exit_price:.0f} PnL={pnl_amount:+,.0f}")
                reason_cn = {'MODEL': '模型逆转', 'TP': '止盈', 'TRAIL': '移动止损', 
                             'STOP': '止损', 'HARD': '硬止损'}.get(reason, reason)
                reason_bi = '%s|%s' % (reason_cn, reason)
                log_trade('V31', sym_key, 'CLOSE', d, entry, exit_price, vol, pnl_amount,
                          reason_bi, '反转计数=%d/%d' % (pos.get('_rev_count', 0), cfg['confirm_bars']),
                          state['cash'], state['cash'])
                del state['positions'][sym_key]
                save_state(state)
                
                # Feishu alert
                try:
                    pnl_sign = '+' if pnl_amount > 0 else ''
                    color = 'green' if pnl_amount > 0 else 'red'
                    send_alert(
                        f"{emoji} {reason_cn} | {sym_key}",
                        f"{dir_cn} {vol}手 @ {entry} → {exit_price:.0f}\n"
                        f"盈亏: {pnl_sign}{pnl_amount:,.0f} ({pnl_pct:+.1%})\n"
                        f"余额: ¥{state['cash']:,.0f}",
                        color=color, pin=True
                    )
                except: pass

        # ===== SIGNAL GENERATION (NEW ENTRIES) =====
        for sym_key, cfg in SYMBOLS.items():
            if sym_key in traded_today or sym_key in state['positions']:
                continue

            df = daily_dfs.get(sym_key)
            if df is None: continue

            feats = build_features(df, len(df)-1, 60)
            if feats is None: continue

            model_key = sym_key
            if model_key not in models: continue

            try:
                p = models[model_key].predict_proba(feats.reshape(1, -1))[0]
                prob = float(p[1])
            except: continue

            signal_dir = 'LONG' if prob > 0.5 else 'SHORT'
            confidence = prob if prob > 0.5 else (1 - prob)
            price = float(df.iloc[-1]['close'])
            # 开仓用实时价
            try:
                from realtime_data import get_realtime_quote
                rt = get_realtime_quote(sym_key)
                if rt and rt.get('price', 0) > 0: price = rt['price']
            except: pass
            pos_size = calc_position_size(sym_key, df)
            if pos_size == 0: continue

            # V31 entry stop: hard_atr-based
            atr = calc_atr(df, len(df)-1, 20)
            if atr is None: continue
            
            stop_dist = atr * cfg['hard_atr']
            if signal_dir == 'LONG':
                entry_stop = price - stop_dist
            else:
                entry_stop = price + stop_dist

            # Correct TP using actual RR config (V31 fix: was using trail_atr+trail_dist)
            display_rr = cfg.get('rr', 3.5)  # fallback 3.5 if rr not set
            if signal_dir == 'LONG':
                display_tp = price + (price - entry_stop) * display_rr
            else:
                display_tp = price - (entry_stop - price) * display_rr

            margin = pos_size * price * cfg['multiplier'] * 0.15
            emoji = '🟢' if signal_dir == 'LONG' else '🔴'
            dir_cn = '做多' if signal_dir == 'LONG' else '做空'
            risk_pts = abs(price - entry_stop)
            reward_pts = abs(display_tp - price)
            
            print(f"\n  ═══ V31操作建议 ═══")
            print(f"  {emoji} {sym_key.upper()} {dir_cn}信号")
            print(f"  入场: {price:.0f}  |  硬止损: {entry_stop:.0f}  |  参考目标: {display_tp:.0f}")
            print(f"  仓位: {pos_size}手  |  保证金: ¥{margin/10000:.1f}万  |  置信度: {confidence:.1%}")
            print(f"  动态管理: 移动止损{cfg['trail_atr']}ATR启动 | 保本{cfg['be_atr']}ATR")
            print(f"  {'─'*42}")

            # Margin check
            cost_pct = cfg['cost']
            margin_used = margin
            if margin_used > state['cash']:
                print(f"    ⚠️ 保证金不足 (需{margin_used/10000:.1f}万, 可用{state['cash']/10000:.1f}万)")
                continue

            # Check stop already hit today
            try:
                today_str2 = datetime.now().strftime('%Y-%m-%d')
                minute_df = ak.futures_zh_minute_sina(symbol=sym_key.upper(), period='5')
                if minute_df is not None and len(minute_df) > 0:
                    minute_df['dt'] = pd.to_datetime(minute_df['datetime'])
                    today_df = minute_df[minute_df['dt'].dt.strftime('%Y-%m-%d') == today_str2]
                    if len(today_df) > 0:
                        today_high = float(today_df['high'].max())
                        today_low = float(today_df['low'].min())
                        already_stopped = (signal_dir == 'LONG' and today_low <= entry_stop) or \
                                          (signal_dir == 'SHORT' and today_high >= entry_stop)
                        if already_stopped:
                            print(f"    ⚠️ 今日已触发止损，跳过开仓")
                            continue
            except: pass
            
            state['positions'][sym_key] = {
                'dir': signal_dir, 'vol': pos_size,
                'entry': price, 'entry_time': now.isoformat(),
                'stop': entry_stop, 'take_profit': display_tp,
                'cost_pct': cost_pct,
                '_rev_count': 0, 'bar_count': 0,
            }
            state['cash'] -= margin_used
            traded_today.add(sym_key)
            save_state(state)
            print(f"    ✅ 开仓 [V31动态] 硬止损={entry_stop:.0f} 参考目标={display_tp:.0f}")
            
            log_event(f"OPEN {sym_key} {signal_dir} {pos_size}手 @{price:.0f} HARD_STOP={entry_stop:.0f}")
            log_trade('V31', sym_key, 'OPEN', signal_dir, price, 0, pos_size, 0,
                      '模型信号|SIGNAL', '概率=%.3f_置信=%.2f_止损=%.0f' % (prob, confidence, entry_stop),
                      state['cash'], state['cash'])
            
            # Feishu alert
            try:
                send_alert(
                    f"{emoji} 开仓 [V25] | {sym_key}",
                    f"{dir_cn} {pos_size}手 @ {price:.0f}\n"
                    f"硬止损: {entry_stop:.0f}\n"
                    f"动态管理: 移动{cfg['trail_atr']}ATR | 保本{cfg['be_atr']}ATR\n"
                    f"保证金: {margin_used/10000:.1f}万",
                    color='blue', pin=True
                )
            except: pass

        # Record equity
        total = state['cash']
        for k, p in state['positions'].items():
            c2 = SYMBOLS.get(k, {}).get('multiplier', 10)
            total += p['vol'] * p['entry'] * c2 * 0.15
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
