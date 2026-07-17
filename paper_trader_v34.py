#!/usr/bin/env python3
"""
Prophet Futures — Paper Trading Engine V32
模型驱动动态交易: 持有/加仓/减仓/反手
V34 = 紧止损宽止盈 + v31模型 + 模型动态决策
独立状态文件 paper_state_v34.json
"""
import sys, os, time, json, signal, pickle
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
from trade_logger import log_trade

try:
    from feishu_send import send_alert, send_scan
except Exception as _fe:
    import sys
    print(f"[WARN] feishu_send import failed: {_fe}. Feishu alerts disabled.", file=sys.stderr, flush=True)

# ===== CONFIG =====
SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'cost': 0.0006, 'multiplier': 16,
        'max_pos': 6, 'max_total': 12,
        'atr_stop': 1.0,  # V34 网格最优(0717)
        'rr': 6.0,         # V34 宽止盈
        'add_conf': 0.70,  # V34 更难加仓
        'add_atr': 3.0,    # 加仓: 浮盈>3ATR
        'reduce_conf': 0.45,  # V34 更早减仓
        'reverse_conf': 0.0,  # V34 反手关闭
        'trail_atr': 1.5,     # V34 更早追踪
        'be_atr': 0.8,        # V34 更早保本
        'min_hold': 5,        # 最小持仓bar
    },
}

CAPITAL = 300000
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_state_v34.json')
TRADE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_journal_v32.log')
BAR_INTERVAL = 60  # 1 min scan

running = True
def signal_handler(sig, frame):
    global running; running = False
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ===== DATA =====
def fetch_history(code, days=1200):
    end = datetime.now(); start = end - timedelta(days=days)
    try:
        df = ak.futures_main_sina(symbol=code, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except Exception as e:
        print(f'  Fetch error {code}: {e}'); return None

def build_features(df, idx, window=60):
    if idx < window+5: return None
    w = df.iloc[idx-window:idx+1]
    c = w['close'].values.astype(float); o = w['open'].values.astype(float)
    h = w['high'].values.astype(float); l = w['low'].values.astype(float)
    v = w['volume'].values.astype(float); oi = w['oi'].values.astype(float)
    oc = float((o[-1]-c[-2])/c[-2]) if idx>=1 else 0.0
    f = [oc, abs(oc)]
    for lag in [1,3,5,10,20]: f.append(float((c[-1]-c[-lag-1])/c[-lag-1] if len(c)>lag else 0))
    for p in [5,10,20,60]: ma=np.mean(c[-min(p,len(c)):]);f.append(float((c[-1]-ma)/ma))
    f.append(float(np.std(c[-20:])/np.mean(c[-20:])))
    f.append(float((h[-1]-l[-1])/c[-1]))
    vm=np.mean(v[-20:])if np.mean(v[-20:])>0 else 1;f.append(float(v[-1]/vm))
    f.append(float(oi[-1]/np.mean(oi[-20:]))if len(oi)>=20 and np.mean(oi[-20:])>0 else 1)
    e12=c[-1];e26=c[-1]
    for j in range(len(c)-2,-1,-1):e12=(2/13)*c[j]+(11/13)*e12;e26=(2/27)*c[j]+(25/27)*e26
    f.append(float((e12-e26)/c[-1]))
    dd=np.diff(c[-15:]);g=float(dd[dd>0].sum())if len(dd[dd>0])>0 else 0
    lo=float(abs(dd[dd<0].sum()))if len(dd[dd<0])>0 else 1e-10
    f.append(float(100-100/(1+g/lo)if lo>0 else 50))
    bb=np.std(c[-20:]);m20=np.mean(c[-20:]);f.append(float((c[-1]-m20)/(2*bb+1e-10)))
    f.append(float(c[-1]/1000.0))
    # V34: +3维基本面(期现价差/猪粮比Z/周指变化) — 与v5_backtest_fund一致
    try:
        from fund_features import get_fund_features
        f.extend(get_fund_features(float(c[-1])))
    except Exception:
        f.extend([0.0, 0.0, 0.0])
    return np.array(f, dtype=np.float32)

def calc_atr(df, idx, period=20):
    if idx < period: return None
    vals = [abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])) for i in range(idx-period+1, idx+1)]
    return np.mean(vals)

# ===== STATE =====
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'cash': CAPITAL, 'positions': {}, 'trades': [], 'equity_history': []}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2, default=str)

def log_event(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(TRADE_LOG, 'a') as f: f.write('[%s] %s\n' % (ts, msg))

def is_trading_time(now=None):
    if now is None: now = datetime.now()
    if now.weekday() >= 5: return False
    t = now.hour * 60 + now.minute
    return (540 <= t < 615) or (630 <= t < 690) or (810 <= t < 900)

def calc_position_size(capital, price, atr, cfg):
    atr_pct = atr/price
    if atr_pct < 0.01: lev = 3.0
    elif atr_pct < 0.02: lev = 2.0
    elif atr_pct < 0.03: lev = 1.5
    elif atr_pct < 0.05: lev = 0.5
    else: lev = 0
    base = max(1, int(lev * (cfg['max_pos']//2))) if lev > 0 else 0
    ratio = capital / CAPITAL
    lots = max(0, int(base * ratio))
    if capital < 100000: lots = max(1, lots//2) if lots>0 else 0
    return min(lots, cfg['max_pos'])

# ===== MAIN =====
def main():
    global running
    # PID锁: 防止重复启动
    import fcntl
    lock_fd = open('/tmp/paper_v34.lock', 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid())); lock_fd.flush()
    except (IOError, OSError):
        print('V34已在运行,退出')
        sys.exit(0)
    
    print('=' * 60)
    print('  Prophet V34 — 动态加仓/减仓/反手引擎')
    print('  Capital: ¥%s  |  %s' % (format(CAPITAL, ','), datetime.now().strftime('%Y-%m-%d %H:%M')))
    print('  LH: ATR×0.5 加仓>70%&3ATR 反手OFF')
    print('=' * 60)

    # Load models
    models = {}
# V34 uses newly trained model from v5 backtest
    MODEL_MAP = {'lh2609': 'v34_fund_xgb.pkl'}
    for sym_key in SYMBOLS:
        mp = os.path.join(MODEL_DIR, MODEL_MAP.get(sym_key, sym_key+'_xgb.pkl'))
        if os.path.exists(mp):
            with open(mp, 'rb') as f: models[sym_key] = pickle.load(f)
            print('  %s: loaded' % sym_key)

    # Load state
    state = load_state()
    print('\nCash: ¥%s' % format(int(state['cash']), ','))
    if state['positions']:
        for k, v in state['positions'].items():
            if isinstance(v, list):
                total_vol = sum(p['vol'] for p in v)
                d = v[0]['dir']
                ae = sum(p['entry']*p['vol'] for p in v)/total_vol
                print('  %s: %s %d手(%d仓) @ %.0f' % (k, d, total_vol, len(v), ae))
            else:
                print('  %s: %s %d手 @ %s' % (k, v['dir'], v['vol'], v['entry']))

    print('\n' + '='*60)
    print('  V34 引擎就绪 (独立状态: paper_state_v34.json)')
    print('='*60 + '\n')
    
    # 恢复检查：启动时补执行已穿止损
    try:
        from trader_recovery import run_recovery
        run_recovery(STATE_FILE, SYMBOLS, 'V32')
    except Exception as e:
        print(f'  [V34] 恢复检查跳过: {e}')
    
    traded_today = set()
    today_str = datetime.now().strftime('%Y%m%d')

    while running:
        now = datetime.now()
        current_date = now.strftime('%Y%m%d')
        if current_date != today_str:
            traded_today.clear()
            today_str = current_date
            for v in state['positions'].values():
                if isinstance(v, list):
                    for p in v: p['bar_count'] = p.get('bar_count', 0)
                else:
                    v['bar_count'] = v.get('bar_count', 0)

        if not is_trading_time(now):
            if now.second == 0:
                print('[%s] Non-trading, waiting...' % now.strftime('%H:%M:%S'))
            for _ in range(min(BAR_INTERVAL, 30)):
                if not running: break
                time.sleep(1)
            continue

        print('\n[%s] V34 Scanning...' % now.strftime('%H:%M:%S'))

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

        # Fetch daily data
        daily_dfs = {}
        for sym_key in SYMBOLS:
            cfg = SYMBOLS[sym_key]
            df = fetch_history(cfg['code'], 1200)
            if df is not None and len(df) >= 100:
                daily_dfs[sym_key] = df

        # ===== V34 管理现有持仓 =====
        positions = state['positions']
        for sym_key in list(positions.keys()):
            pos_list = positions[sym_key]  # list of sub-positions
            cfg = SYMBOLS.get(sym_key)
            if not cfg: continue
            
            df = daily_dfs.get(sym_key)
            if df is None: continue
            
            price = float(df.iloc[-1]['close'])
            # 止损检查用实时价(日线收盘可能是昨天价)
            try:
                from realtime_data import get_realtime_quote
                rt = get_realtime_quote(sym_key)
                if rt and rt.get('price', 0) > 0:
                    price = rt['price']
            except:
                pass
            high = price
            low = price
            atr = calc_atr(df, len(df)-1, 20)
            if atr is None: continue
            
            feats = build_features(df, len(df)-1, 60)
            if feats is None: continue
            model_key = sym_key
            if model_key not in models: continue
            try:
                prob = float(models[model_key].predict_proba(feats.reshape(1,-1))[0][1])
            except: continue
            
            cur_dir = 'LONG' if prob > 0.5 else 'SHORT'
            conf = prob if prob > 0.5 else 1-prob
            
            surviving = []
            for pos in pos_list:
                d = pos['dir']; entry = pos['entry']; vol = pos['vol']
                pnl_pts = price-entry if d=='LONG' else entry-price
                pnl_atr = pnl_pts/atr if atr>0 else 0
                margin = vol * entry * cfg['multiplier'] * 0.15
                
                trail = pos.get('_trail', entry)
                
                if d == 'LONG':
                    hard_stop = price - atr * cfg['atr_stop']
                    if pnl_atr > cfg['trail_atr']: trail = max(trail, price - atr*(cfg['atr_stop']-0.3))
                    if pnl_atr > cfg['be_atr']: trail = max(trail, entry)
                    eff_stop = max(hard_stop, trail)
                    
                    should_reduce = (cur_dir=='LONG' and conf<cfg['reduce_conf'])
                    should_reverse = (prob < cfg['reverse_conf'])
                    
                    if low <= eff_stop:
                        ep = eff_stop
                        pnl = margin * (((ep-entry)/entry)/0.15 - cfg['cost']*2)
                        state['cash'] += margin + pnl
                        trade = {'sym': sym_key, 'dir': d, 'entry': entry, 'exit': ep, 'vol': vol,
                                 'pnl': pnl, 'type': 'STOP', 'time': now.isoformat()}
                        state.setdefault('trades', []).append(trade)
                        emoji = '🟢' if pnl>0 else '🔴'
                        print('  %s [V34-STOP] %s %s %d手 @%.0f→%.0f PnL=%+.0f 余额=¥%s' % (
                            emoji, sym_key, d, vol, entry, ep, pnl, format(int(state['cash']), ',')))
                        log_event('V34 STOP %s %s %d手 @%s→%s PnL=%+.0f' % (sym_key, d, vol, entry, ep, pnl))
                        log_trade('V32', sym_key, 'CLOSE', d, entry, ep, vol, pnl,
                                  '止损|STOP', '有效止损=%.1f_硬止损=%.1f_追踪=%.1f_概率=%.3f' % (eff_stop, hard_stop, trail, prob),
                                  state['cash'], state['cash'])
                        try: send_alert('%s [V34] 止损 | %s' % (emoji, sym_key),
                            '%s %d手 @%s→%s\nPnL=%+.0f\n余额 ¥%s' % (d,vol,entry,ep,pnl,format(int(state['cash']),',')),
                            color='red' if pnl<0 else 'green', pin=True)
                        except: pass
                    elif should_reverse:
                        pnl = margin * (pnl_pts/entry/0.15 - cfg['cost']*2)
                        state['cash'] += margin + pnl
                        trade = {'sym': sym_key, 'dir': d, 'entry': entry, 'exit': price, 'vol': vol,
                                 'pnl': pnl, 'type': 'REVERSE', 'time': now.isoformat()}
                        state.setdefault('trades', []).append(trade)
                        print('  🔴 [V34-REV] %s %s→反手平仓 @%.0f PnL=%+.0f' % (sym_key, d, price, pnl))
                        log_event('V34 REV %s %s %d手 @%s→%s PnL=%+.0f' % (sym_key, d, vol, entry, price, pnl))
                        log_trade('V32', sym_key, 'CLOSE', d, entry, price, vol, pnl,
                                  '反手|REVERSE', '概率=%.3f<%.2f_反手阈值' % (prob, cfg['reverse_conf']),
                                  state['cash'], state['cash'])
                        try: send_alert('🔴 [V34] 反手 | %s' % sym_key,
                            '平%s %d手 @%s→%.0f\nPnL=%+.0f' % (d,vol,entry,price,pnl), color='red', pin=True)
                        except: pass
                    elif should_reduce and vol > 1:
                        cut = vol // 2
                        released = margin * (cut/vol)
                        pnl = released * (pnl_pts/entry/0.15 - cfg['cost'])
                        state['cash'] += released + pnl
                        trade = {'sym': sym_key, 'dir': d, 'entry': entry, 'exit': price, 'vol': cut,
                                 'pnl': pnl, 'type': 'REDUCE', 'time': now.isoformat()}
                        state.setdefault('trades', []).append(trade)
                        print('  🟡 [V34-RED] %s 减仓 %d→%d手 @%.0f PnL=%+.0f' % (sym_key, vol, vol-cut, price, pnl))
                        log_event('V34 RED %s %s %d→%d手 @%.0f PnL=%+.0f' % (sym_key, d, vol, vol-cut, price, pnl))
                        log_trade('V32', sym_key, 'REDUCE', d, entry, price, cut, pnl,
                                  '减仓|REDUCE', '置信=%.2f<%.2f_浮盈ATR=%.2f' % (conf, cfg['reduce_conf'], pnl_atr),
                                  state['cash'], state['cash'])
                        pos['vol'] = vol - cut
                        pos['_trail'] = trail
                        surviving.append(pos)
                    else:
                        pos['_trail'] = trail
                        surviving.append(pos)
                else:  # SHORT
                    hard_stop = price + atr * cfg['atr_stop']
                    if pnl_atr > cfg['trail_atr']: trail = min(trail, price + atr*(cfg['atr_stop']-0.3))
                    if pnl_atr > cfg['be_atr']: trail = min(trail, entry)
                    eff_stop = min(hard_stop, trail)
                    
                    should_reduce = (cur_dir=='SHORT' and conf<cfg['reduce_conf'])
                    should_reverse = (prob > 1-cfg['reverse_conf'])
                    
                    if high >= eff_stop:
                        ep = eff_stop
                        pnl = margin * (((entry-ep)/entry)/0.15 - cfg['cost']*2)
                        state['cash'] += margin + pnl
                        trade = {'sym': sym_key, 'dir': d, 'entry': entry, 'exit': ep, 'vol': vol,
                                 'pnl': pnl, 'type': 'STOP', 'time': now.isoformat()}
                        state.setdefault('trades', []).append(trade)
                        emoji = '🟢' if pnl>0 else '🔴'
                        print('  %s [V34-STOP] %s %s %d手 @%.0f→%.0f PnL=%+.0f' % (emoji, sym_key, d, vol, entry, ep, pnl))
                        log_event('V34 STOP %s %s %d手 @%s→%s PnL=%+.0f' % (sym_key, d, vol, entry, ep, pnl))
                        log_trade('V32', sym_key, 'CLOSE', d, entry, ep, vol, pnl,
                                  '止损|STOP', '有效止损=%.1f_硬止损=%.1f_追踪=%.1f_概率=%.3f' % (eff_stop, hard_stop, trail, prob),
                                  state['cash'], state['cash'])
                        try: send_alert('%s [V34] 止损 | %s' % (emoji, sym_key),
                            '%s %d手 @%s→%.0f\nPnL=%+.0f' % (d,vol,entry,ep,pnl), color='red' if pnl<0 else 'green', pin=True)
                        except: pass
                    elif should_reverse:
                        pnl = margin * (pnl_pts/entry/0.15 - cfg['cost']*2)
                        state['cash'] += margin + pnl
                        trade = {'sym': sym_key, 'dir': d, 'entry': entry, 'exit': price, 'vol': vol,
                                 'pnl': pnl, 'type': 'REVERSE', 'time': now.isoformat()}
                        state.setdefault('trades', []).append(trade)
                        print('  🔴 [V34-REV] %s %s→反手平仓 @%.0f PnL=%+.0f' % (sym_key, d, price, pnl))
                        log_event('V34 REV %s %s %d手 @%s→%s PnL=%+.0f' % (sym_key, d, vol, entry, price, pnl))
                        log_trade('V32', sym_key, 'CLOSE', d, entry, price, vol, pnl,
                                  '反手|REVERSE', '概率=%.3f<%.2f_反手阈值' % (prob, cfg['reverse_conf']),
                                  state['cash'], state['cash'])
                        try: send_alert('🔴 [V34] 反手 | %s' % sym_key,
                            '平%s %d手 @%s→%.0f\nPnL=%+.0f' % (d,vol,entry,price,pnl), color='red', pin=True)
                        except: pass
                    elif should_reduce and vol > 1:
                        cut = vol // 2
                        released = margin * (cut/vol)
                        pnl = released * (pnl_pts/entry/0.15 - cfg['cost'])
                        state['cash'] += released + pnl
                        trade = {'sym': sym_key, 'dir': d, 'entry': entry, 'exit': price, 'vol': cut,
                                 'pnl': pnl, 'type': 'REDUCE', 'time': now.isoformat()}
                        state.setdefault('trades', []).append(trade)
                        print('  🟡 [V34-RED] %s 减仓 %d→%d手 @%.0f PnL=%+.0f' % (sym_key, vol, vol-cut, price, pnl))
                        log_event('V34 RED %s %s %d→%d手 @%.0f PnL=%+.0f' % (sym_key, d, vol, vol-cut, price, pnl))
                        log_trade('V32', sym_key, 'REDUCE', d, entry, price, cut, pnl,
                                  '减仓|REDUCE', '置信=%.2f<%.2f_浮盈ATR=%.2f' % (conf, cfg['reduce_conf'], pnl_atr),
                                  state['cash'], state['cash'])
                        pos['vol'] = vol - cut
                        pos['_trail'] = trail
                        surviving.append(pos)
                    else:
                        pos['_trail'] = trail
                        surviving.append(pos)
            
            positions[sym_key] = surviving
            # Clean empty
            if not surviving:
                del positions[sym_key]

        # ===== V34 开仓/加仓 =====
        for sym_key in SYMBOLS:
            if sym_key in traded_today and sym_key in positions: continue
            
            cfg = SYMBOLS[sym_key]
            df = daily_dfs.get(sym_key)
            if df is None: continue
            
            price = float(df.iloc[-1]['close'])
            atr = calc_atr(df, len(df)-1, 20)
            if atr is None: continue
            
            feats = build_features(df, len(df)-1, 60)
            if feats is None: continue
            if sym_key not in models: continue
            try:
                prob = float(models[sym_key].predict_proba(feats.reshape(1,-1))[0][1])
            except: continue
            
            sd = 'LONG' if prob > 0.5 else 'SHORT'
            conf = prob if prob > 0.5 else 1-prob
            if conf < 0.65: continue  # V34 高置信门槛(网格最优)
            conf = prob if prob > 0.5 else 1-prob
            
            ps = calc_position_size(state['cash'], price, atr, cfg)
            if ps == 0: continue
            
            cur_positions = positions.get(sym_key, [])
            total_lots = sum(p['vol'] for p in cur_positions)
            max_total = min(int(cfg['max_total'] * state['cash']/CAPITAL), cfg['max_total'])
            max_total = max(1, max_total)
            
            if total_lots + ps > max_total: continue
            
            stop_dist = atr * cfg['atr_stop']
            margin_per_lot = price * cfg['multiplier'] * 0.15
            total_margin = ps * margin_per_lot
            if total_margin > state['cash'] * 0.8: continue
            
            if not cur_positions:
                # 新开仓
                try:
                    minute_df = ak.futures_zh_minute_sina(symbol=sym_key.upper(), period='5')
                    today_str2 = datetime.now().strftime('%Y-%m-%d')
                    if minute_df is not None and len(minute_df) > 0:
                        minute_df['dt'] = pd.to_datetime(minute_df['datetime'])
                        today_df = minute_df[minute_df['dt'].dt.strftime('%Y-%m-%d') == today_str2]
                        if len(today_df) > 0:
                            today_high = float(today_df['high'].max())
                            today_low = float(today_df['low'].min())
                except: today_high = today_low = None
                
                if sd == 'LONG':
                    entry_stop = price - stop_dist
                    if today_low is not None and today_low <= entry_stop:
                        print('  ⚠️ %s 今日已触发止损，跳过' % sym_key); continue
                    state['cash'] -= total_margin
                    cur_positions.append({'dir': 'LONG', 'entry': price, 'vol': ps,
                        '_entry_time': now.isoformat(), '_trail': entry_stop})
                    positions[sym_key] = cur_positions
                    traded_today.add(sym_key)
                    marg = ps * price * cfg['multiplier'] * 0.15
                    print('  🟢 [V34] 开多 %s %d手 @%.0f 止损%.0f 保证金¥%.1f万' % (sym_key, ps, price, entry_stop, marg/10000))
                    log_event('V34 OPEN %s LONG %d手 @%s STOP=%s' % (sym_key, ps, price, entry_stop))
                    log_trade('V32', sym_key, 'OPEN', 'LONG', price, 0, ps, 0,
                              '模型信号|SIGNAL', '概率=%.3f_置信=%.2f_ATR=%.1f_stop=%.0f' % (prob, conf, atr, entry_stop),
                              state['cash'], state['cash'])
                    try: send_alert('🟢 [V34] 开多 | %s' % sym_key,
                        '%d手 @%.0f\n止损%.0f\n保证金¥%.1f万' % (ps, price, entry_stop, marg/10000),
                        color='blue', pin=True)
                    except: pass
                else:
                    entry_stop = price + stop_dist
                    if today_high is not None and today_high >= entry_stop:
                        print('  ⚠️ %s 今日已触发止损，跳过' % sym_key); continue
                    state['cash'] -= total_margin
                    cur_positions.append({'dir': 'SHORT', 'entry': price, 'vol': ps,
                        '_entry_time': now.isoformat(), '_trail': entry_stop})
                    positions[sym_key] = cur_positions
                    traded_today.add(sym_key)
                    marg = ps * price * cfg['multiplier'] * 0.15
                    print('  🔴 [V34] 开空 %s %d手 @%.0f 止损%.0f 保证金¥%.1f万' % (sym_key, ps, price, entry_stop, marg/10000))
                    log_event('V34 OPEN %s SHORT %d手 @%s STOP=%s' % (sym_key, ps, price, entry_stop))
                    log_trade('V32', sym_key, 'OPEN', 'SHORT', price, 0, ps, 0,
                              '模型信号|SIGNAL', '概率=%.3f_置信=%.2f_ATR=%.1f_stop=%.0f' % (prob, conf, atr, entry_stop),
                              state['cash'], state['cash'])
                    try: send_alert('🔴 [V34] 开空 | %s' % sym_key,
                        '%d手 @%.0f\n止损%.0f\n保证金¥%.1f万' % (ps, price, entry_stop, marg/10000),
                        color='blue', pin=True)
                    except: pass
            else:
                # 加仓判断
                existing_dir = cur_positions[0]['dir']
                if sd != existing_dir: continue  # 不加反向
                
                avg_entry = np.mean([p['entry'] for p in cur_positions])
                pnl_atr = (price-avg_entry)/atr if existing_dir=='LONG' else (avg_entry-price)/atr
                
                if conf > cfg['add_conf'] and pnl_atr > cfg['add_atr']:
                    if existing_dir == 'LONG':
                        entry_stop = price - stop_dist
                        state['cash'] -= total_margin
                        cur_positions.append({'dir': 'LONG', 'entry': price, 'vol': ps,
                            '_entry_time': now.isoformat(), '_trail': entry_stop})
                    else:
                        entry_stop = price + stop_dist
                        state['cash'] -= total_margin
                        cur_positions.append({'dir': 'SHORT', 'entry': price, 'vol': ps,
                            '_entry_time': now.isoformat(), '_trail': entry_stop})
                    positions[sym_key] = cur_positions
                    total_now = sum(p['vol'] for p in cur_positions)
                    print('  ➕ [V34] 加仓 %s +%d手 @%.0f 共%d手 浮盈%.1fATR' % (sym_key, ps, price, total_now, pnl_atr))
                    log_event('V34 ADD %s +%d手 @%s 共%d手' % (sym_key, ps, price, total_now))
                    log_trade('V32', sym_key, 'ADD', existing_dir, price, 0, ps, 0,
                              '加仓|ADD', '置信=%.2f>%.2f_浮盈ATR=%.2f>%.1f' % (conf, cfg['add_conf'], pnl_atr, cfg['add_atr']),
                              state['cash'], state['cash'])
                    try: send_alert('➕ [V34] 加仓 | %s' % sym_key,
                        '+%d手 @%.0f 共%d手\n浮盈%.1fATR' % (ps, price, total_now, pnl_atr),
                        color='green', pin=True)
                    except: pass

        # Record equity (with unrealized PnL)
        total_equity = state['cash']
        for sym_key, pos_list in state['positions'].items():
            cfg = SYMBOLS.get(sym_key, {})
            mult = cfg.get('multiplier', 10)
            df = daily_dfs.get(sym_key)
            cur_price = float(df.iloc[-1]['close']) if df is not None else None
            for p in (pos_list if isinstance(pos_list, list) else [pos_list]):
                total_equity += p['vol'] * p['entry'] * mult * 0.15
                if cur_price is not None:
                    total_equity += (cur_price - p['entry']) * p['vol'] * mult if p['dir'] == 'LONG' else (p['entry'] - cur_price) * p['vol'] * mult
        state['equity_history'].append({'time': now.isoformat(), 'equity': round(total_equity, 2)})

        save_state(state)

        # Wait
        for _ in range(BAR_INTERVAL):
            if not running: break
            time.sleep(1)

    print('\nShutting down V32...')
    save_state(state)
    print('Final equity: ¥%s' % format(int(state['cash']), ','))
    print('Total trades: %d' % len(state.get('trades', [])))

if __name__ == '__main__':
    main()
