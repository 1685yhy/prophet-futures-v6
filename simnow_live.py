#!/usr/bin/env python3
"""Prophet Futures — SimNow Live Trader v26
动态止盈止损: 模型驱动退出 + 移动止损 + 保本
CTP实盘连接SimNow, 用XGBoost模型生成信号
"""
import sys, os, time, signal, json
import ctypes
_libc = ctypes.CDLL('libc.so.6')
_libc.setlocale(0, b'C')
os.environ['LC_ALL'] = 'C'
os.environ['LANG'] = 'C'
os.environ['LANGUAGE'] = 'C'
import locale
locale.setlocale(locale.LC_ALL, 'C')
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb
from features import build_features, fetch_daily, calc_atr, CONTRACT_MAP as _FEAT_MAP
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Direction, Offset, OrderType
from vnpy_ctp import CtpGateway

# ===== CONFIG (V26) =====
SYMBOLS = {
    'lh2609': {
        'vt': 'DCE.lh2609', 'name': 'LH', 'cost': 0.0006, 'multiplier': 16,
        'max_pos': 3,
        # V26 exit params
        'hard_atr': 0.8, 'trail_atr': 2.0, 'be_atr': 1.0,
        'trail_dist': 1.5, 'model_low': 0.35, 'model_high': 0.65,
        'confirm_bars': 2, 'min_hold': 3,
    },
    'jm2609': {
        'vt': 'DCE.jm2609', 'name': 'JM', 'cost': 0.0011, 'multiplier': 60,
        'max_pos': 4,
        # V26 exit params
        'hard_atr': 1.8, 'trail_atr': 3.0, 'be_atr': 2.0,
        'trail_dist': 2.5, 'model_low': 0.30, 'model_high': 0.70,
        'confirm_bars': 3, 'min_hold': 5,
    },
}

CAPITAL = 300000
VOL_LOOKBACK = 20
MAX_LEVERAGE = 3.0
MIN_LEVERAGE = 0.5

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
running = True

def signal_handler(sig, frame):
    global running; running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===== DATA (imported from features.py) =====
# fetch_daily → fetch_daily, build_features, calc_atr 统一从 features.py 导入

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
    base = cfg['max_pos']
    return max(1, int(leverage * base)) if leverage > 0 else 0

# ===== V26 SIGNAL GENERATION =====
def generate_signals(models):
    signals = {}
    for sym_key, cfg in SYMBOLS.items():
        main_code = cfg['name'] + '0'
        df = fetch_daily(main_code, 1200)
        if df is None or len(df) < 100: continue
        
        feats = build_features(df, len(df)-1, 60)
        if feats is None: continue
        
        model_key = sym_key
        if model_key not in models: continue
        
        try:
            prob = float(models[model_key].predict_proba(feats.reshape(1, -1))[0][1])
        except: continue
        
        signal_dir = 'LONG' if prob > 0.5 else 'SHORT'
        confidence = prob if prob > 0.5 else (1 - prob)
        price = float(df.iloc[-1]['close'])
        pos_size = calc_position_size(sym_key, df)
        
        # V26 entry stop: hard_atr
        atr = calc_atr(df, len(df)-1, 20)
        if atr is None: continue
        stop_dist = atr * cfg['hard_atr']
        
        if signal_dir == 'LONG':
            entry_stop = price - stop_dist
            # V26 fix: use actual RR config, not trail_atr+trail_dist
            display_rr = cfg.get('rr', 3.5)
            display_tp = price + (price - entry_stop) * display_rr
        else:
            entry_stop = price + stop_dist
            display_rr = cfg.get('rr', 3.5)
            display_tp = price - (entry_stop - price) * display_rr
        
        signals[sym_key] = {
            'signal': signal_dir, 'confidence': confidence,
            'price': price, 'pos_size': pos_size,
            'entry_stop': round(entry_stop, 1),
            'display_tp': round(display_tp, 1),
            'df': df, 'cfg': cfg, 'atr': atr,
        }
    return signals

# ===== MAIN =====
def main():
    global running
    
    print("="*60)
    print("  Prophet v26 — SimNow Live Trader")
    print("  Dynamic stop/take-profit | CTP connected")
    print("="*60)
    
    # Load models
    models = {}
    print("Loading models...")
    import pickle
    for sym_key in SYMBOLS:
        mp = os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')
        if os.path.exists(mp):
            with open(mp, 'rb') as f: models[sym_key] = pickle.load(f)
            print(f"  {sym_key}: loaded")
        else:
            print(f"  {sym_key}: no model at {mp}")
    
    # VNPY setup
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(CtpGateway)
    
    ctp_setting = {
        "用户名": "266887", "密码": "asdfghjkl123!!",
        "经纪商代码": "9999", "交易服务器": "182.254.243.31:30001",
        "行情服务器": "182.254.243.31:30011",
        "产品名称": "simnow_client_test", "授权编码": "0000000000000000",
        "产品信息": "prophet_futures_v26"
    }
    
    print("\nConnecting to SimNow CTP...")
    from threading import Event
    from vnpy.trader.event import EVENT_LOG, EVENT_ACCOUNT
    login_event = Event()
    
    def on_login(event):
        data = str(event.data) if hasattr(event, 'data') else ''
        if '登录成功' in data: login_event.set()
    event_engine.register(EVENT_LOG, on_login)
    event_engine.register(EVENT_ACCOUNT, lambda e: login_event.set())
    
    main_engine.connect(ctp_setting, "CTP")
    print("  CTP connecting, waiting for login...")
    
    for i in range(120):
        time.sleep(0.5)
        if login_event.is_set():
            print("  ✅ CTP login confirmed!")
            break
        if i % 20 == 0: print(f"  ...{i//2}s", flush=True)
    
    if not login_event.is_set():
        print("  ⚠️ Login not confirmed after 60s, continuing anyway...")
    
    event_engine.unregister(EVENT_LOG, on_login)
    
    # Subscribe to market data
    for sym_key, cfg in SYMBOLS.items():
        parts = cfg['vt'].split('.')
        from vnpy.trader.object import SubscribeRequest
        from vnpy.trader.constant import Exchange
        exchange = Exchange.DCE if parts[0] == 'DCE' else Exchange.CZCE
        req = SubscribeRequest(symbol=parts[1], exchange=exchange)
        main_engine.subscribe(req, "CTP")
        print(f"  Subscribed: {sym_key} ({cfg['vt']})")
    
    print("\n" + "="*60)
    print("  V26 Dynamic Loop — Model + Trail + Breakeven")
    print("  LH: hardATR×0.8 trail2ATR be1ATR model0.35/0.65")
    print("  JM: hardATR×1.8 trail3ATR be2ATR model0.30/0.70")
    print("="*60 + "\n")
    
    TRADED_TODAY = {}
    positions = {}  # {sym_key: {dir, vol, entry, entry_time, entry_stop, display_tp, _trail_stop, bar_count, _rev_count}}
    today_str = datetime.now().strftime('%Y%m%d')
    BAR_INTERVAL = 300
    
    # Import vnpy types once
    from vnpy.trader.object import OrderRequest
    from vnpy.trader.constant import Exchange
    
    while running:
        now = datetime.now()
        current_date = now.strftime('%Y%m%d')
        if current_date != today_str:
            TRADED_TODAY.clear()
            today_str = current_date
            for p in positions.values():
                p['bar_count'] = 0
        
        # Trading hours check
        wd = now.weekday(); h, m = now.hour, now.minute
        t = h * 60 + m
        in_trading = wd < 5 and ((540 <= t < 615) or (630 <= t < 690) or (810 <= t < 900))
        
        if not in_trading:
            time.sleep(BAR_INTERVAL)
            continue
        
        print(f"\n[{now.strftime('%H:%M:%S')}] V26 Scan — {len(positions)} positions...")
        
        # ===== 1. CHECK EXISTING POSITIONS (EXIT LOGIC) =====
        for sym_key in list(positions.keys()):
            pos = positions[sym_key]
            cfg = SYMBOLS[sym_key]
            d = pos['dir']; entry = pos['entry']; vol = pos['vol']
            
            # Fetch fresh data for exit check
            main_code = cfg['name'] + '0'
            df = fetch_daily(main_code, 60)
            if df is None or len(df) < 20: continue
            
            price = float(df.iloc[-1]['close'])
            atr = calc_atr(df, len(df)-1, 20)
            if atr is None: continue
            
            pos['bar_count'] = pos.get('bar_count', 0) + 1
            bc = pos['bar_count']
            
            # Model reversal check
            feats = build_features(df, len(df)-1, 60)
            model_exit = False
            if feats is not None and sym_key in models:
                try:
                    prob = float(models[sym_key].predict_proba(feats.reshape(1, -1))[0][1])
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
                    model_exit = (pos.get('_rev_count', 0) >= cfg['confirm_bars']) and (bc >= cfg['min_hold'])
                except: pass
            
            # Trailing stop
            hard_stop_dist = atr * cfg['hard_atr']
            trail_stop = pos.get('_trail_stop', None)
            
            exit_trigger = False
            reason = 'HARD'
            
            if d == 'LONG':
                hard_stop = price - hard_stop_dist
                if trail_stop is None:
                    trail_stop = hard_stop
                if bc >= cfg['min_hold'] and price > entry + atr * cfg['trail_atr']:
                    new_trail = price - atr * cfg['trail_dist']
                    trail_stop = max(trail_stop, new_trail)
                if bc >= cfg['min_hold'] and price > entry + atr * cfg['be_atr']:
                    trail_stop = max(trail_stop, entry)
                pos['_trail_stop'] = trail_stop
                eff_stop = max(hard_stop, trail_stop)
                
                exit_trigger = (price <= eff_stop) or (price >= pos.get('display_tp', 1e9)) or model_exit
                if exit_trigger:
                    if model_exit: reason = 'MODEL'
                    elif price >= pos.get('display_tp', 1e9): reason = 'TP'
                    elif trail_stop >= hard_stop: reason = 'TRAIL'
                    else: reason = 'HARD'
            else:  # SHORT
                hard_stop = price + hard_stop_dist
                if trail_stop is None:
                    trail_stop = hard_stop
                if bc >= cfg['min_hold'] and price < entry - atr * cfg['trail_atr']:
                    new_trail = price + atr * cfg['trail_dist']
                    trail_stop = min(trail_stop, new_trail)
                if bc >= cfg['min_hold'] and price < entry - atr * cfg['be_atr']:
                    trail_stop = min(trail_stop, entry)
                pos['_trail_stop'] = trail_stop
                eff_stop = min(hard_stop, trail_stop)
                
                exit_trigger = (price >= eff_stop) or (price <= pos.get('display_tp', 0)) or model_exit
                if exit_trigger:
                    if model_exit: reason = 'MODEL'
                    elif price <= pos.get('display_tp', 0): reason = 'TP'
                    elif trail_stop <= hard_stop: reason = 'TRAIL'
                    else: reason = 'HARD'
            
            if exit_trigger:
                pnl_pts = price - entry if d == 'LONG' else entry - price
                pnl = pnl_pts * vol * cfg['multiplier']
                reason_cn = {'MODEL': '模型逆转', 'TP': '止盈', 'TRAIL': '移动止损', 'HARD': '硬止损'}.get(reason, reason)
                emoji = '🟢' if pnl > 0 else '🔴'
                
                parts = cfg['vt'].split('.')
                exchange = Exchange.DCE if parts[0] == 'DCE' else Exchange.CZCE
                exit_dir = Direction.SHORT if d == 'LONG' else Direction.LONG
                
                # Use aggressive price for exit (slightly beyond stop to ensure fill)
                if d == 'LONG':
                    exit_px = eff_stop - atr * 0.05  # 5% ATR below stop
                else:
                    exit_px = eff_stop + atr * 0.05  # 5% ATR above stop
                
                try:
                    req = OrderRequest(
                        symbol=parts[1], exchange=exchange,
                        direction=exit_dir, offset=Offset.CLOSE,
                        price=round(float(exit_px), 0), volume=int(vol),
                        type=OrderType.LIMIT
                    )
                    main_engine.send_order(req, "CTP")
                    print(f"  {emoji} [{reason_cn}] {sym_key} {d} {vol}手 @{entry}→{price:.0f} PnL={pnl:+,.0f}")
                    
                    # Also send Feishu alert
                    try:
                        from feishu_send import send_alert
                        send_alert(f"{emoji} [{reason_cn}] SimNow | {sym_key}",
                            f"{'做多' if d=='LONG' else '做空'} {vol}手 @{entry}→{price:.0f}\nPnL={pnl:+,.0f}",
                            color='green' if pnl>0 else 'red', pin=True)
                    except: pass
                    
                    del positions[sym_key]
                    TRADED_TODAY[sym_key] = now
                except Exception as e:
                    print(f"    Exit order error: {e}")
        
        # ===== 2. SIGNAL GENERATION (NEW ENTRIES) =====
        try:
            sigs = generate_signals(models)
        except Exception as e:
            print(f"  Signal error: {e}")
            time.sleep(BAR_INTERVAL)
            continue
        
        for sym_key, signal in sigs.items():
            if sym_key in TRADED_TODAY or sym_key in positions: continue
            if signal['pos_size'] == 0: continue
            
            emoji = '🟢' if signal['signal'] == 'LONG' else '🔴'
            price = signal['price']
            pos_size = signal['pos_size']
            entry_stop = signal['entry_stop']
            display_tp = signal['display_tp']
            cfg = signal['cfg']
            atr = signal['atr']
            
            margin_est = pos_size * price * cfg['multiplier'] * 0.15
            
            print(f"  {emoji} {sym_key} {signal['signal']} {pos_size}手 @ {price:.0f} "
                  f"(约{margin_est/10000:.1f}万) conf={signal['confidence']:.2f}")
            print(f"    V26: 止损={entry_stop:.0f} | 止盈={display_tp:.0f} | "
                  f"移动{cfg['trail_atr']}ATR | 保本{cfg['be_atr']}ATR")
            
            # Check stop already hit (using minute data)
            try:
                today_str2 = datetime.now().strftime('%Y-%m-%d')
                minute_df = ak.futures_zh_minute_sina(symbol=sym_key.upper(), period='5')
                if minute_df is not None and len(minute_df) > 0:
                    minute_df['dt'] = pd.to_datetime(minute_df['datetime'])
                    today_df = minute_df[minute_df['dt'].dt.strftime('%Y-%m-%d') == today_str2]
                    if len(today_df) > 0:
                        th = float(today_df['high'].max()); tl = float(today_df['low'].min())
                        already = (signal['signal'] == 'LONG' and tl <= entry_stop) or \
                                  (signal['signal'] == 'SHORT' and th >= entry_stop)
                        if already:
                            print(f"    ⚠️ 当日已触发止损，跳过")
                            continue
            except: pass
            
            # Place entry order via CTP
            parts = cfg['vt'].split('.')
            exchange = Exchange.DCE if parts[0] == 'DCE' else Exchange.CZCE
            entry_dir = Direction.LONG if signal['signal'] == 'LONG' else Direction.SHORT
            
            # Use limit price 0.2% beyond current for better fill chance
            if signal['signal'] == 'LONG':
                order_px = round(price * 1.002, 0)  # bid slightly above
            else:
                order_px = round(price * 0.998, 0)  # bid slightly below
            
            try:
                req = OrderRequest(
                    symbol=parts[1], exchange=exchange,
                    direction=entry_dir, offset=Offset.OPEN,
                    price=float(order_px), volume=int(pos_size),
                    type=OrderType.LIMIT
                )
                main_engine.send_order(req, "CTP")
                TRADED_TODAY[sym_key] = now
                positions[sym_key] = {
                    'dir': signal['signal'], 'vol': pos_size,
                    'entry': price, 'entry_time': now.isoformat(),
                    'entry_stop': entry_stop, 'display_tp': display_tp,
                    '_trail_stop': None, 'bar_count': 0, '_rev_count': 0,
                }
                print(f"    ✅ 已发单 [V26] @{order_px:.0f}")
            except Exception as e:
                print(f"    Order error: {e}")
        
        for _ in range(BAR_INTERVAL):
            if not running: break
            time.sleep(1)
    
    print("\nShutting down...")
    main_engine.close()

if __name__ == '__main__':
    main()
