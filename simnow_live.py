#!/usr/bin/env python3
"""Prophet Fut6 SimNow Live Trader — v23 Struct Stop"""
import sys, os, time, signal, json
# Fix CTP locale crash — set C locale BEFORE any vnpy imports
os.environ['LC_ALL'] = 'C'
os.environ['LANG'] = 'C'
os.environ['LANGUAGE'] = 'C'
import locale
locale.setlocale(locale.LC_ALL, 'C')
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Direction, Offset, OrderType
from vnpy_ctp import CtpGateway

# ===== CONFIG =====
SYMBOLS = {
    'lh2609': {'vt': 'DCE.lh2609', 'name': 'LH', 'cost': 0.0006, 'multiplier': 16,
               'params': (200, 5, 0.05, 60), 'rev': False, 'struct_n': 26, 'rr': 4},
    'jm2609': {'vt': 'DCE.jm2609', 'name': 'JM', 'cost': 0.0011, 'multiplier': 60,
               'params': (100, 4, 0.03, 60), 'rev': False, 'struct_n': 24, 'rr': 3},
}

CAPITAL = 300000
MAX_POSITION = {'lh2609': 3, 'jm2609': 5}
VOL_LOOKBACK = 20
MAX_LEVERAGE = 3.0
MIN_LEVERAGE = 0.5

# Models directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
running = True

def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ===== DATA =====
def fetch_history(symbol_code, days=1000):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = ak.futures_main_sina(symbol=symbol_code, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for col in ['open','high','low','close','volume','oi']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except Exception as e:
        print(f"Fetch error {symbol_code}: {e}")
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
    atr_vals = []
    for i in range(max(0, len(df)-VOL_LOOKBACK), len(df)):
        atr_vals.append(abs(df.iloc[i]['high'] - df.iloc[i]['low']))
    if not atr_vals: return 1
    atr = np.mean(atr_vals)
    price = df.iloc[-1]['close']
    atr_pct = atr / price
    if atr_pct < 0.01: leverage = MAX_LEVERAGE
    elif atr_pct < 0.02: leverage = 2.0
    elif atr_pct < 0.03: leverage = 1.5
    elif atr_pct < 0.05: leverage = MIN_LEVERAGE
    else: leverage = 0
    base = MAX_POSITION.get(sym_key, 1)
    return max(1, int(leverage * base)) if leverage > 0 else 0

# ===== TRADING SIGNAL =====
def generate_signals(models, symbol_map):
    signals = {}
    df_cache = {}
    
    for sym_key, cfg in SYMBOLS.items():
        main_code = symbol_map.get(sym_key, {}).get('code', sym_key[:2]+'0')
        df = fetch_history(main_code, 1200)
        if df is None or len(df) < 100: continue
        df_cache[sym_key] = df
        
        n_est, depth, lr, win = cfg['params']
        rev = cfg['rev']
        struct_n = cfg.get('struct_n', 20)
        rr_val = cfg.get('rr', 3)
        
        # Build feature
        f = build_features(df, len(df)-1, win)
        if f is None: continue
        
        # Get model prediction
        model_key = sym_key
        if model_key not in models: continue
        
        prob = 0.5
        try:
            model_data = models[model_key]
            if isinstance(model_data, dict):
                probs = []
                for name, model in model_data.items():
                    p = model.predict_proba(f.reshape(1, -1))[0]
                    probs.append(p)
                avg_prob = np.mean(probs)
                prob = avg_prob[1]
            else:
                p = model_data.predict_proba(f.reshape(1, -1))[0]
                prob = p[1]
        except: continue
        
        signal = 'LONG' if prob > 0.5 else 'SHORT'
        if rev: signal = 'SHORT' if signal == 'LONG' else 'LONG'
        
        confidence = prob if prob > 0.5 else (1 - prob)
        current_price = float(df.iloc[-1]['close'])
        pos_size = calc_position_size(sym_key, df)
        
        signals[sym_key] = {
            'signal': signal, 'confidence': confidence,
            'price': current_price, 'pos_size': pos_size,
            'df': df, 'struct_n': struct_n, 'rr': rr_val, 'cfg': cfg
        }
    
    return signals

# ===== MAIN =====
def main():
    global running
    
    # Load models
    models = {}
    print("Loading models...")
    import pickle
    for sym_key in SYMBOLS:
        model_path = os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[sym_key] = pickle.load(f)
            print(f"  {sym_key}: loaded")
        else:
            print(f"  {sym_key}: no model at {model_path}")
    
    # Symbol code mapping
    symbol_map = {
        'lh2609': {'code': 'LH0'}, 'jm2609': {'code': 'JM0'},
    }
    
    # VNPY
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(CtpGateway)
    
    ctp_setting = {
        "用户名": "266887", "密码": "9999",
        "经纪商代码": "9999", "交易服务器": "182.254.243.31:30001",
        "行情服务器": "182.254.243.31:30011",
        "产品名称": "simnow_client_test", "授权编码": "0000000000000000",
        "产品信息": ""
    }
    
    # Try all 3 groups if first one fails
    CTP_SERVERS = [
        ("182.254.243.31:30001", "182.254.243.31:30011"),
        ("182.254.243.31:30002", "182.254.243.31:30012"),
        ("182.254.243.31:30003", "182.254.243.31:30013"),
        ("182.254.243.31:40001", "182.254.243.31:40011"),  # 7x24
    ]
    
    print("\nConnecting to SimNow CTP...")
    
    # Set up login detection via VNPY event system
    from threading import Event
    from vnpy.trader.event import EVENT_LOG, EVENT_ACCOUNT
    login_event = Event()
    
    def on_login_check(event):
        data = str(event.data) if hasattr(event, 'data') else ''
        if '登录成功' in data or 'login' in data.lower():
            login_event.set()
    
    event_engine.register(EVENT_LOG, on_login_check)
    # Account info arrival also means login done
    event_engine.register(EVENT_ACCOUNT, lambda e: login_event.set())
    
    main_engine.connect(ctp_setting, "CTP")
    print("  CTP gateway connecting, waiting for login...")
    
    # Wait up to 60s for login
    for i in range(120):
        time.sleep(0.5)
        if login_event.is_set():
            print("  ✅ CTP login confirmed!")
            break
        if i == 0: print("  ...waiting", end='', flush=True)
        elif i % 20 == 0: print(f" {i//2}s", end='', flush=True)
    
    if not login_event.is_set():
        print(f"\n  ⚠️ Login not confirmed after 60s")
    else:
        print()
    
    event_engine.unregister(EVENT_LOG, on_login_check)
    
    # Subscribe
    for sym_key, cfg in SYMBOLS.items():
        parts = cfg['vt'].split('.')
        from vnpy.trader.object import SubscribeRequest
        from vnpy.trader.constant import Exchange
        exchange = Exchange.DCE if parts[0] == 'DCE' else Exchange.CZCE
        req = SubscribeRequest(symbol=parts[1], exchange=exchange)
        main_engine.subscribe(req, "CTP")
        print(f"  Subscribed: {sym_key} ({cfg['vt']})")
    
    print("\n" + "="*60)
    print("  Trading loop started (v23 Struct Stop)")
    print("="*60 + "\n")
    
    positions = {}
    active_orders = {}
    TRADED_TODAY = {}
    today_str = datetime.now().strftime('%Y%m%d')
    BAR_INTERVAL = 300  # 5 minutes
    
    while running:
        now = datetime.now()
        current_date = now.strftime('%Y%m%d')
        if current_date != today_str:
            TRADED_TODAY.clear()
            today_str = current_date
        
        # Trading hours check (simplified)
        weekday = now.weekday()
        hour = now.hour
        
        if weekday >= 5:
            time.sleep(BAR_INTERVAL)
            continue
        
        print(f"\n[{now.strftime('%H:%M:%S')}] Signal scan...")
        
        try:
            sigs = generate_signals(models, symbol_map)
        except Exception as e:
            print(f"  Signal error: {e}")
            time.sleep(BAR_INTERVAL)
            continue
        
        for sym_key, signal in sigs.items():
            if sym_key in TRADED_TODAY:
                continue
            if sym_key in positions:
                continue
            
            df = signal['df']
            emoji = '\U0001F7E2' if signal['signal'] == 'LONG' else '\U0001F534'
            cost_pct = signal['cfg']['cost'] * 100
            rr_val = signal['rr']
            pos_size = signal['pos_size']
            
            margin_est = pos_size * (int(signal['price']) * signal['cfg'].get('multiplier', 10) * 0.15)
            
            print(f"  {emoji} {sym_key} {signal['signal']} {pos_size}手 @ {signal['price']:.0f} "
                  f"(约{margin_est/10000:.1f}万保证金) (conf={signal['confidence']:.2f}) "
                  f"RR=1:{rr_val}")
            
            if pos_size == 0:
                continue
            
            # Struct stop
            struct_n = signal['struct_n']
            if signal['signal'] == 'LONG':
                struct_lows = [float(df.iloc[k]['low']) for k in range(max(0, len(df)-struct_n), len(df))]
                stop_price = min(struct_lows)
                stop_dir = Direction.SHORT
            else:
                struct_highs = [float(df.iloc[k]['high']) for k in range(max(0, len(df)-struct_n), len(df))]
                stop_price = max(struct_highs)
                stop_dir = Direction.LONG
            
            stop_price = round(stop_price, 1)
            
            # Place entry order
            cfg = signal['cfg']
            parts = cfg['vt'].split('.')
            from vnpy.trader.object import OrderRequest
            from vnpy.trader.constant import Exchange
            exchange = Exchange.DCE if parts[0] == 'DCE' else Exchange.CZCE
            
            entry_dir = Direction.LONG if signal['signal'] == 'LONG' else Direction.SHORT
            
            try:
                req = OrderRequest(
                    symbol=parts[1], exchange=exchange,
                    direction=entry_dir, offset=Offset.OPEN,
                    price=float(signal['price']), volume=int(pos_size),
                    type=OrderType.LIMIT
                )
                vt_orderid = main_engine.send_order(req, "CTP")
                TRADED_TODAY[sym_key] = now
                positions[sym_key] = {'direction': signal['signal'], 'volume': pos_size, 'entry': signal['price']}
                print(f"    ENTER {sym_key} {signal['signal']} {pos_size}手 @ {signal['price']:.0f}")
                
                # Place stop order
                req_stop = OrderRequest(
                    symbol=parts[1], exchange=exchange,
                    direction=stop_dir, offset=Offset.CLOSE,
                    price=stop_price, volume=int(pos_size),
                    type=OrderType.LIMIT
                )
                main_engine.send_order(req_stop, "CTP")
                print(f"    STOP {sym_key} @ {stop_price:.0f} ({struct_n}-bar {'low' if signal['signal']=='LONG' else 'high'})")
            except Exception as e:
                print(f"    Order error: {e}")
        
        for _ in range(BAR_INTERVAL):
            if not running: break
            time.sleep(1)
    
    print("\nShutting down...")
    main_engine.close()

if __name__ == '__main__':
    main()
