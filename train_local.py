#!/usr/bin/env python3
"""Prophet Local Trainer — Train models locally, deploy to server"""
import sys, os, json, time, joblib
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit

# === CONFIG ===
SYMBOLS = {
    'LH': {'code': 'LH0', 'name': '生猪', 'cost': 0.0006, 'rev': False,
           'has_night': False,
           'params': {'n_est': 200, 'depth': 5, 'lr': 0.05, 'win': 60}},
    'JM': {'code': 'JM0', 'name': '焦煤', 'cost': 0.0011, 'rev': False,
           'has_night': True,
           'params': {'n_est': 100, 'depth': 4, 'lr': 0.03, 'win': 60}},
    'RM': {'code': 'RM0', 'name': '菜粕', 'cost': 0.0011, 'rev': True,
           'has_night': True,
           'params': {'n_est': 100, 'depth': 5, 'lr': 0.03, 'win': 120}},
}

RR = 3.0          # Risk:Reward
TRAIN_DAYS = 2000  # Days of history for training
OUTPUT_DIR = '/home/a/prophet_futures/prophet_futures/models'
RISK = {
    'max_pos': 3.0,         # Max position leverage
    'min_pos': 0.5,         # Min position leverage
    'vol_filter': True,     # Skip high-volatility periods
    'vol_max': 0.05,        # Max ATR% to trade
    'time_exit': 10,        # Exit after N bars if no target/stop hit
    'partial_tp': False,     # Disabled - simplifies backtest
    'min_volume_ratio': 0.5, # Skip if volume < 50% of average
}

# ══════════════════ DATA ══════════════════
def fetch_data(sym_code, days=TRAIN_DAYS):
    end = datetime.now()
    start = end - timedelta(days=days + 200)
    df = ak.futures_main_sina(symbol=sym_code, start_date=start.strftime('%Y%m%d'),
                               end_date=end.strftime('%Y%m%d'))
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for c in ['open','high','low','close','volume','oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)

# ══════════════════ FEATURES ══════════════════
def build_features(df, idx, cfg, L=60):
    if idx < L + 5: return None
    w = df.iloc[idx-L:idx+1]
    c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values
    v = w['volume'].values; oi_vals = w['oi'].values
    
    f = []
    # Session features
    if idx >= 1:
        f.append((o[-1]-c[-2])/c[-2])
        f.append(abs((o[-1]-c[-2])/c[-2]))
    else:
        f.extend([0, 0])
    f.append(1.0 if cfg['has_night'] else 0.0)
    
    # Returns
    for lag in [1, 3, 5, 10, 20]:
        f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    
    # MAs
    for p in [5, 10, 20, 60]:
        ma = np.mean(c[-min(p, len(c)):])
        f.append((c[-1]-ma)/ma)
    
    # Volatility
    f.append(np.std(c[-20:]) / np.mean(c[-20:]))
    f.append((h[-1]-l[-1])/c[-1])
    
    # Volume & OI
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
    f.append(v[-1]/vma)
    f.append(oi_vals[-1]/np.mean(oi_vals[-20:]) if len(oi_vals) >= 20 and np.mean(oi_vals[-20:]) > 0 else 1)
    f.append((oi_vals[-1]-oi_vals[-6])/np.mean(oi_vals[-20:]) if len(oi_vals) >= 20 and np.mean(oi_vals[-20:]) > 0 else 0)
    
    # MACD + RSI
    ema12 = c[-1]; ema26 = c[-1]
    for i in range(len(c)-2, -1, -1):
        ema12 = (2/13)*c[i] + (11/13)*ema12
        ema26 = (2/27)*c[i] + (25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    
    deltas = np.diff(c[-15:])
    gains = deltas[deltas > 0].sum() if len(deltas[deltas > 0]) > 0 else 0
    losses = abs(deltas[deltas < 0].sum()) if len(deltas[deltas < 0]) > 0 else 1e-10
    f.append(100 - 100/(1+gains/losses) if losses > 0 else 50)
    
    # Bollinger + Seasonality
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:])
    f.append((c[-1]-ma20)/(2*bb+1e-10))
    
    ds = str(df.iloc[idx]['date'])
    try:
        m = int(ds[5:7])
        f.append(np.sin(2*np.pi*m/12))
        f.append(np.cos(2*np.pi*m/12))
    except:
        f.extend([0, 0])
    
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

# ══════════════════ TRAINING ══════════════════
def train_models(df, cfg):
    """Train XGBoost + LightGBM + CatBoost on full data"""
    win = cfg['params']['win']
    rev = cfg['rev']
    
    # Build training data (use only first 60% for training - strict temporal split)
    split = int(len(df) * 0.6)
    if split < win + 100: return None
    
    X, y = [], []
    for i in range(win, split - 1):
        f = build_features(df, i, cfg, win)
        if f is None: continue
        label = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
        if rev: label = 1 - label
        X.append(f)
        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"  Training on {len(X)} samples, {X.shape[1]} features")
    
    models = {}
    p = cfg['params']
    
    # XGBoost
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=p['n_est'], max_depth=p['depth'],
        learning_rate=p['lr'], use_label_encoder=False,
        eval_metric='logloss', verbosity=0, random_state=42)
    xgb_model.fit(X, y)
    models['xgb'] = xgb_model
    print(f"  XGBoost trained in {time.time()-t0:.1f}s")
    
    # LightGBM
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=p['n_est'], max_depth=p['depth'],
        learning_rate=p['lr'], verbosity=-1, random_state=42)
    lgb_model.fit(X, y)
    models['lgb'] = lgb_model
    print(f"  LightGBM trained in {time.time()-t0:.1f}s")
    
    # CatBoost
    t0 = time.time()
    cb_model = CatBoostClassifier(
        iterations=min(p['n_est'], 200), depth=p['depth'],
        learning_rate=p['lr'], verbose=0, random_seed=42)
    cb_model.fit(X, y)
    models['cb'] = cb_model
    print(f"  CatBoost trained in {time.time()-t0:.1f}s")
    
    return models

# ══════════════════ BACKTEST ══════════════════
def backtest(df, models, cfg):
    """Walk-forward backtest with risk management"""
    win = cfg['params']['win']
    cost = cfg['cost']; rev = cfg['rev']
    n = len(df)
    train_size = int(n * 0.6)
    if train_size < 200: return None
    
    results = []; pnls = []
    N_WF = 100  # Reduced for speed
    
    for run in range(min(N_WF, (n - train_size) // 20)):
        split = train_size + run * 20
        if split + 20 > n: break
        
        test_start = split
        test_end = min(split + 20, n)
        
        for j in range(test_start, test_end - 1):
            f = build_features(df, j, cfg, win)
            if f is None: continue
            
            # Ensemble prediction (average of 3 models)
            probs = []
            for name, model in models.items():
                prob = model.predict_proba(f.reshape(1, -1))[0][1]
                probs.append(prob)
            avg_prob = np.mean(probs)
            
            # Confidence filter
            conf = avg_prob if avg_prob > 0.5 else 1 - avg_prob
            if conf < 0.52: continue  # Skip low confidence
            
            pred = 1 if avg_prob > 0.5 else 0
            if rev: pred = 1 - pred
            
            entry = df.iloc[j]['close']
            
            # Volume filter
            vol_ratio = df.iloc[j]['volume'] / df.iloc[max(0, j-20):j+1]['volume'].mean() if j > 0 else 1
            if vol_ratio < RISK['min_volume_ratio']: continue
            
            # ATR-based position sizing
            atr_vals = [abs(df.iloc[k]['high'] - df.iloc[k]['low']) 
                       for k in range(max(0, j-20), j+1)]
            atr = np.mean(atr_vals) if atr_vals else entry * 0.02
            atr_pct = atr / entry
            
            # Volatility filter
            if RISK['vol_filter'] and atr_pct > RISK['vol_max']: continue
            
            # Position size
            if atr_pct < 0.01: pos = RISK['max_pos']
            elif atr_pct < 0.02: pos = 2.0
            elif atr_pct < 0.03: pos = 1.5
            elif atr_pct < 0.04: pos = RISK['min_pos']
            else: pos = 0
            if pos == 0: continue
            
            # Dynamic ATR stop (1x ATR for stop, 3x for target)
            atr_stop = atr_pct  # 1x daily ATR as stop
            stop_pct = max(cost, min(0.02, atr_stop))  # floor=cost, cap=2%
            target_pct = RR * stop_pct
            
            # Exit simulation
            fp = df.iloc[j+1:min(j+RISK['time_exit']+1, n)]['close'].values
            if len(fp) == 0: continue
            
            for px in fp:
                if pred == 1:
                    current_pnl = (px - entry) / entry
                else:
                    current_pnl = (entry - px) / entry
                
                if current_pnl >= target_pct:
                    pnls.append((current_pnl - cost) * pos)
                    results.append(1)
                    break
                elif current_pnl <= -stop_pct:
                    pnls.append(-stop_pct * pos)
                    results.append(0)
                    break
            else:
                # Time exit
                lp = fp[-1] if len(fp) > 0 else entry
                if pred == 1:
                    final_pnl = (lp - entry) / entry - cost
                else:
                    final_pnl = (entry - lp) / entry - cost
                results.append(1 if final_pnl > 0 else 0)
                pnls.append(final_pnl * pos)
    
    if not results: return None
    
    total = len(results); nw = sum(results)
    wr = nw / total * 100
    tp = sum(pnls)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = np.max((peak - cum) / (peak + 1e-10)) * 100 if len(cum) > 0 else 0
    
    gp = sum(p for p, w in zip(pnls, results) if w == 1)
    gl = abs(sum(p for p, w in zip(pnls, results) if w == 0))
    pf = gp / gl if gl > 0 else 999
    
    test_days = n - train_size
    years = test_days / 252
    ann_ret = ((1+tp)**(1/years)-1)*100 if years > 0 and tp > -1 else tp/years*100 if years > 0 else 0
    
    return {
        'trades': total, 'wr': round(wr, 1),
        'cum_ret': round(tp * 100, 1), 'ann_ret': round(ann_ret, 1),
        'dd': round(dd, 1), 'pf': round(pf, 2)
    }

# ══════════════════ MAIN ══════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("  Prophet Local Trainer — Multi-Model + Risk Management")
    print(f"  Models: XGBoost + LightGBM + CatBoost (Ensemble)")
    print(f"  Risk: Dynamic ATR stop, Vol filter, Partial TP, Max pos {RISK['max_pos']}x")
    print("=" * 60)
    
    all_results = {}
    
    for sym_key, cfg in SYMBOLS.items():
        print(f"\n{'='*60}")
        print(f"  {sym_key} {cfg['name']}")
        print(f"{'='*60}")
        
        # Fetch data
        print(f"  Fetching {TRAIN_DAYS}d data...")
        df = fetch_data(cfg['code'], TRAIN_DAYS)
        print(f"  Data: {len(df)} rows, {df.iloc[0]['date']} -> {df.iloc[-1]['date']}")
        
        # Train models
        models = train_models(df, cfg)
        
        # Save models
        for name, model in models.items():
            path = f"{OUTPUT_DIR}/{sym_key}_{name}.pkl"
            joblib.dump(model, path)
            print(f"  Saved: {path}")
        
        # Backtest with risk management
        print(f"  Backtesting...")
        bt = backtest(df, models, cfg)
        if bt:
            all_results[sym_key] = bt
            print(f"  Result: {bt['trades']}t {bt['wr']}% "
                  f"Ann={bt['ann_ret']:+.1f}%/yr DD={bt['dd']}% PF={bt['pf']}")
    
    # Save results
    with open(f'{OUTPUT_DIR}/training_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"  Models saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")
    for sym, r in all_results.items():
        print(f"  {sym}: {r['trades']}t {r['wr']}% Ann={r['ann_ret']:+.1f}%/yr DD={r['dd']}% PF={r['pf']}")
    
    print(f"\n  Deploy with: scp {OUTPUT_DIR}/* root@47.102.42.238:/root/prophet_futures/models/")

if __name__ == '__main__':
    main()
