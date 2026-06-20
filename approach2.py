#!/usr/bin/env python3
"""LH Approach 2: 30-min intraday training + testing (available 6 months)"""
import numpy as np, pandas as pd, json, time
import akshare as ak, xgboost as xgb

m30 = ak.futures_zh_minute_sina(symbol='LH2609', period='30')
m30['datetime'] = pd.to_datetime(m30['datetime'])
m30 = m30.sort_values('datetime').reset_index(drop=True)
print(f'30-min data: {len(m30)} rows  {m30["datetime"].min()} -> {m30["datetime"].max()}')

CAP = 300000; mult = 16; base = 3; cost = 0.0006
win = 60; ne = 200; d = 5; lr = 0.05
SM = [1.5, 2.0, 2.5]; RR = [3.0, 4.0, 5.0]; CF = [0.50, 0.52]

def build_f_intraday(df_local, idx, lookback=13):
    """Features from 30-min bars: lookback=13 => ~6.5 hours"""
    if idx < lookback + 5: return None
    w = df_local.iloc[idx-lookback:idx+1]
    c = w['close'].values; h = w['high'].values; l = w['low'].values
    v = w['volume'].values
    f = []
    # Price momentum
    for lag in [1, 3, 5, 13]:
        f.append((c[-1] - c[-lag-1]) / c[-lag-1] if len(c) > lag else 0)
    # Moving averages
    for p in [3, 5, 13]:
        ma = np.mean(c[-min(p, len(c)):]); f.append((c[-1] - ma) / ma)
    # Volatility
    f.append(np.std(c[-5:]) / np.mean(c[-5:]) if np.mean(c[-5:]) > 0 else 0)
    # Range
    f.append((h[-1] - l[-1]) / c[-1])
    # Volume
    vma = np.mean(v[-5:]) if np.mean(v[-5:]) > 0 else 1; f.append(v[-1] / vma)
    # Trend
    trend = np.polyfit(range(len(c[-13:])), c[-13:], 1)[0]; f.append(trend / c[-1])
    # RSI-like
    dd_ = np.diff(c[-8:]); gain = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
    loss = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
    f.append(100 - 100/(1+gain/loss))
    # BB
    bb_std = np.std(c[-13:]); bb_ma = np.mean(c[-13:]); f.append((c[-1]-bb_ma)/(2*bb_std+1e-10))
    # Time of day (sin/cos encode)
    hour = m30.iloc[idx]['datetime'].hour; minute = m30.iloc[idx]['datetime'].minute
    tod = (hour * 60 + minute) / (24 * 60)
    f.append(np.sin(2 * np.pi * tod)); f.append(np.cos(2 * np.pi * tod))
    # Price level
    f.append(c[-1] / 1000.0)
    return np.array(f, dtype=np.float32)

# Train/test split: first 80% train, last 20% test
split = int(len(m30) * 0.8)
train_m30 = m30.iloc[:split]; test_m30 = m30.iloc[split:]

# Train on intraday features
X, y = [], []
for i in range(13, len(train_m30) - 3):
    f = build_f_intraday(train_m30, i)
    if f is None: continue
    X.append(f)
    # Label: next 3 bars (1.5 hours) up or down
    future_close = train_m30.iloc[min(i+3, len(train_m30)-1)]['close']
    current_close = train_m30.iloc[i]['close']
    y.append(1 if future_close > current_close else 0)

model = xgb.XGBClassifier(n_estimators=ne, max_depth=d, learning_rate=lr,
                           use_label_encoder=False, eval_metric='logloss',
                           verbosity=0, random_state=42)
model.fit(np.array(X), np.array(y))
print(f'Trained on {len(X)} intraday samples')

# Test: 30-min level execution with re-entry
print(f'\nApproach 2: Intraday model + 30-min execution, {len(SM)*len(RR)*len(CF)} combos')
print(f'{"Stop":<7} {"RR":<5} {"Conf":<6} {"NetPnL":<9} {"Ann":<8} {"DD":<7} {"Exits":<6} {"WR":<6}')

results = []
for sm in SM:
    for rr_val in RR:
        for conf in CF:
            capital = CAP; peak = CAP; max_dd = 0; exits = 0; wins = 0
            positions = []
            
            for i in range(len(test_m30)):
                bar = test_m30.iloc[i]; o = bar['open']; h = bar['high']; l = bar['low']
                
                # ATR from recent 30-min range
                start_idx = max(0, i - 13)
                recent_range = [abs(test_m30.iloc[k]['high'] - test_m30.iloc[k]['low']) for k in range(start_idx, i+1)]
                atr = np.mean(recent_range) if recent_range else o * 0.005
                ap = atr / o
                if ap < 0.005: lev = 3.0
                elif ap < 0.01: lev = 2.0
                elif ap < 0.015: lev = 1.5
                elif ap < 0.03: lev = 0.5
                else: lev = 0
                
                cap_ratio = max(0.3, capital / CAP)
                max_lots = max(1, int(lev * base * cap_ratio)) if lev > 0 else 0
                
                # Exit
                new_positions = []
                for ep2, dr2, lots2 in positions:
                    sd_val = sm * atr; td_val = rr_val * sd_val
                    hit = False; pnl2 = 0
                    if dr2 == 'LONG':
                        if h >= ep2 + td_val: pnl2 = td_val * mult * lots2 - 2*cost*ep2*mult*lots2; hit = True
                        elif l <= ep2 - sd_val: pnl2 = -sd_val * mult * lots2 - 2*cost*ep2*mult*lots2; hit = True
                    else:
                        if l <= ep2 - td_val: pnl2 = td_val * mult * lots2 - 2*cost*ep2*mult*lots2; hit = True
                        elif h >= ep2 + sd_val: pnl2 = -sd_val * mult * lots2 - 2*cost*ep2*mult*lots2; hit = True
                    if hit:
                        capital += pnl2; exits += 1
                        if pnl2 > 0: wins += 1
                    else:
                        new_positions.append((ep2, dr2, lots2))
                positions = new_positions
                
                if capital > peak: peak = capital
                dd_pct = (peak - capital) / peak * 100
                if dd_pct > max_dd: max_dd = dd_pct
                
                # Entry
                if max_lots > 0 and i < len(test_m30) - 5:
                    f = build_f_intraday(test_m30, i)
                    if f is not None:
                        prob = model.predict_proba(f.reshape(1, -1))[0]
                        c2 = prob[1] if prob[1] > 0.5 else 1 - prob[1]
                        if c2 >= conf:
                            dr = 'LONG' if prob[1] > 0.5 else 'SHORT'
                            positions.append((o, dr, max_lots))
            
            pnl = capital - CAP
            test_days = len(test_m30) / 13  # ~13 bars per day
            yrs = test_days / 252
            if yrs > 0 and capital > 0 and pnl > -CAP * 0.9:
                ann = ((capital / CAP) ** (1.0 / yrs) - 1) * 100
            else:
                ann = -100
            wr = wins / exits * 100 if exits > 0 else 0
            
            label = '  <--' if pnl > 0 else ''
            print(f'{sm:<7.1f} {rr_val:<5.1f} {conf:<6.2f} {pnl/10000:<+8.1f}万 {ann:<+7.1f}% {max_dd:<6.1f}% {exits:<6} {wr:<5.0f}%{label}')
            results.append({'sm': sm, 'rr': rr_val, 'conf': conf, 'pnl': pnl, 'ann': ann, 'dd': max_dd, 'exits': exits, 'wr': wr})

results.sort(key=lambda x: -x['pnl'])
with open('/home/a/prophet_futures/prophet_futures/approach2_intraday.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('\n=== TOP 3 ===')
for r in results[:3]:
    print(f'  stop={r["sm"]:.1f}xATR RR={r["rr"]} conf={r["conf"]} '
          f'pnl={r["pnl"]/10000:+.1f}万 ann={r["ann"]:+.0f}% DD={r["dd"]:.0f}% {r["exits"]}t WR={r["wr"]:.0f}%')
print('Saved: approach2_intraday.json')
