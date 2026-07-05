#!/usr/bin/env python3
"""
Prophet Futures v28 — 模型驱动动态交易
基底: V25 ATR止损 (LH 1.5× JM 2.0×)
动态: 每根K线模型重新判断 → 持有/加仓/减仓/反手/移止损
验证: 500次 Walk-Forward + 滚动训练
"""
import numpy as np, pandas as pd, pickle, os, time, json
from datetime import datetime, timedelta
import akshare as ak
import xgboost as xgb

# ===== CONFIG =====
SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'multiplier': 16, 'cost': 0.0006,
        'max_pos': 6, 'max_total': 12,  # v28: 总仓位上限
        'atr_stop_mult': 1.5,  # V25 base
        'atr_period': 20,
        'rr_base': 4.0,
        # 动态阈值
        'add_conf': 0.65,      # 加仓置信度 (>0.65做多或<0.35做空)
        'add_atr': 2.0,        # 盈利>2ATR才加仓
        'reduce_conf': 0.55,   # 减仓置信度 (方向不变但信心降低)
        'reverse_conf': 0.35,  # 反手阈值 (prob<0.35做多→转空, >0.65做空→转多)
        'trail_atr': 2.0,      # 盈利>2ATR启动移动止损
        'be_atr': 1.0,         # 盈利>1ATR保本
        'min_hold': 3,         # 最小持仓bar数
    },
    'jm2609': {
        'code': 'JM0', 'name': 'JM', 'multiplier': 60, 'cost': 0.0011,
        'max_pos': 4, 'max_total': 8,
        'atr_stop_mult': 2.0,
        'atr_period': 20,
        'rr_base': 3.5,
        'add_conf': 0.65,
        'add_atr': 2.5,
        'reduce_conf': 0.55,
        'reverse_conf': 0.30,
        'trail_atr': 3.0,
        'be_atr': 2.0,
        'min_hold': 5,
    },
}

def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx-window:idx+1]
    c = w['close'].values.astype(float)
    o = w['open'].values.astype(float)
    h = w['high'].values.astype(float)
    l = w['low'].values.astype(float)
    v = w['volume'].values.astype(float)
    oi = w['oi'].values.astype(float)
    f = []
    if idx >= 1:
        f.append(float((o[-1]-c[-2])/c[-2])); f.append(abs(f[-1]))
    else:
        f.extend([0.0, 0.0])
    for lag in [1,3,5,10,20]:
        f.append(float((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0))
    for p in [5,10,20,60]:
        ma = np.mean(c[-min(p,len(c)):])
        f.append(float((c[-1]-ma)/ma))
    f.append(float(np.std(c[-20:])/np.mean(c[-20:])))
    f.append(float((h[-1]-l[-1])/c[-1]))
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
    f.append(float(v[-1]/vma))
    f.append(float(oi[-1]/np.mean(oi[-20:])) if len(oi)>=20 and np.mean(oi[-20:])>0 else 1)
    ema12 = c[-1]; ema26 = c[-1]
    for j in range(len(c)-2, -1, -1):
        ema12 = (2/13)*c[j]+(11/13)*ema12; ema26 = (2/27)*c[j]+(25/27)*ema26
    f.append(float((ema12-ema26)/c[-1]))
    dd_ = np.diff(c[-15:])
    g = float(dd_[dd_>0].sum()) if len(dd_[dd_>0])>0 else 0
    lo = float(abs(dd_[dd_<0].sum())) if len(dd_[dd_<0])>0 else 1e-10
    f.append(float(100-100/(1+g/lo) if lo>0 else 50))
    bb = np.std(c[-20:])
    ma20 = np.mean(c[-20:])
    f.append(float((c[-1]-ma20)/(2*bb+1e-10)))
    f.append(float(c[-1]/1000.0))
    return np.array(f, dtype=np.float32)

def calc_atr(df, idx, period=20):
    if idx < period: return None
    vals = []
    for i in range(idx-period+1, idx+1):
        vals.append(abs(float(df.iloc[i]['high'])-float(df.iloc[i]['low'])))
    return np.mean(vals)

def train_model(X, y):
    if len(X) < 50: return None
    m = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=1, verbosity=0
    )
    m.fit(np.array(X)[-1000:], np.array(y)[-1000:])  # last 1000 samples
    return m

def make_label(df, idx, horizon=1):
    """Label: next day close vs current close"""
    if idx + horizon >= len(df): return None
    c0 = float(df.iloc[idx]['close'])
    c1 = float(df.iloc[idx+horizon]['close'])
    return 1 if c1 > c0 else 0

def run_v25(df, model, cfg):
    """V25 baseline: ATR stop + fixed RR TP, 单一持仓"""
    trades = []
    pos = None
    warmup = 70
    
    for i in range(warmup, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        try:
            prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
        except: continue
        
        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high'])
        low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, cfg['atr_period'])
        if atr is None or price <= 0: continue
        
        if pos:
            d, entry, stop, tp, entry_i, vol = pos
            if d == 'LONG':
                if low <= stop:
                    pnl = ((stop-entry)/entry) - cfg['cost']*2
                    trades.append({'pnl': pnl, 'bars': i-entry_i, 'type': 'STOP'})
                    pos = None
                elif high >= tp:
                    pnl = ((tp-entry)/entry) - cfg['cost']*2
                    trades.append({'pnl': pnl, 'bars': i-entry_i, 'type': 'TP'})
                    pos = None
            else:
                if high >= stop:
                    pnl = ((entry-stop)/entry) - cfg['cost']*2
                    trades.append({'pnl': pnl, 'bars': i-entry_i, 'type': 'STOP'})
                    pos = None
                elif low <= tp:
                    pnl = ((entry-tp)/entry) - cfg['cost']*2
                    trades.append({'pnl': pnl, 'bars': i-entry_i, 'type': 'TP'})
                    pos = None
        
        if pos is None:
            atr_pct = atr/price
            if atr_pct < 0.01: lev = 3.0
            elif atr_pct < 0.02: lev = 2.0
            elif atr_pct < 0.03: lev = 1.5
            elif atr_pct < 0.05: lev = 0.5
            else: lev = 0
            ps = max(1, int(lev*(cfg['max_pos']//2))) if lev > 0 else 0
            if ps > 0:
                sd = 'LONG' if prob > 0.5 else 'SHORT'
                sd2 = atr * cfg['atr_stop_mult']
                if sd == 'LONG':
                    s_val = price - sd2
                    t_val = price + sd2*cfg['rr_base']
                    if low <= s_val: continue
                else:
                    s_val = price + sd2
                    t_val = price - sd2*cfg['rr_base']
                    if high >= s_val: continue
                pos = (sd, price, s_val, t_val, i, ps)
    
    if pos:
        d, entry, stop, tp, entry_i, vol = pos
        lp = float(df.iloc[-1]['close'])
        pnl = ((lp-entry)/entry if d=='LONG' else (entry-lp)/entry) - cfg['cost']*2
        trades.append({'pnl': pnl, 'bars': len(df)-1-entry_i, 'type': 'EOD'})
    return trades

def run_v28(df, model, cfg):
    """V28: 模型驱动动态决策 — 持有/加仓/减仓/反手"""
    trades = []
    positions = []  # list of (dir, entry, trail_stop, entry_i, vol)
    warmup = 70
    total_lots = 0
    rev_bars = 0  # consecutive reversal bars
    last_dir = None
    
    for i in range(warmup, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        try:
            prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
        except: continue
        
        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high'])
        low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, cfg['atr_period'])
        if atr is None or price <= 0: continue
        
        cur_dir = 'LONG' if prob > 0.5 else 'SHORT'
        conf = prob if prob > 0.5 else 1-prob
        
        # ===== 1. 管理现有持仓 =====
        surviving = []
        for pos in positions:
            d, entry, trail, entry_i, vol = pos
            bars = i - entry_i
            pnl_pct = (price-entry)/entry if d=='LONG' else (entry-price)/entry
            pnl_atr = pnl_pct * entry / atr if atr > 0 else 0
            
            if d == 'LONG':
                hard_stop = price - atr*cfg['atr_stop_mult']
                # 移动止损
                if pnl_atr > cfg['trail_atr']:
                    trail = max(trail, price - atr*(cfg['atr_stop_mult']-0.3))
                if pnl_atr > cfg['be_atr']:
                    trail = max(trail, entry)
                eff_stop = max(hard_stop, trail)
                
                # 减仓判断: 方向不变但信心降低
                should_reduce = (cur_dir == 'LONG' and conf < cfg['reduce_conf'] and bars >= cfg['min_hold'])
                # 反手判断: 模型确认转空
                should_reverse = (prob < cfg['reverse_conf'] and bars >= cfg['min_hold'])
                
                if low <= eff_stop:
                    # 止损触发
                    ep = eff_stop
                    pnl = ((ep-entry)/entry) - cfg['cost']*2
                    trades.append({'pnl': pnl, 'bars': bars, 'type': 'STOP', 'vol': vol})
                    total_lots -= vol
                    rev_bars += 1 if prob < 0.5 else 0
                elif should_reverse and rev_bars >= 2:
                    # 确认反转 → 平多
                    pnl = pnl_pct - cfg['cost']*2
                    trades.append({'pnl': pnl, 'bars': bars, 'type': 'REVERSE', 'vol': vol})
                    total_lots -= vol
                    rev_bars += 1
                elif should_reduce and vol > 1:
                    # 减半仓
                    reduce_vol = vol // 2
                    pnl = pnl_pct - cfg['cost']
                    trades.append({'pnl': pnl*0.5, 'bars': bars, 'type': 'REDUCE', 'vol': reduce_vol})
                    total_lots -= reduce_vol
                    surviving.append((d, entry, trail, entry_i, vol-reduce_vol))
                    rev_bars = 0
                else:
                    surviving.append((d, entry, trail, entry_i, vol))
                    rev_bars = 0 if cur_dir == 'LONG' else rev_bars
            else:  # SHORT
                hard_stop = price + atr*cfg['atr_stop_mult']
                if -pnl_atr > cfg['trail_atr']:
                    trail = min(trail, price + atr*(cfg['atr_stop_mult']-0.3))
                if -pnl_atr > cfg['be_atr']:
                    trail = min(trail, entry)
                eff_stop = min(hard_stop, trail)
                
                should_reduce = (cur_dir == 'SHORT' and conf < cfg['reduce_conf'] and bars >= cfg['min_hold'])
                should_reverse = (prob > 1-cfg['reverse_conf'] and bars >= cfg['min_hold'])
                
                if high >= eff_stop:
                    ep = eff_stop
                    pnl = ((entry-ep)/entry) - cfg['cost']*2
                    trades.append({'pnl': pnl, 'bars': bars, 'type': 'STOP', 'vol': vol})
                    total_lots -= vol
                    rev_bars += 1 if prob > 0.5 else 0
                elif should_reverse and rev_bars >= 2:
                    pnl = pnl_pct - cfg['cost']*2
                    trades.append({'pnl': pnl, 'bars': bars, 'type': 'REVERSE', 'vol': vol})
                    total_lots -= vol
                    rev_bars += 1
                elif should_reduce and vol > 1:
                    reduce_vol = vol // 2
                    pnl = pnl_pct - cfg['cost']
                    trades.append({'pnl': pnl*0.5, 'bars': bars, 'type': 'REDUCE', 'vol': reduce_vol})
                    total_lots -= reduce_vol
                    surviving.append((d, entry, trail, entry_i, vol-reduce_vol))
                    rev_bars = 0
                else:
                    surviving.append((d, entry, trail, entry_i, vol))
                    rev_bars = 0 if cur_dir == 'SHORT' else rev_bars
        
        positions = surviving
        
        # ===== 2. 开仓/加仓 =====
        atr_pct = atr/price
        if atr_pct < 0.01: lev = 3.0
        elif atr_pct < 0.02: lev = 2.0
        elif atr_pct < 0.03: lev = 1.5
        else: lev = 0.5
        ps = max(1, int(lev*(cfg['max_pos']//2))) if lev > 0 else 0
        
        if ps > 0 and total_lots + ps <= cfg['max_total']:
            sd = 'LONG' if prob > 0.5 else 'SHORT'
            sd2 = atr * cfg['atr_stop_mult']
            
            if not positions:
                # 无持仓 → 开新仓
                if sd == 'LONG':
                    s_val = price - sd2
                    if low > s_val:
                        positions.append((sd, price, s_val, i, ps))
                        total_lots += ps
                else:
                    s_val = price + sd2
                    if high < s_val:
                        positions.append((sd, price, s_val, i, ps))
                        total_lots += ps
            else:
                # 有持仓 → 判断是否加仓
                existing_dir = positions[0][0]
                if sd == existing_dir:
                    # 同向 → 需要在盈利且高置信度时才加
                    avg_entry = np.mean([p[1] for p in positions])
                    pnl_atr = (price-avg_entry)/atr if sd=='LONG' else (avg_entry-price)/atr
                    if conf > cfg['add_conf'] and pnl_atr > cfg['add_atr']:
                        if sd == 'LONG':
                            s_val = price - sd2
                            if low > s_val:
                                positions.append((sd, price, s_val, i, ps))
                                total_lots += ps
                        else:
                            s_val = price + sd2
                            if high < s_val:
                                positions.append((sd, price, s_val, i, ps))
                                total_lots += ps
                else:
                    # 反向信号 → 不开，等REVERSE触发
                    pass
    
    # EOD close all
    lp = float(df.iloc[-1]['close'])
    for pos in positions:
        d, entry, trail, entry_i, vol = pos
        pnl = ((lp-entry)/entry if d=='LONG' else (entry-lp)/entry) - cfg['cost']*2
        trades.append({'pnl': pnl, 'bars': len(df)-1-entry_i, 'type': 'EOD', 'vol': vol})
    
    return trades

def analyze_trades(name, trades):
    if not trades: return {'trades': 0, 'wr': 0, 'pnl': 0, 'pf': 0, 'mdd': 0}, "无交易"
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins)/len(trades)
    total_pnl = sum(t['pnl'] for t in trades)
    gw = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = gw/gl if gl > 0 else 99
    
    eq = 1.0; peak = 1.0; mdd = 0
    for t in trades:
        eq *= (1+t['pnl'])
        peak = max(peak, eq)
        mdd = min(mdd, (eq-peak)/peak)
    
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    avg_bars = np.mean([t['bars'] for t in trades])
    total_vol = sum(t.get('vol', 1) for t in trades)
    types = {}
    for t in trades: types[t['type']] = types.get(t['type'],0)+1
    
    d = {'trades': len(trades), 'wr': wr, 'pnl': total_pnl, 'pf': pf, 'mdd': mdd,
         'avg_win': avg_win, 'avg_loss': avg_loss, 'avg_bars': avg_bars,
         'total_vol': total_vol, 'types': types}
    
    s = f"{name}: {len(trades)}笔 WR={wr:.0%} 收益={total_pnl*100:+.1f}% PF={pf:.2f} MDD={mdd*100:.1f}%"
    return d, s

# ===== MAIN =====
print("="*60)
print("  Prophet v28 — 模型驱动动态交易")
print(f"  500-WF 回测 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*60)

for sym_key, cfg in SYMBOLS.items():
    print(f"\n{'='*60}")
    print(f"  {sym_key} ({cfg['name']}) — V25 vs V28")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"  📡 取数据...")
    try:
        end = datetime.now(); start = end - timedelta(days=1500)
        df = ak.futures_main_sina(symbol=cfg['code'], start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        print(f"  ✅ {len(df)}行 {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
    except Exception as e:
        print(f"  ❌ 数据: {e}"); continue
    
    # Precompute features + labels
    print(f"  🔄 预计算特征+标签...")
    X_all, y_all = [], []
    for i in range(70, len(df)-1):
        feats = build_features(df, i, 60)
        if feats is not None:
            X_all.append(feats)
            y_all.append(1 if float(df.iloc[i+1]['close']) > float(df.iloc[i]['close']) else 0)
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all)
    print(f"  ✅ {len(X_all)}条特征")
    
    n_total = len(X_all)
    n_test = 3  # 3-day test periods for more WF windows
    n_train = 250  # rolling train window
    step = 2  # slide 2 days
    
    v25_results = []
    v28_results = []
    wf_count = 0
    
    # Walk-Forward
    for test_start in range(n_train, n_total - n_test, step):
        if wf_count >= 500: break
        test_end = test_start + n_test
        if test_end > n_total: break
        train_start = max(0, test_start - n_train)
        
        X_train = X_all[train_start:test_start]
        y_train = y_all[train_start:test_start]
        X_test = np.array(X_all[test_start:test_end])
        if len(y_train) < 50: continue
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=1, verbosity=0
        )
        model.fit(X_train[-1000:], y_train[-1000:])
        
        # Get test period data: df rows [test_start, test_end+70+1] for warmup
        test_df_start = test_start
        test_df_end = min(len(df), test_end + 71)
        test_df = df.iloc[test_df_start:test_df_end].copy().reset_index(drop=True)
        
        if len(test_df) < 2: continue
        
        # V25
        t25 = run_v25(test_df, model, cfg)
        d25, _ = analyze_trades('V25', t25)
        if d25['trades'] > 0:
            v25_results.append(d25['pnl'])
        
        # V28
        t28 = run_v28(test_df, model, cfg)
        d28, _ = analyze_trades('V28', t28)
        if d28['trades'] > 0:
            v28_results.append(d28['pnl'])
        
        wf_count += 1
        if wf_count % 100 == 0:
            print(f"    WF {wf_count}/500...", flush=True)
    
    print(f"  ✅ {wf_count}次WF完成")
    
    # Aggregate
    def wf_stats(results, name):
        if not results: return f"{name}: 无有效数据", {'n': 0}
        arr = np.array(results)
        pos = (arr > 0).sum() / len(arr)
        mean = np.mean(arr) * 100
        med = np.median(arr) * 100
        std = np.std(arr) * 100
        worst = np.min(arr) * 100
        best = np.max(arr) * 100
        # Compound
        eq = np.prod(1 + arr)
        total = (eq - 1) * 100
        
        # MDD
        cum = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(cum)
        mdd = np.min((cum - peak) / peak) * 100
        
        return f"{name}: {len(arr)}周期 正收益{pos:.0%} 均值{mean:+.1f}% 中位{med:+.1f}% " \
               f"累计{total:+.1f}% MDD{mdd:.1f}% 最差{worst:+.1f}% 最佳{best:+.1f}%", {
            'n': len(arr), 'pos_rate': pos, 'mean': mean, 'median': med,
            'total': total, 'mdd': mdd, 'worst': worst, 'best': best, 'std': std
        }
    
    s25, d25s = wf_stats(v25_results, 'V25')
    s28, d28s = wf_stats(v28_results, 'V28')
    
    print(f"\n  {'─'*55}")
    print(f"  {s25}")
    print(f"  {s28}")
    
    if d25s and d28s and d25s['n'] > 0 and d28s['n'] > 0:
        print(f"  {'─'*55}")
        diff = d28s['total'] - d25s['total']
        mdd_diff = d28s['mdd'] - d25s['mdd']
        # How often V28 beats V25
        better_count = sum(1 for a,b in zip(v28_results, v25_results) if a > b)
        better_pct = better_count / len(v25_results)
        winner = 'V28 🏆' if d28s['total'] > d25s['total'] else 'V25 🏆'
        print(f"  {winner} | 收益差{diff:+.1f}% | 回撤差{mdd_diff:+.1f}% | V28胜率{better_pct:.0%}")

print(f"\n{'='*60}")
print(f"  V28 500-WF 完成")
print(f"{'='*60}")
