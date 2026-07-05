#!/usr/bin/env python3
"""
Prophet v26_tuned — 动态止盈止损回测 (调参版)
vs v25 对比

关键调整:
1. 模型退出阈值放宽: prob<0.35/>0.65 (不再是0.45/0.55)
2. 需连续2根K线确认反转才退出
3. 最小持仓3根K线(避免今日开明日出)
4. 移动止损: 盈利>2ATR才启动
5. 保本止损: 盈利>1ATR移到开仓价
6. 不同品种不同参数
"""
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import pickle, json, os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'multiplier': 16,
        'v25_atr_mult': 1.5, 'v25_rr': 4,
        'v26_hard_atr': 0.8,      # 硬止损ATR倍数
        'v26_trail_atr': 2.0,     # 移动止损启动: 盈利>2ATR
        'v26_be_atr': 1.0,        # 保本启动: 盈利>1ATR
        'v26_trail_dist': 1.5,    # 移动止损距离: ATR×1.5
        'v26_model_low': 0.35,    # 做多→退出: prob<0.35
        'v26_model_high': 0.65,   # 做空→退出: prob>0.65
        'v26_confirm_bars': 2,    # 反转确认需要连续K线数
        'v26_min_hold': 3,        # 最小持仓K线数
        'max_pos': 6,
    },
    'jm2609': {
        'code': 'JM0', 'name': 'JM', 'multiplier': 60,
        'v25_atr_mult': 2.0, 'v25_rr': 3.5,
        'v26_hard_atr': 1.8,      # JM波动大, 硬止损宽 (1.0→1.8)
        'v26_trail_atr': 3.0,     # 盈利>3ATR才移动止损
        'v26_be_atr': 2.0,        # 保本: 盈利>2ATR
        'v26_trail_dist': 2.5,    # 移动止损距离
        'v26_model_low': 0.30,    # 做多→退出 prob<0.30
        'v26_model_high': 0.70,   # 做空→退出 prob>0.70
        'v26_confirm_bars': 3,    # 反转确认3根K线
        'v26_min_hold': 5,        # 最小持仓5根
        'max_pos': 4,
    },
}

def build_features(df, idx, window=60):
    if idx < window + 5: return None
    w = df.iloc[idx-window:idx+1]
    c = w['close'].values; o = w['open'].values
    h = w['high'].values; l = w['low'].values
    v = w['volume'].values; oi = w['oi'].values
    f = []
    if idx >= 1:
        f.append((o[-1]-c[-2])/c[-2]); f.append(abs(f[-1]))
    else:
        f.extend([0, 0])
    for lag in [1,3,5,10,20]:
        f.append((c[-1]-c[-lag-1])/c[-lag-1] if len(c) > lag else 0)
    for p in [5,10,20,60]:
        ma = np.mean(c[-min(p,len(c)):]); f.append((c[-1]-ma)/ma)
    f.append(np.std(c[-20:])/np.mean(c[-20:]))
    f.append((h[-1]-l[-1])/c[-1])
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
    f.append(v[-1]/vma)
    f.append(oi[-1]/np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1)
    ema12 = c[-1]; ema26 = c[-1]
    for j in range(len(c)-2, -1, -1):
        ema12 = (2/13)*c[j] + (11/13)*ema12; ema26 = (2/27)*c[j] + (25/27)*ema26
    f.append((ema12-ema26)/c[-1])
    dd_ = np.diff(c[-15:])
    g = dd_[dd_>0].sum() if len(dd_[dd_>0]) > 0 else 0
    lo = abs(dd_[dd_<0].sum()) if len(dd_[dd_<0]) > 0 else 1e-10
    f.append(100 - 100/(1+g/lo) if lo > 0 else 50)
    bb = np.std(c[-20:]); ma20 = np.mean(c[-20:])
    f.append((c[-1]-ma20)/(2*bb+1e-10))
    f.append(c[-1]/1000.0)
    return np.array(f, dtype=np.float32)

def calc_atr(df, idx, period=20):
    if idx < period: return None
    vals = [abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
            for i in range(idx-period+1, idx+1)]
    return np.mean(vals)

def fetch_history(code, days=1200):
    end = datetime.now(); start = end - timedelta(days=days)
    try:
        df = ak.futures_main_sina(symbol=code, start_date=start.strftime('%Y%m%d'),
                                   end_date=end.strftime('%Y%m%d'))
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except Exception as e:
        print(f"  Fetch error {code}: {e}"); return None

def run_backtest(df, model, cfg):
    results = {'v25': [], 'v26': []}
    v25_pos = None; v26_pos = None
    warmup = 70
    
    # V26 model reversal tracking
    reversal_count = 0  # consecutive bars confirming reversal
    
    for i in range(warmup, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        try:
            prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
        except: continue
        
        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high']); low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, 20)
        if atr is None or price <= 0: continue
        
        atr_pct = atr / price
        if atr_pct < 0.01: leverage = 3.0
        elif atr_pct < 0.02: leverage = 2.0
        elif atr_pct < 0.03: leverage = 1.5
        elif atr_pct < 0.05: leverage = 0.5
        else: leverage = 0
        pos_size = max(1, int(leverage * (cfg['max_pos'] // 2))) if leverage > 0 else 0
        
        # ===== V25 =====
        if v25_pos:
            d, entry, stop, tp, entry_i, vol = v25_pos
            hit_stop = (d == 'LONG' and low <= stop) or (d == 'SHORT' and high >= stop)
            hit_tp = (d == 'LONG' and high >= tp) or (d == 'SHORT' and low <= tp)
            if hit_stop or hit_tp:
                et = 'STOP' if hit_stop else 'TP'
                ep = stop if hit_stop else tp
                pnl = ((ep-entry)/entry if d=='LONG' else (entry-ep)/entry) - 0.0012
                results['v25'].append({
                    'dir': d, 'entry': entry, 'exit': ep, 'type': et,
                    'pnl_pct': pnl, 'bars': i-entry_i,
                    'entry_date': str(df.iloc[entry_i]['date']),
                    'exit_date': str(df.iloc[i]['date']),
                })
                v25_pos = None
        
        if v25_pos is None and pos_size > 0:
            sd = 'LONG' if prob > 0.5 else 'SHORT'
            sd2 = atr * cfg['v25_atr_mult']
            if sd == 'LONG':
                sp = price - sd2; tp = price + (price-sp)*cfg['v25_rr']
            else:
                sp = price + sd2; tp = price - (sp-price)*cfg['v25_rr']
            ah = (sd=='LONG' and low<=sp) or (sd=='SHORT' and high>=sp)
            if not ah: v25_pos = (sd, price, sp, tp, i, pos_size)
        
        # ===== V26 TUNED =====
        if v26_pos:
            d, entry, trail_stop, entry_i, vol, peak_prob = v26_pos
            bars_held = i - entry_i
            
            hard_stop_dist = atr * cfg['v26_hard_atr']
            
            if d == 'LONG':
                hard_stop = price - hard_stop_dist
                
                # Model reversal check (need consecutive confirmation)
                is_reversal = prob < cfg['v26_model_low']
                if is_reversal:
                    reversal_count += 1
                else:
                    reversal_count = 0
                model_exit = (reversal_count >= cfg['v26_confirm_bars']) and (bars_held >= cfg['v26_min_hold'])
                
                # Trailing stop
                if price > entry + atr * cfg['v26_trail_atr']:
                    new_trail = price - atr * cfg['v26_trail_dist']
                    trail_stop = max(trail_stop, new_trail)
                # Breakeven
                if price > entry + atr * cfg['v26_be_atr']:
                    trail_stop = max(trail_stop, entry)
                
                effective_stop = max(hard_stop, trail_stop)
                exit_trigger = (low <= effective_stop) or model_exit
                if exit_trigger:
                    exit_price = effective_stop if low <= effective_stop else price
                    if model_exit: er = 'MODEL'
                    elif trail_stop >= hard_stop: er = 'TRAIL'
                    else: er = 'HARD'
            else:
                hard_stop = price + hard_stop_dist
                
                is_reversal = prob > cfg['v26_model_high']
                if is_reversal: reversal_count += 1
                else: reversal_count = 0
                model_exit = (reversal_count >= cfg['v26_confirm_bars']) and (bars_held >= cfg['v26_min_hold'])
                
                if price < entry - atr * cfg['v26_trail_atr']:
                    new_trail = price + atr * cfg['v26_trail_dist']
                    trail_stop = min(trail_stop, new_trail)
                if price < entry - atr * cfg['v26_be_atr']:
                    trail_stop = min(trail_stop, entry)
                
                effective_stop = min(hard_stop, trail_stop)
                exit_trigger = (high >= effective_stop) or model_exit
                if exit_trigger:
                    exit_price = effective_stop if high >= effective_stop else price
                    if model_exit: er = 'MODEL'
                    elif trail_stop <= hard_stop: er = 'TRAIL'
                    else: er = 'HARD'
            
            if exit_trigger:
                pnl = ((exit_price-entry)/entry if d=='LONG' else (entry-exit_price)/entry) - 0.0012
                results['v26'].append({
                    'dir': d, 'entry': entry, 'exit': exit_price, 'type': er,
                    'pnl_pct': pnl, 'bars': bars_held,
                    'entry_date': str(df.iloc[entry_i]['date']),
                    'exit_date': str(df.iloc[i]['date']),
                })
                v26_pos = None; reversal_count = 0
        
        if v26_pos is None and pos_size > 0:
            sd = 'LONG' if prob > 0.5 else 'SHORT'
            if sd == 'LONG':
                ts = price - atr * (cfg['v26_hard_atr'] + 0.5)
            else:
                ts = price + atr * (cfg['v26_hard_atr'] + 0.5)
            if sd == 'LONG': ah = low <= (price - atr * cfg['v26_hard_atr'])
            else: ah = high >= (price + atr * cfg['v26_hard_atr'])
            if not ah: 
                v26_pos = (sd, price, ts, i, pos_size, prob)
                reversal_count = 0
    
    # Close open
    lp = float(df.iloc[-1]['close']); li = len(df)-1
    for pos, v in [(v25_pos, 'v25'), (v26_pos, 'v26')]:
        if pos:
            if v == 'v25': d, entry, stop, tp, entry_i, vol = pos
            else: d, entry, trail_stop, entry_i, vol, peak_prob = pos
            pnl = ((lp-entry)/entry if d=='LONG' else (entry-lp)/entry) - 0.0012
            results[v].append({
                'dir': d, 'entry': entry, 'exit': lp, 'type': 'EOD',
                'pnl_pct': pnl, 'bars': li-entry_i,
                'entry_date': str(df.iloc[entry_i]['date']),
                'exit_date': str(df.iloc[-1]['date']),
            })
    
    return results

def print_report(name, trades):
    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    if not trades: print(f"    无交易"); return
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    wr = len(wins)/len(trades) if trades else 0
    total_pnl = sum(t['pnl_pct'] for t in trades)
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    avg_bars = np.mean([t['bars'] for t in trades])
    mw = max(t['pnl_pct'] for t in trades); ml = min(t['pnl_pct'] for t in trades)
    gw = sum(t['pnl_pct'] for t in wins); gl = abs(sum(t['pnl_pct'] for t in losses))
    pf = gw/gl if gl>0 else 99
    
    ets = {}
    for t in trades: ets[t['type']] = ets.get(t['type'],0)+1
    
    print(f"    交易次数: {len(trades)}")
    print(f"    胜率: {wr:.0%} ({len(wins)}W/{len(losses)}L)")
    print(f"    总收益: {total_pnl*100:+.1f}%")
    print(f"    平均盈利: {avg_win*100:+.1f}% | 平均亏损: {avg_loss*100:+.1f}%")
    print(f"    最大盈利: {mw*100:+.1f}% | 最大亏损: {ml*100:+.1f}%")
    print(f"    盈亏比: {abs(avg_win/avg_loss) if avg_loss!=0 else 99:.1f}")
    print(f"    盈利因子: {pf:.2f}")
    print(f"    平均持仓: {avg_bars:.0f}根K线")
    print(f"    退出方式: {ets}")
    if len(trades) >= 5:
        print(f"    最近5笔:")
        for t in trades[-5:]:
            e = '🟢' if t['pnl_pct']>0 else '🔴'
            print(f"      {e} {t['dir']:5s} {t['entry_date']} {t['entry']:.0f}→{t['exit_date']} {t['exit']:.0f} [{t['type']:5s}] {t['pnl_pct']*100:+.1f}% ({t['bars']}b)")

if __name__ == '__main__':
    print("="*60)
    print("  Prophet v25 vs v26_tuned — 动态止损止盈 (调参)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    for sym_key, cfg in SYMBOLS.items():
        print(f"\n{'='*60}")
        print(f"  {sym_key.upper()} ({cfg['name']}) — {cfg['code']}")
        print(f"{'='*60}")
        mp = os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')
        if not os.path.exists(mp): print(f"  ❌ Model not found"); continue
        with open(mp, 'rb') as f: model = pickle.load(f)
        print(f"  ✅ Model: {os.path.basename(mp)}")
        print(f"  📡 Fetching {cfg['code']} 1200d...")
        df = fetch_history(cfg['code'], 1200)
        if df is None or len(df)<80: print(f"  ❌ Data: {len(df) if df is not None else 0}"); continue
        print(f"  ✅ {len(df)}行, {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
        print(f"  🔄 Backtesting...")
        results = run_backtest(df, model, cfg)
        
        print_report("V25 — 固定ATR止损 + RR止盈", results['v25'])
        print_report("V26 — 模型确认反转 + 移动止损(调参)", results['v26'])
        
        v25_t = sum(t['pnl_pct'] for t in results['v25'])*100
        v26_t = sum(t['pnl_pct'] for t in results['v26'])*100
        v25_w = len([t for t in results['v25'] if t['pnl_pct']>0])/max(len(results['v25']),1)
        v26_w = len([t for t in results['v26'] if t['pnl_pct']>0])/max(len(results['v26']),1)
        
        # Max drawdown calculation
        def calc_mdd(trades):
            if not trades: return 0
            eq = 1.0; peak = 1.0; mdd = 0
            for t in trades:
                eq *= (1+t['pnl_pct'])
                peak = max(peak, eq)
                mdd = min(mdd, (eq-peak)/peak)
            return mdd
        
        v25_mdd = calc_mdd(results['v25'])*100
        v26_mdd = calc_mdd(results['v26'])*100
        
        print(f"\n  {'='*55}")
        print(f"  ⚡ 对比总结")
        print(f"  {'='*55}")
        print(f"  {'指标':<18} {'V25':>12} {'V26':>12} {'改善':>12}")
        print(f"  {'─'*54}")
        print(f"  {'总收益':<18} {v25_t:>+11.1f}% {v26_t:>+11.1f}% {v26_t-v25_t:>+11.1f}%")
        print(f"  {'胜率':<18} {v25_w:>11.0%} {v26_w:>11.0%} {(v26_w-v25_w)*100:>+11.1f}pp")
        print(f"  {'交易次数':<18} {len(results['v25']):>12} {len(results['v26']):>12}")
        print(f"  {'最大回撤':<18} {v25_mdd:>+11.1f}% {v26_mdd:>+11.1f}% {v26_mdd-v25_mdd:>+11.1f}%")
    
    print(f"\n{'='*60}")
    print(f"  全部回测完成")
    print(f"{'='*60}")
