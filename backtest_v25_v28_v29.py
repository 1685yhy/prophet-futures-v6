#!/usr/bin/env python3
"""Prophet Futures — V25/V28/V29 完整回测 (使用当前模型文件)
严格复用 v28_full_wf.py 的交易逻辑: run_v25 / run_v28 + compound_stats
唯一区别: 不滚动训练，直接用部署的模型文件
"""
import sys, os, pickle, time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import build_features

CAPITAL = 300000
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# ── 配置 (和纸盘完全一致) ──
V25_CFG = {
    'lh2609': {'code':'LH0','name':'LH','multiplier':16,'cost':0.0006,'max_pos':6,
               'hard_atr':0.8,'trail_atr':2.0,'be_atr':1.0,'trail_dist':1.5,'rr':4.0,
               'model_low':0.35,'model_high':0.65,'confirm_bars':2,'min_hold':3},
    'jm2609': {'code':'JM0','name':'JM','multiplier':60,'cost':0.0011,'max_pos':4,
               'hard_atr':1.8,'trail_atr':3.0,'be_atr':2.0,'trail_dist':2.5,'rr':3.5,
               'model_low':0.30,'model_high':0.70,'confirm_bars':3,'min_hold':5},
}

V28_CFG = {
    'lh2609': {'code':'LH0','name':'LH','multiplier':16,'cost':0.0006,
               'max_pos':6,'max_total':12,'atr_stop':1.5,'rr':4.0,
               'add_conf':0.65,'add_atr':2.0,'reduce_conf':0.55,'reverse_conf':0.35,
               'trail_atr':2.0,'be_atr':1.0,'min_hold':3},
    'jm2609': {'code':'JM0','name':'JM','multiplier':60,'cost':0.0011,
               'max_pos':4,'max_total':8,'atr_stop':2.0,'rr':3.5,
               'add_conf':0.65,'add_atr':2.5,'reduce_conf':0.55,'reverse_conf':0.30,
               'trail_atr':3.0,'be_atr':2.0,'min_hold':5},
}


def fetch_data(code, days=1500):
    import akshare as ak, pandas as pd
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=days)
    df = ak.futures_main_sina(symbol=code,
                               start_date=start.strftime('%Y%m%d'),
                               end_date=end.strftime('%Y%m%d'))
    df.columns = ['date','open','high','low','close','volume','oi','settle']
    for c in ['open','high','low','close','volume','oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)


def calc_atr(df, idx, period=20):
    if idx < period: return None
    tr = [max(float(df.iloc[i]['high'])-float(df.iloc[i]['low']),
              abs(float(df.iloc[i]['high'])-float(df.iloc[i-1]['close'])),
              abs(float(df.iloc[i]['low'])-float(df.iloc[i-1]['close'])))
          for i in range(idx-period+1, idx+1)]
    return float(np.mean(tr))


# ── V25 逻辑: 精准复用 v28_full_wf.py run_v25 ──
def run_v25(df, model, cfg):
    trades = []; pos = None
    for i in range(70, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        try: prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
        except: continue
        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high'])
        low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, 20)
        if atr is None or price <= 0: continue

        if pos:
            d, entry, stop, tp, entry_i, vol = pos
            if d == 'LONG':
                if low <= stop:
                    pnl = ((stop-entry)/entry) - cfg['cost']*2
                    trades.append({'pnl':pnl, 'bars':i-entry_i, 'type':'STOP'}); pos = None
                elif high >= tp:
                    pnl = ((tp-entry)/entry) - cfg['cost']*2
                    trades.append({'pnl':pnl, 'bars':i-entry_i, 'type':'TP'}); pos = None
            else:
                if high >= stop:
                    pnl = ((entry-stop)/entry) - cfg['cost']*2
                    trades.append({'pnl':pnl, 'bars':i-entry_i, 'type':'STOP'}); pos = None
                elif low <= tp:
                    pnl = ((entry-tp)/entry) - cfg['cost']*2
                    trades.append({'pnl':pnl, 'bars':i-entry_i, 'type':'TP'}); pos = None

        if pos is None:
            sd2 = atr * cfg['hard_atr']
            sd = 'LONG' if prob > 0.5 else 'SHORT'
            if sd == 'LONG':
                sv = price - sd2; tv = price + sd2 * cfg['rr']
                if low > sv: pos = (sd, price, sv, tv, i, 1)
            else:
                sv = price + sd2; tv = price - sd2 * cfg['rr']
                if high < sv: pos = (sd, price, sv, tv, i, 1)

    if pos:
        d, entry, stop, tp, entry_i, vol = pos
        lp = float(df.iloc[-1]['close'])
        pnl = ((lp-entry)/entry if d=='LONG' else (entry-lp)/entry) - cfg['cost']*2
        trades.append({'pnl':pnl, 'bars':len(df)-1-entry_i, 'type':'EOD'})
    return trades


# ── V28 逻辑: 精准复用 v28_full_wf.py run_v28 ──
def run_v28(df, model, cfg):
    trades = []; positions = []; total_lots = 0; rev_bars = 0
    for i in range(70, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        try: prob = float(model.predict_proba(feats.reshape(1,-1))[0][1])
        except: continue
        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high'])
        low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, 20)
        if atr is None or price <= 0: continue

        cur_dir = 'LONG' if prob > 0.5 else 'SHORT'
        conf = prob if prob > 0.5 else 1 - prob

        surviving = []
        for pos in positions:
            d, entry, trail, entry_i, vol = pos
            bars = i - entry_i
            pnl_pct = (price-entry)/entry if d=='LONG' else (entry-price)/entry
            pnl_atr = pnl_pct * entry / atr if atr > 0 else 0

            if d == 'LONG':
                hs = price - atr * cfg['atr_stop']
                if pnl_atr > cfg['trail_atr']:
                    trail = max(trail, price - atr*(cfg['atr_stop']-0.3))
                if pnl_atr > cfg['be_atr']:
                    trail = max(trail, entry)
                es = max(hs, trail)
                should_reduce = (cur_dir=='LONG' and conf < cfg['reduce_conf'] and bars >= cfg['min_hold'])
                should_reverse = (prob < cfg['reverse_conf'] and bars >= cfg['min_hold'])
                if low <= es:
                    ep = es; pnl = ((ep-entry)/entry) - cfg['cost']*2
                    trades.append({'pnl':pnl, 'bars':bars, 'type':'STOP', 'vol':vol})
                    total_lots -= vol
                    rev_bars += 1 if prob < 0.5 else 0
                elif should_reverse and rev_bars >= 2:
                    pnl = pnl_pct - cfg['cost']*2
                    trades.append({'pnl':pnl, 'bars':bars, 'type':'REVERSE', 'vol':vol})
                    total_lots -= vol; rev_bars += 1
                elif should_reduce and vol > 1:
                    rv = vol // 2; pnl = pnl_pct - cfg['cost']
                    trades.append({'pnl':pnl*0.5, 'bars':bars, 'type':'REDUCE', 'vol':rv})
                    total_lots -= rv
                    surviving.append((d, entry, trail, entry_i, vol-rv)); rev_bars = 0
                else:
                    surviving.append((d, entry, trail, entry_i, vol))
                    rev_bars = 0 if cur_dir=='LONG' else rev_bars
            else:
                hs = price + atr * cfg['atr_stop']
                if -pnl_atr > cfg['trail_atr']:
                    trail = min(trail, price + atr*(cfg['atr_stop']-0.3))
                if -pnl_atr > cfg['be_atr']:
                    trail = min(trail, entry)
                es = min(hs, trail)
                should_reduce = (cur_dir=='SHORT' and conf < cfg['reduce_conf'] and bars >= cfg['min_hold'])
                should_reverse = (prob > 1-cfg['reverse_conf'] and bars >= cfg['min_hold'])
                if high >= es:
                    ep = es; pnl = ((entry-ep)/entry) - cfg['cost']*2
                    trades.append({'pnl':pnl, 'bars':bars, 'type':'STOP', 'vol':vol})
                    total_lots -= vol
                    rev_bars += 1 if prob > 0.5 else 0
                elif should_reverse and rev_bars >= 2:
                    pnl = pnl_pct - cfg['cost']*2
                    trades.append({'pnl':pnl, 'bars':bars, 'type':'REVERSE', 'vol':vol})
                    total_lots -= vol; rev_bars += 1
                elif should_reduce and vol > 1:
                    rv = vol // 2; pnl = pnl_pct - cfg['cost']
                    trades.append({'pnl':pnl*0.5, 'bars':bars, 'type':'REDUCE', 'vol':rv})
                    total_lots -= rv
                    surviving.append((d, entry, trail, entry_i, vol-rv)); rev_bars = 0
                else:
                    surviving.append((d, entry, trail, entry_i, vol))
                    rev_bars = 0 if cur_dir=='SHORT' else rev_bars

        positions = surviving

        atr_pct = atr / price
        if atr_pct < 0.01: lev = 3.0
        elif atr_pct < 0.02: lev = 2.0
        elif atr_pct < 0.03: lev = 1.5
        else: lev = 0.5
        ps = max(1, int(lev * (cfg['max_pos'] // 2))) if lev > 0 else 0

        if ps > 0 and total_lots + ps <= cfg['max_total']:
            sd = 'LONG' if prob > 0.5 else 'SHORT'
            sd2 = atr * cfg['atr_stop']
            if not positions:
                if sd == 'LONG':
                    sv = price - sd2
                    if low > sv: positions.append((sd, price, sv, i, ps)); total_lots += ps
                else:
                    sv = price + sd2
                    if high < sv: positions.append((sd, price, sv, i, ps)); total_lots += ps
            else:
                ed = positions[0][0]
                if sd == ed:
                    avg_entry = np.mean([p[1] for p in positions])
                    pa = (price-avg_entry)/atr if sd=='LONG' else (avg_entry-price)/atr
                    if conf > cfg['add_conf'] and pa > cfg['add_atr']:
                        if sd == 'LONG':
                            sv = price - sd2
                            if low > sv: positions.append((sd, price, sv, i, ps)); total_lots += ps
                        else:
                            sv = price + sd2
                            if high < sv: positions.append((sd, price, sv, i, ps)); total_lots += ps

    lp = float(df.iloc[-1]['close'])
    for pos in positions:
        d, entry, trail, entry_i, vol = pos
        pnl = ((lp-entry)/entry if d=='LONG' else (entry-lp)/entry) - cfg['cost']*2
        trades.append({'pnl':pnl, 'bars':len(df)-1-entry_i, 'type':'EOD', 'vol':vol})
    return trades


def compound_stats(trades):
    if not trades: return {'wr':0, 'eq':1.0, 'mdd':0, 'n':0, 'pf':0,
                           'avg_win':0, 'avg_loss':0, 'pnl_sum':0}
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins) / len(trades)
    eq = 1.0; peak = 1.0; mdd = 0
    for t in trades:
        eq *= (1 + t['pnl']); peak = max(peak, eq); mdd = min(mdd, (eq-peak)/peak)
    gw = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = gw/gl if gl > 0 else 99
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    return {'wr':wr, 'eq':eq, 'mdd':mdd, 'n':len(trades), 'pf':pf,
            'avg_win':avg_win, 'avg_loss':avg_loss, 'pnl_sum':eq-1}


def report(name, trades, capital=CAPITAL):
    s = compound_stats(trades)
    return {
        '最终权益': f'¥{capital * s["eq"]:,.0f}',
        '总收益率': f'{(s["eq"]-1)*100:+.1f}%',
        '最大回撤': f'{s["mdd"]*100:.1f}%',
        '交易次数': s['n'],
        '胜率': f'{s["wr"]*100:.1f}%',
        '平均盈利': f'{s["avg_win"]*100:+.2f}%',
        '平均亏损': f'{s["avg_loss"]*100:.2f}%',
        '盈亏比': f'{s["pf"]:.2f}',
    }


def main():
    import pandas as pd
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 30)

    print("=" * 70)
    print("Prophet Futures — V25/V28/V29 当前部署模型回测")
    print(f"资金: ¥{CAPITAL:,} | 逻辑: 严格复用 v28_full_wf.py | 模型: 部署文件")
    print("=" * 70)

    for sym_name, sym, code in [('LH 生猪', 'lh2609', 'LH0'), ('JM 焦煤', 'jm2609', 'JM0')]:
        print(f"\n{'─'*70}")
        print(f"  {sym_name} ({sym})")
        print(f"{'─'*70}")

        print(f"  获取数据...", end=' ', flush=True)
        df = fetch_data(code, days=1500)
        print(f"{len(df)} 条日线")

        results = {}

        # V25 params
        v25_cfg = V25_CFG[sym]

        # 模型: V25用校准模型, V28用旧模型, V29用新模型
        model_files = {
            'V25': f'{sym}_xgb_calibrated.pkl',
            'V28': f'{sym}_xgb.pkl',
            'V29': f'{sym}_xgb_new.pkl',
        }

        for ver, fname in model_files.items():
            path = os.path.join(MODEL_DIR, fname)
            if not os.path.exists(path):
                print(f"  {ver}: 模型 {fname} 不存在, 跳过")
                continue

            model = pickle.load(open(path, 'rb'))
            print(f"  {ver} 回测...", end=' ', flush=True)
            t0 = time.time()

            if ver == 'V25':
                trades = run_v25(df, model, v25_cfg)
            else:
                trades = run_v28(df, model, V28_CFG[sym])

            results[ver] = report(ver, trades)
            print(f"完成 ({time.time()-t0:.1f}s, {len(trades)}笔)")

        if results:
            df_r = pd.DataFrame(results).T
            print(f"\n  {'指标':<10}", end='')
            for v in results: print(f"{v:>14}", end='')
            print(f"\n  {'─'*10}─" + "─"*14*len(results))
            for metric in results[list(results.keys())[0]]:
                print(f"  {metric:<10}", end='')
                for v in results:
                    print(f"{results[v][metric]:>14}", end='')
                print()

    print(f"\n{'='*70}")
    print("回测完成 — 逻辑 = v28_full_wf.py, 模型 = 当前部署文件")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
