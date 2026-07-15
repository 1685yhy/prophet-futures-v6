#!/usr/bin/env python3
import sys, os, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_v4 import fd, bf, S, MD, eq, ls, lv28, lv29, lv30, send, md
from realtime_data import get_realtime_quote

# 实时价
prices = {}
for sk in ['lh2609', 'jm2609']:
    rt = get_realtime_quote(sk)
    prices[sk] = rt['price'] if rt else None

ele = [md('**📊 纸盘状态 + 模型评估 | 7/14**'), md('')]

# ── 表1: 止损状态 ──
ele.append(md('**止损状态**'))
rows = ['| 版本 | 品种 | 方向 | 手数 | 均价 | 止损 | 现价 | 距止损 |']
rows.append('|------|------|------|------|------|------|------|------|')
ver_list = [
    ('V25', ls(), 'stop'),
    ('V28', lv28(), '_trail'),
    ('V29', lv29(), '_trail'),
    ('V30', lv30(), '_trail'),
]
for ver_name, st, mkey in ver_list:
    for sk, pl in st.get('positions', {}).items():
        cfg = S.get(sk, {})
        cur = prices.get(sk)
        if isinstance(pl, list):
            tv = sum(p['vol'] for p in pl); d = pl[0]['dir']
            ae = sum(p['entry']*p['vol'] for p in pl)/tv; sl = pl[0].get(mkey, 0)
        else:
            tv = pl['vol']; d = pl['dir']; ae = pl['entry']; sl = pl.get(mkey, 0)
        dist = abs(cur - sl) if cur else 0
        triggered = (d == 'LONG' and cur and cur <= sl) or (d == 'SHORT' and cur and cur >= sl)
        flag = '🔴触发!' if triggered else f'距{dist:.0f}'
        rows.append(f'| {ver_name} | {cfg.get("cn",sk)} | {"多" if d=="LONG" else "空"} | {tv} | {ae:.0f} | {sl:.0f} | {cur:.0f} | {flag} |')
ele.append(md('\n'.join(rows)))
ele.append(md(''))

# ── 表2: 模型准确率 + 盈亏 ──
ele.append(md('**模型准确率(6/22起日线) + 盈亏**'))
rows2 = ['| 版本 | LH准确 | JM准确 | 平均 | 权益 | 盈亏 |']
rows2.append('|------|------|------|------|------|------|')
models = [
    ('V25', '_xgb.pkl', ls()),
    ('V28', '_xgb.pkl', lv28()),
    ('V29', '_xgb_new.pkl', lv29()),
    ('V30', '_xgb_calibrated.pkl', lv30()),
]
all_r = []
for ver, msuffix, st in models:
    accs = {}
    for sk in ['lh2609', 'jm2609']:
        df = fd(S[sk]['code'])
        if df is None: continue
        mp = MD + '/' + sk + msuffix
        if not os.path.exists(mp): continue
        m = pickle.load(open(mp, 'rb'))
        correct = total = 0
        df['dt'] = pd.to_datetime(df['date'])
        for i in range(65, len(df)-1):
            if str(df.iloc[i]['date']) < '2026-06-22': continue
            ft = bf(df, i, 60)
            if ft is None: continue
            try:
                prob = float(m.predict_proba(ft.reshape(1, -1))[0][1])
            except: continue
            if (prob > 0.5) == (float(df.iloc[i+1]['close']) > float(df.iloc[i]['close'])):
                correct += 1
            total += 1
        accs[sk] = (correct, total)
    
    lh_c, lh_t = accs.get('lh2609', (0,1))
    jm_c, jm_t = accs.get('jm2609', (0,1))
    lh_a = lh_c/max(lh_t,1)*100
    jm_a = jm_c/max(jm_t,1)*100
    avg_a = (lh_a + jm_a) / 2
    
    equity = eq(st, prices)
    pnl = equity - 300000
    pnl_pct = pnl / 300000 * 100
    
    rows2.append(f'| {ver} | {lh_a:.0f}% | {jm_a:.0f}% | {avg_a:.0f}% | ¥{equity:,.0f} | {pnl:+,.0f}({pnl_pct:+.1f}%)')
    all_r.append((ver, avg_a, pnl_pct))

ele.append(md('\n'.join(rows2)))
ele.append(md(''))

# 排名
by_acc = sorted(all_r, key=lambda x: -x[1])
by_pnl = sorted(all_r, key=lambda x: -x[2])
ele.append(md(f'**准确率**: {" > ".join(f"{r[0]}({r[1]:.0f}%)" for r in by_acc)}'))
ele.append(md(f'**盈亏**: {" > ".join(f"{r[0]}({r[2]:+.1f}%)" for r in by_pnl)}'))

send('📊 纸盘+模型评估', ele, 'blue')
print('OK')
