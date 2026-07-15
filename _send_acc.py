#!/usr/bin/env python3
import sys, os, pickle, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_v4 import fd, bf, S, MD, eq, ls, lv28, lv29, lv30, send, md
from realtime_data import get_realtime_quote

CAPITAL = 300000
MULT = {'lh2609': 16, 'jm2609': 60}

models = [
    ('V25/V28', '_xgb.pkl'),
    ('V29', '_xgb_new.pkl'),
    ('V30', '_xgb_calibrated.pkl'),
]

ele = [md('**模型评估 | 7/8~7/14**'), md('')]

# ── 表1: 逐日明细(含涨幅) ──
ele.append(md('**逐日预测** (多=预测涨 空=预测跌 ✓=对 ✗=错)'))

for sk in ['lh2609', 'jm2609']:
    cfg = S[sk]
    df = fd(cfg['code'])
    if df is None: continue
    df['dt'] = pd.to_datetime(df['date'])
    
    ele.append(md(''))
    ele.append(md('**' + cfg['cn'] + '**'))
    rows = ['| 日期 | 收盘 | 次日涨跌 | V25/V28 | V29 | V30 |']
    rows.append('|------|------|------|------|------|------|')
    
    for i in range(65, len(df)-1):
        d_str = str(df.iloc[i]['date'])
        if d_str < '2026-07-08': continue
        price = float(df.iloc[i]['close'])
        next_p = float(df.iloc[i+1]['close'])
        chg = next_p - price
        chg_pct = (next_p - price) / price * 100
        actual = ('涨+' if chg > 0 else '跌') + str(int(abs(chg))) + '点(' + ('+' if chg>0 else '') + '{:.1f}'.format(chg_pct) + '%)'
        ft = bf(df, i, 60)
        if ft is None: continue
        
        row = '| ' + d_str[5:] + ' | ' + str(int(price)) + ' | ' + actual + ' '
        for mlbl, msuffix in models:
            mp = MD + '/' + sk + msuffix
            if not os.path.exists(mp): continue
            m = pickle.load(open(mp, 'rb'))
            prob = float(m.predict_proba(ft.reshape(1, -1))[0][1])
            pred_up = prob > 0.5
            ok = pred_up == (chg > 0)
            pred = '多' if pred_up else '空'
            mark = '✓' if ok else '✗'
            row += '| ' + pred + mark + '(' + str(int(max(prob,1-prob)*100)) + '%) '
        row += '|'
        rows.append(row)
    ele.append(md('\n'.join(rows)))

ele.append(md(''))

# ── 表2: 准确率 ──
ele.append(md('**准确率汇总**'))
rows2 = ['| 模型 | 生猪 | 焦煤 | 合计 |']
rows2.append('|------|------|------|------|')
all_r = []

for mlbl, msuffix in models:
    lh_c = lh_t = jm_c = jm_t = 0
    for sk in ['lh2609', 'jm2609']:
        df = fd(S[sk]['code'])
        if df is None: continue
        df['dt'] = pd.to_datetime(df['date'])
        mp = MD + '/' + sk + msuffix
        if not os.path.exists(mp): continue
        m = pickle.load(open(mp, 'rb'))
        correct = total = 0
        for i in range(65, len(df)-1):
            if str(df.iloc[i]['date']) < '2026-07-08': continue
            ft = bf(df, i, 60)
            if ft is None: continue
            try:
                prob = float(m.predict_proba(ft.reshape(1, -1))[0][1])
            except: continue
            if (prob > 0.5) == (float(df.iloc[i+1]['close']) > float(df.iloc[i]['close'])): correct += 1
            total += 1
        if sk == 'lh2609': lh_c, lh_t = correct, total
        else: jm_c, jm_t = correct, total
    
    all_c = lh_c + jm_c; all_t = lh_t + jm_t
    rows2.append('| **' + mlbl + '** | ' + str(lh_c) + '/' + str(lh_t) + '(' + str(int(lh_c/max(lh_t,1)*100)) + '%) | ' + str(jm_c) + '/' + str(jm_t) + '(' + str(int(jm_c/max(jm_t,1)*100)) + '%) | ' + str(all_c) + '/' + str(all_t) + '(' + str(int(all_c/max(all_t,1)*100)) + '%) |')
    all_r.append((mlbl, all_c/max(all_t,1)*100))

ele.append(md('\n'.join(rows2)))
ele.append(md(''))

# ── 表3: 动态市值盈亏 —— 用真实权益 ──
prices = {}
for sk in ['lh2609', 'jm2609']:
    rt = get_realtime_quote(sk)
    prices[sk] = rt['price'] if rt else None

ele.append(md('**账户市值变动(7/8→7/14)**'))
rows3 = ['| 版本 | 7/8市值 | 7/14市值 | 净盈亏 | 现金 | 持仓 |']
rows3.append('|------|------|------|------|------|------|')

ver_config = [
    ('V25', ls(), 'paper_state.json'),
    ('V28', lv28(), 'paper_state_v28.json'),
    ('V29', lv29(), 'paper_state_v29.json'),
    ('V30', lv30(), 'paper_state_v30.json'),
]

for ver_name, st_now, fname in ver_config:
    with open(fname) as f:
        raw = json.load(f)
    
    # 7/8 权益: 从交易反推
    trades_all = raw.get('trades', [])
    
    # 累计交易盈亏到7/7
    pnl_to_0707 = 0
    for t in trades_all:
        t_time = t.get('time', '') or t.get('exit_time', '')
        if t_time and t_time[:10] < '2026-07-08':
            pnl_to_0707 += t.get('pnl', 0) or t.get('pnl_amount', 0)
    
    # V25 有 equity_history 可以直接查
    eq_hist = raw.get('equity_history', [])
    eq_0708 = None
    for e in eq_hist:
        if e['time'][:10] == '2026-07-08':
            if eq_0708 is None:
                eq_0708 = e['equity']
    
    if eq_0708 is None:
        # 重建: 30万 + 7/8前交易盈亏
        eq_0708 = CAPITAL + pnl_to_0707
    
    # 当前市值
    eq_now = eq(st_now, prices)
    
    net = eq_now - eq_0708
    
    # 当前现金
    cash_now = st_now['cash']
    
    # 持仓描述
    pos_desc = []
    for sk_k, pl in st_now.get('positions', {}).items():
        if isinstance(pl, list):
            tv = sum(p['vol'] for p in pl); d = pl[0]['dir']
            ae = sum(p['entry']*p['vol'] for p in pl) / tv
            pos_desc.append(('多' if d=='LONG' else '空') + str(tv))
        else:
            pos_desc.append(('多' if pl['dir']=='LONG' else '空') + str(pl['vol']))
    pos_str = ','.join(pos_desc) if pos_desc else '空'
    
    rows3.append('| **' + ver_name + '** | ' + str(int(eq_0708)) + ' | ' + str(int(eq_now)) + ' | ' + ('+' if net>=0 else '') + str(int(net)) + '(' + ('+' if net>0 else '') + '{:.1f}'.format(net/eq_0708*100) + '%) | ' + str(int(cash_now)) + ' | ' + pos_str + ' |')

ele.append(md('\n'.join(rows3)))
ele.append(md(''))

by_acc = sorted(all_r, key=lambda x: -x[1])
rank_str = ' > '.join(r[0] + '(' + str(int(r[1])) + '%)' for r in by_acc)
ele.append(md('**准确率**: ' + rank_str))

send('模型评估 7/8~7/14', ele, 'blue')
print('OK')
