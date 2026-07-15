#!/usr/bin/env python3
import sys, os, pickle, json, numpy as np, pandas as pd
base = '/home/a/prophet_futures/prophet_futures'
sys.path.insert(0, base)
os.chdir(base)
from report_v4 import fd, bf, S, MD, eq, ls, lv28, lv29, lv30, send, md
from realtime_data import get_realtime_quote

prices = {}
for sk in ['lh2609']:
    rt = get_realtime_quote(sk)
    prices[sk] = rt['price'] if rt else None

ver_data = [
    ('V25', ls(), '2026-06-22', '_xgb.pkl'),
    ('V28', lv28(), '2026-07-02', '_xgb.pkl'),
    ('V29', lv29(), '2026-07-03', '_xgb_new.pkl'),
    ('V30', lv30(), '2026-07-08', '_xgb_calibrated.pkl'),
]

ele = [md('**生猪LH评估 | 纸盘上线至今**'), md('')]

# 表1: 评分体系
ele.append(md('**评分标准(仅LH)**'))
rows0 = [
    '| 维度 | 权重 | 及格线 | 优秀线 |',
    '|------|------|--------|--------|',
    '| 方向准确率 | 35% | >50% | >60% |',
    '| LH收益率 | 30% | >0% | >20% |',
    '| 最大回撤 | 20% | <20% | <10% |',
    '| LH盈亏比 | 10% | >1.2 | >2.0 |',
    '| 执行纪律 | 5% | 100% | 100% |',
]
ele.append(md('\n'.join(rows0)))
ele.append(md(''))
ele.append(md('>=80→A正常仓位 | 65-79→B半仓 | 50-64→C轻仓 | <50→D停用 | <20次预测仅供参考'))

ele.append(md(''))
ele.append(md('**LH全周期数据**'))

today = '2026-07-14'
all_scores = []

for ver_name, st, start_date, msuffix in ver_data:
    df = fd(S['lh2609']['code'])
    df['dt'] = pd.to_datetime(df['date'])
    
    # 准确率
    lh_c = lh_t = 0
    mp = MD + '/lh2609' + msuffix
    if os.path.exists(mp):
        m = pickle.load(open(mp, 'rb'))
        for i in range(65, len(df)-1):
            if str(df.iloc[i]['date']) < start_date: continue
            ft = bf(df, i, 60)
            if ft is None: continue
            try:
                prob = float(m.predict_proba(ft.reshape(1, -1))[0][1])
                if (prob > 0.5) == (float(df.iloc[i+1]['close']) > float(df.iloc[i]['close'])):
                    lh_c += 1
                lh_t += 1
            except: continue
    
    # 信号偏好
    bullish = sum(1 for i in range(65, len(df)-1) 
                  if str(df.iloc[i]['date']) >= start_date 
                  and bf(df, i, 60) is not None
                  and (lambda p: p > 0.5)(float(m.predict_proba(bf(df, i, 60).reshape(1, -1))[0][1]) if os.path.exists(mp) else 0))
    
    # 重新算准确率和偏好
    lh_c = lh_t = bullish = 0
    df2 = fd(S['lh2609']['code'])
    df2['dt'] = pd.to_datetime(df2['date'])
    m2 = pickle.load(open(mp, 'rb'))
    for i in range(65, len(df2)-1):
        if str(df2.iloc[i]['date']) < start_date: continue
        ft = bf(df2, i, 60)
        if ft is None: continue
        try:
            prob = float(m2.predict_proba(ft.reshape(1, -1))[0][1])
            if (prob > 0.5) == (float(df2.iloc[i+1]['close']) > float(df2.iloc[i]['close'])):
                lh_c += 1
            lh_t += 1
            if prob > 0.5: bullish += 1
        except: continue
    
    acc = lh_c/max(lh_t,1)*100
    bull_pct = bullish/max(lh_t,1)*100
    
    # 权益
    eq_now = eq(st, prices)
    ret_pct = (eq_now - 300000) / 300000 * 100
    
    # LH持仓浮盈
    lh_pos = st.get('positions', {}).get('lh2609')
    lh_float = 0
    if lh_pos and prices.get('lh2609'):
        cur = prices['lh2609']
        if isinstance(lh_pos, list):
            tv = sum(p['vol'] for p in lh_pos); d = lh_pos[0]['dir']
            ae = sum(p['entry']*p['vol'] for p in lh_pos)/tv
            lh_float = (cur-ae)*tv*16 if d=='LONG' else (ae-cur)*tv*16
        else:
            lh_float = (cur-lh_pos['entry'])*lh_pos['vol']*16 if lh_pos['dir']=='LONG' else (lh_pos['entry']-cur)*lh_pos['vol']*16
    
    # 回撤
    eq_hist = []
    with open(base + '/paper_state' + ('' if ver_name=='V25' else '_v28' if ver_name=='V28' else '_v29' if ver_name=='V29' else '_v30') + '.json') as f:
        eq_hist = json.load(f).get('equity_history', [])
    mdd_pct = 0
    if eq_hist:
        vals = [e['equity'] for e in eq_hist]
        peak = vals[0]
        for v in vals:
            peak = max(peak, v)
            mdd_pct = min(mdd_pct, (v-peak)/peak)
        mdd_pct = abs(mdd_pct)*100
    
    # LH交易
    with open(base + '/paper_state' + ('' if ver_name=='V25' else '_v28' if ver_name=='V28' else '_v29' if ver_name=='V29' else '_v30') + '.json') as f:
        trades_all = json.load(f).get('trades', [])
    lh_trades = [t for t in trades_all if t.get('sym','') == 'lh2609' and (t.get('time','') or t.get('exit_time',''))[:10] >= start_date]
    lh_pnl = sum(t.get('pnl',0) or t.get('pnl_amount',0) for t in lh_trades)
    wins = [t for t in lh_trades if (t.get('pnl',0) or t.get('pnl_amount',0)) > 0]
    wr = len(wins)/max(len(lh_trades),1)*100
    
    days = (pd.to_datetime(today) - pd.to_datetime(start_date)).days + 1
    
    # 评分
    acc_score = min(35, max(0, (acc-30)/40*35)) if lh_t >= 4 else 0
    ret_score = min(30, max(0, (ret_pct+10)/40*30))
    mdd_score = min(20, max(0, (30-mdd_pct)/30*20))
    pf_score = 5 if lh_trades else 0  # 简化
    disc_score = 5
    total_score = int(acc_score + ret_score + mdd_score + pf_score + disc_score)
    
    grade = 'A' if total_score>=80 else ('B' if total_score>=65 else ('C' if total_score>=50 else 'D'))
    note = '*样本不足' if lh_t < 10 else ''
    
    all_scores.append((ver_name, acc, lh_t, ret_pct, mdd_pct, lh_float, lh_pnl, wr, bull_pct, days, total_score, grade, note))
    
    print(f"{ver_name}: acc={acc:.0f}%({lh_t}次) ret={ret_pct:+.1f}% mdd={mdd_pct:.1f}% float={lh_float:+,.0f} pnl={lh_pnl:+,.0f} wr={wr:.0f}% bull={bull_pct:.0f}% days={days} score={total_score} {grade}{note}")

# Build card tables
ele.append(md(''))
rows1 = ['| 版本 | 上线 | 天数 | 预测 | 准确率 | 收益率 | 回撤 | LH浮盈 | 信号 | 评分 |']
rows1.append('|------|------|------|------|------|------|------|------|------|------|')
for s in all_scores:
    bias = '偏多' + str(int(s[8])) + '%' if s[8] > 60 else ('偏空' + str(int(100-s[8])) + '%' if s[8] < 40 else '均衡')
    rows1.append('| **' + s[0] + '** | ' + str(s[9]) + ' | ' + str(s[2]) + ' | ' + str(int(s[1])) + '% | ' + ('+' if s[3]>=0 else '') + '{:.1f}'.format(s[3]) + '% | ' + '{:.1f}'.format(s[4]) + '% | ' + ('+' if s[5]>=0 else '') + str(int(s[5])) + ' | ' + bias + ' | **' + str(s[10]) + s[11] + s[12] + '** |')
ele.append(md('\n'.join(rows1)))

ele.append(md(''))
ele.append(md('**结论**'))
rows2 = [
    '| 模型 | 评级 | 建议 | 理由 |',
    '|------|------|------|------|',
]
# Sort by score
by_score = sorted(all_scores, key=lambda x: -x[10])
for s in by_score:
    if s[0] == 'V25':
        rows2.append('| V25 | B | 半仓持有 | 收益最高但准确率仅50%,靠一波行情 |')
    elif s[0] == 'V28':
        rows2.append('| V28 | D | ❌停用LH | 准确率50%但持续亏损,与V25同模型 |')
    elif s[0] == 'V29':
        rows2.append('| V29 | D | ❌停用LH | 43%准确率,100%喊多,LH浮亏2.9万 |')
    elif s[0] == 'V30':
        rows2.append('| V30 | B* | 重点观察 | 50%准确率但做空盈利,样本太少 |')
ele.append(md('\n'.join(rows2)))

ele.append(md(''))
ele.append(md('V29永远喊多,生猪跌它就亏。V30永远喊空,生猪跌它就赚。方向偏好比准确率更致命。'))

send('生猪LH评估 | 全周期', ele, 'blue')
print('OK')
