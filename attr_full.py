#!/usr/bin/env python3
"""全归因 — 年度 + 牛市段 + 多空拆解, trades存盘复用
用法: python attr_full.py v33|v35
"""
import sys, json
from collections import defaultdict
import pandas as pd

sys.path.insert(0, '.')
mode = sys.argv[1] if len(sys.argv) > 1 else 'v33'

import v5_backtest as bt
if mode == 'v35':
    import v35_backtest  # noqa — patches bt at import

bt.SPECS['LH']['reverse_conf'] = 0.0
data = bt.fetch_data()
lh_df, jm_df = data['LH'], data['JM']
stats, bt_obj = bt.run_combined_backtest(lh_df, jm_df, strategy='V32')

MKT = {2021: '熊-46%', 2022: '含大牛+92%', 2023: '震荡-12%',
       2024: '含中牛+43%', 2025: '震荡-10%', 2026: '震荡+1%'}
BULL1 = (pd.Timestamp('2022-04-11'), pd.Timestamp('2022-10-18'), '大牛市+92%')
BULL2 = (pd.Timestamp('2024-01-10'), pd.Timestamp('2024-08-05'), '中级牛+43%')

# trades → 带真实日期/方向/品种
rows = []
for t in bt_obj.trades:
    idx = int(t['date'])
    src = lh_df if t.get('name') == 'LH' else jm_df
    if idx >= len(src): continue
    rows.append({'date': str(pd.to_datetime(src.iloc[idx]['date']).date()),
                 'sym': t.get('name'), 'dir': t.get('direction'),
                 'pnl': float(t['pnl_abs']), 'type': t.get('type')})

name = 'V35(LSTM)' if mode == 'v35' else 'V33(XGB无反手)'
json.dump({'name': name, 'total_return': stats['total_return'], 'mdd': stats['mdd'],
           'trades': rows}, open(f'attr_{mode}.json', 'w'), ensure_ascii=False)

print(f'RESULT {name} 总{stats["total_return"]:+.0f}% 回撤{stats["mdd"]:.0f}%')

by_year = defaultdict(float); n_year = defaultdict(int)
for r in rows:
    yr = int(r['date'][:4])
    by_year[yr] += r['pnl']; n_year[yr] += 1
print('— 分年度 —')
for yr in sorted(by_year):
    print(f'  {yr}[{MKT.get(yr, "?")}]: ¥{by_year[yr]:+,.0f} ({n_year[yr]}笔)')

for b0, b1, lbl in [BULL1, BULL2]:
    seg = [r for r in rows if b0 <= pd.Timestamp(r['date']) <= b1 and r['sym'] == 'LH']
    pnl = sum(r['pnl'] for r in seg)
    long_pnl = sum(r['pnl'] for r in seg if r['dir'] == 'LONG')
    short_pnl = sum(r['pnl'] for r in seg if r['dir'] == 'SHORT')
    nl = len([r for r in seg if r['dir'] == 'LONG']); ns = len([r for r in seg if r['dir'] == 'SHORT'])
    print(f'— {lbl} ({b0.date()}~{b1.date()}) LH —')
    print(f'  合计¥{pnl:+,.0f} | 做多¥{long_pnl:+,.0f}({nl}笔) | 做空¥{short_pnl:+,.0f}({ns}笔)')
