#!/usr/bin/env python3
"""分年度归因 — 独立进程跑单版本(避免reload污染patch链)
用法: python yearly_attr.py v33|v35
"""
import sys
from collections import defaultdict
import pandas as pd

sys.path.insert(0, '.')
mode = sys.argv[1] if len(sys.argv) > 1 else 'v33'

import v5_backtest as bt
if mode == 'v35':
    import v35_backtest  # noqa: F401 — patches bt at import

bt.SPECS['LH']['reverse_conf'] = 0.0
data = bt.fetch_data()
lh_df, jm_df = data['LH'], data['JM']
stats, bt_obj = bt.run_combined_backtest(lh_df, jm_df, strategy='V32')

MKT = {2021: '熊-46%', 2022: '震荡+14%', 2023: '震荡-12%',
       2024: '震荡-7%', 2025: '震荡-10%', 2026: '震荡+1%'}

by_year = defaultdict(float); n_year = defaultdict(int)
for t in bt_obj.trades:
    idx = int(t['date'])
    src = lh_df if t.get('name') == 'LH' else jm_df
    if idx >= len(src): continue
    yr = pd.to_datetime(src.iloc[idx]['date']).year
    by_year[yr] += t['pnl_abs']; n_year[yr] += 1

name = 'V35(LSTM)' if mode == 'v35' else 'V33(XGB无反手)'
print(f'RESULT {name} 总{stats["total_return"]:+.0f}% 回撤{stats["mdd"]:.0f}%')
for yr in sorted(by_year):
    print(f'  {yr}[{MKT.get(yr, "JM早期")}]: ¥{by_year[yr]:+,.0f} ({n_year[yr]}笔)')
