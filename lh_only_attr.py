#!/usr/bin/env python3
"""纯LH回测 — JM信号全部注掉(predict返回None→零交易), 与纸盘配置一致
用法: python lh_only_attr.py v33|v35
"""
import sys, json
from collections import defaultdict
import pandas as pd

sys.path.insert(0, '.')
mode = sys.argv[1] if len(sys.argv) > 1 else 'v33'

import v5_backtest as bt
if mode == 'v35':
    import v35_backtest  # noqa — patches bt at import

# ── 注掉JM: 无信号=永不开仓 ──
_orig_predict = bt.ContinuousBacktest.predict
def predict_lh_only(self, models, name, idx):
    if name == 'JM':
        return None
    return _orig_predict(self, models, name, idx)
bt.ContinuousBacktest.predict = predict_lh_only

bt.SPECS['LH']['reverse_conf'] = 0.0
data = bt.fetch_data()
lh_df, jm_df = data['LH'], data['JM']
stats, bt_obj = bt.run_combined_backtest(lh_df, jm_df, strategy='V32')

MKT = {2021: '熊-46%', 2022: '含大牛+92%', 2023: '震荡-12%',
       2024: '含中牛+43%', 2025: '震荡-10%', 2026: '震荡+1%'}
BULLS = [(pd.Timestamp('2022-04-11'), pd.Timestamp('2022-10-18'), '大牛+92%'),
         (pd.Timestamp('2024-01-10'), pd.Timestamp('2024-08-05'), '中牛+43%')]

rows = []
for t in bt_obj.trades:
    idx = int(t['date'])
    if t.get('name') != 'LH' or idx >= len(lh_df): continue
    rows.append({'date': str(pd.to_datetime(lh_df.iloc[idx]['date']).date()),
                 'dir': t.get('direction'), 'pnl': float(t['pnl_abs'])})

name = 'V35纯LH(LSTM)' if mode == 'v35' else 'V33纯LH(XGB)'
json.dump({'name': name, 'total_return': stats['total_return'], 'mdd': stats['mdd'],
           'trades': rows}, open(f'attr_lh_{mode}.json', 'w'), ensure_ascii=False)

jm_leak = [t for t in bt_obj.trades if t.get('name') == 'JM']
print(f"RESULT {name} 总{stats['total_return']:+.1f}% 回撤{stats['mdd']:.1f}% "
      f"LH{len(rows)}笔 JM泄漏{len(jm_leak)}笔 最终¥{stats['final_equity']:,.0f}")

by_year = defaultdict(float); n_year = defaultdict(int)
for r in rows:
    yr = int(r['date'][:4]); by_year[yr] += r['pnl']; n_year[yr] += 1
print('— 分年度(LH) —')
for yr in sorted(by_year):
    print(f'  {yr}[{MKT.get(yr, "?")}]: ¥{by_year[yr]:+,.0f} ({n_year[yr]}笔)')

for b0, b1, lbl in BULLS:
    seg = [r for r in rows if b0 <= pd.Timestamp(r['date']) <= b1]
    lp = sum(r['pnl'] for r in seg if r['dir'] == 'LONG')
    sp = sum(r['pnl'] for r in seg if r['dir'] == 'SHORT')
    nl = len([r for r in seg if r['dir'] == 'LONG']); ns = len([r for r in seg if r['dir'] == 'SHORT'])
    print(f'— {lbl}: 合计¥{lp+sp:+,.0f} | 多¥{lp:+,.0f}({nl}) | 空¥{sp:+,.0f}({ns}) —')
