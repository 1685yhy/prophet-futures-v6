#!/usr/bin/env python3
"""牛市改造实验 — 基于纯LH最强配置(V28系: atr1.5/rev0.35)
A基线: 原样(已知-11.1%)
B类平衡训练: XGB scale_pos_weight=空样本/多样本 → 消除空头偏置
C趋势过滤器: price>MA60禁空 / price<MA60禁多 → 牛市强制顺势
D=B+C组合
用法: python bull_experiment.py A|B|C|D
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
mode = sys.argv[1] if len(sys.argv) > 1 else 'A'

import v5_backtest as bt

# 纯LH: JM注掉
_orig_predict = bt.ContinuousBacktest.predict
def _pred_lh(self, models, name, idx):
    if name == 'JM':
        return None
    r = _orig_predict(self, models, name, idx)
    if r is None or mode not in ('C', 'D'):
        return r
    # C/D: 趋势过滤器
    df = self.lh_df
    if idx < 60: return r
    ma60 = float(df['close'].iloc[idx-59:idx+1].mean())
    price = float(df['close'].iloc[idx])
    if price > ma60 and r['direction'] == 'SHORT':
        return None   # 均线上禁空
    if price < ma60 and r['direction'] == 'LONG':
        return None   # 均线下禁多
    return r
bt.ContinuousBacktest.predict = _pred_lh

# B/D: 类平衡训练(消除空头偏置)
if mode in ('B', 'D'):
    _orig_train = bt.train_xgb
    def train_balanced(X, y, params):
        import xgboost as xgb
        n_pos = max(int(np.sum(y == 1)), 1)
        n_neg = max(int(np.sum(y == 0)), 1)
        model = xgb.XGBClassifier(
            n_estimators=params['n_est'], max_depth=params['depth'],
            learning_rate=params['lr'], subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos,   # 类平衡
            random_state=42, n_jobs=1, verbosity=0)
        model.fit(X, y)
        return model
    bt.train_xgb = train_balanced

# V28系纸盘参数
bt.SPECS['LH']['atr_stop'] = 1.5
bt.SPECS['LH']['rr'] = 4.0
bt.SPECS['LH']['reverse_conf'] = 0.35

data = bt.fetch_data()
lh_df = data['LH']
stats, bt_obj = bt.run_combined_backtest(data['LH'], data['JM'], strategy='V32')

rows = []
for t in bt_obj.trades:
    idx = int(t['date'])
    if t.get('name') != 'LH' or idx >= len(lh_df): continue
    rows.append({'date': str(pd.to_datetime(lh_df.iloc[idx]['date']).date()),
                 'dir': t.get('direction'), 'pnl': float(t['pnl_abs'])})

NAMES = {'A': 'A基线(V28系)', 'B': 'B类平衡训练', 'C': 'C趋势过滤器', 'D': 'D平衡+过滤'}
b0, b1 = '2022-04-11', '2022-10-18'
seg = [r for r in rows if b0 <= r['date'] <= b1]
lp = sum(r['pnl'] for r in seg if r['dir'] == 'LONG')
sp = sum(r['pnl'] for r in seg if r['dir'] == 'SHORT')
nl = len([r for r in seg if r['dir'] == 'LONG'])
ns = len([r for r in seg if r['dir'] == 'SHORT'])
print(f"RESULT {NAMES[mode]}: 总{stats['total_return']:+.1f}% 回撤{stats['mdd']:.1f}% "
      f"{len(rows)}笔 | 大牛段: ¥{lp+sp:+,.0f} 多{nl}/空{ns}(多¥{lp:+,.0f} 空¥{sp:+,.0f})")
