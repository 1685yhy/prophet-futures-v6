#!/usr/bin/env python3
"""纸盘10版本全矩阵纯LH回测 + 牛市段拆解
每配置独立子进程(防patch污染)。参数与纸盘代码逐一对齐。
用法: python matrix_backtest.py            # runner: 顺序跑全部并汇总
      python matrix_backtest.py --one V32  # 单配置(子进程模式)
"""
import sys, json, subprocess
from collections import defaultdict

# 纸盘参数对齐表 (feat: tech=19维XGB / fund=22维XGB / lstm=360维LSTM)
CONFIGS = {
    'V25/V31': {'strategy': 'V31', 'atr': 1.5, 'rr': 4.0, 'rev': None, 'feat': 'tech',
                'note': '固定止损止盈+模型退出'},
    'V28/29/30': {'strategy': 'V32', 'atr': 1.5, 'rr': 4.0, 'rev': 0.35, 'feat': 'tech',
                  'note': '动态+反手0.35(引擎滚动重训近似,3版模型差异未复刻)'},
    'V32': {'strategy': 'V32', 'atr': 0.5, 'rr': 6.0, 'rev': 0.25, 'feat': 'tech',
            'note': '紧止损+反手0.25'},
    'V32b': {'strategy': 'V32', 'atr': 0.5, 'rr': 5.0, 'rev': 0.25, 'feat': 'tech',
             'max_pos': 3, 'max_total': 6, 'note': '半仓版'},
    'V33': {'strategy': 'V32', 'atr': 0.5, 'rr': 6.0, 'rev': 0.0, 'feat': 'tech',
            'note': '紧止损+反手OFF'},
    'V34': {'strategy': 'V32', 'atr': 1.0, 'rr': 6.0, 'rev': 0.0, 'feat': 'fund',
            'entry_conf': 0.65, 'note': '22维基本面+conf0.65'},
    'V35': {'strategy': 'V32', 'atr': 1.5, 'rr': 4.0, 'rev': 0.0, 'feat': 'lstm',
            'note': 'LSTM序列'},
}

import pandas as pd
BULLS = [('2022-04-11', '2022-10-18', '大牛+92%'),
         ('2024-01-10', '2024-08-05', '中牛+43%')]

def run_one(key):
    cfg = CONFIGS[key]
    sys.path.insert(0, '.')
    if cfg['feat'] == 'fund':
        import v5_backtest_fund as bt
    else:
        import v5_backtest as bt
        if cfg['feat'] == 'lstm':
            import v35_backtest  # noqa — patches v5_backtest

    # JM注掉
    _orig = bt.ContinuousBacktest.predict
    def _p(self, models, name, idx):
        return None if name == 'JM' else _orig(self, models, name, idx)
    bt.ContinuousBacktest.predict = _p

    spec = bt.SPECS['LH']
    spec['atr_stop'] = cfg['atr']; spec['rr'] = cfg['rr']
    if cfg['rev'] is not None: spec['reverse_conf'] = cfg['rev']
    if 'max_pos' in cfg: spec['max_pos'] = cfg['max_pos']; spec['max_total'] = cfg['max_total']
    if 'entry_conf' in cfg: spec['entry_conf'] = cfg['entry_conf']

    data = bt.fetch_data()
    lh_df = data['LH']
    stats, bt_obj = bt.run_combined_backtest(data['LH'], data['JM'], strategy=cfg['strategy'])

    rows = []
    for t in bt_obj.trades:
        idx = int(t['date'])
        if t.get('name') != 'LH' or idx >= len(lh_df): continue
        rows.append({'date': str(pd.to_datetime(lh_df.iloc[idx]['date']).date()),
                     'dir': t.get('direction'), 'pnl': float(t['pnl_abs'])})

    out = {'key': key, 'note': cfg['note'], 'total_return': stats['total_return'],
           'mdd': stats['mdd'], 'n': len(rows), 'final': stats['final_equity'], 'bulls': {}}
    for b0, b1, lbl in BULLS:
        seg = [r for r in rows if b0 <= r['date'] <= b1]
        out['bulls'][lbl] = {
            'pnl': sum(r['pnl'] for r in seg),
            'long': sum(r['pnl'] for r in seg if r['dir'] == 'LONG'),
            'short': sum(r['pnl'] for r in seg if r['dir'] == 'SHORT'),
            'nl': len([r for r in seg if r['dir'] == 'LONG']),
            'ns': len([r for r in seg if r['dir'] == 'SHORT'])}
    print('JSON:' + json.dumps(out, ensure_ascii=False))

if __name__ == '__main__':
    if '--one' in sys.argv:
        run_one(sys.argv[sys.argv.index('--one') + 1])
    else:
        results = []
        for key in CONFIGS:
            print(f'>>> {key} ...', flush=True)
            r = subprocess.run([sys.executable, __file__, '--one', key],
                               capture_output=True, text=True, timeout=3600)
            for line in r.stdout.split('\n'):
                if line.startswith('JSON:'):
                    results.append(json.loads(line[5:]))
                    break
            else:
                print(f'  FAIL {key}: {r.stderr[-300:]}', flush=True)
        json.dump(results, open('matrix_results.json', 'w'), ensure_ascii=False, indent=2)
        print('\n=== 纯LH全矩阵(2021-2026, ¥30万连续) ===')
        print(f'{"版本":10s} {"总收益":>8s} {"回撤":>7s} {"大牛段":>10s} {"多/空笔":>8s} {"中牛段":>10s}')
        for r in results:
            b1 = r['bulls']['大牛+92%']; b2 = r['bulls']['中牛+43%']
            print(f"{r['key']:10s} {r['total_return']:+7.1f}% {r['mdd']:6.1f}% "
                  f"¥{b1['pnl']:+9,.0f} {b1['nl']}/{b1['ns']:2d} ¥{b2['pnl']:+9,.0f}")
