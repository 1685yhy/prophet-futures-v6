#!/usr/bin/env python3
"""场景化标准回测 — 11版本(7配置)×4场景, 每段独立¥30万连续本金
约束: 段前禁交易(predict None), 训练滚动只用过去(引擎原生), JM注掉, 无前视
输出: 段收益/年化/回撤/胜率/笔数
用法: runner无参 | 子进程 --one <cfg> <scene>
"""
import sys, json, subprocess
import numpy as np
import pandas as pd

CONFIGS = {
    'V25/V31':   {'strategy': 'V31', 'atr': 1.5, 'rr': 4.0, 'rev': None, 'feat': 'tech'},
    'V28/29/30': {'strategy': 'V32', 'atr': 1.5, 'rr': 4.0, 'rev': 0.35, 'feat': 'tech'},
    'V32':       {'strategy': 'V32', 'atr': 0.5, 'rr': 6.0, 'rev': 0.25, 'feat': 'tech'},
    'V32b':      {'strategy': 'V32', 'atr': 0.5, 'rr': 5.0, 'rev': 0.25, 'feat': 'tech',
                  'max_pos': 3, 'max_total': 6},
    'V33':       {'strategy': 'V32', 'atr': 0.5, 'rr': 6.0, 'rev': 0.0, 'feat': 'tech'},
    'V34':       {'strategy': 'V32', 'atr': 1.0, 'rr': 6.0, 'rev': 0.0, 'feat': 'fund',
                  'entry_conf': 0.65},
    'V35':       {'strategy': 'V32', 'atr': 1.5, 'rr': 4.0, 'rev': 0.0, 'feat': 'lstm'},
    'V36':       {'strategy': 'V32', 'atr': 2.0, 'rr': 4.0, 'rev': 0.35, 'feat': 'tech',
                  'balanced': True, 'ma_filter': 90},
}
SCENES = {
    '牛市':   ('2022-04-11', '2022-10-18'),
    '熊市':   ('2022-10-18', '2023-12-05'),
    '震荡':   ('2026-01-02', '2026-07-17'),
    '全程':   (None, None),
}

def run_one(cfg_key, scene_key):
    cfg = CONFIGS[cfg_key]
    b0, b1 = SCENES[scene_key]
    sys.path.insert(0, '.')
    if cfg['feat'] == 'fund':
        import v5_backtest_fund as bt
    else:
        import v5_backtest as bt
        if cfg['feat'] == 'lstm':
            import v35_backtest  # noqa

    # 类平衡训练(V36)
    if cfg.get('balanced'):
        def train_bal(X, y, params):
            import xgboost as xgb
            n_pos = max(int(np.sum(y == 1)), 1); n_neg = max(int(np.sum(y == 0)), 1)
            m = xgb.XGBClassifier(n_estimators=params['n_est'], max_depth=params['depth'],
                learning_rate=params['lr'], subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=n_neg / n_pos, random_state=42, n_jobs=1, verbosity=0)
            m.fit(X, y); return m
        bt.train_xgb = train_bal

    data = bt.fetch_data()
    lh_df, jm_df = data['LH'], data['JM']
    dates = pd.to_datetime(lh_df['date'])
    if b1:  # 裁到段末(段后数据不进回测)
        end_idx = dates[dates <= b1].index[-1]
        lh_df = lh_df.iloc[:end_idx + 1].reset_index(drop=True)
        dates = pd.to_datetime(lh_df['date'])
    start_ts = pd.Timestamp(b0) if b0 else None
    ma_n = cfg.get('ma_filter')

    _orig = bt.ContinuousBacktest.predict
    def _pred(self, models, name, idx):
        if name == 'JM': return None                      # 纯LH
        if start_ts is not None and pd.to_datetime(self.lh_df.iloc[idx]['date']) < start_ts:
            return None                                    # 段前禁交易
        r = _orig(self, models, name, idx)
        if r is None or not ma_n: return r
        if idx < ma_n: return r
        ma_v = float(self.lh_df['close'].iloc[idx - ma_n + 1:idx + 1].mean())
        price = float(self.lh_df['close'].iloc[idx])
        if price > ma_v and r['direction'] == 'SHORT': return None
        if price < ma_v and r['direction'] == 'LONG': return None
        return r
    bt.ContinuousBacktest.predict = _pred

    spec = bt.SPECS['LH']
    spec['atr_stop'] = cfg['atr']; spec['rr'] = cfg['rr']
    if cfg['rev'] is not None: spec['reverse_conf'] = cfg['rev']
    if 'max_pos' in cfg: spec['max_pos'] = cfg['max_pos']; spec['max_total'] = cfg['max_total']
    if 'entry_conf' in cfg: spec['entry_conf'] = cfg['entry_conf']

    stats, bt_obj = bt.run_combined_backtest(lh_df, jm_df, strategy=cfg['strategy'])

    # 段内LH交易统计(独立¥30万)
    CAP = 300000.0
    pnls = []
    for t in bt_obj.trades:
        idx = int(t['date'])
        if t.get('name') != 'LH' or idx >= len(lh_df): continue
        d = pd.to_datetime(lh_df.iloc[idx]['date'])
        if start_ts is not None and d < start_ts: continue
        pnls.append(float(t['pnl_abs']))
    total_pnl = sum(pnls)
    ret = total_pnl / CAP * 100
    # 段内回撤(已实现PnL曲线, 与引擎get_stats同口径)
    run_eq = CAP; peak = CAP; mdd = 0.0
    for p in pnls:
        run_eq += p; peak = max(peak, run_eq)
        mdd = min(mdd, (run_eq - peak) / peak * 100)
    closed = [p for p in pnls if p != 0]
    wr = len([p for p in closed if p > 0]) / len(closed) * 100 if closed else 0.0
    # 年化: 按段自然日
    d0 = pd.Timestamp(b0) if b0 else dates.iloc[0]
    d1 = pd.Timestamp(b1) if b1 else dates.iloc[-1]
    days = max((d1 - d0).days, 1)
    ann = ((1 + ret / 100) ** (365.0 / days) - 1) * 100 if ret > -100 else -100.0
    print('JSON:' + json.dumps({'cfg': cfg_key, 'scene': scene_key, 'ret': round(ret, 1),
        'ann': round(ann, 1), 'mdd': round(mdd, 1), 'wr': round(wr, 0),
        'n': len(closed), 'days': days}, ensure_ascii=False))

if __name__ == '__main__':
    if '--one' in sys.argv:
        i = sys.argv.index('--one')
        run_one(sys.argv[i + 1], sys.argv[i + 2])
    else:
        results = []
        for ck in CONFIGS:
            for sk in SCENES:
                print(f'>>> {ck} × {sk}', flush=True)
                r = subprocess.run([sys.executable, __file__, '--one', ck, sk],
                                   capture_output=True, text=True, timeout=5400)
                for line in r.stdout.split('\n'):
                    if line.startswith('JSON:'):
                        results.append(json.loads(line[5:])); break
                else:
                    print(f'  FAIL: {r.stderr[-200:]}', flush=True)
        json.dump(results, open('scene_results.json', 'w'), ensure_ascii=False, indent=2)
        print('\n=== 场景矩阵(每段独立¥30万) ===')
        print(f'{"版本":10s} {"场景":4s} {"收益":>8s} {"年化":>8s} {"回撤":>7s} {"胜率":>5s} {"笔数":>4s}')
        for r in results:
            print(f"{r['cfg']:10s} {r['scene']:4s} {r['ret']:+7.1f}% {r['ann']:+7.1f}% "
                  f"{r['mdd']:6.1f}% {r['wr']:4.0f}% {r['n']:4d}")
