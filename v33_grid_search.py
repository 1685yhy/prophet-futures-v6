#!/usr/bin/env python3
"""V33策略网格搜索 — 为22维基本面模型找匹配策略
网格: atr_stop{0.5,1.0,1.5} × reverse_conf{off,0.25,0.35}
评分: return*(1+mdd/100) 风险调整
"""
import sys, json
sys.path.insert(0, '.')
import v5_backtest_fund as btf

data = btf.fetch_data()
lh, jm = data['LH'], data['JM']

results = []
for atr_s in [0.5, 1.0, 1.5]:
    for rev in [0.0, 0.25, 0.35]:  # 0.0 = 反手关闭
        btf.SPECS['LH']['reverse_conf'] = rev
        try:
            stats, _ = btf.run_combined_backtest(lh, jm, strategy='V32', atr_stop_lh=atr_s)
            ret, mdd, n = stats['total_return'], stats['mdd'], stats['n_trades']
            score = ret * (1 + mdd / 100)
            results.append({'atr_stop': atr_s, 'reverse': rev, 'return': ret,
                            'mdd': mdd, 'trades': n, 'score': round(score, 1)})
            print(f"atr={atr_s} rev={rev if rev else 'off'}: "
                  f"收益{ret:+.1f}% 回撤{mdd:.1f}% {n}笔 分={score:.0f}", flush=True)
        except Exception as e:
            print(f"atr={atr_s} rev={rev}: FAIL {e}", flush=True)

results.sort(key=lambda x: x['score'], reverse=True)
json.dump(results, open('v33_grid_results.json', 'w'), indent=2)

print('\n=== TOP3 (风险调整分) ===')
for r in results[:3]:
    print(f"  atr={r['atr_stop']} rev={r['reverse'] or 'off'}: "
          f"{r['return']:+.1f}%/{r['mdd']:.1f}% 分={r['score']}")
print(f"\n基线参照: 纯技术19维+766%/-38% 分={766*(1-0.379):.0f}")
