#!/usr/bin/env python3
"""V33策略网格v2 — SPECS直改(atr参数第一版未生效的修正)
网格: atr_stop{0.5,1.0,1.5} × entry_conf{0.55,0.60,0.65}, 反手固定off(v1已证明off最优)
"""
import sys, json
sys.path.insert(0, '.')
import v5_backtest_fund as btf

data = btf.fetch_data()
lh, jm = data['LH'], data['JM']

btf.SPECS['LH']['reverse_conf'] = 0.0  # v1结论: 反手off
results = []
for atr_s in [0.5, 1.0, 1.5]:
    for ec in [0.55, 0.60, 0.65]:
        btf.SPECS['LH']['atr_stop'] = atr_s
        btf.SPECS['LH']['entry_conf'] = ec
        try:
            stats, _ = btf.run_combined_backtest(lh, jm, strategy='V32')
            ret, mdd, n = stats['total_return'], stats['mdd'], stats['n_trades']
            score = ret * (1 + mdd / 100)
            results.append({'atr_stop': atr_s, 'entry_conf': ec, 'return': ret,
                            'mdd': mdd, 'trades': n, 'score': round(score, 1)})
            print(f"atr={atr_s} conf={ec}: 收益{ret:+.1f}% 回撤{mdd:.1f}% {n}笔 分={score:.0f}", flush=True)
        except Exception as e:
            print(f"atr={atr_s} conf={ec}: FAIL {e}", flush=True)

results.sort(key=lambda x: x['score'], reverse=True)
json.dump(results, open('v33_grid_v2_results.json', 'w'), indent=2)

print('\n=== TOP3 ===')
for r in results[:3]:
    print(f"  atr={r['atr_stop']} conf={r['entry_conf']}: {r['return']:+.1f}%/{r['mdd']:.1f}% 分={r['score']}")
print(f"\nv1最优(rev=off默认参数): +674.1%/-50.4% 分=335")
print(f"基线纯技术19维: +766%/-37.9% 分=476")
