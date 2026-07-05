#!/usr/bin/env python3
"""概率校准验证 — 可靠性图 + ECE 指标
对比 校准前 vs 校准后 的预测概率是否更接近真实频率
"""
import sys, os, pickle, json
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import build_features, get_daily_history, SYMBOL_MAP

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
N_BINS = 10  # 10个概率桶
SYMBOLS = ['lh2609', 'jm2609']


def load_all_daily_data(sym_key):
    """加载全部历史日线"""
    df = get_daily_history(sym_key, 1200)
    if df is None:
        return None
    for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['close']).reset_index(drop=True)


def compute_ece_and_buckets(probs, labels, n_bins=10):
    """计算 ECE + 分桶统计"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    buckets = []
    total = len(probs)

    for b in range(n_bins):
        mask = bin_indices == b
        n_b = mask.sum()
        if n_b == 0:
            buckets.append({
                'bin_low': bin_edges[b],
                'bin_high': bin_edges[b+1],
                'n': 0, 'avg_prob': None, 'actual_freq': None,
                'gap': None
            })
            continue
        avg_prob = float(probs[mask].mean())
        actual_freq = float(labels[mask].mean())
        gap = actual_freq - avg_prob
        ece += (n_b / total) * abs(gap)
        buckets.append({
            'bin_low': float(bin_edges[b]),
            'bin_high': float(bin_edges[b+1]),
            'n': int(n_b),
            'avg_prob': round(avg_prob, 4),
            'actual_freq': round(actual_freq, 4),
            'gap': round(gap, 4),
        })
    return ece, buckets


def verify_symbol(sym_key):
    """对一个品种做完整验证"""
    name = SYMBOL_MAP[sym_key]['name']
    print(f"\n{'='*60}")
    print(f"  {name} ({sym_key}) — 概率校准验证")
    print(f"{'='*60}")

    df = load_all_daily_data(sym_key)
    if df is None or len(df) < 200:
        print("  ❌ 数据不足")
        return None

    # 用最后30%的数据做验证（既不在训练集也不在校准集）
    split_val = int(len(df) * 0.7)
    window = 60

    X_val, y_val = [], []
    for i in range(split_val, len(df) - 1):
        feats = build_features(df, i, window)
        if feats is None:
            continue
        label = 1 if df.iloc[i + 1]['close'] > df.iloc[i]['close'] else 0
        X_val.append(feats)
        y_val.append(label)

    if len(X_val) < 50:
        print(f"  ❌ 验证集太小 ({len(X_val)} 样本)")
        return None

    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val)

    # ── 基准：真实涨跌频率 ──
    actual_up_rate = y_val.mean()
    print(f"\n  📊 验证集: {len(X_val)} 条日线 ({df.iloc[split_val]['date']} → {df.iloc[-2]['date']})")
    print(f"  真实上涨比例: {actual_up_rate:.1%} (基线)")

    results = {}

    # ── 未校准模型 ──
    uncalib_path = os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')
    calib_path = os.path.join(MODEL_DIR, f'{sym_key}_xgb_calibrated.pkl')

    for label, path in [('未校准', uncalib_path), ('校准后(Platt)', calib_path)]:
        if not os.path.exists(path):
            print(f"\n  ⚠️ {label}: 模型文件不存在 ({path})")
            continue

        with open(path, 'rb') as f:
            model = pickle.load(f)

        probs = model.predict_proba(X_val)[:, 1]
        ece, buckets = compute_ece_and_buckets(probs, y_val, N_BINS)

        # 置信度分布
        confidences = np.where(probs > 0.5, probs, 1 - probs)
        mean_conf = confidences.mean()
        extreme_ratio = (confidences > 0.7).mean()

        print(f"\n  ── {label} ──")
        print(f"  ECE (期望校准误差): {ece:.4f}  {'✅ 良好' if ece < 0.05 else '⚠️ 偏差较大' if ece < 0.15 else '❌ 严重失准'}")
        print(f"  平均置信度: {mean_conf:.1%}")
        print(f"  高置信占比(>70%): {extreme_ratio:.1%}")
        print(f"\n  {'区间':<16} {'样本':>6} {'预测概率':>10} {'实际频率':>10} {'偏差':>8}")
        print(f"  {'─'*16} {'─'*6} {'─'*10} {'─'*10} {'─'*8}")
        for b in buckets:
            if b['n'] == 0:
                continue
            gap_str = f"{b['gap']:+.3f}" if b['gap'] is not None else 'N/A'
            marker = ' ⚠️' if b['gap'] and abs(b['gap']) > 0.1 else ''
            print(f"  {b['bin_low']:.0%}-{b['bin_high']:.0%}     {b['n']:>5}  {b['avg_prob']:>9.1%}  {b['actual_freq']:>9.1%}  {gap_str:>8}{marker}")

        results[label] = {
            'ece': round(ece, 4),
            'mean_confidence': round(float(mean_conf), 4),
            'extreme_ratio': round(float(extreme_ratio), 4),
            'buckets': buckets,
        }

    # ── 对比总结 ──
    if '未校准' in results and '校准后(Platt)' in results:
        uncal = results['未校准']
        cal = results['校准后(Platt)']
        ece_change = cal['ece'] - uncal['ece']
        arrow = '↓ 改善' if ece_change < 0 else '↑ 恶化'
        print(f"\n  📈 对比总结:")
        print(f"  {'指标':<20} {'未校准':>10} {'校准后':>10} {'变化':>10}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
        print(f"  {'ECE':<20} {uncal['ece']:>10.4f} {cal['ece']:>10.4f} {ece_change:>+10.4f} {arrow}")
        print(f"  {'平均置信度':<20} {uncal['mean_confidence']:>9.1%} {cal['mean_confidence']:>9.1%} "
              f"{(cal['mean_confidence']-uncal['mean_confidence']):>+9.1%}")
        print(f"  {'高置信占比':<20} {uncal['extreme_ratio']:>9.1%} {cal['extreme_ratio']:>9.1%} "
              f"{(cal['extreme_ratio']-uncal['extreme_ratio']):>+9.1%}")

    return results


def main():
    print("🧪 Prophet Futures — 概率校准验证")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   分桶: {N_BINS} 个概率区间")
    print(f"   ECE = Σ (|实际频率 - 预测概率| × 样本占比)  — 越小越好，<0.05 为良好校准")

    for sym_key in SYMBOLS:
        verify_symbol(sym_key)

    print(f"\n{'='*60}")
    print("  解读:")
    print("  - ECE < 0.05  → 概率可信，模型不自欺欺人")
    print("  - 高置信占比下降 → 校准让模型收敛了过度自信")
    print("  - 预测概率 ≈ 实际频率 → 看到76%概率时，真的约76%会涨")
    print("=" * 60)


if __name__ == '__main__':
    main()
