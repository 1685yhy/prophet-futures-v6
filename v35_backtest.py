#!/usr/bin/env python3
"""V35回测 — LSTM接入v5_backtest完整V32引擎(反手OFF)
合法替换(无前视, 与V33/V34同协议):
  1. build_features: LH→360维展平序列(60日×6), JM保持19维(价格<5000判定)
  2. train_xgb: 360维输入→TorchSeqClassifier(30天滚动重训), 其余→原XGB
引擎逻辑/风控/资金曲线零改动。
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import v5_backtest as bt
from v35_torch_model import TorchSeqClassifier

SEQ, NF = 60, 6
_seq_cache = {}

def _seq_matrix(df):
    """整表逐日6维序列特征(相对量)"""
    key = id(df)
    if key in _seq_cache:
        return _seq_cache[key]
    c = df['close'].values.astype(float)
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    v = df['volume'].values.astype(float)
    oi = df['oi'].values.astype(float)
    n = len(df)
    m = np.zeros((n, NF), dtype=np.float32)
    for i in range(1, n):
        m[i, 0] = (c[i] - c[i-1]) / c[i-1]
        m[i, 1] = (h[i] - l[i]) / c[i]
        m[i, 2] = (o[i] - c[i-1]) / c[i-1]
        vm = v[max(0, i-20):i].mean() or 1
        m[i, 3] = min(v[i] / vm, 5.0)
        om = oi[max(0, i-20):i].mean() or 1
        m[i, 4] = min(oi[i] / om, 3.0)
        m[i, 5] = (c[i] - c[max(0, i-5)]) / c[max(0, i-5)] if i >= 5 else 0.0
    _seq_cache[key] = m
    return m

_orig_build = bt.build_features
_orig_train = bt.train_xgb

def build_features_v35(df, idx, window=60):
    if float(df['close'].iloc[-1]) < 5000:          # JM: 原19维
        return _orig_build(df, idx, window)
    if idx < SEQ + 10:
        return None
    m = _seq_matrix(df)
    return m[idx-SEQ+1:idx+1].flatten()             # LH: 360维序列

def train_v35(X, y, params):
    X = np.asarray(X)
    if X.shape[1] == SEQ * NF:                       # LH→LSTM
        return TorchSeqClassifier().fit(X, y)
    return _orig_train(X, y, params)                 # JM→XGB

bt.build_features = build_features_v35
bt.train_xgb = train_v35

if __name__ == '__main__':
    bt.SPECS['LH']['reverse_conf'] = 0.0            # V33最优: 反手OFF
    data = bt.fetch_data()
    stats, _ = bt.run_combined_backtest(data['LH'], data['JM'], strategy='V32')
    score = stats['total_return'] * (1 + stats['mdd'] / 100)
    print(f"\nV35(LSTM+完整V32引擎,反手OFF): 收益{stats['total_return']:+.1f}% "
          f"回撤{stats['mdd']:.1f}% 交易{stats['n_trades']}笔 "
          f"最终¥{stats['final_equity']:,.0f} 分={score:.0f}")
    print('对照: V33纯技术+反手OFF +837%/-38.0% 分519 | V34基本面 +633%/-39.5% 分383')
