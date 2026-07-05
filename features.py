#!/usr/bin/env python3
"""
Prophet Futures — 共享特征工程模块
单一 build_features 实现，所有脚本从此导入，确保一致性。

合约映射: 明确指定合约代码,避免主力切换时数据错乱。
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import akshare as ak

# 合约代码映射 (akshare symbol → 具体合约)
# 使用 futures_main_sina 获取主力合约,但本映射用于验证和备选
CONTRACT_MAP = {
    'lh2609': {'code': 'LH0', 'fut': 'LH2609', 'name': 'LH'},
    'jm2609': {'code': 'JM0', 'fut': 'JM2609', 'name': 'JM'},
}


def fetch_daily(code, days=1200):
    """获取日线数据, code 可以是 'LH0'/'JM0' 等 akshare 主力代码"""
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = ak.futures_main_sina(
            symbol=code,
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d')
        )
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'oi', 'settle']
        for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except Exception as e:
        print(f"  Fetch error {code}: {e}")
        return None


def build_features(df, idx, window=60):
    """
    从日线 DataFrame 构建 19 维特征向量。
    所有脚本统一使用此函数 —— 不要再各自实现。
    
    特征列表:
    0: 隔夜涨跌幅
    1: 隔夜涨跌幅绝对值
    2-6: lag 1/3/5/10/20 期收益率
    7-10: MA5/10/20/60 偏离
    11: 20日波动率
    12: 日内振幅
    13: 相对成交量
    14: 相对持仓量
    15: MACD (EMA12-EMA26)
    16: RSI-14
    17: 布林带位置
    18: 价格/1000
    """
    if idx < window + 5:
        return None
    
    w = df.iloc[idx - window:idx + 1]
    c = w['close'].values.astype(float)
    o = w['open'].values.astype(float)
    h = w['high'].values.astype(float)
    l = w['low'].values.astype(float)
    v = w['volume'].values.astype(float)
    oi_v = w['oi'].values.astype(float)
    
    f = []
    if idx >= 1:
        f.append(float((o[-1] - c[-2]) / c[-2]))
        f.append(abs(f[-1]))
    else:
        f.extend([0.0, 0.0])
    
    for lag in [1, 3, 5, 10, 20]:
        f.append(float((c[-1] - c[-lag - 1]) / c[-lag - 1] if len(c) > lag else 0.0))
    
    for p in [5, 10, 20, 60]:
        ma = np.mean(c[-min(p, len(c)):])
        f.append(float((c[-1] - ma) / ma))
    
    f.append(float(np.std(c[-20:]) / np.mean(c[-20:])))
    f.append(float((h[-1] - l[-1]) / c[-1]))
    
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1.0
    f.append(float(v[-1] / vma))
    
    f.append(float(oi_v[-1] / np.mean(oi_v[-20:]))
             if len(oi_v) >= 20 and np.mean(oi_v[-20:]) > 0 else 1.0)
    
    # MACD
    e12 = c[-1]
    e26 = c[-1]
    for j in range(len(c) - 2, -1, -1):
        e12 = (2 / 13) * c[j] + (11 / 13) * e12
        e26 = (2 / 27) * c[j] + (25 / 27) * e26
    f.append(float((e12 - e26) / c[-1]))
    
    # RSI-14
    dd = np.diff(c[-15:])
    g = float(dd[dd > 0].sum()) if len(dd[dd > 0]) > 0 else 0.0
    lo = float(abs(dd[dd < 0].sum())) if len(dd[dd < 0]) > 0 else 1e-10
    f.append(float(100 - 100 / (1 + g / lo) if lo > 0 else 50.0))
    
    # Bollinger
    bb = np.std(c[-20:])
    m20 = np.mean(c[-20:])
    f.append(float((c[-1] - m20) / (2 * bb + 1e-10)))
    
    # Price scale
    f.append(float(c[-1] / 1000.0))
    
    return np.array(f, dtype=np.float32)


def calc_atr(df, idx, period=20):
    """计算 ATR"""
    if idx < period:
        return None
    vals = [
        abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
        for i in range(idx - period + 1, idx + 1)
    ]
    return np.mean(vals)


def get_latest_prediction(sym_key, model_dir):
    """获取最新模型预测 — 返回 (prob, direction, confidence, price, atr)"""
    import pickle
    import os
    
    cfg = CONTRACT_MAP[sym_key]
    mp = os.path.join(model_dir, f'{sym_key}_xgb.pkl')
    if not os.path.exists(mp):
        return None
    
    df = fetch_daily(cfg['code'], 1200)
    if df is None or len(df) < 100:
        return None
    
    feats = build_features(df, len(df) - 1, 60)
    if feats is None:
        return None
    
    with open(mp, 'rb') as f:
        model = pickle.load(f)
    
    prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
    direction = 'LONG' if prob > 0.5 else 'SHORT'
    confidence = prob if prob > 0.5 else (1 - prob)
    price = float(df.iloc[-1]['close'])
    atr = calc_atr(df, len(df) - 1, 20)
    
    return {
        'prob': prob,
        'direction': direction,
        'confidence': confidence,
        'price': price,
        'atr': atr,
        'df': df,
        'last_date': str(df.iloc[-1]['date']),
    }
