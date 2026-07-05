#!/usr/bin/env python3
"""实时行情数据获取 — 共享模块"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 品种映射: sym_key → (realtime_name, minute_symbol, daily_code)
SYMBOL_MAP = {
    'lh2609': {'rt_name': '生猪',    'min_sym': 'LH2609', 'daily_code': 'LH0',  'name': 'LH 生猪'},
    'jm2609': {'rt_name': '焦煤',    'min_sym': 'JM2609', 'daily_code': 'JM0',  'name': 'JM 焦煤'},
}

def get_realtime_quote(sym_key):
    """获取实时报价 — 秒级刷新
    
    Returns: dict with {price, open, high, low, volume, position, ticktime, changepercent}
    失败返回 None
    """
    info = SYMBOL_MAP.get(sym_key)
    if not info: return None
    try:
        df = ak.futures_zh_realtime(symbol=info['rt_name'])
        # 找匹配合约
        match = df[df['symbol'] == info['min_sym']]
        if len(match) == 0:
            return None
        row = match.iloc[0]
        return {
            'price': float(row['trade']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'volume': int(row['volume']),
            'position': int(row['position']),
            'ticktime': str(row['ticktime']),
            'changepercent': float(row['changepercent']) if row['changepercent'] else 0,
            'prev_close': float(row.get('preclose', 0)),
        }
    except Exception as e:
        return None

def get_minute_history(sym_key, minutes=60):
    """获取分钟级K线历史（用于计算ATR、短期趋势）
    
    Returns: DataFrame with [datetime, open, high, low, close, volume]
    """
    info = SYMBOL_MAP.get(sym_key)
    if not info: return None
    try:
        df = ak.futures_zh_minute_sina(symbol=info['min_sym'], period='1')
        df = df.tail(minutes)
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except:
        return None

def get_daily_history(sym_key, days=500):
    """获取日线历史（用于模型训练）
    
    Returns: DataFrame with [date, open, high, low, close, volume, oi]
    """
    info = SYMBOL_MAP.get(sym_key)
    if not info: return None
    end = datetime.now()
    start = end - timedelta(days=days + 50)
    try:
        df = ak.futures_main_sina(
            symbol=info['daily_code'],
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d')
        )
        df.columns = ['date','open','high','low','close','volume','oi','settle']
        for c in ['open','high','low','close','volume','oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except:
        return None

def compute_atr_from_minutes(df, window=20):
    """从分钟K线计算ATR"""
    if df is None or len(df) < 5:
        return None
    atr_vals = [abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
                for i in range(max(0, len(df)-window), len(df))]
    return np.mean(atr_vals) if atr_vals else None

def get_current_price(sym_key):
    """快捷获取当前价"""
    q = get_realtime_quote(sym_key)
    return q['price'] if q else None


def build_features(df, idx, window=60):
    """从日线DataFrame构建19维特征向量（公共函数）

    输入: df(DataFrame, 需有 open/high/low/close/volume/oi 列), idx(int), window(int)
    输出: np.array(shape=(19,), dtype=float32) 或 None
    """
    if idx < window + 5:
        return None
    w = df.iloc[idx - window:idx + 1]
    c = w['close'].values.astype(float)
    o = w['open'].values.astype(float)
    h = w['high'].values.astype(float)
    l_low = w['low'].values.astype(float)
    v = w['volume'].values.astype(float)
    oi_v = w['oi'].values.astype(float)

    f = []
    if idx >= 1:
        f.append(float((o[-1] - c[-2]) / c[-2]))
        f.append(abs(f[-1]))
    else:
        f.extend([0.0, 0.0])

    for lag in [1, 3, 5, 10, 20]:
        f.append(float((c[-1] - c[-lag - 1]) / c[-lag - 1] if len(c) > lag else 0))

    for p in [5, 10, 20, 60]:
        ma_val = np.mean(c[-min(p, len(c)):])
        f.append(float((c[-1] - ma_val) / ma_val))

    f.append(float(np.std(c[-20:]) / np.mean(c[-20:])))
    f.append(float((h[-1] - l_low[-1]) / c[-1]))

    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
    f.append(float(v[-1] / vma))
    f.append(float(oi_v[-1] / np.mean(oi_v[-20:])) if len(oi_v) >= 20 and np.mean(oi_v[-20:]) > 0 else 1)

    e12 = c[-1]
    e26 = c[-1]
    for j in range(len(c) - 2, -1, -1):
        e12 = (2 / 13) * c[j] + (11 / 13) * e12
        e26 = (2 / 27) * c[j] + (25 / 27) * e26
    f.append(float((e12 - e26) / c[-1]))

    dd = np.diff(c[-15:])
    g = float(dd[dd > 0].sum()) if len(dd[dd > 0]) > 0 else 0
    lo = float(abs(dd[dd < 0].sum())) if len(dd[dd < 0]) > 0 else 1e-10
    f.append(float(100 - 100 / (1 + g / lo) if lo > 0 else 50))

    bb = np.std(c[-20:])
    m20 = np.mean(c[-20:])
    f.append(float((c[-1] - m20) / (2 * bb + 1e-10)))
    f.append(float(c[-1] / 1000.0))

    return np.array(f, dtype=np.float32)
