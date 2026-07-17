#!/usr/bin/env python3
"""V34实时基本面特征 — 与v5_backtest_fund.build_features的3维完全一致
日频数据, 当日磁盘缓存, 失败返回零向量(与回测except路径一致)
"""
import os, json, time
from datetime import datetime

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.fund_cache.json')

def _fetch_raw():
    """拉取三个数据源原始序列(日频)"""
    import akshare as ak, requests, pandas as pd
    from datetime import timedelta
    out = {}
    # 1. 现货指数(成交均价)
    idx_df = ak.index_hog_spot_price()
    out['idx'] = [[str(d), float(v)] for d, v in zip(idx_df['日期'], idx_df['成交均价']) if v == v]
    # 2. 猪粮比
    rt = ak.futures_hog_supply(symbol='猪粮比价')
    out['ratio'] = [[str(d), float(v)] for d, v in zip(rt['date'], rt['value']) if v == v]
    # 3. 周价格指数
    r = requests.post('https://xt.yangzhu.vip/data/getmapdata',
                      params={'ptype': '6', 'areno': '-1'}, timeout=15,
                      headers={'User-Agent': 'Mozilla/5.0'})
    wk = []
    for row in r.json()['data']:
        yr = int(row[0][:4]); w = int(row[0].split('第')[1].split('周')[0])
        d = datetime(yr, 1, 1) + timedelta(weeks=w - 1, days=3)
        wk.append([d.strftime('%Y-%m-%d'), float(row[1])])
    out['week'] = wk
    return out

def _load():
    """当日缓存优先, 过期重拉"""
    today = datetime.now().strftime('%Y-%m-%d')
    if os.path.exists(CACHE):
        try:
            c = json.load(open(CACHE))
            if c.get('date') == today:
                return c['data']
        except Exception:
            pass
    try:
        data = _fetch_raw()
        json.dump({'date': today, 'data': data}, open(CACHE, 'w'))
        return data
    except Exception:
        # 拉取失败: 用旧缓存兜底
        if os.path.exists(CACHE):
            try:
                return json.load(open(CACHE))['data']
            except Exception:
                pass
        return None

def get_fund_features(cur_close):
    """返回3维: [期现价差率, 猪粮比20期Z, 周指4周变化率] — 失败全0(与回测一致)"""
    import numpy as np
    data = _load()
    if not data or not cur_close or cur_close <= 0:
        return [0.0, 0.0, 0.0]
    f = []
    # 1. 期现价差
    try:
        ia = data['idx'][-1][1]
        f.append((ia * 1000 - cur_close) / cur_close)
    except Exception:
        f.append(0.0)
    # 2. 猪粮比Z
    try:
        vals = [v for _, v in data['ratio'][-20:]]
        m, s = float(np.mean(vals)), float(np.std(vals))
        f.append((vals[-1] - m) / s if s > 1e-6 else 0.0)
    except Exception:
        f.append(0.0)
    # 3. 周指变化率
    try:
        wv = [v for _, v in data['week'][-5:]]
        f.append((wv[-1] - wv[0]) / max(abs(wv[0]), 0.01) if len(wv) >= 2 else 0.0)
    except Exception:
        f.append(0.0)
    return f

if __name__ == '__main__':
    t0 = time.time()
    f = get_fund_features(11765.0)
    print(f'3维基本面: {[round(x, 4) for x in f]} ({time.time()-t0:.1f}s)')
    t0 = time.time()
    f = get_fund_features(11765.0)
    print(f'缓存二读: {[round(x, 4) for x in f]} ({time.time()-t0:.2f}s)')
