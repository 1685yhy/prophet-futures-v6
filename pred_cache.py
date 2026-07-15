#!/usr/bin/env python3
"""预生成模型预测缓存 — 每5分钟更新，扫描和报告直接读取"""
import sys, os, json, pickle, numpy as np
from datetime import datetime

PROJ = '/home/a/prophet_futures/prophet_futures'
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from realtime_data import get_daily_history, build_features, SYMBOL_MAP

MODEL_DIR = os.path.join(PROJ, 'models')
CACHE_FILE = os.path.join(MODEL_DIR, 'pred_cache.json')

VERSIONS = {
    'V25': '_xgb.pkl',
    'V28': '_xgb.pkl',
    'V29': '_xgb_new.pkl',
    'V30': '_xgb_calibrated.pkl',
}

cache = {}
for ver, suffix in VERSIONS.items():
    for sym in ['lh2609', 'jm2609']:
        mp = os.path.join(MODEL_DIR, f'{sym}{suffix}')
        if not os.path.exists(mp): continue
        try:
            with open(mp, 'rb') as f:
                model = pickle.load(f)
            df = get_daily_history(sym, 1200)
            if df is None or len(df) < 100: continue
            feats = build_features(df, len(df)-1, 60)
            if feats is None: continue
            prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
            key = f'{ver}_{sym}'
            cache[key] = {
                'prob': round(prob, 4),
                'direction': 'LONG' if prob > 0.5 else 'SHORT',
                'confidence': round(prob if prob > 0.5 else 1-prob, 4),
                'date': str(df.iloc[-1]['date']),
                'time': datetime.now().strftime('%H:%M:%S'),
            }
        except Exception as e:
            cache[key] = {'error': str(e)[:100]}

with open(CACHE_FILE, 'w') as f:
    json.dump(cache, f, indent=2, ensure_ascii=False)
print(f'Cache updated: {len(cache)} entries')
