#!/usr/bin/env python3
"""模型预测缓存 — 用系统 python3 预计算，避免 venv ld.so 问题"""
import os, sys, json, pickle, numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '/home/a/prophet_futures/prophet_futures')
from realtime_data import get_daily_history, build_features

MODEL_DIR = '/home/a/prophet_futures/prophet_futures/models'
CACHE_FILE = '/home/a/prophet_futures/prophet_futures/models/pred_cache.json'
SYMBOLS = {'lh2609': 'LH0', 'jm2609': 'JM0'}

results = {}
for sym_key, code in SYMBOLS.items():
    for ver, suffix in [('V25', '_xgb.pkl'), ('V28', '_xgb.pkl'),
                         ('V29', '_xgb_new.pkl'), ('V30', '_xgb_calibrated.pkl')]:
        key = f'{sym_key}_{ver}'
        mp = os.path.join(MODEL_DIR, f'{sym_key}{suffix}')
        if not os.path.exists(mp):
            results[key] = None
            continue
        try:
            with open(mp, 'rb') as f:
                model = pickle.load(f)
            df = get_daily_history(sym_key, 1200)
            if df is None or len(df) < 100:
                results[key] = None
                continue
            feats = build_features(df, len(df)-1, 60)
            if feats is None:
                results[key] = None
                continue
            prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
            results[key] = {
                'prob': round(prob, 4),
                'direction': 'LONG' if prob > 0.5 else 'SHORT',
                'confidence': round(prob if prob > 0.5 else 1-prob, 4),
                'date': str(df.iloc[-1]['date']),
                'model': os.path.basename(mp),
            }
        except Exception as e:
            results[key] = {'error': str(e)}

with open(CACHE_FILE, 'w') as f:
    json.dump({'updated': datetime.now().isoformat(), 'predictions': results}, f, indent=2)

print(f'Cache updated: {sum(1 for v in results.values() if v)} predictions')
for k, v in sorted(results.items()):
    if v:
        print(f'  {k}: {v["direction"]} {v["prob"]:.1%} ({v["date"]})')
    else:
        print(f'  {k}: NONE')
