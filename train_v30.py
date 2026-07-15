#!/usr/bin/env python3
"""Prophet Futures V30 — 波动率预测 v2
  预测目标: 明天是否是高波动日（振幅>1.5倍20日均振幅）
  用处: 高波动→减仓，低波动→可以拿大仓位
"""
import sys, os, pickle, json, time
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta
import akshare as ak
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import SYMBOL_MAP

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
TRAIN_DAYS = 600
RELATED = {'c0': 'C0', 'm0': 'M0'}


def fetch_daily(code, days=600):
    end = datetime.now(); start = end - timedelta(days=days + 50)
    try:
        df = ak.futures_main_sina(symbol=code, start_date=start.strftime('%Y%m%d'),
                                   end_date=end.strftime('%Y%m%d'))
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'oi', 'settle']
        for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except:
        return None


def build_features_v30(df, idx, related_data, window=60):
    """24维特征（同前）"""
    if idx < window + 5: return None
    w = df.iloc[idx - window:idx + 1]
    c = w['close'].values.astype(float); o = w['open'].values.astype(float)
    h = w['high'].values.astype(float); l_low = w['low'].values.astype(float)
    v = w['volume'].values.astype(float); oi_v = w['oi'].values.astype(float)
    f = []
    if idx >= 1:
        f.append(float((o[-1] - c[-2]) / c[-2])); f.append(abs(f[-1]))
    else:
        f.extend([0.0, 0.0])
    for lag in [1, 3, 5, 10, 20]:
        f.append(float((c[-1] - c[-lag - 1]) / c[-lag - 1] if len(c) > lag else 0))
    for p in [5, 10, 20, 60]:
        ma = np.mean(c[-min(p, len(c)):])
        f.append(float((c[-1] - ma) / ma))
    f.append(float(np.std(c[-20:]) / np.mean(c[-20:])))
    f.append(float((h[-1] - l_low[-1]) / c[-1]))
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
    f.append(float(v[-1] / vma))
    f.append(float(oi_v[-1] / np.mean(oi_v[-20:])) if len(oi_v) >= 20 and np.mean(oi_v[-20:]) > 0 else 1)
    e12 = c[-1]; e26 = c[-1]
    for j in range(len(c) - 2, -1, -1):
        e12 = (2 / 13) * c[j] + (11 / 13) * e12
        e26 = (2 / 27) * c[j] + (25 / 27) * e26
    f.append(float((e12 - e26) / c[-1]))
    dd = np.diff(c[-15:])
    g = float(dd[dd > 0].sum()) if len(dd[dd > 0]) > 0 else 0
    lo = float(abs(dd[dd < 0].sum())) if len(dd[dd < 0]) > 0 else 1e-10
    f.append(float(100 - 100 / (1 + g / lo) if lo > 0 else 50))
    bb = np.std(c[-20:]); m20 = np.mean(c[-20:])
    f.append(float((c[-1] - m20) / (2 * bb + 1e-10)))
    f.append(float(c[-1] / 1000.0))
    for rkey in ['c0', 'm0']:
        rdf = related_data.get(rkey)
        if rdf is not None and len(rdf) > 20:
            rc = rdf['close'].values.astype(float)
            f.append(float((rc[-1] - rc[-4]) / rc[-4]) if len(rc) > 3 else 0.0)
            f.append(float((rc[-1] - np.mean(rc[-min(20, len(rc)):])) / np.mean(rc[-min(20, len(rc)):])))
        else:
            f.extend([0.0, 0.0])
    atr_vals = [abs(h[i] - l_low[i]) for i in range(len(c) - 20, len(c))]
    f.append(float(np.mean(atr_vals)))
    return np.array(f, dtype=np.float32)


def make_target_volatility(df, idx, mult=1.5):
    """
    目标: 明天的振幅(high-low)是否超过最近20天平均振幅的 mult 倍
    返回 1=高波动, 0=正常
    """
    # 今天之前的20天平均振幅
    start = max(0, idx - 20)
    ranges = [abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
              for i in range(start, idx + 1)]
    avg_range = np.mean(ranges) if ranges else 0
    if avg_range <= 0: return 0

    # 明天的振幅
    if idx + 1 >= len(df): return 0
    tomorrow_range = abs(float(df.iloc[idx + 1]['high']) - float(df.iloc[idx + 1]['low']))

    return 1 if tomorrow_range > avg_range * mult else 0


def train_v30(sym_key, related_data):
    name = SYMBOL_MAP[sym_key]['name']
    code = SYMBOL_MAP[sym_key].get('daily_code', 'LH0' if 'lh' in sym_key else 'JM0')
    print(f"\n{'─'*50}")
    print(f"  {name} V30 波动率预测训练")
    print(f"{'─'*50}")

    df = fetch_daily(code, TRAIN_DAYS)
    if df is None or len(df) < 200:
        print("  ❌ 数据不足"); return None

    X, y = [], []
    for i in range(60, len(df) - 1):
        feats = build_features_v30(df, i, related_data)
        if feats is None: continue
        label = make_target_volatility(df, i, 1.5)
        X.append(feats); y.append(label)

    if len(X) < 100:
        print(f"  ❌ 样本不足 ({len(X)})"); return None

    pos_ratio = sum(y) / len(y)
    print(f"  数据: {len(df)}条 | 样本: {len(X)} | 高波动日占比: {pos_ratio:.1%}")

    split = int(len(X) * 0.8)
    X_train = np.array(X[:split], dtype=np.float32); y_train = np.array(y[:split])
    X_calib = np.array(X[split:], dtype=np.float32); y_calib = np.array(y[split:])

    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss', verbosity=0, random_state=42,
        scale_pos_weight=(1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1,  # 处理不平衡
    )
    model.fit(X_train, y_train)
    print(f"  训练: {time.time()-t0:.1f}s | 特征: {X_train.shape[1]}维 | 深度: 3")

    prob = model.predict_proba(X_calib)[:, 1]
    pred = (prob > 0.5).astype(int)
    acc = (pred == y_calib).mean()
    # 高波动召回率
    high_idx = y_calib == 1
    recall = (pred[high_idx] == 1).mean() if high_idx.sum() > 0 else 0
    print(f"  准确率: {acc:.1%} | 高波动召回率: {recall:.1%} | 校准集: {len(y_calib)}样本")

    calib = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calib.fit(np.array(X, dtype=np.float32), np.array(y))
    print(f"  校准: Platt Scaling (cv=5)")

    model_path = os.path.join(MODEL_DIR, f'{sym_key}_xgb_v30.pkl')
    with open(model_path, 'wb') as f: pickle.dump(calib, f)
    print(f"  💾 已保存: {os.path.basename(model_path)}")

    # 预测当前
    last_idx = len(df) - 2
    last_feats = build_features_v30(df, last_idx, related_data)
    if last_feats is not None:
        last_prob = float(calib.predict_proba(last_feats.reshape(1, -1))[0][1])
        label = '⚠️ 高波动' if last_prob > 0.5 else '✅ 正常'
        print(f"  🔮 当前预测(明天高波动): {last_prob:.1%} → {label}")

    return {'samples': len(X), 'accuracy': round(acc, 4), 'recall': round(recall, 4)}


def main():
    print("🧪 Prophet V30 — 波动率预测训练")
    print(f"   目标: 明天振幅 > 1.5倍 20日均振幅")
    print(f"   模型: XGBoost(max_depth=3) + 类别平衡 + Platt校准")

    related = {}
    for key, code in RELATED.items():
        df = fetch_daily(code, TRAIN_DAYS)
        if df is not None:
            related[key] = df
            print(f"  {key}({code}): {len(df)}条")
        else:
            print(f"  {key}({code}): 加载失败")

    for sk in ['lh2609', 'jm2609']:
        train_v30(sk, related)

    print(f"\n{'='*50}")
    print("  V30模型已更新 → models/{sym}_xgb_v30.pkl (波动率预测)")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
