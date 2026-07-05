#!/usr/bin/env python3
"""Prophet Futures — 每周模型重训脚本
生成与生产环境兼容的 XGBoost 模型文件 (19特征)
用法: python train_weekly.py [--dry-run]
"""
import sys, os, json, time
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import pickle
import xgboost as xgb

# 导入公共特征函数（与生产一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import build_features, get_daily_history, SYMBOL_MAP

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
TRAIN_DAYS = 1200

SYMBOLS = ['lh2609', 'jm2609']
PARAMS = {
    'lh2609': {'n_est': 200, 'depth': 5, 'lr': 0.05},
    'jm2609': {'n_est': 100, 'depth': 4, 'lr': 0.03},
}


def load_data(sym_key):
    """加载日线数据"""
    df = get_daily_history(sym_key, TRAIN_DAYS)
    if df is None or len(df) < 200:
        print(f"  ❌ 数据不足 ({len(df) if df is not None else 0} 条)")
        return None
    for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)
    return df


def build_training_data(df, win=60):
    """用前60%数据构建训练集（严格时序分割）"""
    split = int(len(df) * 0.6)
    if split < win + 100:
        return None, None

    X, y = [], []
    for i in range(win, split - 1):
        feats = build_features(df, i, win)
        if feats is None:
            continue
        # 标签：下一天涨=1，跌=0
        label = 1 if df.iloc[i + 1]['close'] > df.iloc[i]['close'] else 0
        X.append(feats)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


def train_model(X, y, params):
    """训练 XGBoost"""
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=params['n_est'],
        max_depth=params['depth'],
        learning_rate=params['lr'],
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        random_state=42,
    )
    model.fit(X, y)
    elapsed = time.time() - t0
    return model, elapsed


def quick_validate(model, df, win=60):
    """简单验证：用后40%数据测试准确率"""
    split = int(len(df) * 0.6)
    if split >= len(df) - 20:
        return None

    correct, total = 0, 0
    for i in range(split, len(df) - 1):
        feats = build_features(df, i, win)
        if feats is None:
            continue
        pred = model.predict(feats.reshape(1, -1))[0]
        actual = 1 if df.iloc[i + 1]['close'] > df.iloc[i]['close'] else 0
        if pred == actual:
            correct += 1
        total += 1

    if total == 0:
        return None
    return correct / total


def main():
    dry_run = '--dry-run' in sys.argv
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Prophet Futures — 每周模型重训")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  特征: 19维 (与生产一致)")
    print(f"  输出: {OUTPUT_DIR}/")
    if dry_run:
        print("  ⚠️ DRY RUN — 不保存文件")
    print("=" * 60)

    results = {}
    all_passed = True

    for sym_key in SYMBOLS:
        name = SYMBOL_MAP[sym_key]['name']
        params = PARAMS[sym_key]
        print(f"\n{'─' * 40}")
        print(f"  {name} ({sym_key})")
        print(f"{'─' * 40}")

        # 1. 加载数据
        print(f"  加载日线数据...")
        df = load_data(sym_key)
        if df is None:
            all_passed = False
            continue
        print(f"  ✅ {len(df)} 条 ({df.iloc[0]['date']} → {df.iloc[-1]['date']})")

        # 2. 构建训练集
        print(f"  构建特征...")
        X, y = build_training_data(df)
        if X is None or len(X) < 100:
            print(f"  ❌ 样本不足")
            all_passed = False
            continue
        print(f"  ✅ {len(X)} 样本, {X.shape[1]} 特征")

        # 3. 训练
        print(f"  训练 XGBoost ({params['n_est']}树, 深度{params['depth']})...")
        model, elapsed = train_model(X, y, params)
        print(f"  ✅ 训练完成 ({elapsed:.1f}s)")

        # 4. 验证
        print(f"  验证准确率...")
        acc = quick_validate(model, df)
        if acc:
            print(f"  ✅ 准确率: {acc:.1%}")
        else:
            print(f"  ⚠️ 无法验证（数据不足）")

        # 5. 保存
        path = os.path.join(OUTPUT_DIR, f'{sym_key}_xgb.pkl')
        if not dry_run:
            # 先备份旧模型
            if os.path.exists(path):
                backup = path + '.bak'
                os.rename(path, backup)
                print(f"  📦 旧模型已备份: {backup}")

            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  💾 已保存: {path}")
        else:
            print(f"  🔍 DRY RUN — 跳过保存")

        results[sym_key] = {
            'name': name,
            'samples': len(X),
            'features': X.shape[1],
            'accuracy': acc,
            'train_time': round(elapsed, 1),
            'data_end': str(df.iloc[-1]['date']),
        }

    # 汇总
    print(f"\n{'=' * 60}")
    print("  训练汇总")
    print(f"{'=' * 60}")
    for sym_key, r in results.items():
        acc_str = f"{r['accuracy']:.1%}" if r['accuracy'] else 'N/A'
        print(f"  {r['name']}: {r['samples']}样本 {r['features']}特征 "
              f"准确率={acc_str} 数据截止={r['data_end']}")
    print()

    if dry_run:
        print("  ⚠️ DRY RUN 完成 — 未保存任何文件")
    else:
        print("  ✅ 所有模型已更新")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
