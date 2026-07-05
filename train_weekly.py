#!/usr/bin/env python3
"""Prophet Futures — 每周模型重训 + 对比报告
用法: python train_weekly.py [--dry-run] [--switch] [--calibrate]

默认行为: 训新模型但不替换 → 保存为 {sym}_xgb_new.pkl，生成对比报告
--switch: 替换当前生效模型（需要人工确认）
--calibrate: 训练后做概率校准（Platt Scaling），生成 {sym}_xgb_calibrated.pkl
--dry-run: 不保存文件，只验证流程
"""
import sys, os, json, time
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import pickle
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import build_features, get_daily_history, SYMBOL_MAP

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
TRAIN_DAYS = 1200

SYMBOLS = ['lh2609', 'jm2609']
PARAMS = {
    'lh2609': {'n_est': 200, 'depth': 5, 'lr': 0.05},
    'jm2609': {'n_est': 100, 'depth': 4, 'lr': 0.03},
}

FEAT_NAMES = [
    '开盘缺口', '|缺口|', '1日涨跌', '3日涨跌', '5日涨跌',
    '10日涨跌', '20日涨跌', 'MA5偏离', 'MA10偏离', 'MA20偏离',
    'MA60偏离', '波动率(20)', '日内振幅', '量比(20)',
    '持仓比(20)', 'MACD', 'RSI(14)', '布林带', '价格/1000'
]

# 概率校准参数
CALIB_METHOD = 'sigmoid'   # 'sigmoid' (Platt, 适合小数据) 或 'isotonic' (需更多数据)
CALIB_CV = 3               # 交叉验证折数


def load_data(sym_key):
    df = get_daily_history(sym_key, TRAIN_DAYS)
    if df is None or len(df) < 200:
        return None
    for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)
    return df


def build_training_data(df, win=60):
    split = int(len(df) * 0.6)
    if split < win + 100:
        return None, None
    X, y = [], []
    for i in range(win, split - 1):
        feats = build_features(df, i, win)
        if feats is None:
            continue
        label = 1 if df.iloc[i + 1]['close'] > df.iloc[i]['close'] else 0
        X.append(feats)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def train_xgb(X, y, params):
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
    return model, time.time() - t0


def validate(model, df, win=60):
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
    return correct / total if total > 0 else None


def calibrate_model(model, X, y, method='sigmoid', cv=3):
    """用 Platt Scaling 或 Isotonic Regression 校准概率
    
    method: 'sigmoid' (Platt, 推荐小数据集) 或 'isotonic'
    cv: 交叉验证折数（数据少时用 cv=3 或 5）
    
    返回: 校准后的模型 (CalibratedClassifierCV)
    """
    t0 = time.time()
    calib = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
        n_jobs=1,
    )
    calib.fit(X, y)
    elapsed = time.time() - t0
    return calib, elapsed


def predict_current(model, sym_key):
    """用当前最新日线做预测"""
    df = get_daily_history(sym_key, 1200)
    if df is None:
        return None
    feats = build_features(df, len(df) - 1, 60)
    if feats is None:
        return None
    prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
    direction = 'LONG' if prob > 0.5 else 'SHORT'
    confidence = prob if prob > 0.5 else (1 - prob)
    return {
        'prob': prob,
        'direction': direction,
        'confidence': confidence,
        'data_date': str(df.iloc[-1]['date']),
    }


def load_model(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def compare_importances(old_imp, new_imp, threshold=0.005):
    """返回变化超过阈值的特征"""
    changes = []
    for i, name in enumerate(FEAT_NAMES):
        delta = new_imp[i] - old_imp[i]
        if abs(delta) >= threshold:
            changes.append({
                'name': name,
                'old': round(float(old_imp[i]), 4),
                'new': round(float(new_imp[i]), 4),
                'delta': round(float(delta), 4),
            })
    changes.sort(key=lambda x: abs(x['delta']), reverse=True)
    return changes


def main():
    dry_run = '--dry-run' in sys.argv
    do_switch = '--switch' in sys.argv
    do_calibrate = '--calibrate' in sys.argv
    now = datetime.now()
    date_tag = now.strftime('%Y%m%d_%H%M')
    backup_dir = os.path.join(OUTPUT_DIR, 'backups', date_tag)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not dry_run:
        os.makedirs(backup_dir, exist_ok=True)

    print("=" * 60)
    print("  Prophet Futures — 每周模型重训")
    print(f"  时间: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  特征: 19维")
    print(f"  备份: {backup_dir}/")
    if dry_run:
        print("  ⚠️  DRY RUN — 不保存文件")
    if do_switch:
        print("  ⚡ 将替换当前生效模型")
    else:
        print("  📋 仅训新模型，不替换当前")
    print("=" * 60)

    comparison = {}
    all_passed = True

    for sym_key in SYMBOLS:
        name = SYMBOL_MAP[sym_key]['name']
        params = PARAMS[sym_key]
        model_path = os.path.join(OUTPUT_DIR, f'{sym_key}_xgb.pkl')
        new_path = os.path.join(OUTPUT_DIR, f'{sym_key}_xgb_new.pkl')

        print(f"\n{'─' * 40}")
        print(f"  {name} ({sym_key})")
        print(f"{'─' * 40}")

        # ── 1. 备份旧模型 + 元数据 ──
        old_model = load_model(model_path)
        old_pred = None
        old_imp = None

        if old_model:
            old_pred = predict_current(old_model, sym_key)
            old_imp = old_model.feature_importances_
            if not dry_run:
                cp_path = os.path.join(backup_dir, f'{sym_key}_xgb_old.pkl')
                with open(cp_path, 'wb') as f:
                    pickle.dump(old_model, f)
            dir_cn = '做多' if (old_pred['direction'] == 'LONG' if old_pred else False) else '做空'
            prob_str = f"{old_pred['prob']:.1%}" if old_pred else 'N/A'
            print(f"  📦 旧模型: {dir_cn} {prob_str} (数据: {old_pred.get('data_date', '?') if old_pred else '?'})")
        else:
            print(f"  📦 无旧模型（首次训练）")

        # ── 2. 加载数据 ──
        print(f"  加载日线数据...")
        df = load_data(sym_key)
        if df is None:
            all_passed = False
            continue
        print(f"  ✅ {len(df)} 条 ({df.iloc[0]['date']} → {df.iloc[-1]['date']})")

        # ── 3. 训练 ──
        X, y = build_training_data(df)
        if X is None or len(X) < 100:
            print(f"  ❌ 样本不足")
            all_passed = False
            continue
        print(f"  训练 XGBoost ({params['n_est']}树, {len(X)}样本)...")
        model, elapsed = train_xgb(X, y, params)
        print(f"  ✅ 训练完成 ({elapsed:.1f}s)")

        # ── 3b. 概率校准（可选）──
        calibrated_model = None
        calibrate_time = 0
        if do_calibrate:
            print(f"  校准 (Platt Scaling, cv={CALIB_CV})...")
            calibrated_model, calibrate_time = calibrate_model(model, X, y, method=CALIB_METHOD, cv=CALIB_CV)
            print(f"  ✅ 校准完成 ({calibrate_time:.1f}s)")

        # ── 4. 验证 ──
        acc = validate(model, df)
        acc_str = f"{acc:.1%}" if acc else 'N/A'
        print(f"  验证准确率: {acc_str}")

        calib_acc = None
        if calibrated_model:
            calib_acc = validate(calibrated_model, df)
            calib_acc_str = f"{calib_acc:.1%}" if calib_acc else 'N/A'
            print(f"  校准后验证准确率: {calib_acc_str}")

        # ── 5. 新模型预测 ──
        new_pred = predict_current(model, sym_key)
        new_imp = model.feature_importances_

        if new_pred:
            dir_cn = '做多' if new_pred['direction'] == 'LONG' else '做空'
            print(f"  🔮 新模型预测: {dir_cn} {new_pred['prob']:.1%}")

        calib_pred = None
        if calibrated_model:
            calib_pred = predict_current(calibrated_model, sym_key)
            if calib_pred:
                dir_cn = '做多' if calib_pred['direction'] == 'LONG' else '做空'
                print(f"  🔧 校准后预测: {dir_cn} {calib_pred['prob']:.1%}")

        # ── 6. 保存 ──
        if not dry_run:
            # 保存新模型
            with open(new_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  💾 已保存: {new_path}")

            # 保存校准模型
            if calibrated_model:
                calib_path = os.path.join(OUTPUT_DIR, f'{sym_key}_xgb_calibrated.pkl')
                with open(calib_path, 'wb') as f:
                    pickle.dump(calibrated_model, f)
                print(f"  🔧 已保存校准模型: {calib_path}")

            # 备份新模型
            cp_path = os.path.join(backup_dir, f'{sym_key}_xgb_new.pkl')
            with open(cp_path, 'wb') as f:
                pickle.dump(model, f)

            # 如果要替换
            if do_switch and os.path.exists(new_path):
                os.replace(new_path, model_path)
                print(f"  ⚡ 已替换生效模型: {model_path}")

        # ── 7. 对比报告 ──
        changes = []
        if old_imp is not None and new_imp is not None:
            changes = compare_importances(old_imp, new_imp)
            if changes:
                print(f"\n  📊 特征重要性变化 (>{0.005:.0%}):")
                for c in changes[:5]:
                    arrow = '↑' if c['delta'] > 0 else '↓'
                    print(f"    {arrow} {c['name']:<12} {c['old']:.4f} → {c['new']:.4f} ({c['delta']:+.4f})")

        comparison[sym_key] = {
            'name': name,
            'samples': len(X),
            'accuracy': round(acc, 4) if acc else None,
            'train_time': round(elapsed, 1),
            'data_end': str(df.iloc[-1]['date']),
            'old_pred': old_pred,
            'new_pred': new_pred,
            'imp_changes': changes if old_imp is not None else [],
        }

        if not dry_run:
            # 保存完整元数据到备份目录
            meta = {k: v for k, v in comparison[sym_key].items() if k != 'old_model'}
            # 将 prediction 转为可序列化格式
            for key in ['old_pred', 'new_pred']:
                if meta[key]:
                    meta[key] = {
                        'prob': round(meta[key]['prob'], 4),
                        'direction': meta[key]['direction'],
                        'confidence': round(meta[key]['confidence'], 4),
                        'data_date': meta[key]['data_date'],
                    }
            with open(os.path.join(backup_dir, f'{sym_key}_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    # ══════════════════ 最终对比报告 ══════════════════
    print(f"\n{'=' * 60}")
    print("  📋 对比报告")
    print(f"{'=' * 60}")

    for sym_key, c in comparison.items():
        print(f"\n  {c['name']} ({sym_key})")
        print(f"  训练: {c['samples']}样本 | 验证准确率: {c['accuracy']:.1%}" if c['accuracy'] else f"  训练: {c['samples']}样本")
        print(f"  数据截止: {c['data_end']}")

        old_p = c['old_pred']
        new_p = c['new_pred']

        if old_p and new_p:
            old_dir = '做多' if old_p['direction'] == 'LONG' else '做空'
            new_dir = '做多' if new_p['direction'] == 'LONG' else '做空'
            changed = '⚠️ 方向翻转!' if old_p['direction'] != new_p['direction'] else '✅ 方向一致'
            print(f"  旧模型: {old_dir} {old_p['prob']:.1%} (数据: {old_p['data_date']})")
            print(f"  新模型: {new_dir} {new_p['prob']:.1%} → {changed}")
        elif new_p:
            print(f"  新模型: {'做多' if new_p['direction']=='LONG' else '做空'} {new_p['prob']:.1%}")

        if c['imp_changes']:
            print(f"  特征变化:")
            for ch in c['imp_changes'][:3]:
                arrow = '↑' if ch['delta'] > 0 else '↓'
                print(f"    {arrow} {ch['name']} {ch['old']:.4f}→{ch['new']:.4f}")

    print()

    if dry_run:
        print("  ⚠️  DRY RUN — 未保存任何文件")
    elif do_switch:
        print("  ⚡ 新模型已生效")
        print(f"  📦 旧模型备份: {backup_dir}/")
    else:
        print(f"  ✅ 新模型已生成（未替换当前）")
        print(f"  📂 备份目录: {backup_dir}/")
        print(f"  🖐  如需切换: python train_weekly.py --switch")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
