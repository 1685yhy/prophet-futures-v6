#!/usr/bin/env python3
"""
Prophet Futures V33 — 基本面增强模型训练
==========================================
在 V32 19维技术特征基础上，融合 akshare 基本面数据：
  - 养殖成本 (futures_hog_cost)
  - 存栏量 (futures_hog_supply)
  - 外三元现货价格 (spot_hog_year_trend_soozhu)

新增6维基本面特征 (20-25)，总计25维。
Walk-forward: 300天训练 / 30天测试 / 15天步进
XGBoost 三分类
对比 19维技术特征 baseline
"""
import sys, os, time, pickle, json
import numpy as np
import pandas as pd
import xgboost as xgb
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 参数 ──
TRAIN_WINDOW = 300   # 训练窗口（天）
TEST_WINDOW  = 30    # 测试窗口（天）
STEP         = 15    # 步进（天）
MIN_TRAIN    = 150   # 最少训练样本

# ── 特征名 ──
TECH_FEAT_NAMES = [
    '开盘缺口','|缺口|','1日涨跌','3日涨跌','5日涨跌','10日涨跌','20日涨跌',
    'MA5偏离','MA10偏离','MA20偏离','MA60偏离','波动率','日内振幅','量比',
    '持仓比','MACD','RSI','布林带','价格/1000'
]
FUND_FEAT_NAMES = [
    '现货-期货价差率','养殖成本5日变化率','存栏量14日变化率',
    '现货价格14日动量','猪粮比(成本/期货)','期货-成本价差'
]
ALL_FEAT_NAMES = TECH_FEAT_NAMES + FUND_FEAT_NAMES


# ═══════════════════════════════════════════════════════
# PHASE 1: 数据获取
# ═══════════════════════════════════════════════════════

def fetch_daily_history(days=1200):
    """获取 LH 主力合约日线数据"""
    end = datetime.now()
    start = end - timedelta(days=days + 50)
    try:
        df = ak.futures_main_sina(
            symbol='LH0',
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d')
        )
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'oi', 'settle']
        for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"  ❌ 日线获取失败: {e}")
        return None


def fetch_fundamental_data():
    """获取基本面数据：成本、存栏、现货价格"""
    result = {}
    
    # 1. 养殖成本
    print("  📡 获取养殖成本 (futures_hog_cost)...")
    try:
        cost_df = ak.futures_hog_cost()
        cost_df = cost_df.rename(columns={'date': 'date', 'value': 'cost_value'})
        cost_df['date'] = pd.to_datetime(cost_df['date'])
        cost_df = cost_df.sort_values('date').reset_index(drop=True)
        result['cost'] = cost_df
        print(f"    ✅ {len(cost_df)}行  {cost_df['date'].iloc[0].strftime('%Y-%m-%d')} → {cost_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"    ⚠️ 失败: {e}")
        result['cost'] = None
    
    # 2. 存栏量
    print("  📡 获取存栏量 (futures_hog_supply)...")
    try:
        supply_df = ak.futures_hog_supply()
        supply_df = supply_df.rename(columns={'date': 'date', 'value': 'supply_value'})
        supply_df['date'] = pd.to_datetime(supply_df['date'])
        supply_df = supply_df.sort_values('date').reset_index(drop=True)
        result['supply'] = supply_df
        print(f"    ✅ {len(supply_df)}行  {supply_df['date'].iloc[0].strftime('%Y-%m-%d')} → {supply_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"    ⚠️ 失败: {e}")
        result['supply'] = None
    
    # 3. 外三元现货价格
    print("  📡 获取现货价格 (spot_hog_year_trend_soozhu)...")
    try:
        spot_df = ak.spot_hog_year_trend_soozhu()
        spot_df = spot_df.rename(columns={'日期': 'date', '价格': 'spot_price'})
        spot_df['date'] = pd.to_datetime(spot_df['date'])
        spot_df = spot_df.sort_values('date').reset_index(drop=True)
        result['spot'] = spot_df
        print(f"    ✅ {len(spot_df)}行  {spot_df['date'].iloc[0].strftime('%Y-%m-%d')} → {spot_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"    ⚠️ 失败: {e}")
        result['spot'] = None
    
    return result


def merge_fundamental(df, fund_data):
    """将基本面数据按日期合并到日线DataFrame"""
    df = df.copy()
    
    if fund_data.get('cost') is not None:
        cost = fund_data['cost'][['date', 'cost_value']].copy()
        # merge_asof: 取最近的不超过该日期的成本数据
        df = pd.merge_asof(df.sort_values('date'), cost.sort_values('date'),
                           on='date', direction='backward')
    else:
        df['cost_value'] = np.nan
    
    if fund_data.get('supply') is not None:
        supply = fund_data['supply'][['date', 'supply_value']].copy()
        df = pd.merge_asof(df.sort_values('date'), supply.sort_values('date'),
                           on='date', direction='backward')
    else:
        df['supply_value'] = np.nan
    
    if fund_data.get('spot') is not None:
        spot = fund_data['spot'][['date', 'spot_price']].copy()
        df = pd.merge_asof(df.sort_values('date'), spot.sort_values('date'),
                           on='date', direction='backward')
    else:
        df['spot_price'] = np.nan
    
    # Forward fill missing fundamental values
    for col in ['cost_value', 'supply_value', 'spot_price']:
        df[col] = df[col].ffill()
    
    return df.sort_values('date').reset_index(drop=True)


# ═══════════════════════════════════════════════════════
# PHASE 2: 特征工程
# ═══════════════════════════════════════════════════════

def build_tech_features(df, idx, window=60):
    """构建19维技术特征（与 realtime_data.py 完全一致）"""
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
    # 1-2: 开盘缺口
    if idx >= 1:
        f.append(float((o[-1] - c[-2]) / c[-2]))
        f.append(abs(f[-1]))
    else:
        f.extend([0.0, 0.0])
    
    # 3-7: 收益率
    for lag in [1, 3, 5, 10, 20]:
        f.append(float((c[-1] - c[-lag - 1]) / c[-lag - 1] if len(c) > lag else 0))
    
    # 8-11: MA偏离
    for p in [5, 10, 20, 60]:
        ma_val = np.mean(c[-min(p, len(c)):])
        f.append(float((c[-1] - ma_val) / ma_val))
    
    # 12: 波动率
    f.append(float(np.std(c[-20:]) / np.mean(c[-20:])))
    
    # 13: 日内振幅
    f.append(float((h[-1] - l_low[-1]) / c[-1]))
    
    # 14: 量比
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
    f.append(float(v[-1] / vma))
    
    # 15: 持仓比
    f.append(float(oi_v[-1] / np.mean(oi_v[-20:])) if len(oi_v) >= 20 and np.mean(oi_v[-20:]) > 0 else 1)
    
    # 16: MACD
    e12 = c[-1]; e26 = c[-1]
    for j in range(len(c) - 2, -1, -1):
        e12 = (2 / 13) * c[j] + (11 / 13) * e12
        e26 = (2 / 27) * c[j] + (25 / 27) * e26
    f.append(float((e12 - e26) / c[-1]))
    
    # 17: RSI
    dd = np.diff(c[-15:])
    g = float(dd[dd > 0].sum()) if len(dd[dd > 0]) > 0 else 0
    lo = float(abs(dd[dd < 0].sum())) if len(dd[dd < 0]) > 0 else 1e-10
    f.append(float(100 - 100 / (1 + g / lo) if lo > 0 else 50))
    
    # 18: 布林带
    bb = np.std(c[-20:]); m20 = np.mean(c[-20:])
    f.append(float((c[-1] - m20) / (2 * bb + 1e-10)))
    
    # 19: 价格水平
    f.append(float(c[-1] / 1000.0))
    
    return np.array(f, dtype=np.float32)


def build_fund_features(df, idx):
    """构建6维基本面特征 (20-25)"""
    if idx < 20:
        return None
    
    f = []
    close = float(df.iloc[idx]['close'])
    
    # 20: 现货-期货价差率
    spot = df.iloc[idx].get('spot_price', np.nan)
    if pd.notna(spot) and close > 0:
        f.append(float((spot - close) / close))
    else:
        f.append(0.0)
    
    # 21: 养殖成本5日变化率
    cost_val = df.iloc[idx].get('cost_value', np.nan)
    if pd.notna(cost_val) and idx >= 5:
        cost_5d = df.iloc[idx - 5].get('cost_value', np.nan)
        if pd.notna(cost_5d) and cost_5d > 0:
            f.append(float((cost_val - cost_5d) / cost_5d))
        else:
            f.append(0.0)
    else:
        f.append(0.0)
    
    # 22: 存栏量14日变化率
    supply_val = df.iloc[idx].get('supply_value', np.nan)
    if pd.notna(supply_val) and idx >= 14:
        supply_14d = df.iloc[idx - 14].get('supply_value', np.nan)
        if pd.notna(supply_14d) and supply_14d > 0:
            f.append(float((supply_val - supply_14d) / supply_14d))
        else:
            f.append(0.0)
    else:
        f.append(0.0)
    
    # 23: 现货价格14日动量
    if pd.notna(spot) and idx >= 14:
        spot_14d = df.iloc[idx - 14].get('spot_price', np.nan)
        if pd.notna(spot_14d) and spot_14d > 0:
            f.append(float((spot - spot_14d) / spot_14d))
        else:
            f.append(0.0)
    else:
        f.append(0.0)
    
    # 24: 猪粮比（用成本/期货价作为代理）
    if pd.notna(cost_val) and close > 0 and cost_val > 0:
        f.append(float(cost_val / close))
    else:
        f.append(0.0)
    
    # 25: 期货-成本价差
    if pd.notna(cost_val) and cost_val > 0:
        f.append(float(close / cost_val - 1))
    else:
        f.append(0.0)
    
    return np.array(f, dtype=np.float32)


def build_all_features(df, idx, window=60):
    """构建完整25维特征向量"""
    tech = build_tech_features(df, idx, window)
    fund = build_fund_features(df, idx)
    
    if tech is None or fund is None:
        return None
    
    return np.concatenate([tech, fund]).astype(np.float32)


# ═══════════════════════════════════════════════════════
# PHASE 3: 三分类标签
# ═══════════════════════════════════════════════════════

def make_labels(df, min_idx, max_idx):
    """生成三分类标签：基于未来1日收益率的 tertile 分桶
    
    返回: labels dict {idx: class} 和 thresholds
    class 2: 强势上涨 (top tertile)
    class 1: 横盘震荡 (middle tertile)  
    class 0: 弱势下跌 (bottom tertile)
    """
    returns = []
    idx_list = []
    for i in range(min_idx, min(max_idx, len(df) - 1)):
        ret = (float(df.iloc[i + 1]['close']) - float(df.iloc[i]['close'])) / float(df.iloc[i]['close'])
        returns.append(ret)
        idx_list.append(i)
    
    if len(returns) < 30:
        return {}, None
    
    returns = np.array(returns)
    lo_thresh = np.percentile(returns, 33.33)
    hi_thresh = np.percentile(returns, 66.67)
    
    labels = {}
    for i, ret in zip(idx_list, returns):
        if ret > hi_thresh:
            labels[i] = 2
        elif ret < lo_thresh:
            labels[i] = 0
        else:
            labels[i] = 1
    
    thresholds = {'lo': float(lo_thresh), 'hi': float(hi_thresh)}
    return labels, thresholds


# ═══════════════════════════════════════════════════════
# PHASE 4: Walk-forward 训练 + 评估
# ═══════════════════════════════════════════════════════

def walk_forward_train(df, fund_feat_enabled=True):
    """Walk-forward 训练评估
    
    Returns: list of fold results
    """
    n = len(df)
    results = []
    
    # 计算全局标签（使用全部数据做 tertile 分桶更稳定）
    min_feat_idx = 80  # 需要至少这么多数据才能构建特征
    global_labels, thresholds = make_labels(df, min_feat_idx, n)
    if thresholds is None:
        print("  ❌ 标签计算失败")
        return []
    
    print(f"  三分类阈值: 下跌<{thresholds['lo']:.4f}, 横盘[{thresholds['lo']:.4f},{thresholds['hi']:.4f}], 上涨>{thresholds['hi']:.4f}")
    
    # Walk-forward
    fold_id = 0
    test_start = min_feat_idx + TRAIN_WINDOW
    
    while test_start + TEST_WINDOW <= n:
        train_end = test_start
        train_start = max(min_feat_idx, train_end - TRAIN_WINDOW)
        test_end = min(test_start + TEST_WINDOW, n)
        
        # 准备训练数据
        X_train, y_train = [], []
        for i in range(train_start, train_end - 1):
            if fund_feat_enabled:
                feats = build_all_features(df, i)
            else:
                feats = build_tech_features(df, i)
            
            if feats is not None and i in global_labels:
                X_train.append(feats)
                y_train.append(global_labels[i])
        
        # 准备测试数据
        X_test, y_test, test_indices = [], [], []
        for i in range(test_start, test_end - 1):
            if fund_feat_enabled:
                feats = build_all_features(df, i)
            else:
                feats = build_tech_features(df, i)
            
            if feats is not None and i in global_labels:
                X_test.append(feats)
                y_test.append(global_labels[i])
                test_indices.append(i)
        
        if len(X_train) < MIN_TRAIN or len(X_test) < 10:
            test_start += STEP
            continue
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test)
        
        # 训练 XGBoost 三分类
        n_classes = len(set(y_train))
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=n_classes,
            random_state=42, verbosity=0, n_jobs=1
        )
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        
        results.append({
            'fold': fold_id,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': float(acc),
            'model': model,
        })
        
        fold_id += 1
        test_start += STEP
    
    return results


def train_final_model(df, fund_feat_enabled=True):
    """在所有可用数据上训练最终模型并返回特征重要性"""
    min_feat_idx = 80
    n = len(df)
    
    global_labels, thresholds = make_labels(df, min_feat_idx, n)
    if thresholds is None:
        return None, None, None
    
    X_all, y_all = [], []
    for i in range(min_feat_idx, n - 1):
        if fund_feat_enabled:
            feats = build_all_features(df, i)
        else:
            feats = build_tech_features(df, i)
        
        if feats is not None and i in global_labels:
            X_all.append(feats)
            y_all.append(global_labels[i])
    
    if len(X_all) < MIN_TRAIN:
        return None, None, None
    
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all)
    
    n_classes = len(set(y_all))
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=n_classes,
        random_state=42, verbosity=0, n_jobs=1
    )
    model.fit(X_all, y_all)
    
    # 特征重要性
    feat_imp = model.feature_importances_
    
    # 最后一行的特征
    last_feats = None
    if fund_feat_enabled:
        last_feats = build_all_features(df, n - 2)
    else:
        last_feats = build_tech_features(df, n - 2)
    
    return model, feat_imp, last_feats, thresholds


# ═══════════════════════════════════════════════════════
# PHASE 5: 当前预测
# ═══════════════════════════════════════════════════════

def predict_current(model, df, fund_feat_enabled=True):
    """对当前最新数据做预测"""
    n = len(df)
    
    # 尝试最后几个能构建特征的索引
    for offset in [2, 3, 4, 5]:
        idx = n - offset
        if idx < 80:
            continue
        
        if fund_feat_enabled:
            feats = build_all_features(df, idx)
        else:
            feats = build_tech_features(df, idx)
        
        if feats is not None:
            proba = model.predict_proba(feats.reshape(1, -1))[0]
            pred_class = int(np.argmax(proba))
            
            class_names = {0: '📉 看跌', 1: '↔️ 震荡', 2: '📈 看涨'}
            return {
                'prediction': class_names.get(pred_class, f'class_{pred_class}'),
                'probabilities': {f'class_{i}': float(p) for i, p in enumerate(proba)},
                'date': str(df.iloc[idx]['date']),
                'close': float(df.iloc[idx]['close']),
            }
    
    return None


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🧬 Prophet Futures V33 — 基本面增强模型训练")
    print("  生猪 LH — XGBoost 三分类")
    print("=" * 60)
    
    # ── 1. 获取日线数据 ──
    print("\n📡 PHASE 1: 获取日线数据")
    df = fetch_daily_history(1200)
    if df is None or len(df) < 200:
        print("  ❌ 日线数据不足，退出")
        return
    print(f"  ✅ LH 日线: {len(df)}行  {df['date'].iloc[0].strftime('%Y-%m-%d')} → {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # ── 2. 获取基本面数据并合并 ──
    print("\n📡 PHASE 2: 获取基本面数据")
    fund_data = fetch_fundamental_data()
    df_full = merge_fundamental(df, fund_data)
    
    has_fund = fund_data['cost'] is not None or fund_data['supply'] is not None or fund_data['spot'] is not None
    print(f"  基本面数据可用: {'✅ 是' if has_fund else '⚠️ 部分缺失'}")
    print(f"  合并后: {len(df_full)}行, 列: {list(df_full.columns)}")
    
    # ── 3. Walk-forward baseline (仅技术特征) ──
    print("\n📊 PHASE 3: Walk-forward Baseline (19维技术特征)")
    t0 = time.time()
    results_baseline = walk_forward_train(df_full, fund_feat_enabled=False)
    t_baseline = time.time() - t0
    
    if results_baseline:
        accs = [r['accuracy'] for r in results_baseline]
        print(f"  Baseline: {len(results_baseline)} folds | 平均准确率: {np.mean(accs):.2%} (±{np.std(accs):.2%})")
        print(f"  各fold: {[f'{a:.2%}' for a in accs]}")
    else:
        print("  ⚠️ Baseline 无有效结果")
    
    # ── 4. Walk-forward 增强版 ──
    print("\n📊 PHASE 4: Walk-forward V33 (25维 技术+基本面)")
    t0 = time.time()
    results_v33 = walk_forward_train(df_full, fund_feat_enabled=True)
    t_v33 = time.time() - t0
    
    if results_v33:
        accs = [r['accuracy'] for r in results_v33]
        print(f"  V33: {len(results_v33)} folds | 平均准确率: {np.mean(accs):.2%} (±{np.std(accs):.2%})")
        print(f"  各fold: {[f'{a:.2%}' for a in accs]}")
    else:
        print("  ⚠️ V33 无有效结果")
    
    # ── 5. 训练最终模型 ──
    print("\n🔧 PHASE 5: 训练最终模型 & 特征重要性")
    final_model, feat_imp, last_feats, thresholds = train_final_model(df_full, fund_feat_enabled=True)
    
    feat_importance_ranking = []
    if feat_imp is not None:
        ranked = sorted(
            zip(ALL_FEAT_NAMES, feat_imp),
            key=lambda x: x[1], reverse=True
        )
        print("\n  特征重要性排名 (降序):")
        for rank, (name, imp) in enumerate(ranked, 1):
            tag = "🔬" if name in FUND_FEAT_NAMES else "📊"
            print(f"  {rank:2d}. {tag} {name}: {imp:.4f}")
            feat_importance_ranking.append({'rank': rank, 'name': name, 'importance': float(imp), 'type': 'fundamental' if name in FUND_FEAT_NAMES else 'technical'})
    
    # ── 6. 当前预测 ──
    print("\n🔮 PHASE 6: 当前预测")
    current_pred = None
    if final_model is not None:
        current_pred = predict_current(final_model, df_full, fund_feat_enabled=True)
        if current_pred:
            print(f"  日期: {current_pred['date']}")
            print(f"  收盘价: {current_pred['close']:.0f}")
            print(f"  预测方向: {current_pred['prediction']}")
            probs = current_pred['probabilities']
            print(f"  概率: P(跌)={probs['class_0']:.3f}  P(震)={probs['class_1']:.3f}  P(涨)={probs['class_2']:.3f}")
        else:
            print("  ⚠️ 无法生成当前预测")
    
    # ── 7. 保存模型 ──
    if final_model is not None:
        model_path = os.path.join(MODEL_DIR, 'v33_fundamental_xgb.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)
        print(f"\n💾 模型已保存: {model_path}")
    
    # ── 8. 准确率对比表 ──
    print("\n" + "=" * 60)
    print("  📋 V33 结果汇总")
    print("=" * 60)
    
    bl_mean = np.mean([r['accuracy'] for r in results_baseline]) if results_baseline else 0
    v33_mean = np.mean([r['accuracy'] for r in results_v33]) if results_v33 else 0
    
    print(f"\n  {'指标':<20} {'Baseline(19维)':<18} {'V33(25维)':<18} {'差异':<10}")
    print(f"  {'─'*20} {'─'*18} {'─'*18} {'─'*10}")
    print(f"  {'平均准确率':<20} {bl_mean:<18.2%} {v33_mean:<18.2%} {v33_mean-bl_mean:>+.2%}")
    print(f"  {'Fold数':<20} {len(results_baseline):<18} {len(results_v33):<18}")
    print(f"  {'训练耗时':<20} {t_baseline:<18.1f}s {t_v33:<18.1f}s")
    
    # ── 9. 飞书通知 ──
    print("\n📨 发送飞书通知...")
    try:
        sys.path.insert(0, BASE_DIR)
        from feishu_card import send_card, md, note
        
        # 构建基本面特征排名
        fund_rank_lines = []
        for item in feat_importance_ranking:
            if item['type'] == 'fundamental':
                fund_rank_lines.append(f"  第{item['rank']}名: {item['name']} ({item['importance']:.4f})")
        
        fund_rank_text = "\n".join(fund_rank_lines) if fund_rank_lines else "  无基本面特征排名数据"
        
        # 当前预测信息
        pred_text = ""
        if current_pred:
            probs = current_pred['probabilities']
            pred_text = (
                f"**当前预测** (基于 {current_pred['date']} 收盘 {current_pred['close']:.0f}):\n"
                f"方向: **{current_pred['prediction']}**\n"
                f"P(跌)={probs['class_0']:.1%} | P(震)={probs['class_1']:.1%} | P(涨)={probs['class_2']:.1%}"
            )
        
        elements = [
            md(f"**🧬 V33 基本面增强模型** — 生猪 LH · XGBoost 三分类\n"),
            md(
                f"**准确率对比**\n"
                f"| 版本 | 准确率 | Folds |\n"
                f"|------|--------|-------|\n"
                f"| Baseline (19维技术) | {bl_mean:.2%} | {len(results_baseline)} |\n"
                f"| V33 (25维 技术+基本面) | {v33_mean:.2%} | {len(results_v33)} |\n"
                f"| **提升** | **{v33_mean-bl_mean:+.2%}** | |"
            ),
            md(
                f"**基本面特征重要性排名**\n{fund_rank_text}"
            ),
        ]
        
        if pred_text:
            elements.append(md(pred_text))
        
        elements.append(md(f"━━━━━━━━━━━━\n训练窗口{TRAIN_WINDOW}天 | 测试{TEST_WINDOW}天 | 步进{STEP}天\n模型: models/v33_fundamental_xgb.pkl"))
        
        ok, msg = send_card(
            title="🧬 V33 基本面增强模型训练完成",
            elements=elements,
            template="purple",
            subtitle=f"LH 生猪 · 准确率 {v33_mean:.1%} (Baseline {bl_mean:.1%})"
        )
        if ok:
            print(f"  ✅ 飞书卡片已发送: {msg}")
        else:
            print(f"  ⚠️ 飞书发送失败: {msg}")
    except Exception as e:
        print(f"  ⚠️ 飞书通知异常: {e}")
    
    print("\n" + "=" * 60)
    print("  ✅ V33 训练完成!")
    print(f"  模型: models/v33_fundamental_xgb.pkl")
    print(f"  准确率: {v33_mean:.2%} (vs Baseline {bl_mean:.2%}, 提升 {v33_mean-bl_mean:+.2%})")
    print("=" * 60)


if __name__ == '__main__':
    main()
