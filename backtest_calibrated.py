#!/usr/bin/env python3
"""Prophet Futures — Walk-Forward 回测
对比 未校准 vs 校准后 模型在真实交易模拟中的表现
"""
import sys, os, pickle, json
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import build_features, get_daily_history, SYMBOL_MAP

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
SYMBOLS = ['lh2609', 'jm2609']
INITIAL_CASH = 300000
TRAIN_WINDOW = 600  # 用最近600天训练
RETRAIN_EVERY = 60   # 每60天重训一次

def simulate(model, df, start_idx, window=60):
    """用模型在测试集上模拟交易，返回权益曲线和统计"""
    equity = [INITIAL_CASH]
    trades = []
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0

    for i in range(start_idx, len(df) - 1):
        feats = build_features(df, i, window)
        if feats is None:
            equity.append(equity[-1])
            continue

        prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
        price = float(df.iloc[i]['close'])
        next_price = float(df.iloc[i + 1]['close'])

        # 交易信号
        if prob > 0.55:
            target = 1   # LONG
        elif prob < 0.45:
            target = -1  # SHORT
        else:
            target = 0   # FLAT

        # 平仓
        if position != 0 and target != position:
            pnl = (price - entry_price) * position
            trades.append({'entry': entry_price, 'exit': price, 'pnl': pnl, 'dir': 'LONG' if position == 1 else 'SHORT'})
            equity.append(equity[-1] + pnl)
            position = 0
            entry_price = 0

        # 开仓
        if position == 0 and target != 0:
            position = target
            entry_price = price
            equity.append(equity[-1])
        elif position == 0:
            equity.append(equity[-1])
        else:
            # 持仓中，按盯市算浮动
            floating = (price - entry_price) * position
            equity.append(equity[-1] + floating - (equity[-1] - (equity[-2] if len(equity) > 1 else INITIAL_CASH)))

    # 强制平仓
    if position != 0:
        last_price = float(df.iloc[-1]['close'])
        pnl = (last_price - entry_price) * position
        trades.append({'entry': entry_price, 'exit': last_price, 'pnl': pnl, 'dir': 'LONG' if position == 1 else 'SHORT'})

    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[returns != 0]  # 去掉0收益（空仓日）

    win_trades = [t for t in trades if t['pnl'] > 0]
    loss_trades = [t for t in trades if t['pnl'] < 0]

    total_return = (equity[-1] - INITIAL_CASH) / INITIAL_CASH
    max_eq = np.maximum.accumulate(equity)
    dd = (equity - max_eq) / max_eq
    max_dd = abs(dd.min())

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0

    return {
        'total_return': total_return,
        'final_equity': equity[-1],
        'sharpe': sharpe,
        'max_dd': max_dd,
        'n_trades': len(trades),
        'win_rate': len(win_trades) / len(trades) if trades else 0,
        'avg_win': np.mean([t['pnl'] for t in win_trades]) if win_trades else 0,
        'avg_loss': np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0,
        'profit_factor': abs(sum(t['pnl'] for t in win_trades) / sum(t['pnl'] for t in loss_trades)) if loss_trades and sum(t['pnl'] for t in loss_trades) != 0 else (999 if win_trades else 0),
        'equity_curve': equity.tolist(),
        'trades': trades,
    }


def run_backtest(sym_key):
    name = SYMBOL_MAP[sym_key]['name']
    print(f"\n{'='*60}")
    print(f"  {name} Walk-Forward 回测")
    print(f"{'='*60}")

    # 加载数据
    df = get_daily_history(sym_key, 1200)
    if df is None or len(df) < 800:
        print("  ❌ 数据不足")
        return None

    for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)
    print(f"  数据: {len(df)} 条 ({df.iloc[0]['date']} → {df.iloc[-1]['date']})")

    # 确定回测起点（留够训练窗口）
    test_start = TRAIN_WINDOW + 100
    print(f"  训练窗口: {TRAIN_WINDOW}天 | 重训间隔: {RETRAIN_EVERY}天")
    print(f"  回测期: {df.iloc[test_start]['date']} → {df.iloc[-1]['date']} ({len(df)-test_start}天)")

    results = {}

    for model_label, model_path in [
        ('未校准 (旧模型)', os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')),
        ('校准后 (Platt)', os.path.join(MODEL_DIR, f'{sym_key}_xgb_calibrated.pkl')),
    ]:
        if not os.path.exists(model_path):
            print(f"\n  ⚠️ {model_label}: 模型不存在 ({model_path})")
            continue

        with open(model_path, 'rb') as f:
            base_model = pickle.load(f)

        print(f"\n  ── {model_label} ──")

        # Walk-forward: 滚动训练+测试
        all_trades = []
        equity_history = [INITIAL_CASH]
        current_idx = test_start
        n_retrains = 0

        while current_idx < len(df) - 1:
            # 重训
            train_end = current_idx
            train_start = max(0, train_end - TRAIN_WINDOW)
            if (n_retrains == 0 or (current_idx - test_start) % RETRAIN_EVERY == 0) and train_end - train_start >= 200:
                X_train, y_train = [], []
                for i in range(train_start + 60, train_end - 1):
                    feats = build_features(df, i, 60)
                    if feats is None:
                        continue
                    label = 1 if df.iloc[i + 1]['close'] > df.iloc[i]['close'] else 0
                    X_train.append(feats)
                    y_train.append(label)

                if len(X_train) >= 100:
                    try:
                        import xgboost as xgb
                        model = xgb.XGBClassifier(
                            n_estimators=100, max_depth=4, learning_rate=0.05,
                            use_label_encoder=False, eval_metric='logloss',
                            verbosity=0, random_state=42,
                        )
                        model.fit(np.array(X_train, dtype=np.float32), np.array(y_train))
                        # 如果是校准模型，用训练集的最后20%做校准
                        if 'calibrated' in model_path:
                            from sklearn.calibration import CalibratedClassifierCV
                            calib_split = int(len(X_train) * 0.8)
                            if calib_split > 50:
                                model.fit(np.array(X_train[:calib_split], dtype=np.float32), np.array(y_train[:calib_split]))
                                model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                                model.fit(np.array(X_train[calib_split:], dtype=np.float32), np.array(y_train[calib_split:]))
                        n_retrains += 1
                    except Exception as e:
                        current_idx += 1
                        continue
                else:
                    model = base_model  # fallback

            # 用当前模型预测下一段（到下次重训）
            segment_end = min(current_idx + RETRAIN_EVERY, len(df) - 1)
            for i in range(current_idx, segment_end):
                feats = build_features(df, i, 60)
                if feats is None:
                    equity_history.append(equity_history[-1])
                    continue

                prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
                price = float(df.iloc[i]['close'])
                next_price = float(df.iloc[i + 1]['close'])

                if prob > 0.55:
                    target = 1
                elif prob < 0.45:
                    target = -1
                else:
                    target = 0

                # 简单策略：每天判断方向，有信号就持仓到反信号
                # 这里用逐日盯市算术
                pnl = 0
                if target == 1:
                    pnl = next_price - price
                elif target == -1:
                    pnl = price - next_price

                equity_history.append(equity_history[-1] + pnl)
                if pnl != 0:
                    all_trades.append({'entry': price, 'exit': next_price, 'pnl': pnl,
                                       'dir': 'LONG' if target == 1 else 'SHORT' if target == -1 else 'FLAT'})

            current_idx = segment_end

        eq = np.array(equity_history)
        rets = np.diff(eq) / eq[:-1]
        rets = rets[rets != 0]

        win_t = [t for t in all_trades if t['pnl'] > 0]
        loss_t = [t for t in all_trades if t['pnl'] < 0]
        total_ret = (eq[-1] - INITIAL_CASH) / INITIAL_CASH
        max_eq = np.maximum.accumulate(eq)
        dd = (eq - max_eq) / max_eq
        max_dd = abs(dd.min())
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 0 and np.std(rets) > 0 else 0

        win_rate = len(win_t) / len(all_trades) if all_trades else 0
        avg_w = np.mean([t['pnl'] for t in win_t]) if win_t else 0
        avg_l = np.mean([t['pnl'] for t in loss_t]) if loss_t else 0
        pf = abs(sum(t['pnl'] for t in win_t) / sum(t['pnl'] for t in loss_t)) if loss_t and sum(t['pnl'] for t in loss_t) != 0 else 999

        print(f"  总收益: {total_ret:+.1%} | 最终权益: ¥{eq[-1]:,.0f}")
        print(f"  Sharpe: {sharpe:.2f} | 最大回撤: {max_dd:.1%}")
        print(f"  交易次数: {len(all_trades)} | 胜率: {win_rate:.1%}")
        print(f"  平均盈利: {avg_w:+.0f} | 平均亏损: {avg_l:+.0f} | 盈亏比: {pf:.1f}")

        results[model_label] = {
            'total_return': round(total_ret, 4),
            'sharpe': round(sharpe, 2),
            'max_dd': round(max_dd, 4),
            'n_trades': len(all_trades),
            'win_rate': round(win_rate, 4),
            'avg_win': round(float(avg_w), 0),
            'avg_loss': round(float(avg_l), 0),
            'profit_factor': round(pf, 1),
        }

    # 对比
    if len(results) == 2:
        print(f"\n  📊 对比:")
        for metric, fmt in [('total_return', '{:+.1%}'), ('sharpe', '{:.2f}'), ('max_dd', '{:.1%}'),
                             ('win_rate', '{:.1%}'), ('n_trades', '{}'), ('profit_factor', '{:.1f}')]:
            v1 = results['未校准 (旧模型)'][metric]
            v2 = results['校准后 (Platt)'][metric]
            print(f"  {metric:<12} 未校准: {fmt.format(v1)}  →  校准后: {fmt.format(v2)}")

    return results


def main():
    print("🧪 Prophet Futures — Walk-Forward 回测")
    print(f"   策略: prob>0.55做多 / prob<0.45做空 / 中间观望")
    print(f"   训练窗口: {TRAIN_WINDOW}天 | 重训: 每{RETRAIN_EVERY}天 | 初始资金: ¥{INITIAL_CASH:,}")

    all_results = {}
    for sym_key in SYMBOLS:
        results = run_backtest(sym_key)
        if results:
            all_results[sym_key] = results

    print(f"\n{'='*60}")
    print("  回测完成")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    main()
