#!/usr/bin/env python3
"""Prophet Futures — 严格 Walk-Forward 回测
逐日滚动重训，只用过去数据做预测，模拟真实交易
"""
import sys, os
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import get_daily_history, build_features, SYMBOL_MAP

INITIAL_CASH = 300000
SYMBOLS = ['lh2609', 'jm2609']
MIN_TRAIN = 300
RETRAIN_EVERY = 20
WIN = 60


def walk_forward(sym_key, use_calibration=False):
    name = SYMBOL_MAP[sym_key]['name']

    df = get_daily_history(sym_key, 1200)
    if df is None or len(df) < MIN_TRAIN + 50:
        return None
    for c in ['open','high','low','close','volume','oi']:
        df[c] = df[c].apply(float)
    df = df.dropna(subset=['close']).reset_index(drop=True)

    cash = INITIAL_CASH
    equity = [cash]
    position = 0
    entry = 0
    stop_price = 0
    trades = []
    last_retrain = 0

    for today in range(MIN_TRAIN, len(df) - 1):
        # 重训
        if today - last_retrain >= RETRAIN_EVERY:
            X_train, y_train = [], []
            for i in range(WIN, today - 1):
                feats = build_features(df, i, WIN)
                if feats is None:
                    continue
                label = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
                X_train.append(feats)
                y_train.append(label)

            if len(X_train) < 100:
                if today == MIN_TRAIN:
                    return None
            else:
                try:
                    model = xgb.XGBClassifier(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        eval_metric='logloss', verbosity=0, random_state=42
                    )
                    if use_calibration and len(X_train) >= 200:
                        model = xgb.XGBClassifier(
                            n_estimators=100, max_depth=4, learning_rate=0.05,
                            eval_metric='logloss', verbosity=0, random_state=42
                        )
                        model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
                        model.fit(np.array(X_train, dtype=np.float32), np.array(y_train))
                    else:
                        model.fit(np.array(X_train, dtype=np.float32), np.array(y_train))
                    last_retrain = today
                except:
                    if today == MIN_TRAIN:
                        return None

        # 预测
        feats = build_features(df, today, WIN)
        if feats is None:
            equity.append(equity[-1])
            continue

        prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
        price = df.iloc[today]['close']
        next_price = df.iloc[today + 1]['close']

        # ATR
        atr_s = max(0, today - 20)
        atr_v = [abs(df.iloc[i]['high'] - df.iloc[i]['low']) for i in range(atr_s, today + 1)]
        atr = np.mean(atr_v) if atr_v else price * 0.005

        # 持仓管理
        if position != 0:
            if position == 1:
                trail = price - atr * 1.5
                if trail > stop_price:
                    stop_price = trail
                hit = price <= stop_price
            else:
                trail = price + atr * 1.5
                if trail < stop_price:
                    stop_price = trail
                hit = price >= stop_price

            reverse = (position == 1 and prob < 0.4) or (position == -1 and prob > 0.6)

            if hit or reverse:
                pnl = (price - entry) * position
                trades.append({'pnl': pnl})
                cash += pnl
                position = 0
                equity.append(cash)
                continue

            floating = (price - entry) * position
            equity.append(cash + floating)
            continue

        # 开仓
        if prob > 0.55:
            position = 1
            entry = price
            stop_price = price - atr * 1.5
            equity.append(cash)
        elif prob < 0.45:
            position = -1
            entry = price
            stop_price = price + atr * 1.5
            equity.append(cash)
        else:
            equity.append(cash)

    # 强制平仓
    if position != 0:
        last_price = df.iloc[-1]['close']
        pnl = (last_price - entry) * position
        trades.append({'pnl': pnl})

    eq = np.array(equity)
    total_ret = (eq[-1] - INITIAL_CASH) / INITIAL_CASH
    max_eq = np.maximum.accumulate(eq)
    dd = (eq - max_eq) / (max_eq + 1)
    max_dd = abs(dd.min())
    rets = np.diff(eq) / (eq[:-1] + 1)
    rets = rets[rets != 0]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0

    n = len(trades)
    win_t = [t for t in trades if t['pnl'] > 0]
    loss_t = [t for t in trades if t['pnl'] <= 0]
    wr = len(win_t) / n if n > 0 else 0
    aw = np.mean([t['pnl'] for t in win_t]) if win_t else 0
    al = np.mean([t['pnl'] for t in loss_t]) if loss_t else 0

    print(f"\n  {name} {'校准' if use_calibration else '未校准'}")
    print(f"  回测: {df.iloc[MIN_TRAIN]['date']} → {df.iloc[-1]['date']} ({len(df)-MIN_TRAIN}天)")
    print(f"  收益: {total_ret:+.1%} | DD: {max_dd:.1%} | Sharpe: {sharpe:.2f}")
    print(f"  交易: {n}笔 | 胜率: {wr:.1%} | 均盈: {aw:.0f} | 均亏: {al:.0f}")

    return {'total_return': total_ret, 'max_dd': max_dd, 'sharpe': sharpe,
            'n_trades': n, 'win_rate': wr}


def main():
    print("🧪 严格 Walk-Forward 回测")
    print(f"   逐日滚动重训 | 每{RETRAIN_EVERY}天重训 | 最少{MIN_TRAIN}天训练")
    print("   策略: prob>0.55做多 / <0.45做空 / 1.5ATR移动止损 / prob<0.4或>0.6反手")

    for sk in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"  {SYMBOL_MAP[sk]['name']}")
        walk_forward(sk, use_calibration=False)
        walk_forward(sk, use_calibration=True)

    print(f"\n{'='*50}")
    print("  注意: 回测≠实盘，过往不保证未来。")


if __name__ == '__main__':
    main()
