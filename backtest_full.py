#!/usr/bin/env python3
"""Prophet Futures — 完整 Walk-Forward 回测
用全部历史数据，模拟 V25/V28/V29 的真实交易策略
"""
import sys, os, pickle
import numpy as np, pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import build_features, get_daily_history, SYMBOL_MAP

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
INITIAL_CASH = 300000
SYMBOLS = ['lh2609', 'jm2609']

# 策略参数
V25_PARAMS = {
    'lh2609': {'stop_mult': 0.8, 'rr': 4, 'max_pos': 6, 'multiplier': 16,
               'model_reverse_long': 0.35, 'model_reverse_short': 0.65,
               'trail_atr': 2.0, 'be_atr': 1.0, 'mg': 0.15},
    'jm2609': {'stop_mult': 1.8, 'rr': 3.5, 'max_pos': 4, 'multiplier': 60,
               'model_reverse_long': 0.30, 'model_reverse_short': 0.70,
               'trail_atr': 3.0, 'be_atr': 2.0, 'mg': 0.15},
}

V28_PARAMS = {
    'lh2609': {'stop_mult': 1.5, 'rr': 4, 'max_pos': 6, 'multiplier': 16,
               'add_conf': 0.65, 'add_atr': 2.0, 'reduce_conf': 0.55,
               'reverse_conf_l': 0.35, 'reverse_conf_s': 0.65,
               'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3, 'mg': 0.15},
    'jm2609': {'stop_mult': 2.0, 'rr': 3.5, 'max_pos': 4, 'multiplier': 60,
               'add_conf': 0.65, 'add_atr': 2.5, 'reduce_conf': 0.55,
               'reverse_conf_l': 0.30, 'reverse_conf_s': 0.70,
               'trail_atr': 3.0, 'be_atr': 2.0, 'min_hold': 5, 'mg': 0.15},
}


def compute_atr(df, idx, window=20):
    """计算 idx 处的 ATR"""
    start = max(0, idx - window)
    atr_vals = [abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
                for i in range(start, idx + 1)]
    return np.mean(atr_vals) if atr_vals else 0


def sim_v25(model, sym_key, df, start_idx, window=60):
    """V25 策略：固定止损 + 模型退出 + 移动止损"""
    cfg = V25_PARAMS[sym_key]
    cash = INITIAL_CASH
    equity = [cash]
    trades = []
    position = 0  # +/- number of lots
    entry = 0
    stop = 0
    tp = 0
    highest_profit = 0  # trailing stop tracking
    trail_activated = False
    be_activated = False

    for i in range(start_idx, len(df)):
        feats = build_features(df, i, window)
        price = float(df.iloc[i]['close'])
        atr = compute_atr(df, i, 20)
        if atr <= 0:
            atr = price * 0.005

        prob = float(model.predict_proba(feats.reshape(1, -1))[0][1]) if feats is not None else 0.5

        if position != 0:
            # 检查止损
            hit_stop = (position > 0 and price <= stop) or (position < 0 and price >= stop)
            # 检查模型退出
            model_exit = (position > 0 and prob < cfg['model_reverse_long']) or \
                         (position < 0 and prob > cfg['model_reverse_short'])

            if hit_stop or model_exit:
                pnl = (price - entry) * position * abs(position) * cfg['multiplier']
                cash += pnl
                trades.append({'exit_i': i, 'entry': entry, 'exit': price, 'pnl': pnl,
                               'dir': 'LONG' if position > 0 else 'SHORT', 'lots': abs(position),
                               'reason': 'stop' if hit_stop else 'model_exit'})
                position = 0
                highest_profit = 0
                trail_activated = False
                be_activated = False
                equity.append(cash)
                continue

            # 移动止损
            current_profit = (price - entry) * (1 if position > 0 else -1)
            if current_profit > highest_profit:
                highest_profit = current_profit

            if position > 0:
                if not be_activated and current_profit >= cfg['be_atr'] * atr * abs(position):
                    stop = max(stop, entry)
                    be_activated = True
                if not trail_activated and current_profit >= cfg['trail_atr'] * atr * abs(position):
                    trail_activated = True
                if trail_activated:
                    stop = max(stop, price - cfg['stop_mult'] * atr * 0.7)
            else:
                if not be_activated and current_profit >= cfg['be_atr'] * atr * abs(position):
                    stop = min(stop, entry)
                    be_activated = True
                if not trail_activated and current_profit >= cfg['trail_atr'] * atr * abs(position):
                    trail_activated = True
                if trail_activated:
                    stop = min(stop, price + cfg['stop_mult'] * atr * 0.7)

            # 盯市浮动
            floating = (price - entry) * position * abs(position) * cfg['multiplier']
            equity.append(cash + floating)
            continue

        # 无持仓：开仓条件
        if feats is not None and prob > 0.55:
            lots = max(1, cfg['max_pos'] // 2)
            entry = price
            stop = price - cfg['stop_mult'] * atr
            tp = price + (price - stop) * cfg['rr']
            margin = lots * price * cfg['multiplier'] * cfg['mg']
            if cash >= margin:
                cash -= margin
                position = lots
                highest_profit = 0
                be_activated = False
                trail_activated = False
                equity.append(cash)
            else:
                equity.append(cash)
        elif feats is not None and prob < 0.45:
            lots = max(1, cfg['max_pos'] // 2)
            entry = price
            stop = price + cfg['stop_mult'] * atr
            tp = price - (stop - price) * cfg['rr']
            margin = lots * price * cfg['multiplier'] * cfg['mg']
            if cash >= margin:
                cash -= margin
                position = -lots
                highest_profit = 0
                be_activated = False
                trail_activated = False
                equity.append(cash)
            else:
                equity.append(cash)
        else:
            equity.append(cash)

    # 强制平仓
    if position != 0:
        last_price = float(df.iloc[-1]['close'])
        pnl = (last_price - entry) * position * abs(position) * cfg['multiplier']
        cash += pnl
        trades.append({'exit_i': len(df)-1, 'entry': entry, 'exit': last_price, 'pnl': pnl,
                       'dir': 'LONG' if position > 0 else 'SHORT', 'lots': abs(position),
                       'reason': 'force_close'})

    return calc_metrics(equity, trades)


def sim_v28(model, sym_key, df, start_idx, window=60):
    """V28 策略：动态加仓/减仓/反手"""
    cfg = V28_PARAMS[sym_key]
    cash = INITIAL_CASH
    equity = [cash]
    trades = []
    positions = []  # list of {'entry', 'lots', 'dir', 'trail_price'}

    for i in range(start_idx, len(df)):
        feats = build_features(df, i, window)
        price = float(df.iloc[i]['close'])
        atr = compute_atr(df, i, 20)
        if atr <= 0:
            atr = price * 0.005

        prob = float(model.predict_proba(feats.reshape(1, -1))[0][1]) if feats is not None else 0.5
        direction = 'LONG' if prob > 0.5 else 'SHORT'

        if positions:
            d = positions[0]['dir']
            total_lots = sum(p['lots'] for p in positions)
            avg_entry = sum(p['entry'] * p['lots'] for p in positions) / total_lots
            pnl_pts = (price - avg_entry) if d == 'LONG' else (avg_entry - price)
            pnl_atr = pnl_pts / atr if atr > 0 else 0

            # 反手条件
            reverse = (d == 'LONG' and prob < cfg['reverse_conf_l']) or \
                      (d == 'SHORT' and prob > cfg['reverse_conf_s'])

            if reverse:
                # 平掉所有
                for p in positions:
                    pnl = (price - p['entry']) * p['lots'] * cfg['multiplier'] if p['dir'] == 'LONG' else \
                          (p['entry'] - price) * p['lots'] * cfg['multiplier']
                    cash += pnl
                    trades.append({'exit_i': i, 'entry': p['entry'], 'exit': price, 'pnl': pnl,
                                   'dir': p['dir'], 'lots': p['lots'], 'reason': 'reverse'})
                positions = []
                equity.append(cash)
                continue

            # 减仓条件
            if d == direction and prob < cfg['reduce_conf'] and total_lots > 1:
                cut = total_lots // 2
                cut_lots = 0
                while cut_lots < cut and positions:
                    p = positions[0]
                    take = min(cut - cut_lots, p['lots'])
                    pnl = (price - p['entry']) * take * cfg['multiplier'] if p['dir'] == 'LONG' else \
                          (p['entry'] - price) * take * cfg['multiplier']
                    cash += pnl
                    trades.append({'exit_i': i, 'entry': p['entry'], 'exit': price, 'pnl': pnl,
                                   'dir': p['dir'], 'lots': take, 'reason': 'reduce'})
                    p['lots'] -= take
                    cut_lots += take
                    if p['lots'] == 0:
                        positions.pop(0)

            # 加仓条件
            if d == direction and prob > cfg['add_conf'] and pnl_atr > cfg['add_atr'] and total_lots < cfg['max_pos']:
                add = min(1, cfg['max_pos'] - total_lots)
                margin = add * price * cfg['multiplier'] * cfg['mg']
                if cash >= margin:
                    cash -= margin
                    positions.append({'entry': price, 'lots': add, 'dir': d, 'trail_price': price})

            # 移动止损（检查每个子仓）
            for p in positions:
                if p['dir'] == 'LONG':
                    trail = price - cfg['stop_mult'] * atr
                    if trail > p.get('trail_price', p['entry']):
                        p['trail_price'] = trail
                    if price <= p['trail_price']:
                        pnl = (price - p['entry']) * p['lots'] * cfg['multiplier']
                        cash += pnl
                        trades.append({'exit_i': i, 'entry': p['entry'], 'exit': price, 'pnl': pnl,
                                       'dir': p['dir'], 'lots': p['lots'], 'reason': 'trail_stop'})
                        p['lots'] = 0
                else:
                    trail = price + cfg['stop_mult'] * atr
                    if trail < p.get('trail_price', p['entry']):
                        p['trail_price'] = trail
                    if price >= p['trail_price']:
                        pnl = (p['entry'] - price) * p['lots'] * cfg['multiplier']
                        cash += pnl
                        trades.append({'exit_i': i, 'entry': p['entry'], 'exit': price, 'pnl': pnl,
                                       'dir': p['dir'], 'lots': p['lots'], 'reason': 'trail_stop'})
                        p['lots'] = 0
            positions = [p for p in positions if p['lots'] > 0]

            # 盯市
            floating = 0
            for p in positions:
                floating += (price - p['entry']) * p['lots'] * cfg['multiplier'] if p['dir'] == 'LONG' else \
                            (p['entry'] - price) * p['lots'] * cfg['multiplier']
            equity.append(cash + floating)
        else:
            # 无持仓：开仓
            if feats is not None and (prob > 0.55 or prob < 0.45):
                lots = max(1, cfg['max_pos'] // 2)
                d = 'LONG' if prob > 0.5 else 'SHORT'
                margin = lots * price * cfg['multiplier'] * cfg['mg']
                if cash >= margin:
                    cash -= margin
                    positions.append({'entry': price, 'lots': lots, 'dir': d, 'trail_price': price})
                equity.append(cash)
            else:
                equity.append(cash)

    # 强制平仓
    last_price = float(df.iloc[-1]['close'])
    for p in positions:
        pnl = (last_price - p['entry']) * p['lots'] * cfg['multiplier'] if p['dir'] == 'LONG' else \
              (p['entry'] - last_price) * p['lots'] * cfg['multiplier']
        cash += pnl
        trades.append({'exit_i': len(df)-1, 'entry': p['entry'], 'exit': last_price, 'pnl': pnl,
                       'dir': p['dir'], 'lots': p['lots'], 'reason': 'force_close'})

    return calc_metrics(equity, trades)


def calc_metrics(equity, trades):
    eq = np.array(equity)
    win_t = [t for t in trades if t['pnl'] > 0]
    loss_t = [t for t in trades if t['pnl'] <= 0]
    n = len(trades)

    total_ret = (eq[-1] - INITIAL_CASH) / INITIAL_CASH
    max_eq = np.maximum.accumulate(eq)
    dd = (eq - max_eq) / (max_eq + 1)
    max_dd = abs(dd.min())

    returns = np.diff(eq) / (eq[:-1] + 1)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    return {
        'total_return': round(total_ret, 4),
        'final_equity': round(eq[-1], 0),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 4),
        'n_trades': n,
        'win_rate': round(len(win_t) / n, 4) if n > 0 else 0,
        'avg_win': round(float(np.mean([t['pnl'] for t in win_t])), 0) if win_t else 0,
        'avg_loss': round(float(np.mean([t['pnl'] for t in loss_t])), 0) if loss_t else 0,
        'profit_factor': round(abs(sum(t['pnl'] for t in win_t) / sum(t['pnl'] for t in loss_t)), 1) if loss_t and sum(t['pnl'] for t in loss_t) != 0 else 999,
        'total_pnl': round(float(sum(t['pnl'] for t in trades)), 0),
    }


def run(sym_key):
    name = SYMBOL_MAP[sym_key]['name']
    print(f"\n{'='*60}")
    print(f"  {name} — 全版本回测")
    print(f"{'='*60}")

    df = get_daily_history(sym_key, 1200)
    if df is None or len(df) < 400:
        print("  数据不足")
        return None
    for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['close']).reset_index(drop=True)

    # 回测期：后40%数据
    test_start = int(len(df) * 0.6)
    print(f"  数据: {len(df)}条 ({df.iloc[0]['date']} → {df.iloc[-1]['date']})")
    print(f"  回测: {len(df)-test_start}天 ({df.iloc[test_start]['date']} → {df.iloc[-1]['date']})")

    results = {}

    # V25: 旧模型 + 固定策略
    mp = os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')
    if os.path.exists(mp):
        model = pickle.load(open(mp, 'rb'))
        m = sim_v25(model, sym_key, df, test_start)
        results['V25(旧模型)'] = m
        print(f"\n  V25 旧模型  : 收益{m['total_return']:+.1%} | DD{m['max_dd']:.1%} | "
              f"Sharpe{m['sharpe']:.2f} | 胜率{m['win_rate']:.1%} | {m['n_trades']}笔")

    # V28: 旧模型 + 动态策略
    if os.path.exists(mp):
        model = pickle.load(open(mp, 'rb'))
        m = sim_v28(model, sym_key, df, test_start)
        results['V28(旧+动态)'] = m
        print(f"  V28 旧+动态 : 收益{m['total_return']:+.1%} | DD{m['max_dd']:.1%} | "
              f"Sharpe{m['sharpe']:.2f} | 胜率{m['win_rate']:.1%} | {m['n_trades']}笔")

    # V29: 校准模型 + 动态策略
    mp_cal = os.path.join(MODEL_DIR, f'{sym_key}_xgb_calibrated.pkl')
    if os.path.exists(mp_cal):
        model_cal = pickle.load(open(mp_cal, 'rb'))
        m = sim_v28(model_cal, sym_key, df, test_start)
        results['V29(校准+动态)'] = m
        print(f"  V29 校准+动态: 收益{m['total_return']:+.1%} | DD{m['max_dd']:.1%} | "
              f"Sharpe{m['sharpe']:.2f} | 胜率{m['win_rate']:.1%} | {m['n_trades']}笔")

    return results


def main():
    print("🧪 Prophet Futures — 全版本 Walk-Forward 回测")
    print(f"   初始资金: ¥{INITIAL_CASH:,} | 回测期: 数据后40%")

    for sym_key in SYMBOLS:
        run(sym_key)

    print(f"\n{'='*60}")
    print("  回测完成。注意：回测≠实盘，过往表现不保证未来结果。")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
