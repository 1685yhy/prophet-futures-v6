#!/usr/bin/env python3
"""Prophet Futures — V28 vs V29 全量回测对比
使用实际模型文件，在全量历史上跑完整交易模拟
"""
import os, sys, pickle, time, json
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import akshare as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_data import build_features, get_daily_history, SYMBOL_MAP

CAPITAL = 300000
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

CONFIG = {
    'lh2609': {
        'multiplier': 16, 'cost': 0.0006, 'max_pos': 6, 'max_total': 12,
        'atr_stop': 1.5, 'rr': 4.0,
        'add_conf': 0.65, 'add_atr': 2.0,
        'reduce_conf': 0.55, 'reverse_conf': 0.35,
        'trail_atr': 2.0, 'be_atr': 1.0, 'min_hold': 3,
    },
    'jm2609': {
        'multiplier': 60, 'cost': 0.0011, 'max_pos': 4, 'max_total': 8,
        'atr_stop': 2.0, 'rr': 3.5,
        'add_conf': 0.65, 'add_atr': 2.5,
        'reduce_conf': 0.55, 'reverse_conf': 0.30,
        'trail_atr': 3.0, 'be_atr': 2.0, 'min_hold': 5,
    },
}


def calc_atr(df, idx, period=20):
    if idx < period: return None
    vals = [abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
            for i in range(idx - period + 1, idx + 1)]
    return np.mean(vals)


def calc_position_size(price, atr, cfg):
    atr_pct = atr / price if price > 0 else 0
    if atr_pct < 0.01: lev = 3.0
    elif atr_pct < 0.02: lev = 2.0
    elif atr_pct < 0.03: lev = 1.5
    elif atr_pct < 0.05: lev = 0.5
    else: lev = 0
    return max(1, int(lev * (cfg['max_pos'] // 2))) if lev > 0 else 0


def simulate(df, model, cfg, capital=CAPITAL):
    """V28 交易逻辑在回测中模拟"""
    trades = []
    cash = capital
    positions = []  # list of {dir, entry, trail_stop, entry_i, vol}
    equity_curve = [capital]
    warmup = 70
    total_vol = 0

    for i in range(warmup, len(df)):
        feats = build_features(df, i, 60)
        if feats is None: continue
        try:
            prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
        except:
            continue

        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high'])
        low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, 20)
        if atr is None or price <= 0: continue

        cur_dir = 'LONG' if prob > 0.5 else 'SHORT'
        conf = prob if prob > 0.5 else 1 - prob

        # ── 1. 管理现有持仓 ──
        surviving = []
        for pos in positions:
            d = pos['dir']; entry = pos['entry']
            trail = pos.get('trail_stop', 0)
            entry_i = pos['entry_i']; vol = pos['vol']
            bars = i - entry_i
            pnl_pct = (price - entry) / entry if d == 'LONG' else (entry - price) / entry
            pnl_atr = pnl_pct * entry / atr if atr > 0 else 0
            pnl_yuan = pnl_pct * entry * vol * cfg['multiplier']

            if d == 'LONG':
                hard_stop = price - atr * cfg['atr_stop']
                if pnl_atr > cfg['trail_atr']:
                    trail = max(trail, price - atr * (cfg['atr_stop'] - 0.3))
                if pnl_atr > cfg['be_atr']:
                    trail = max(trail, entry)
                effective_stop = max(hard_stop, trail)

                should_reduce = (cur_dir == 'LONG' and conf < cfg['reduce_conf']
                                 and bars >= cfg['min_hold'])
                should_reverse = (prob < cfg['reverse_conf'] and bars >= cfg['min_hold'])

                if low <= effective_stop:
                    ep = effective_stop
                    ret = (ep - entry) / entry - cfg['cost'] * 2
                    trades.append(
                        {'pnl_pct': ret, 'pnl_yuan': ret * entry * vol * cfg['multiplier'],
                         'bars': bars, 'type': 'STOP', 'vol': vol})
                    cash += entry * vol * cfg['multiplier'] * 0.15 + ret * entry * vol * cfg[
                        'multiplier']
                    total_vol -= vol
                elif should_reverse:
                    ret = pnl_pct - cfg['cost'] * 2
                    trades.append(
                        {'pnl_pct': ret, 'pnl_yuan': ret * entry * vol * cfg['multiplier'],
                         'bars': bars, 'type': 'REVERSE', 'vol': vol})
                    cash += entry * vol * cfg['multiplier'] * 0.15 + ret * entry * vol * cfg[
                        'multiplier']
                    total_vol -= vol
                elif should_reduce and vol > 1:
                    cut = vol // 2
                    ret = pnl_pct - cfg['cost']
                    trades.append(
                        {'pnl_pct': ret, 'pnl_yuan': ret * entry * cut * cfg['multiplier'],
                         'bars': bars, 'type': 'REDUCE', 'vol': cut})
                    cash += entry * cut * cfg['multiplier'] * 0.15 + ret * entry * cut * cfg[
                        'multiplier']
                    total_vol -= cut
                    surviving.append(
                        {'dir': d, 'entry': entry, 'trail_stop': trail, 'entry_i': entry_i,
                         'vol': vol - cut})
                else:
                    surviving.append(
                        {'dir': d, 'entry': entry, 'trail_stop': trail, 'entry_i': entry_i,
                         'vol': vol})
            else:  # SHORT
                hard_stop = price + atr * cfg['atr_stop']
                if -pnl_atr > cfg['trail_atr']:
                    trail = min(trail, price + atr * (cfg['atr_stop'] - 0.3))
                if -pnl_atr > cfg['be_atr']:
                    trail = min(trail, entry)
                effective_stop = min(hard_stop, trail)

                should_reduce = (cur_dir == 'SHORT' and conf < cfg['reduce_conf']
                                 and bars >= cfg['min_hold'])
                should_reverse = (prob > 1 - cfg['reverse_conf'] and bars >= cfg['min_hold'])

                if high >= effective_stop:
                    ep = effective_stop
                    ret = (entry - ep) / entry - cfg['cost'] * 2
                    trades.append(
                        {'pnl_pct': ret, 'pnl_yuan': ret * entry * vol * cfg['multiplier'],
                         'bars': bars, 'type': 'STOP', 'vol': vol})
                    cash += entry * vol * cfg['multiplier'] * 0.15 + ret * entry * vol * cfg[
                        'multiplier']
                    total_vol -= vol
                elif should_reverse:
                    ret = pnl_pct - cfg['cost'] * 2
                    trades.append(
                        {'pnl_pct': ret, 'pnl_yuan': ret * entry * vol * cfg['multiplier'],
                         'bars': bars, 'type': 'REVERSE', 'vol': vol})
                    cash += entry * vol * cfg['multiplier'] * 0.15 + ret * entry * vol * cfg[
                        'multiplier']
                    total_vol -= vol
                elif should_reduce and vol > 1:
                    cut = vol // 2
                    ret = pnl_pct - cfg['cost']
                    trades.append(
                        {'pnl_pct': ret, 'pnl_yuan': ret * entry * cut * cfg['multiplier'],
                         'bars': bars, 'type': 'REDUCE', 'vol': cut})
                    cash += entry * cut * cfg['multiplier'] * 0.15 + ret * entry * cut * cfg[
                        'multiplier']
                    total_vol -= cut
                    surviving.append(
                        {'dir': d, 'entry': entry, 'trail_stop': trail, 'entry_i': entry_i,
                         'vol': vol - cut})
                else:
                    surviving.append(
                        {'dir': d, 'entry': entry, 'trail_stop': trail, 'entry_i': entry_i,
                         'vol': vol})

        positions = surviving

        # ── 2. 开仓/加仓 ──
        ps = calc_position_size(price, atr, cfg)
        if ps > 0 and total_vol + ps <= cfg['max_total']:
            sd = conf > 0.5
            sd_dir = 'LONG' if prob > 0.5 else 'SHORT'
            stop_dist = atr * cfg['atr_stop']

            if not positions:
                # 新开仓
                margin = ps * price * cfg['multiplier'] * 0.15
                if margin <= cash * 0.8:
                    if sd_dir == 'LONG':
                        s_val = price - stop_dist
                        if low > s_val:
                            cash -= margin
                            positions.append(
                                {'dir': 'LONG', 'entry': price, 'trail_stop': s_val,
                                 'entry_i': i, 'vol': ps})
                            total_vol += ps
                    else:
                        s_val = price + stop_dist
                        if high < s_val:
                            cash -= margin
                            positions.append(
                                {'dir': 'SHORT', 'entry': price, 'trail_stop': s_val,
                                 'entry_i': i, 'vol': ps})
                            total_vol += ps
            else:
                existing_dir = positions[0]['dir']
                if sd_dir == existing_dir:
                    avg_entry = np.mean([p['entry'] for p in positions])
                    pnl_atr = (price - avg_entry) / atr if sd_dir == 'LONG' else (
                                                                                             avg_entry - price) / atr
                    if conf > cfg['add_conf'] and pnl_atr > cfg['add_atr']:
                        margin = ps * price * cfg['multiplier'] * 0.15
                        if margin <= cash * 0.8:
                            if sd_dir == 'LONG':
                                s_val = price - stop_dist
                                if low > s_val:
                                    cash -= margin
                                    positions.append(
                                        {'dir': 'LONG', 'entry': price, 'trail_stop': s_val,
                                         'entry_i': i, 'vol': ps})
                                    total_vol += ps
                            else:
                                s_val = price + stop_dist
                                if high < s_val:
                                    cash -= margin
                                    positions.append(
                                        {'dir': 'SHORT', 'entry': price, 'trail_stop': s_val,
                                         'entry_i': i, 'vol': ps})
                                    total_vol += ps

        # ── 权益记录 ──
        eq = cash
        for pos in positions:
            pnl = (price - pos['entry']) if pos['dir'] == 'LONG' else (pos['entry'] - price)
            eq += pnl * pos['vol'] * cfg['multiplier']
        equity_curve.append(eq)

    # EOD 平仓
    final_price = float(df.iloc[-1]['close'])
    for pos in positions:
        ret = ((final_price - pos['entry']) / pos['entry'] if pos['dir'] == 'LONG' else (
                                                                                                    pos[
                                                                                                        'entry'] - final_price) /
               pos['entry']) - cfg['cost'] * 2
        trades.append(
            {'pnl_pct': ret, 'pnl_yuan': ret * pos['entry'] * pos['vol'] * cfg['multiplier'],
             'bars': len(df) - 1 - pos['entry_i'], 'type': 'EOD', 'vol': pos['vol']})

    return trades, equity_curve


def analyze(trades, equity_curve, name):
    if not trades:
        return {'name': name, 'trades': 0, 'wr': 0, 'total_pnl_pct': 0, 'total_pnl_yuan': 0,
                'mdd_pct': 0, 'pf': 0, 'sharpe': 0, 'types': {}}, "无交易"

    wins = [t for t in trades if t['pnl_yuan'] > 0]
    losses = [t for t in trades if t['pnl_yuan'] <= 0]
    wr = len(wins) / len(trades) if trades else 0
    total_pnl_yuan = sum(t['pnl_yuan'] for t in trades)
    total_pnl_pct = sum(t['pnl_pct'] for t in trades) if trades else 0

    gw = sum(t['pnl_yuan'] for t in wins)
    gl = abs(sum(t['pnl_yuan'] for t in losses))
    pf = gw / gl if gl > 0 else 99

    eq = np.array(equity_curve)
    mdd = 0
    peak = eq[0]
    for v in eq:
        peak = max(peak, v)
        dd = (v - peak) / peak
        mdd = min(mdd, dd)

    # Sharpe (annualized, assume 252 trading days)
    returns = np.diff(eq) / eq[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    avg_win = np.mean([t['pnl_yuan'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl_yuan']) for t in losses]) if losses else 0
    avg_bars = np.mean([t['bars'] for t in trades])
    final_eq = eq[-1]
    total_return = (final_eq - CAPITAL) / CAPITAL

    types = {}
    for t in trades: types[t['type']] = types.get(t['type'], 0) + 1

    result = {
        'name': name, 'trades': len(trades), 'wr': wr,
        'total_pnl_yuan': total_pnl_yuan, 'total_pnl_pct': total_pnl_pct,
        'total_return': total_return,
        'final_equity': final_eq,
        'mdd_pct': mdd, 'pf': pf, 'sharpe': sharpe,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'avg_bars': avg_bars, 'types': types,
    }
    return result, ""


def print_report(r, model_label):
    """打印单个模型的回测报告"""
    print(f"  {'─' * 50}")
    print(f"  {r['name']} ({model_label})")
    print(f"  {'─' * 50}")
    print(f"  总交易: {r['trades']} 笔")
    print(f"  胜率: {r['wr']:.1%}")
    print(f"  总收益: ¥{r['total_pnl_yuan']:+,.0f}  ({r['total_return']:+.1%})")
    print(f"  最终权益: ¥{r['final_equity']:,.0f}")
    print(f"  最大回撤: {r['mdd_pct']:.1%}")
    print(f"  盈亏比: {r['pf']:.2f}")
    print(f"  年化夏普: {r['sharpe']:.2f}")
    print(f"  平均盈利: ¥{r['avg_win']:,.0f}  平均亏损: ¥{r['avg_loss']:,.0f}")
    print(f"  平均持仓: {r['avg_bars']:.0f} 根K线")
    if r.get('types'):
        types_str = ' '.join(f'{k}={v}' for k, v in sorted(r['types'].items()))
        print(f"  交易类型: {types_str}")


def main():
    print("=" * 65)
    print("  Prophet Futures — V28 vs V29 全量回测对比")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  本金: ¥{CAPITAL:,}")
    print("=" * 65)

    for sym_key in ['lh2609', 'jm2609']:
        cfg = CONFIG[sym_key]
        name = SYMBOL_MAP[sym_key]['name']

        # 加载模型
        old_path = os.path.join(MODEL_DIR, f'{sym_key}_xgb.pkl')
        new_path = os.path.join(MODEL_DIR, f'{sym_key}_xgb_new.pkl')

        old_model = pickle.load(open(old_path, 'rb')) if os.path.exists(old_path) else None
        new_model = pickle.load(open(new_path, 'rb')) if os.path.exists(new_path) else None

        if not old_model or not new_model:
            print(f"\n  ❌ {name} 模型缺失，跳过")
            continue

        # 加载数据
        print(f"\n{'=' * 65}")
        print(f"  {name} ({sym_key})")
        print(f"{'=' * 65}")
        print(f"  加载数据...")
        df = get_daily_history(sym_key, 1200)
        if df is None or len(df) < 100:
            print(f"  ❌ 数据不足")
            continue
        print(f"  ✅ {len(df)} 条 ({df.iloc[0]['date']} → {df.iloc[-1]['date']})")
        print(f"  价格范围: {float(df['close'].min()):.0f} → {float(df['close'].max()):.0f}")

        # 回测
        print(f"\n  回测中...")
        t0 = time.time()

        trades_old, eq_old = simulate(df, old_model, cfg)
        r_old, _ = analyze(trades_old, eq_old, f"V28(旧模型)")

        trades_new, eq_new = simulate(df, new_model, cfg)
        r_new, _ = analyze(trades_new, eq_new, f"V29(新模型)")

        elapsed = time.time() - t0
        print(f"  ✅ 回测完成 ({elapsed:.1f}s)")

        # 报告
        print()
        print_report(r_old, 'lh2609_xgb.pkl')
        print()
        print_report(r_new, 'lh2609_xgb_new.pkl')

        # 对比
        print(f"\n  {'─' * 50}")
        print(f"  📊 对比")
        print(f"  {'─' * 50}")
        if r_old['trades'] > 0 and r_new['trades'] > 0:
            better = 'V29(新) 🏆' if r_new['total_return'] > r_old['total_return'] else 'V28(旧) 🏆'
            print(f"  胜者: {better}")
            print(
                f"  收益差: {r_new['total_return'] - r_old['total_return']:+.1%}  (¥{r_new['total_pnl_yuan'] - r_old['total_pnl_yuan']:+,.0f})")
            print(f"  胜率差: {r_new['wr'] - r_old['wr']:+.1%}")
            print(f"  回撤差: {r_new['mdd_pct'] - r_old['mdd_pct']:+.1%}  (更小更好)")
            print(f"  夏普差: {r_new['sharpe'] - r_old['sharpe']:+.2f}")

    print(f"\n{'=' * 65}")
    print("  回测完成")


if __name__ == '__main__':
    main()
