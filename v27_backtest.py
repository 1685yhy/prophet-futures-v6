#!/usr/bin/env python3
"""Prophet Futures — V27 Formal Backtest
500 Walk-Forward trials per symbol
Standard 19-feature build_features() from paper_trader.py
V26 3-layer exit: hard stop + model reversal + trailing stop
V27: same-direction add-position with total lot cap
"""

import sys, os, pickle, time
import numpy as np
from datetime import datetime, timedelta
import akshare as ak
import pandas as pd

# ===== CONFIG (from paper_trader.py) =====
SYMBOLS = {
    'lh2609': {
        'code': 'LH0', 'name': 'LH', 'cost': 0.0006, 'multiplier': 16,
        'max_pos': 6, 'max_total_lots': 12,
        'hard_atr': 0.8, 'trail_atr': 2.0, 'be_atr': 1.0, 'trail_dist': 1.5,
        'model_low': 0.35, 'model_high': 0.65, 'confirm_bars': 2, 'min_hold': 3,
    },
    'jm2609': {
        'code': 'JM0', 'name': 'JM', 'cost': 0.0011, 'multiplier': 60,
        'max_pos': 4, 'max_total_lots': 8,
        'hard_atr': 1.8, 'trail_atr': 3.0, 'be_atr': 2.0, 'trail_dist': 2.5,
        'model_low': 0.30, 'model_high': 0.70, 'confirm_bars': 3, 'min_hold': 5,
    },
}

MODEL_DIR = '/home/a/prophet_futures/prophet_futures/models'
N_WF = 500
WARMUP = 80
FEATURE_WINDOW = 60


def build_features(df, idx, window=60):
    """EXACT COPY from paper_trader.py lines 77-99"""
    if idx < window + 5:
        return None
    w = df.iloc[idx - window:idx + 1]
    c = w['close'].values
    o = w['open'].values
    h = w['high'].values
    l = w['low'].values
    v = w['volume'].values
    oi = w['oi'].values
    f = []
    if idx >= 1:
        f.append((o[-1] - c[-2]) / c[-2])
        f.append(abs(f[-1]))
    else:
        f.extend([0, 0])
    for lag in [1, 3, 5, 10, 20]:
        f.append((c[-1] - c[-lag - 1]) / c[-lag - 1] if len(c) > lag else 0)
    for p in [5, 10, 20, 60]:
        ma = np.mean(c[-min(p, len(c)):])
        f.append((c[-1] - ma) / ma)
    f.append(np.std(c[-20:]) / np.mean(c[-20:]))
    f.append((h[-1] - l[-1]) / c[-1])
    vma = np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
    f.append(v[-1] / vma)
    f.append(oi[-1] / np.mean(oi[-20:]) if len(oi) >= 20 and np.mean(oi[-20:]) > 0 else 1)
    ema12 = c[-1]
    ema26 = c[-1]
    for j in range(len(c) - 2, -1, -1):
        ema12 = (2 / 13) * c[j] + (11 / 13) * ema12
        ema26 = (2 / 27) * c[j] + (25 / 27) * ema26
    f.append((ema12 - ema26) / c[-1])
    dd_ = np.diff(c[-15:])
    g = dd_[dd_ > 0].sum() if len(dd_[dd_ > 0]) > 0 else 0
    lo = abs(dd_[dd_ < 0].sum()) if len(dd_[dd_ < 0]) > 0 else 1e-10
    f.append(100 - 100 / (1 + g / lo) if lo > 0 else 50)
    bb = np.std(c[-20:])
    ma20 = np.mean(c[-20:])
    f.append((c[-1] - ma20) / (2 * bb + 1e-10))
    f.append(c[-1] / 1000.0)
    return np.array(f, dtype=np.float32)


def calc_atr(df, idx, period=20):
    if idx < period:
        return None
    vals = [
        abs(float(df.iloc[i]['high']) - float(df.iloc[i]['low']))
        for i in range(idx - period + 1, idx + 1)
    ]
    return np.mean(vals)


def fetch_history(code, days=1200):
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = ak.futures_main_sina(
            symbol=code,
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d'),
        )
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'oi', 'settle']
        for c in ['open', 'high', 'low', 'close', 'volume', 'oi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close']).reset_index(drop=True)
    except Exception as e:
        print(f"  Fetch error {code}: {e}")
        return None


def precompute_features(df):
    """Precompute all features for speed (from skill best practice)"""
    n = len(df)
    features = np.zeros((n, 19), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        f = build_features(df, i, FEATURE_WINDOW)
        if f is not None:
            features[i] = f
            valid[i] = True
    return features, valid


def run_v26_single(df, model, cfg, test_start, test_end):
    """V26: single position at a time"""
    trades = []
    pos = None
    rev_count = 0

    for i in range(test_start, test_end):
        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high'])
        low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, 20)
        if atr is None or price <= 0:
            continue

        # Exit existing position (V26 logic)
        if pos:
            d, entry, entry_i, vol, trail_stop, bar_count = pos
            bar_count += 1

            hard_stop_dist = atr * cfg['hard_atr']
            if d == 'LONG':
                hard_stop = price - hard_stop_dist
                # Trailing
                if bar_count >= cfg['min_hold'] and price > entry + atr * cfg['trail_atr']:
                    trail_stop = max(trail_stop, price - atr * cfg['trail_dist'])
                if bar_count >= cfg['min_hold'] and price > entry + atr * cfg['be_atr']:
                    trail_stop = max(trail_stop, entry)
                eff = max(hard_stop, trail_stop)
                exited = low <= eff
            else:
                hard_stop = price + hard_stop_dist
                if bar_count >= cfg['min_hold'] and price < entry - atr * cfg['trail_atr']:
                    trail_stop = min(trail_stop, price + atr * cfg['trail_dist'])
                if bar_count >= cfg['min_hold'] and price < entry - atr * cfg['be_atr']:
                    trail_stop = min(trail_stop, entry)
                eff = min(hard_stop, trail_stop)
                exited = high >= eff

            if exited:
                ep = eff
                pnl = ((ep - entry) / entry if d == 'LONG' else (entry - ep) / entry) - cfg['cost'] * 2
                trades.append({
                    'dir': d, 'entry': entry, 'exit': ep, 'pnl_pct': pnl,
                    'bars': bar_count, 'vol': vol,
                })
                pos = None
                rev_count = 0
            else:
                pos = (d, entry, entry_i, vol, trail_stop, bar_count)

        # Try new entry
        if pos is None:
            try:
                feats = build_features(df, i, FEATURE_WINDOW)
                if feats is None:
                    continue
                prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
            except:
                continue

            atr_pct = atr / price
            if atr_pct < 0.01:
                lev = 3.0
            elif atr_pct < 0.02:
                lev = 2.0
            elif atr_pct < 0.03:
                lev = 1.5
            elif atr_pct < 0.05:
                lev = 0.5
            else:
                lev = 0
            base = cfg['max_pos'] // 2
            size = max(1, int(lev * base)) if lev > 0 else 0

            if size > 0:
                sd = 'LONG' if prob > 0.5 else 'SHORT'
                if sd == 'LONG':
                    ts = price - atr * (cfg['hard_atr'] + 0.5)
                    already_hit = low <= (price - atr * cfg['hard_atr'])
                else:
                    ts = price + atr * (cfg['hard_atr'] + 0.5)
                    already_hit = high >= (price + atr * cfg['hard_atr'])
                if not already_hit:
                    pos = (sd, price, i, size, ts, 0)
                    rev_count = 0

    # EOD close
    if pos:
        d, entry, entry_i, vol, trail_stop, bar_count = pos
        lp = float(df.iloc[test_end - 1]['close'])
        pnl = ((lp - entry) / entry if d == 'LONG' else (entry - lp) / entry) - cfg['cost'] * 2
        trades.append({
            'dir': d, 'entry': entry, 'exit': lp, 'pnl_pct': pnl,
            'bars': bar_count + (test_end - 1 - entry_i), 'vol': vol,
        })

    return trades


def run_v27_addpos(df, model, cfg, test_start, test_end):
    """V27: multiple same-direction positions, independent management"""
    trades = []
    positions = []  # list of (dir, entry, entry_i, vol, trail_stop, bar_count)
    total_lots = 0
    max_total = cfg['max_total_lots']

    for i in range(test_start, test_end):
        price = float(df.iloc[i]['close'])
        high = float(df.iloc[i]['high'])
        low = float(df.iloc[i]['low'])
        atr = calc_atr(df, i, 20)
        if atr is None or price <= 0:
            continue

        # 1. Check exits for all existing positions
        surviving = []
        for pos in positions:
            d, entry, entry_i, vol, trail_stop, bar_count = pos
            bar_count += 1

            hard_stop_dist = atr * cfg['hard_atr']
            if d == 'LONG':
                hard_stop = price - hard_stop_dist
                if bar_count >= cfg['min_hold'] and price > entry + atr * cfg['trail_atr']:
                    trail_stop = max(trail_stop, price - atr * cfg['trail_dist'])
                if bar_count >= cfg['min_hold'] and price > entry + atr * cfg['be_atr']:
                    trail_stop = max(trail_stop, entry)
                eff = max(hard_stop, trail_stop)
                if low <= eff:
                    ep = eff
                    pnl = ((ep - entry) / entry) - cfg['cost'] * 2
                    trades.append({
                        'dir': d, 'entry': entry, 'exit': ep, 'pnl_pct': pnl,
                        'bars': bar_count, 'vol': vol,
                    })
                    total_lots -= vol
                else:
                    surviving.append((d, entry, entry_i, vol, trail_stop, bar_count))
            else:
                hard_stop = price + hard_stop_dist
                if bar_count >= cfg['min_hold'] and price < entry - atr * cfg['trail_atr']:
                    trail_stop = min(trail_stop, price + atr * cfg['trail_dist'])
                if bar_count >= cfg['min_hold'] and price < entry - atr * cfg['be_atr']:
                    trail_stop = min(trail_stop, entry)
                eff = min(hard_stop, trail_stop)
                if high >= eff:
                    ep = eff
                    pnl = ((entry - ep) / entry) - cfg['cost'] * 2
                    trades.append({
                        'dir': d, 'entry': entry, 'exit': ep, 'pnl_pct': pnl,
                        'bars': bar_count, 'vol': vol,
                    })
                    total_lots -= vol
                else:
                    surviving.append((d, entry, entry_i, vol, trail_stop, bar_count))
        positions = surviving

        # 2. Try new entry (V27: allow add to same direction)
        try:
            feats = build_features(df, i, FEATURE_WINDOW)
            if feats is None:
                continue
            prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
        except:
            continue

        atr_pct = atr / price
        if atr_pct < 0.01:
            lev = 3.0
        elif atr_pct < 0.02:
            lev = 2.0
        elif atr_pct < 0.03:
            lev = 1.5
        elif atr_pct < 0.05:
            lev = 0.5
        else:
            lev = 0
        base = cfg['max_pos'] // 2
        size = max(1, int(lev * base)) if lev > 0 else 0

        if size > 0 and total_lots + size <= max_total:
            sd = 'LONG' if prob > 0.5 else 'SHORT'

            # Only add if same direction as existing, or if no positions
            can_add = True
            if positions:
                existing_dir = positions[0][0]
                if sd != existing_dir:
                    can_add = False

            if can_add:
                if sd == 'LONG':
                    ts = price - atr * (cfg['hard_atr'] + 0.5)
                    already_hit = low <= (price - atr * cfg['hard_atr'])
                else:
                    ts = price + atr * (cfg['hard_atr'] + 0.5)
                    already_hit = high >= (price + atr * cfg['hard_atr'])
                if not already_hit:
                    positions.append((sd, price, i, size, ts, 0))
                    total_lots += size

    # EOD close all positions
    lp = float(df.iloc[test_end - 1]['close'])
    for pos in positions:
        d, entry, entry_i, vol, trail_stop, bar_count = pos
        pnl = ((lp - entry) / entry if d == 'LONG' else (entry - lp) / entry) - cfg['cost'] * 2
        trades.append({
            'dir': d, 'entry': entry, 'exit': lp, 'pnl_pct': pnl,
            'bars': bar_count + (test_end - 1 - entry_i), 'vol': vol,
        })

    return trades


def calc_stats(trades):
    """Calculate performance metrics for a set of trades"""
    if not trades:
        return {'trades': 0, 'wr': 0, 'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
                'pf': 0, 'max_dd': 0, 'avg_bars': 0}

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    wr = len(wins) / len(trades)

    total_pnl = sum(t['pnl_pct'] for t in trades)
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    avg_bars = np.mean([t['bars'] for t in trades])

    gw = sum(t['pnl_pct'] for t in wins)
    gl = abs(sum(t['pnl_pct'] for t in losses))
    pf = gw / gl if gl > 0 else 99

    # Cumulative equity for max DD
    eq = [1.0]
    for t in trades:
        eq.append(eq[-1] * (1 + t['pnl_pct']))
    peak = 1.0
    max_dd = 0
    for v in eq:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak)

    # Annualized return (compound)
    total_bars = trades[-1]['bars'] if trades else 1  # rough
    cumulative = np.prod([1 + t['pnl_pct'] for t in trades])
    years = max(total_bars, 1) / 252
    annualized = (cumulative ** (1 / years) - 1) if years > 0 else 0

    return {
        'trades': len(trades),
        'wr': wr,
        'total_pnl': total_pnl * 100,  # percentage
        'cumulative': cumulative,
        'annualized': annualized * 100,
        'avg_win': avg_win * 100,
        'avg_loss': avg_loss * 100,
        'pf': pf,
        'max_dd': max_dd * 100,
        'avg_bars': avg_bars,
        'total_vol': sum(t.get('vol', 1) for t in trades),
    }


def walk_forward_v26(df, features, valid, cfg):
    """500 WF trials: V26"""
    results = []
    n = len(df)

    for trial in range(N_WF):
        np.random.seed(trial)
        n_splits = np.random.randint(3, 6)
        split_points = sorted(
            np.random.choice(range(WARMUP + 200, n - 120), n_splits, replace=False)
        )

        for sp in range(len(split_points) - 1):
            train_end = split_points[sp]
            test_start = split_points[sp]
            test_end = split_points[sp + 1]

            tr_idx = [j for j in range(FEATURE_WINDOW + 5, train_end) if valid[j]]
            if len(tr_idx) < 50:
                continue

            X_tr = features[tr_idx]
            y_tr = np.zeros(len(tr_idx), dtype=int)
            for k, j in enumerate(tr_idx):
                future = df.iloc[j + 1]['close'] if j + 1 < n else df.iloc[j]['close']
                current = df.iloc[j]['close']
                y_tr[k] = 1 if future > current * 1.005 else 0

            if len(set(y_tr)) < 2:
                continue

            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                verbosity=0, random_state=42, n_jobs=4,
            )
            model.fit(X_tr, y_tr)

            trades = run_v26_single(df, model, cfg, test_start, test_end)
            if trades:
                results.append(calc_stats(trades))

        if (trial + 1) % 100 == 0:
            print(f"    V26 WF {trial + 1}/{N_WF}...")
    return results


def walk_forward_v27(df, features, valid, cfg):
    """500 WF trials: V27 (add-position)"""
    results = []
    n = len(df)

    for trial in range(N_WF):
        np.random.seed(trial)
        n_splits = np.random.randint(3, 6)
        split_points = sorted(
            np.random.choice(range(WARMUP + 200, n - 120), n_splits, replace=False)
        )

        for sp in range(len(split_points) - 1):
            train_end = split_points[sp]
            test_start = split_points[sp]
            test_end = split_points[sp + 1]

            tr_idx = [j for j in range(FEATURE_WINDOW + 5, train_end) if valid[j]]
            if len(tr_idx) < 50:
                continue

            X_tr = features[tr_idx]
            y_tr = np.zeros(len(tr_idx), dtype=int)
            for k, j in enumerate(tr_idx):
                future = df.iloc[j + 1]['close'] if j + 1 < n else df.iloc[j]['close']
                current = df.iloc[j]['close']
                y_tr[k] = 1 if future > current * 1.005 else 0

            if len(set(y_tr)) < 2:
                continue

            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                verbosity=0, random_state=42, n_jobs=4,
            )
            model.fit(X_tr, y_tr)

            trades = run_v27_addpos(df, model, cfg, test_start, test_end)
            if trades:
                results.append(calc_stats(trades))

        if (trial + 1) % 100 == 0:
            print(f"    V27 WF {trial + 1}/{N_WF}...")
    return results


def aggregate_results(results, label):
    """Aggregate 500 WF results into summary stats"""
    if not results:
        return {'label': label, 'count': 0}

    trades_list = [r['trades'] for r in results]
    wr_list = [r['wr'] for r in results]
    pnl_list = [r['total_pnl'] for r in results]
    dd_list = [r['max_dd'] for r in results]
    pf_list = [r['pf'] for r in results]
    ann_list = [r['annualized'] for r in results]

    return {
        'label': label,
        'trials': len(results),
        'trades_mean': np.mean(trades_list),
        'trades_total': sum(trades_list),
        'wr_mean': np.mean(wr_list) * 100,
        'wr_std': np.std(wr_list) * 100,
        'pnl_mean': np.mean(pnl_list),
        'pnl_std': np.std(pnl_list),
        'dd_mean': np.mean(dd_list),
        'dd_max': max(dd_list),
        'pf_mean': np.mean(pf_list),
        'ann_mean': np.mean(ann_list),
    }


def main():
    print("=" * 70)
    print("  Prophet Futures — V27 正式回测")
    print(f"  {N_WF}次Walk-Forward | 标准19维特征 | V26三层退出")
    print(f"  开始: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    for sym_key, cfg in SYMBOLS.items():
        print(f"\n{'#' * 70}")
        print(f"#  {sym_key.upper()} ({cfg['name']}) — {cfg['code']}")
        print(f"#  V27: 最大总手数={cfg['max_total_lots']} | 基础开仓={cfg['max_pos']}")
        print(f"{'#' * 70}")

        # 1. Fetch data
        print(f"\n  📡 Fetching {cfg['code']} 1200d...")
        df = fetch_history(cfg['code'], 1200)
        if df is None or len(df) < 200:
            print(f"  ❌ 数据不足: {len(df) if df is not None else 0}行")
            continue
        print(f"  ✅ {len(df)}行, {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")

        # 2. Precompute features
        print(f"  🔄 Precomputing features...")
        t0 = time.time()
        features, valid = precompute_features(df)
        valid_count = valid.sum()
        print(f"  ✅ {valid_count}/{len(df)} valid feature rows ({time.time() - t0:.1f}s)")

        # 3. V26 WF
        print(f"\n  🏃 V26 单一持仓 WF ({N_WF}次)...")
        t0 = time.time()
        v26_results = walk_forward_v26(df, features, valid, cfg)
        v26_agg = aggregate_results(v26_results, 'V26(单仓)')
        v26_time = time.time() - t0
        print(f"  ✅ V26 完成: {v26_agg['trials']}个test周期, {v26_agg['trades_total']}笔交易 ({v26_time:.0f}s)")

        # 4. V27 WF
        print(f"\n  🏃 V27 加仓 WF ({N_WF}次)...")
        t0 = time.time()
        v27_results = walk_forward_v27(df, features, valid, cfg)
        v27_agg = aggregate_results(v27_results, 'V27(加仓)')
        v27_time = time.time() - t0
        print(f"  ✅ V27 完成: {v27_agg['trials']}个test周期, {v27_agg['trades_total']}笔交易 ({v27_time:.0f}s)")

        # 5. Print comparison
        print(f"\n{'─' * 60}")
        print(f"  ⚡ V26 vs V27 对比 — {sym_key.upper()}")
        print(f"{'─' * 60}")
        print(f"  {'指标':<20} {'V26(单仓)':>15} {'V27(加仓)':>15} {'改善':>15}")
        print(f"  {'─' * 60}")
        print(f"  {'Test周期数':<20} {v26_agg['trials']:>15} {v27_agg['trials']:>15}")
        print(f"  {'总交易笔数':<20} {v26_agg['trades_total']:>15} {v27_agg['trades_total']:>15} {v27_agg['trades_total']-v26_agg['trades_total']:>+15}")
        print(f"  {'平均胜率':<20} {v26_agg['wr_mean']:>+14.1f}% {v27_agg['wr_mean']:>+14.1f}% {v27_agg['wr_mean']-v26_agg['wr_mean']:>+14.1f}pp")
        print(f"  {'平均收益(累积)':<20} {v26_agg['pnl_mean']:>+14.2f}% {v27_agg['pnl_mean']:>+14.2f}% {v27_agg['pnl_mean']-v26_agg['pnl_mean']:>+14.2f}%")
        print(f"  {'收益标准差':<20} {v26_agg['pnl_std']:>14.2f}% {v27_agg['pnl_std']:>14.2f}%")
        print(f"  {'平均年化':<20} {v26_agg['ann_mean']:>+14.1f}% {v27_agg['ann_mean']:>+14.1f}% {v27_agg['ann_mean']-v26_agg['ann_mean']:>+14.1f}%")
        print(f"  {'平均回撤':<20} {v26_agg['dd_mean']:>+14.1f}% {v27_agg['dd_mean']:>+14.1f}% {v27_agg['dd_mean']-v26_agg['dd_mean']:>+14.1f}%")
        print(f"  {'最大回撤':<20} {v26_agg['dd_max']:>+14.1f}% {v27_agg['dd_max']:>+14.1f}% {v27_agg['dd_max']-v26_agg['dd_max']:>+14.1f}%")
        print(f"  {'平均盈利因子':<20} {v26_agg['pf_mean']:>14.2f} {v27_agg['pf_mean']:>14.2f} {v27_agg['pf_mean']-v26_agg['pf_mean']:>+14.2f}")
        print(f"{'─' * 60}")

        # V27 vs V26 per-test scatter correlation
        if len(v26_results) == len(v27_results) and len(v26_results) > 10:
            v26_pnl = [r['total_pnl'] for r in v26_results]
            v27_pnl = [r['total_pnl'] for r in v27_results]
            corr = np.corrcoef(v26_pnl, v27_pnl)[0, 1]
            print(f"  V26-V27 收益相关性: {corr:.3f}")

            # V27 better % of tests
            better_count = sum(1 for a, b in zip(v26_pnl, v27_pnl) if b > a)
            print(f"  V27 优于 V26: {better_count}/{len(v26_pnl)} = {better_count/len(v26_pnl):.0%} of tests")

    print(f"\n{'=' * 70}")
    print(f"  全部回测完成 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  V27 = V26同逻辑 + 允许同向加仓(上限LH=12手 JM=8手)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    os.chdir('/home/a/prophet_futures/prophet_futures')
    main()
