#!/usr/bin/env python3
"""
Prophet Futures Cognitive Trading System — Entry Point

Usage:
  python main.py --mode paper_trading [--symbol lh]
  python main.py --mode backtest --date 2025-06-01 [--backtest-days 180] [--symbols rb,lh,sc]
  python main.py --mode build_memory [--symbols rb,i,lh,jd,sc] [--start 20230101]
  python main.py --mode daily_update --symbol lh
  python main.py --mode daily_update --symbol lh --position SHORT,11910,12115,11295,2026-06-09
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging
from tools.llm_utils import load_config, check_llm_connectivity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prophet Futures Cognitive Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper_trading", "build_memory", "daily_update"],
        default="paper_trading",
        help="Operating mode",
    )
    parser.add_argument(
        "--position", default=None,
        help="当前持仓，格式: 方向,入场价,止损价,目标价,入场日期[,手数]  "
             "例: SHORT,11910,12115,11295,2026-06-09,13",
    )
    parser.add_argument("--date",   default=None, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--symbol", default=None, help="Override symbol (paper_trading)")
    parser.add_argument(
        "--symbols", default=None,
        help="Comma-separated symbols for backtest/build_memory (e.g. rb,lh,sc)",
    )
    parser.add_argument(
        "--backtest-days", type=int, default=180,
        help="Number of days for backtest window (default: 180)",
    )
    parser.add_argument(
        "--start", default="20230101",
        help="Start date for build_memory mode (YYYYMMDD, default: 20230101)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--no-log-file", action="store_true")
    return parser.parse_args()


def run_paper_trading(symbol_override=None):
    logger = logging.getLogger(__name__)
    logger.info("Starting PAPER TRADING mode")

    from graph.workflow import get_compiled_workflow
    from graph.state import TradingState
    from tools.memory_store import init_vector_db

    cfg = load_config()
    init_vector_db(cfg.get("advanced", {}).get("memory", {}).get("db_path", "./vector_db"))

    workflow = get_compiled_workflow()
    initial: TradingState = {
        "mode":          "paper_trading",
        "date":          datetime.now().strftime("%Y-%m-%d"),
        "candidates":    [symbol_override] if symbol_override else [],
        "errors":        [],
        "daily_summary": {},
        "final_output":  "",
    }

    final = workflow.invoke(initial)
    print("\n" + final.get("final_output", "No output generated"))

    if final.get("risk_order") and final["risk_order"].orders:
        o = final["risk_order"]
        print(f"\n订单: {len(o.orders)} 笔  最大亏损: {o.max_loss:.2f}  风险: {o.risk_pct:.2%}")
        for i, order in enumerate(o.orders, 1):
            print(f"  [{i}] {order.side} {order.quantity:.1f}手 {order.symbol} "
                  f"@ {order.price or 'MARKET'}")
    else:
        print("\n无交易信号 (WAIT 或执行未触发)")

    return final


def run_backtest(date, symbols, backtest_days):
    logger = logging.getLogger(__name__)
    logger.info("Starting BACKTEST mode: %s, days=%d, symbols=%s", date, backtest_days, symbols)

    from tools.backtest import run_backtest as bt
    from utils.portfolio_analytics import generate_trade_report

    result = bt(date=date, symbols=symbols, backtest_days=backtest_days)

    if "error" in result:
        print(f"回测失败: {result['error']}")
        return result

    print(generate_trade_report(result.get("trades", [])))
    print(f"\n回测区间: 截至 {result['date_range']}, {backtest_days} 天")
    print(f"品种: {', '.join(result['symbols'])}")
    print(f"总交易: {result['total_trades']} 笔")
    print(f"胜率:   {result['win_rate']:.1%}")
    print(f"盈亏比: {result['pl_ratio']:.2f}")
    print(f"夏普:   {result['sharpe_ratio']:.3f}")
    print(f"最大回撤: {result['max_drawdown_pct']:.2f}%")
    print(f"总收益: {result['total_pnl']:+,.2f} 元 ({result['total_return_pct']:+.2f}%)")

    if result.get("trades"):
        print(f"\n最近5笔交易:")
        for t in result["trades"][-5:]:
            print(f"  {t['symbol']} {t['direction']:5s} {t['entry_date']}→{t['exit_date']} "
                  f"PnL={t['pnl']:+.1f} ({t['reason']})")

    return result


def run_build_memory(symbols, start_date):
    logger = logging.getLogger(__name__)
    logger.info("Building historical memory: symbols=%s, start=%s", symbols, start_date)

    from tools.history_builder import build_historical_memory, get_memory_stats
    from tools.llm_utils import load_config

    cfg     = load_config()
    db_path = cfg.get("advanced", {}).get("memory", {}).get("db_path", "./vector_db")

    print(f"构建历史记忆库...")
    print(f"品种: {symbols}")
    print(f"起始: {start_date}  存储: {db_path}")
    print()

    count = build_historical_memory(
        symbols=symbols,
        start_date=start_date,
        end_date=datetime.now().strftime("%Y%m%d"),
        db_path=db_path,
    )

    stats = get_memory_stats(db_path)
    print(f"\n完成！写入 {count} 条新记录")
    print(f"记忆库总量: {stats['total_records']} 条")
    if stats.get("symbols"):
        print(f"已覆盖品种: {', '.join(stats['symbols'])}")
    return {"written": count, "stats": stats}


def run_daily_update(symbol: str = "lh", position_str: str = None):
    """
    每日更新模式：输出次日方向预测 + 持仓管理建议。

    Args:
        symbol:       品种代码（如 lh、jd、bu、ma）
        position_str: 持仓字符串，格式 "方向,入场价,止损价,目标价,入场日期"
                      例: "SHORT,11910,12115,11295,2026-06-09"
    """
    logger = logging.getLogger(__name__)
    from datetime import date as date_cls
    from tools.market_data import get_kline, get_realtime_quote
    from tools.indicators import calc_indicators
    from tools.next_day_predictor import predict_next_day
    from tools.position_manager import get_position_advice, format_position_report
    from tools.cycle_detector import get_lh_signal_conditions, get_generic_signal_conditions
    from tools.hog_fundamentals import get_hog_fundamentals, format_fundamentals_report
    from tools.backtest import get_lot_size
    import pandas as pd

    print()
    print(f"{'='*55}")
    print(f"  先知期货认知交易系统 — 每日更新")
    print(f"  {date_cls.today()}  品种: {symbol.upper()}")
    print(f"{'='*55}")
    print()

    # ── 拉取数据 ────────────────────────────────────────────────────────────
    kline = get_kline(symbol, "daily", 120)
    df = pd.DataFrame({
        "open":  kline.opens,  "high":   kline.highs,
        "low":   kline.lows,   "close":  kline.closes,
        "volume":kline.volumes,
    })
    if kline.open_interests:
        df["oi"] = kline.open_interests

    ind  = calc_indicators(df)
    atr  = ind["atr14"]
    cur  = ind["current_close"]

    try:
        q    = get_realtime_quote(symbol)
        cur  = q.last_price
    except Exception:
        pass

    prev_close = ind["prev_close"]

    # ── 基本面数据（生猪专项）────────────────────────────────────────────────
    fundamentals = None
    if symbol.lower() in ("lh",):
        try:
            fundamentals = get_hog_fundamentals()
        except Exception as e:
            logger.warning("基本面数据获取失败: %s", e)

    # ── 次日方向预测（含基本面）─────────────────────────────────────────────
    pred = predict_next_day(df, ind, fundamentals=fundamentals)

    # ── 趋势信号（含基本面）─────────────────────────────────────────────────
    if symbol.lower() in ("lh",):
        trend_sig = get_lh_signal_conditions(df, ind, fundamentals=fundamentals)
    else:
        trend_sig = get_generic_signal_conditions(symbol, df, ind)

    # ── 持仓解析 ────────────────────────────────────────────────────────────
    position = None
    if position_str:
        try:
            parts = [p.strip() for p in position_str.split(",")]
            position = {
                "direction":  parts[0].upper(),
                "entry":      float(parts[1]),
                "stop":       float(parts[2]),
                "target":     float(parts[3]),
                "entry_date": parts[4] if len(parts) > 4 else str(date_cls.today()),
                "qty":        float(parts[5]) if len(parts) > 5 else 1.0,
            }
        except Exception as e:
            logger.warning("持仓解析失败（格式: 方向,入场,止损,目标,日期）: %s", e)

    # ── 输出 ─────────────────────────────────────────────────────────────────
    # ── 基本面报告（生猪专项）────────────────────────────────────────────────
    if fundamentals:
        print(format_fundamentals_report(fundamentals))
        print()

    lot_size = get_lot_size(symbol)

    if position:
        advice = get_position_advice(position, cur, pred, ind, atr, prev_close,
                                     lot_size=lot_size)
        report = format_position_report(symbol.upper(), position, cur, pred, advice, {
            "cycle":  trend_sig.get("conditions", {}).get("cycle", "N/A"),
            "signal": trend_sig.get("signal", "WAIT"),
        })
        print(report)
    else:
        # 无持仓：只显示信号扫描
        dir_cn = {"UP": "偏多↑", "DOWN": "偏空↓", "NEUTRAL": "中性→"}.get(pred["direction"], "?")
        print(f"【今日行情（无持仓）】")
        print(f"  {symbol.upper()} 当前价: {cur:.0f}  ATR: {atr:.1f}")
        print()
        print(f"【次日方向预测】")
        print(f"  方向: {dir_cn}  置信度: {pred['confidence']:.0%}  得分: {pred['score']}/10")
        for sig in pred.get("key_signals", [])[:4]:
            print(f"    · {sig}")
        print(f"  关键支撑: {pred['support']:.0f}  关键压力: {pred['resistance']:.0f}")
        print(f"  {pred['action_advice']}")
        print()

    # ── 趋势信号状态 ────────────────────────────────────────────────────────
    print(f"【趋势信号扫描】")
    conds = trend_sig.get("conditions", {})
    sig   = trend_sig.get("signal", "WAIT")
    cycle = conds.get("cycle", "N/A")

    if sig != "WAIT":
        print(f"  ✅ {sig} 信号触发！置信度: {trend_sig.get('confidence', 0):.0%}")
        print(f"  建议: 入场方向={sig}，止损={trend_sig.get('stop_atr_mult',1.5)}×ATR，"
              f"目标={trend_sig.get('target_atr_mult',2.5)}×ATR，"
              f"持仓约{trend_sig.get('hold_days',5)}日")
    else:
        print(f"  ⏸ 当前无趋势入场信号（大周期: {cycle}）")
        missing = []
        if not conds.get("ma_bear") and not conds.get("ma_bull"):
            missing.append("均线排列不明确")
        if not conds.get("macd_neg") and not conds.get("macd_pos"):
            missing.append("MACD方向未确认")
        if conds.get("oi_trend") not in ("REDUCING","ACCUMULATING"):
            missing.append("OI趋势FLAT")
        if not conds.get("no_noise"):
            missing.append("换仓噪音期")
        if missing:
            print(f"  未满足条件: {' | '.join(missing)}")
    print()

    # ── 风险提示 ────────────────────────────────────────────────────────────
    risk = pred.get("risk_note", "")
    if risk and risk != "当前无特别风险提示":
        print(f"【⚠ 风险提示】")
        print(f"  {risk}")
        print()


def main():
    args   = parse_args()
    setup_logging(level=args.log_level, log_to_file=not args.no_log_file)
    logger = logging.getLogger(__name__)

    cfg = load_config()
    logger.info(
        "Prophet Futures starting | mode=%s | provider=%s | model=%s",
        args.mode,
        cfg.get("system", {}).get("llm_provider"),
        cfg.get("system", {}).get("llm_model"),
    )

    # LLM 连通性检查（paper_trading 模式才探针，其他模式不需要）
    if args.mode == "paper_trading":
        available = check_llm_connectivity()
        if available:
            print("LLM: 可用 ✓ — 将使用 AI 分析")
        else:
            print("LLM: 不可用 — 将使用规则 fallback（设置 ANTHROPIC_API_KEY 后可启用 AI）")

    # 解析品种列表
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = cfg.get("markets", {}).get("futures", ["rb", "i", "sc", "lh", "jd"])[:5]

    # 执行
    if args.mode == "paper_trading":
        result = run_paper_trading(symbol_override=args.symbol)
        sys.exit(0 if "error" not in result else 1)

    elif args.mode == "backtest":
        date = args.date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        result = run_backtest(date=date, symbols=symbols, backtest_days=args.backtest_days)
        sys.exit(0 if "error" not in result else 1)

    elif args.mode == "build_memory":
        result = run_build_memory(symbols=symbols, start_date=args.start)
        sys.exit(0)

    elif args.mode == "daily_update":
        symbol = args.symbol or "lh"
        run_daily_update(symbol=symbol, position_str=args.position)
        sys.exit(0)


if __name__ == "__main__":
    main()
