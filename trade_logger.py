#!/usr/bin/env python3
"""
统一的交易日志记录器 — 所有版本共用。
写入 CSV: trades_master.csv
每笔交易记录版本、时间、品种、操作、方向、开仓价、平仓价、手数、盈亏、触发类型、触发详情、现金、权益。
"""
import csv, os
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trades_master.csv')

COLUMNS = [
    'version', 'time', 'symbol', 'action', 'dir', 'entry', 'exit', 'vol',
    'pnl', 'trigger', 'trigger_detail', 'cash', 'equity'
]


def log_trade(version, symbol, action, direction,
              entry, exit_price, vol, pnl,
              trigger, trigger_detail,
              cash, equity):
    """记录一笔交易到统一 CSV。

    Args:
        version:  版本号，如 'V28', 'V31'
        symbol:   品种，如 'lh2609', 'jm2609'
        action:   操作，'OPEN' / 'CLOSE' / 'ADD' / 'REDUCE'
        direction: 方向，'LONG' / 'SHORT'
        entry:    开仓价（OPEN/ADD 时填入场价，CLOSE/REDUCE 填原始开仓价）
        exit_price: 平仓价（OPEN/ADD 时填 0）
        vol:      手数
        pnl:      已实现盈亏（OPEN/ADD 时填 0）
        trigger:  触发类型，'SIGNAL' / 'STOP' / 'REVERSE' / 'REDUCE' / 'ADD' / 'MODEL' / 'TP' / 'TRAIL' / 'HARD'
        trigger_detail: 触发详情，如 'prob=0.34<0.35_rev_count=2/2'、'conf=0.45<0.55'、'pnl_atr=2.3>2.0&conf=0.72>0.65'
        cash:     交易后现金余额
        equity:   交易后总权益（含浮动盈亏）
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [
        version, ts, symbol, action, direction,
        round(entry, 1) if entry else 0,
        round(exit_price, 1) if exit_price else 0,
        vol,
        round(pnl, 2) if pnl else 0,
        trigger,
        trigger_detail,
        round(cash, 2),
        round(equity, 2),
    ]

    file_exists = os.path.exists(CSV_PATH)
    try:
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(COLUMNS)
            writer.writerow(row)
    except Exception as e:
        # 避免日志写入失败导致交易中断
        import sys
        print(f'[trade_logger ERROR] {e}', file=sys.stderr, flush=True)
