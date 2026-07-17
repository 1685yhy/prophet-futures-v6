#!/bin/bash
# 盘中守护：每分钟检查所有纸盘，死了立刻重启
cd /home/a/prophet_futures/prophet_futures

TARGETS=(
  "paper_trader.py:V25:paper_trader.py"
  "paper_trader_v28.py:V28:paper_trader_v28.py"
  "paper_trader_v29.py:V29:paper_trader_v29.py"
  "paper_trader_v30.py:V30:paper_trader_v30.py"
  "paper_trader_v31.py:V31:paper_trader_v31.py"
  "paper_trader_v32.py:V32:paper_trader_v32.py"
  "paper_trader_v32b.py:V32b:paper_trader_v32b.py"
  "paper_trader_v33.py:V33:paper_trader_v33.py"
  "paper_trader_v34.py:V34:paper_trader_v34.py"
  "paper_trader_v35.py:V35:paper_trader_v35.py"
  "paper_trader_v36.py:V36:paper_trader_v36.py"
)

for target in "${TARGETS[@]}"; do
  IFS=':' read -r file label pattern <<< "$target"
  alive=$(pgrep -fc "$pattern" 2>/dev/null)
  if [ "$alive" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] ⚠️ $label 挂了，重启..."
    nohup .venv/bin/python -u "$file" >> "/home/a/prophet_futures/logs/guard_${label}.log" 2>&1 &
  fi
done
