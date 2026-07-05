#!/bin/bash
# Prophet Futures — Start All Paper Traders
set -e
cd /home/a/prophet_futures/prophet_futures
source .venv/bin/activate

pkill -f "paper_trader.py" 2>/dev/null || true
pkill -f "paper_trader_v28.py" 2>/dev/null || true
sleep 2

nohup python3 -u paper_trader.py > /home/a/prophet_futures/logs/paper_v25.log 2>&1 &
echo "V25 PID $!"
nohup python3 -u paper_trader_v28.py > /home/a/prophet_futures/logs/paper_v28.log 2>&1 &
echo "V28 PID $!"
echo "done"
