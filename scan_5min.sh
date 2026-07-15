#!/bin/bash
# 5分钟扫描 — Cron wrapper
# WSL workaround: stub libcuda prevents glibc assertion crash from /usr/lib/wsl/lib/libcuda.so.1
cd /home/a/prophet_futures/prophet_futures
export LD_LIBRARY_PATH=/home/a/prophet_futures/prophet_futures/lib:$LD_LIBRARY_PATH
source .venv/bin/activate
python3 scan_5min.py
