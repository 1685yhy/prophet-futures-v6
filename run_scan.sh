#!/bin/bash
# Wrapper to run scan_5min.py working around glibc 2.39 ld.so bug
# The bug triggers when running from the project directory due to
# shared library loading order. Running from /tmp avoids the issue.

PROJECT_DIR="/home/a/prophet_futures/prophet_futures"
VENV_PYTHON="$PROJECT_DIR/.venv-system/bin/python3"
SCRIPT="$PROJECT_DIR/scan_5min.py"

cd /tmp && PYTHONPATH="$PROJECT_DIR" exec "$VENV_PYTHON" "$SCRIPT"
