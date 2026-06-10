"""Logging configuration for Prophet Futures system."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> None:
    """Configure structured logging for the entire system."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"prophet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    for noisy in ["httpx", "httpcore", "urllib3", "anthropic", "openai", "langchain"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("Prophet Futures logging initialized (level=%s)", level)


# 兼容旧版调用
def setup_logger(name: str = "prophet_futures") -> logging.Logger:
    return logging.getLogger(name)
