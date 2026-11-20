"""Logging for the server and tools: `YYYY-MM-DD HH:MM:SS.mmm LEVEL [tsumi] message` on stdout.
Level from TSUMI_LOG_LEVEL (default INFO; DEBUG adds per-layer loading and per-request details)."""
import logging
import sys

from .env import env


def get_logger(name="tsumi"):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(f"%(asctime)s.%(msecs)03d %(levelname)-5s [{name}] %(message)s",
                                         "%Y-%m-%d %H:%M:%S"))
        lg.addHandler(h)
        lg.setLevel(env("LOG_LEVEL", "INFO").upper())
        lg.propagate = False
    return lg
