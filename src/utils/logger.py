"""
RunLogger utility (migrated from original eit_.py).
=====================================================

Provides structured logging to console and text log files for experiment runs.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
from typing import Optional


class RunLogger:
    """
    Structured logger for pipeline execution runs.
    """

    def __init__(self, log_dir: str = "logs", run_name: str = "eit_run") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{run_name}.log"

        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)

            # File handler
            fh = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

    def info(self, msg: str) -> None:
        """Log informational message to stdout and persistent file log."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message to stdout and persistent file log."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message to stdout and persistent file log."""
        self.logger.error(msg)
