from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# -----------------------------------------
# Paths
# -----------------------------------------

# Go up 2 levels: core/ -> orchestrator/ -> project root
BASE_DIR = Path(__file__).resolve().parents[2]

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "orchestrator.log"


def _setup_root_logger() -> logging.Logger:
    """
    Create root 'orchestrator' logger with:
    - File handler (logs/orchestrator.log, rotated)
    - Console handler
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers twice if this is imported multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with rotation (5MB, keep 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create root logger once
_root_logger = _setup_root_logger()


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Use this in any module:
        from orchestrator.core.logger import get_logger
        logger = get_logger(__name__)
    """
    if not name or name == "orchestrator":
        return _root_logger

    return _root_logger.getChild(name)


if __name__ == "__main__":
    # Quick test: run this file directly to see a log line.
    log = get_logger(__name__)
    log.info("Logger test: orchestrator logger is working.")
